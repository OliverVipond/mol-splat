# Molecule‑Constrained Gaussian Splatting (MC‑3GS)

A Python package plan for reconstructing 3D scenes using **molecule instances** composed of **Gaussian splats**, with the constraint:

* **Within a molecule**: all Gaussians of the same **atom type** or **bond type** share the same view‑dependent colour model.
* **Across molecules**: each molecule has its own colour scheme (independent parameters).

The goal is to retain the speed + differentiability of 3D Gaussian Splatting while enforcing semantic/stylistic coherence at the molecule level.

---

## 1. Scope and deliverables

### Deliverables

* `mc3gs` Python package (PyTorch) with:

  * Data pipeline (images → camera poses → initial scene)
  * Molecule templates (RDKit → local Gaussian templates)
  * Constrained scene model (molecule instances with parameter tying)
  * Differentiable Gaussian renderer interface
  * Training loop + checkpointing
  * Export (gaussians + molecule instance transforms + colour schemes)
  * Viewer hooks (optional)

### Non‑goals (initially)

* Full mesh extraction
* Physically based lighting / relighting
* Dynamic scenes

---

## 2. Mathematical formulation

### 2.1 Inputs

* RGB images: ({ I_k }_{k=1}^K), (I_k : \Omega \subset \mathbb{R}^2 \to \mathbb{R}^3)
* Cameras per image: intrinsics (K_k), extrinsics ((R_k,t_k))
* Projection: for world point (X\in\mathbb{R}^3)

[
X_{cam} = R_k X + t_k,\quad \tilde{u} = K_k X_{cam},\quad u = \Pi_k(X)=\left(\frac{\tilde{u}_x}{\tilde{u}_z}, \frac{\tilde{u}_y}{\tilde{u}_z}\right)\in\mathbb{R}^2.
]

### 2.2 Scene as molecule instances

We represent the scene as (M) molecule instances. Molecule instance (m) is a rigid transform applied to a **template** consisting of (N_m) Gaussian splats.

#### Template geometry (local frame)

For molecule (m), template provides for each splat (i\in{1,\dots,N_m}):

* Local mean: (p_{m,i}\in\mathbb{R}^3)
* Local covariance: (\Sigma^{\text{loc}}_{m,i}\in\mathbb{R}^{3\times 3})
* Type label: (\tau_{m,i}\in{1,\dots,T_m}), where types encode **atom/bond categories** (e.g., C, N, O, aromatic bond, single bond…)

Covariance parameterisation:
[
\Sigma^{\text{loc}}*{m,i} = R^{\text{loc}}*{m,i},\mathrm{diag}(s^{\text{loc}}*{m,i})^2,R^{\text{loc}\top}*{m,i}
]
with (R^{\text{loc}}\in SO(3)), (s^{\text{loc}}\in\mathbb{R}^3_{>0}).

#### Instance pose (world frame)

Each molecule instance has learnable pose:

* Rotation (Q_m\in SO(3))
* Translation (t_m\in\mathbb{R}^3)
* Optional uniform scale (\rho_m>0)

World‑space Gaussians are:
[
\mu_{m,i} = Q_m (\rho_m p_{m,i}) + t_m
]
[
\Sigma_{m,i} = Q_m,(\rho_m^2\Sigma^{\text{loc}}_{m,i}),Q_m^\top.
]

Opacity per splat (or per type) is (\alpha_{m,i}\in(0,1)).

### 2.3 Constrained radiance (per‑molecule, per‑type)

We model view‑dependent colour using spherical harmonics (SH) up to degree (L). Number of basis functions:
[
B=(L+1)^2.
]

For each molecule (m) and each type (t\in{1,\dots,T_m}), we learn SH coefficients:
[
\mathbf{c}_{m,t}\in\mathbb{R}^{B\times 3}.
]

For a given camera (k), define view direction at Gaussian ((m,i)):
[
\omega_{m,i,k}=\frac{C_k-\mu_{m,i}}{|C_k-\mu_{m,i}|}\in\mathbb{S}^2
]
where (C_k) is camera centre in world coords.

Evaluate SH basis vector (\mathbf{y}(\omega)\in\mathbb{R}^{B}). The RGB radiance for splat ((m,i)) is:
[
\mathrm{RGB}*{m,i,k}(\omega)=\mathbf{y}(\omega)^\top\mathbf{c}*{m,\tau_{m,i}}.
]

**Constraint encoded:** all splats with the same type label (\tau_{m,i}=t) share the same coefficients (\mathbf{c}_{m,t}) within molecule (m).

### 2.4 Projection of 3D Gaussians to 2D Gaussians

For rendering, each 3D Gaussian projects to a 2D Gaussian footprint using local linearisation.

Let (X\sim\mathcal{N}(\mu_{m,i},\Sigma_{m,i})). Define 2D mean:
[
\mu^{2D}*{m,i,k}=\Pi_k(\mu*{m,i}).
]
Let (J_{m,i,k}=\left.\frac{\partial\Pi_k(X)}{\partial X}\right|*{X=\mu*{m,i}}\in\mathbb{R}^{2\times 3}) be the Jacobian of projection at the mean. Then:
[
\Sigma^{2D}*{m,i,k}=J*{m,i,k},\Sigma_{m,i},J_{m,i,k}^\top + \epsilon I_2.
]

### 2.5 Per‑pixel contribution and compositing

At pixel centre (x\in\mathbb{R}^2), the kernel weight:
[
w_{m,i,k}(x)=\exp\left(-\tfrac12(x-\mu^{2D}*{m,i,k})^\top(\Sigma^{2D}*{m,i,k})^{-1}(x-\mu^{2D}*{m,i,k})\right).
]
Opacity contribution:
[
a*{m,i,k}(x)=\alpha_{m,i},w_{m,i,k}(x).
]

Front‑to‑back compositing (sorted by depth in camera (k)):
[
\hat{I}*k(x)=\sum*{j} \Big( T_j(x),a_j(x),\mathrm{RGB}*j(\omega) \Big),\quad T*{j+1}(x)=T_j(x)(1-a_j(x)),\quad T_0(x)=1.
]
Index (j) runs over all splats in depth order.

### 2.6 Optimisation objective

Given rendered images (\hat{I}*k), minimise photometric loss:
[
\mathcal{L}=\sum*{k\in\mathcal{B}}\Big(\lambda_2|\hat{I}_k-I_k|*2^2+\lambda*{ssim}(1-SSIM(\hat{I}*k,I_k))\Big)+\lambda*{reg},\mathcal{R}.
]
Regularisers (\mathcal{R}) recommended:

* Pose damping: (|t_m|^2), rotation magnitude penalty
* Opacity sparsity: (\sum\alpha_{m,i})
* Scale bounds: penalise extreme (\rho_m) and extreme covariance eigenvalues

Optimise parameters:
[
\Theta={Q_m,t_m,\rho_m}*{m=1}^M\cup{\alpha*{m,i}}\cup{\mathbf{c}_{m,t}}
]
using Adam.

---

## 3. Package architecture

### 3.1 Top‑level layout

```
mc3gs/
  __init__.py
  config/
    defaults.py
    schema.py
  data/
    images.py
    cameras.py
    colmap.py
  chemistry/
    rdkit_templates.py
    typing.py
    featurise.py
  model/
    templates.py
    molecule_instance.py
    scene.py
    constraints.py
  render/
    backend.py
    splat_renderer.py
    sh.py
    projection.py
  train/
    losses.py
    optim.py
    trainer.py
    checkpoints.py
  export/
    export_gaussians.py
    export_blender.py
    export_threejs.py
  cli/
    mc3gs_train.py
    mc3gs_export.py
  tests/
  examples/
```

### 3.2 Core data structures

#### `MoleculeTemplate`

* Created from RDKit or from cached template files
* Holds local Gaussian slots

Fields:

* `p_local: Tensor[N,3]`
* `Sigma_local: Tensor[N,3,3]` (or `(R_local, s_local)`)
* `type_id: LongTensor[N]` (atom/bond type per slot)
* `type_vocab: list[str]` mapping type_id → label

#### `MoleculeInstance (nn.Module)`

Learnable parameters:

* `rot: Parameter[3]` (axis‑angle) or quaternion
* `trans: Parameter[3]`
* `log_scale: Parameter[1]` (optional)
* `logit_opacity: Parameter[N]` (or per type)
* `sh_coeffs: Parameter[T, B, 3]`  **(key constraint)**

Methods:

* `world_gaussians() -> (mu[N,3], Sigma[N,3,3], type_id[N], opacity[N])`

#### `Scene (nn.Module)`

* Holds `ModuleList[MoleculeInstance]`
* `gather()` flattens all gaussians and returns indices into per‑molecule SH banks

---

## 4. Rendering backend design

### 4.1 Renderer interface

Define a backend abstraction to allow multiple renderers (fast CUDA vs slower PyTorch reference).

`render/backend.py`:

* `class RenderBackend(Protocol):`

  * `render(mu, Sigma, opacity, sh_index, sh_bank, cameras) -> images`

Inputs:

* `mu: [N,3]`
* `Sigma: [N,3,3]`
* `opacity: [N]`
* `sh_index: [N,2]` where row = `(mol_id, type_id)`
* `sh_bank: list[M] of [T_m, B, 3]`
* `cameras: batch of camera structs`

Key change vs vanilla 3DGS: the renderer gathers SH coefficients through `(mol_id, type_id)`.

### 4.2 SH evaluation module

`render/sh.py`:

* `eval_sh_basis(omega, L) -> y[B]`
* `shade(y[B], coeffs[B,3]) -> rgb[3]`

### 4.3 Projection + Jacobian

`render/projection.py`:

* `project(mu, K, R, t) -> (uv, depth)`
* `projection_jacobian(mu_cam, K) -> J[2,3]`

### 4.4 Reference renderer (correctness)

`render/splat_renderer.py` implements a slow but readable reference:

* compute `Sigma_2D = J Sigma J^T`
* rasterise ellipse bounding box
* compute `w(x)`
* alpha composite

Use for unit tests and debugging.

### 4.5 Fast renderer (production)

Wrap an existing 3DGS CUDA rasteriser:

* extend per‑Gaussian payload to include `mol_id` and `type_id`
* in the shading kernel, fetch `coeffs = sh_bank[mol_id][type_id]`

---

## 5. Chemistry: building templates from RDKit

### 5.1 Atom/bond typing

Define a discrete vocabulary for slot types:

* Atom: element + aromaticity + formal charge class (optional)

  * e.g. `C_sp2_arom`, `N_pos`, `O`, `S`, `F`, `Cl`, …
* Bond: single/double/triple/aromatic

Decide whether bonds are represented as their own Gaussian slots:

* **Atoms only** (simpler): each atom becomes 1–k Gaussians
* **Atoms + bonds** (more legible molecules): bonds become elongated Gaussians along bond axis

### 5.2 Template geometry generation

For each molecule:

* Generate 3D conformer (ETKDG)
* Atom centres: use conformer coordinates
* Atom covariance: spherical or slightly anisotropic; radius from VdW or learned
* Bond gaussians:

  * mean at bond midpoint
  * covariance elongated along bond axis

Outputs per template:

* `p_local[N,3]`
* `Sigma_local[N,3,3]`
* `type_id[N]`
* `type_vocab[T]`

Cache templates to disk (e.g. `.npz` or `.pt`).

---

## 6. Training pipeline

### 6.1 Camera poses

Options:

* Use COLMAP to recover (K_k,R_k,t_k) and sparse points.
* Alternatively accept given poses.

### 6.2 Scene initialisation

Need initial molecule placement and count. For artistic scenes, you control this.

Initialisation strategies:

1. **From SfM points**: sample sparse points, place molecule instances at those points.
2. **From depth / MVS**: place molecules on surfaces.
3. **Procedural**: place molecules on a 3D grid / shell and let optimisation sculpt (needs strong regularisation).

Initial pose:

* `t_m`: sampled from a point cloud / bounding box
* `Q_m`: random small rotations
* `rho_m`: fixed or small range

Initial SH coefficients per (molecule,type):

* Degree 0 initialised from nearby pixel colours (using visibility heuristics) or random palettes.

### 6.3 Losses

`train/losses.py`:

* `rgb_l2`
* `ssim`
* optional perceptual loss (later)

Regularisers:

* opacity sparsity
* pose smoothness / bounding
* overlap penalty (optional): discourage too many molecules occupying same region

### 6.4 Optimisation schedule

* Stage A (stabilise): optimise only `trans`, `rot`, opacity; SH degree 0 only
* Stage B (detail): enable higher SH degrees, enable scale
* Stage C (refine): optional splitting of Gaussians **within templates** (see below)

---

## 7. Handling refinement under molecule constraints

Vanilla 3DGS uses **splitting/pruning** per Gaussian. With molecule templates you have choices:

### 7.1 No splitting (simplest)

Keep templates fixed. Increase model capacity by adding more molecule instances.

### 7.2 Template‑level refinement

Allow each atom slot to be represented by multiple Gaussians (predefined):

* e.g. 3 Gaussians per atom arranged tetrahedrally
* bonds as 2–3 Gaussians along axis

### 7.3 Soft splitting inside molecule

Permit splitting but keep colour tying:

* split a slot into children that inherit the same `type_id`
* children share the same `sh_coeffs[m, type_id]`

This preserves the constraint while increasing spatial resolution.

Pruning:

* prune whole molecule instances whose total contribution stays tiny

---

## 8. Export and rendering recommendations

### 8.1 Exports

* `export_gaussians`: dump full flattened Gaussians (for 3DGS viewers)
* `export_instances`: dump molecule transforms + per‑type colour schemes

Recommended formats:

* `scene.json` with transforms and palettes
* `gaussians.ply` (if you want standard Gaussian viewers)

### 8.2 Viewers / renderers

Interactive:

* Three.js instancing for molecule meshes (atoms/bonds) using exported transforms
* Unity/Unreal instanced rendering for heavier scenes

Offline:

* Blender geometry nodes: instance molecule assets at exported transforms

Training / debug viewer:

* Use the same Gaussian renderer to preview training progress

---

## 9. Implementation plan (milestones)

### Milestone 0 — Skeleton

* Package layout, config schema, CLI entrypoints

### Milestone 1 — Templates

* RDKit conformer generation
* Atom/bond typing
* Template cache format + unit tests

### Milestone 2 — Constrained model

* `MoleculeInstance` + `Scene`
* Correct `gather()` with `(mol_id, type_id)` indexing

### Milestone 3 — Reference renderer

* Pure PyTorch renderer (slow) to validate maths and gradients
* Unit tests: projection Jacobian, SH evaluation, compositing invariants

### Milestone 4 — Fast backend

* Integrate an existing 3DGS CUDA rasteriser
* Modify shading to gather SH via indices

### Milestone 5 — Training loop

* Dataset loader, batching cameras
* Losses, regularisers, checkpointing
* Basic reconstruction demo on a small dataset

### Milestone 6 — Export + artistic pipeline

* Export transforms + palettes
* Three.js / Blender scripts for molecule instancing

---

## 10. Testing strategy

* Deterministic projection tests vs finite differences for Jacobian
* SH basis sanity tests (orthogonality spot checks, coefficient effects)
* Renderer tests on tiny scenes (1–3 gaussians) with known expected pixels
* Gradient flow tests:

  * verify shared SH parameters receive gradients from multiple gaussians of same type
  * verify different molecules’ SH banks remain independent

---

## 11. Notes on constraints and identifiability

Your constraint reduces radiance degrees of freedom. Expect:

* more burden on geometry/opacity to explain fine texture
* stronger stylisation (good)

To avoid degeneracies:

* clamp opacity and covariance eigenvalues
* add mild priors on molecule density and scale
* optionally start with SH degree 0, then increase degree

---

## Appendix A — Recommended parameterisations

### Rotation

Use quaternion or axis‑angle with exponential map.

* axis‑angle (r\in\mathbb{R}^3), (Q=\exp([r]_\times))

### Positive scales / opacities

* scales: (\rho=\exp(\log\rho))
* opacities: (\alpha=\sigma(\text{logit}\alpha))

---

## Appendix B — Minimal configuration example

* SH degree: 0–3
* Template type vocabulary: atom elements + bond classes
* Molecule count: user specified (art) or estimated from scene density
* Regularisation weights tuned for stability
