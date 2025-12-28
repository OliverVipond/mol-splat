"""Marimo notebook demonstrating single molecule rendering with MC-3GS.

Run with: marimo run notebooks/molecule_rendering_demo.py
Or edit with: marimo edit notebooks/molecule_rendering_demo.py
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # MC-3GS: Molecule Rendering Demo

        This notebook demonstrates how to render individual molecules using 
        **Molecule-Constrained Gaussian Splatting (MC-3GS)**.

        We'll render three molecules:
        1. **Water (H₂O)** - Simple molecule with 3 atoms
        2. **Benzene (C₆H₆)** - Aromatic ring structure  
        3. **Complex molecule** - A larger drug-like molecule

        The key constraint in MC-3GS is that:
        - **Atoms of the same type share color** (e.g., all carbons are the same color)
        - **All bonds within a molecule share a single color**
        - **Covariance (shape) is fixed** from the molecular template
        - **Only orientation, scale, and colors are learnable**
        """
    )
    return (mo,)


@app.cell
def _():
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.patches import Ellipse

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    return Ellipse, mpatches, np, plt, torch


@app.cell
def _(mo):
    mo.md("""
    ## 1. Import MC-3GS Components

    We need:
    - `MoleculeTemplate` - Defines the local geometry (positions, covariances, types)
    - `MoleculeInstance` - Learnable instance with pose and colors
    - `TypeVocabulary` - Mapping of atom/bond types to IDs
    - `create_template_from_smiles` - Generate template from SMILES string
    """)
    return


@app.cell
def _():
    from mc3gs.chemistry.rdkit_templates import create_template_from_smiles
    from mc3gs.chemistry.typing import ATOM_COLORS, TypeVocabulary
    from mc3gs.model import MoleculeInstance, MoleculeTemplate
    from mc3gs.render.sh import C0, shade_sh

    print("✓ MC-3GS components imported successfully")
    return (
        C0,
        MoleculeInstance,
        MoleculeTemplate,
        TypeVocabulary,
        create_template_from_smiles,
        shade_sh,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2. Define Molecules

    We'll create templates from SMILES strings for our three example molecules.
    """)
    return


@app.cell
def _():
    # Define molecules with their SMILES strings
    MOLECULES = {
        "Water": "O",
        "Benzene": "c1ccccc1",
        "Complex": "O=C(C1=C(N)[N@]([C@]2=C(C)C(O)=CC=C2C)C3=NC(C)=C(C)C=C31)N",
    }
    return (MOLECULES,)


@app.cell
def _(
    MOLECULES,
    MoleculeTemplate,
    TypeVocabulary,
    create_template_from_smiles,
    mo,
):
    # Create templates from SMILES
    templates = {}
    vocab = TypeVocabulary.default(include_bonds=True)

    for name, smiles in MOLECULES.items():
        try:
            # Get template dict from chemistry module
            template_dict = create_template_from_smiles(
                smiles,
                vocab=vocab,
                include_hydrogens=True,
                include_bonds=True,
            )
            # Convert to MoleculeTemplate
            template = MoleculeTemplate.from_chemistry_template(
                template_dict,
                vocab=vocab,
                name=name,
            )
            templates[name] = template
            mo.output.append(
                mo.md(
                    f"**{name}**: {template.num_gaussians} Gaussians "
                    f"({int((~template.is_bond_mask()).sum())} atoms, "
                    f"{int(template.is_bond_mask().sum())} bonds)"
                )
            )
        except Exception as e:
            mo.output.append(mo.md(f"**{name}**: Failed - {e}"))
    return (templates,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Create Molecule Instances

    Each instance has:
    - **Pose parameters**: translation, rotation, scale
    - **Atom SH bank**: Per-atom-type spherical harmonics coefficients
    - **Bond SH bank**: Single SH coefficient for all bonds
    """)
    return


@app.cell
def _(MoleculeInstance, mo, templates, torch):
    # Create instances with different poses
    instances = {}

    positions = {
        "Water": torch.tensor([0.0, 0.0, 0.0]),
        "Benzene": torch.tensor([5.0, 0.0, 0.0]),
        "Complex": torch.tensor([12.0, 0.0, 0.0]),
    }

    for _name, _template in templates.items():
        instance = MoleculeInstance(
            _template,
            sh_degree=0,  # DC only for simplicity
            init_position=positions[_name],
            init_scale=1.0,
            init_opacity=0.9,
        )
        instances[_name] = instance
        mo.output.append(
            mo.md(
                f"**{_name}** instance created:\n"
                f"- Atom SH params: {instance.atom_sh_bank.sh_coeffs.shape}\n"
                f"- Bond SH params: {instance.bond_sh_bank.sh_coeffs.shape}"
            )
        )
    return (instances,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Visualize Molecule Structure

    Let's visualize the Gaussian splats for each molecule.
    We'll show:
    - Atom positions as colored ellipses (based on atom type)
    - Bond positions as gray ellipses
    - The covariance determines the ellipse shape
    """)
    return


@app.cell
def _(C0, Ellipse, instances, mpatches, np, plt):
    def get_color_from_sh(sh_coeffs):
        """Convert SH DC coefficient to RGB color."""
        # DC component is sh_coeffs[0] * C0
        rgb = sh_coeffs[0] * C0  # sh_coeffs is already numpy
        return np.clip(rgb, 0, 1)

    def plot_molecule_2d(instance, ax, title):
        """Plot a molecule instance as 2D Gaussian ellipses."""
        gaussians = instance.world_gaussians()
        positions = gaussians["positions"].detach().numpy()
        covariances = gaussians["covariances"].detach().numpy()
        sh_coeffs = gaussians["sh_coeffs"].detach().numpy()
        is_bond = gaussians["is_bond"].numpy()

        # Project to XY plane
        for i in range(len(positions)):
            pos = positions[i]
            cov = covariances[i]

            # Get 2D covariance (XY projection)
            cov_2d = cov[:2, :2]

            # Eigendecomposition for ellipse
            eigvals, eigvecs = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            # Scale eigenvalues for visibility
            width = 2 * np.sqrt(eigvals[0]) * 2
            height = 2 * np.sqrt(eigvals[1]) * 2

            # Get color from SH coefficients
            color = get_color_from_sh(sh_coeffs[i])

            # Bonds are slightly transparent
            alpha = 0.6 if is_bond[i] else 0.9

            ellipse = Ellipse(
                (pos[0], pos[1]),
                width,
                height,
                angle=angle,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=alpha,
            )
            ax.add_patch(ellipse)

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (mol_name, mol_instance) in enumerate(instances.items()):
        plot_molecule_2d(mol_instance, axes[idx], mol_name)

        # Auto-scale axes
        mol_positions = mol_instance.world_positions().detach().numpy()
        margin = 2
        axes[idx].set_xlim(mol_positions[:, 0].min() - margin, mol_positions[:, 0].max() + margin)
        axes[idx].set_ylim(mol_positions[:, 1].min() - margin, mol_positions[:, 1].max() + margin)

    # Add legend
    atom_colors = {
        "Carbon": (0.2, 0.2, 0.2),
        "Nitrogen": (0.0, 0.0, 1.0),
        "Oxygen": (1.0, 0.0, 0.0),
        "Hydrogen": (1.0, 1.0, 1.0),
        "Bond": (0.3, 0.3, 0.3),
    }
    legend_patches = [
        mpatches.Patch(facecolor=color, label=label, edgecolor="black", linewidth=0.5)
        for label, color in atom_colors.items()
    ]
    fig.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Demonstrate Color Constraints

    Let's verify that the constraints are working:
    1. **Atoms of the same type share color** within a molecule
    2. **All bonds share a single color** within a molecule
    """)
    return


@app.cell
def _(instances, mo, torch):
    for _name, _instance in instances.items():
        gaussians = _instance.world_gaussians()
        sh = gaussians["sh_coeffs"]
        is_bond = gaussians["is_bond"]
        type_ids = gaussians["type_ids"]

        mo.output.append(mo.md(f"### {_name}"))

        # Check atom color sharing
        atom_indices = torch.where(~is_bond)[0]
        if len(atom_indices) > 1:
            atom_types = type_ids[atom_indices]
            unique_types = atom_types.unique()

            for t in unique_types:
                mask = atom_types == t
                indices_of_type = atom_indices[mask]
                if len(indices_of_type) > 1:
                    sh_same = all(
                        torch.allclose(sh[indices_of_type[0]], sh[i])
                        for i in indices_of_type[1:]
                    )
                    label = _instance.template.type_vocab.get_label(t.item())
                    mo.output.append(
                        mo.md(
                            f"- Atom type `{label}` ({len(indices_of_type)} atoms): "
                            f"Same color = **{sh_same}** ✓"
                        )
                    )

        # Check bond color sharing
        bond_indices = torch.where(is_bond)[0]
        if len(bond_indices) > 1:
            sh_bonds_same = all(
                torch.allclose(sh[bond_indices[0]], sh[i]) for i in bond_indices[1:]
            )
            mo.output.append(
                mo.md(
                    f"- Bonds ({len(bond_indices)} total): "
                    f"Same color = **{sh_bonds_same}** ✓"
                )
            )
        elif len(bond_indices) == 1:
            mo.output.append(mo.md(f"- Bonds (1 total): Single bond"))

        mo.output.append(mo.md("---"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Render with View-Dependent Colors

    Now let's render the molecules from a specific camera viewpoint and
    demonstrate the spherical harmonics shading.
    """)
    return


@app.cell
def _(Ellipse, instances, np, plt, shade_sh, torch):
    def render_molecule_with_sh(instance, camera_pos, ax, title):
        """Render molecule with view-dependent SH shading."""
        gaussians = instance.world_gaussians()
        positions = gaussians["positions"]
        covariances = gaussians["covariances"].detach().numpy()
        sh_coeffs = gaussians["sh_coeffs"]
        is_bond = gaussians["is_bond"].numpy()

        # Compute view directions
        view_dirs = camera_pos - positions
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)

        # Shade using SH
        colors = shade_sh(sh_coeffs, view_dirs, active_degree=0)
        colors = colors.detach().numpy()

        # Clamp colors to valid range
        colors = np.clip(colors, 0, 1)

        positions_np = positions.detach().numpy()

        for i in range(len(positions_np)):
            pos = positions_np[i]
            cov = covariances[i]
            cov_2d = cov[:2, :2]

            eigvals, eigvecs = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width = 2 * np.sqrt(eigvals[0]) * 2
            height = 2 * np.sqrt(eigvals[1]) * 2

            alpha = 0.6 if is_bond[i] else 0.9

            ellipse = Ellipse(
                (pos[0], pos[1]),
                width,
                height,
                angle=angle,
                facecolor=colors[i],
                edgecolor="black",
                linewidth=0.5,
                alpha=alpha,
            )
            ax.add_patch(ellipse)

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Render from different viewpoints
    camera_positions = [
        torch.tensor([0.0, 0.0, 10.0]),  # Front view
        torch.tensor([10.0, 0.0, 0.0]),  # Side view
        torch.tensor([5.0, 5.0, 10.0]),  # Angled view
    ]

    fig2, axes2 = plt.subplots(len(instances), 3, figsize=(15, 5 * len(instances)))

    for row, (mol_name2, instance2) in enumerate(instances.items()):
        for col, cam_pos in enumerate(camera_positions):
            ax2 = axes2[row, col] if len(instances) > 1 else axes2[col]
            view_name = ["Front", "Side", "Angled"][col]
            render_molecule_with_sh(instance2, cam_pos, ax2, f"{mol_name2} - {view_name}")

            positions2 = instance2.world_positions().detach().numpy()
            margin2 = 2
            ax2.set_xlim(positions2[:, 0].min() - margin2, positions2[:, 0].max() + margin2)
            ax2.set_ylim(positions2[:, 1].min() - margin2, positions2[:, 1].max() + margin2)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Summary of Learnable Parameters

    Let's count the total number of learnable parameters for each molecule instance.
    """)
    return


@app.cell
def _(instances, mo):
    mo.output.append(mo.md("| Molecule | Gaussians | Atoms | Bonds | Pose Params | Atom SH | Bond SH | Total |"))
    mo.output.append(mo.md("|----------|-----------|-------|-------|-------------|---------|---------|-------|"))

    for _name, _instance in instances.items():
        n_gaussians = _instance.template.num_gaussians
        n_atoms = int((~_instance._is_bond).sum())
        n_bonds = int(_instance._is_bond.sum())

        # Count parameters
        pose_params = 3 + 3 + 1 + n_gaussians  # translation + rotation + scale + opacities
        atom_sh_params = _instance.atom_sh_bank.sh_coeffs.numel()
        bond_sh_params = _instance.bond_sh_bank.sh_coeffs.numel()
        total = pose_params + atom_sh_params + bond_sh_params

        mo.output.append(
            mo.md(
                f"| {_name} | {n_gaussians} | {n_atoms} | {n_bonds} | "
                f"{pose_params} | {atom_sh_params} | {bond_sh_params} | **{total}** |"
            )
        )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Conclusion

    This demo showed how MC-3GS represents molecules as constrained Gaussian splats:

    1. **Template-based geometry**: Local positions and covariances come from the molecular structure
    2. **Type-based color sharing**: Atoms of the same element share SH coefficients
    3. **Single bond color**: All bonds in a molecule share one color
    4. **Efficient parameterization**: Constraints drastically reduce the number of learnable parameters

    This constraint-based approach enables efficient optimization while maintaining
    chemically meaningful structure in the learned representation.
    """)
    return


if __name__ == "__main__":
    app.run()
