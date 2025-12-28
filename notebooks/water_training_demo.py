"""MC-3GS Training Demo with Water Molecules.

This marimo notebook demonstrates training MC-3GS on synthetic views
of water molecules. We use a simplified setup for fast experimentation.

To run: uv run marimo run notebooks/water_training_demo.py
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # MC-3GS Training Demo: Water Molecules

        This notebook demonstrates training **Molecule-Constrained Gaussian Splatting**
        on synthetic multi-view images of water molecules.

        ## Overview
        1. Generate synthetic training views by rendering water molecules from multiple angles
        2. Initialize a learnable water molecule scene
        3. Train the scene to match the target views
        4. Visualize the optimization progress

        **Note:** This uses a computationally light setup with:
        - Small image resolution (128x128)
        - Few training iterations (200)
        - Single water molecule
        """
    )
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return F, device, np, plt, torch


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. Create Water Molecule Template

        First, we create a water molecule template from SMILES notation.
        This gives us the 3D positions and shapes of atoms and bonds.
        """
    )
    return


@app.cell
def _(device, torch):
    from mc3gs.chemistry.rdkit_templates import create_template_from_smiles
    from mc3gs.chemistry.typing import TypeVocabulary
    from mc3gs.model import MoleculeInstance, MoleculeTemplate, Scene
    from mc3gs.render.sh import C0

    # Create water molecule template
    vocab = TypeVocabulary.default(include_bonds=True)

    template_dict = create_template_from_smiles(
        "O",  # Water SMILES
        vocab=vocab,
        include_hydrogens=True,
        include_bonds=True,
    )

    template = MoleculeTemplate.from_chemistry_template(
        template_dict,
        vocab=vocab,
        name="Water",
    )

    print(f"Water molecule: {template.num_gaussians} Gaussians")
    print(f"  - Atoms: {int((~template.is_bond_mask()).sum())}")
    print(f"  - Bonds: {int(template.is_bond_mask().sum())}")

    # Center the template
    template = template.centered()
    return (
        C0,
        MoleculeInstance,
        MoleculeTemplate,
        Scene,
        create_template_from_smiles,
        TypeVocabulary,
        template,
        template_dict,
        vocab,
    )


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Create Ground Truth Scene

        We create a "ground truth" water molecule with known colors and pose.
        This will be our target to train against.
        """
    )
    return


@app.cell
def _(MoleculeInstance, Scene, device, template, torch):
    # Create ground truth scene with specific colors
    gt_instance = MoleculeInstance(
        template,
        sh_degree=0,  # Just DC for simplicity
        init_position=torch.tensor([0.0, 0.0, 0.0]),
        init_scale=1.0,
        init_opacity=0.95,
    )

    # The instance already has correct CPK colors from initialization
    # (Oxygen=red, Hydrogen=white, Bonds=gray)

    gt_scene = Scene()
    gt_scene.add_instance(gt_instance)
    gt_scene.to(device)

    print("Ground truth scene created with CPK colors")
    return gt_instance, gt_scene


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Generate Synthetic Training Views

        We render the ground truth molecule from multiple camera angles
        to create our training dataset.
        """
    )
    return


@app.cell
def _(device, np, torch):
    # Camera setup - generate cameras orbiting around the molecule
    def create_orbit_cameras(
        n_cameras: int = 8,
        radius: float = 6.0,
        height: float = 2.0,
        width: int = 128,
        height_px: int = 128,
        fov: float = 60.0,
    ):
        """Create cameras orbiting around the origin."""
        cameras = []

        focal = 0.5 * width / np.tan(0.5 * np.radians(fov))
        K = torch.tensor(
            [[focal, 0, width / 2], [0, focal, height_px / 2], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )

        for i in range(n_cameras):
            angle = 2 * np.pi * i / n_cameras

            # Camera position in world coordinates
            cam_x = radius * np.cos(angle)
            cam_y = height
            cam_z = radius * np.sin(angle)
            cam_pos = np.array([cam_x, cam_y, cam_z])

            # Look at origin
            forward = -cam_pos / np.linalg.norm(cam_pos)
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # Rotation matrix (world to camera)
            R = np.stack([right, -up, forward], axis=0)

            # Translation
            t = -R @ cam_pos

            cameras.append(
                {
                    "K": K.clone(),
                    "R": torch.tensor(R, dtype=torch.float32, device=device),
                    "t": torch.tensor(t, dtype=torch.float32, device=device),
                    "center": torch.tensor(cam_pos, dtype=torch.float32, device=device),
                    "width": width,
                    "height": height_px,
                }
            )

        return cameras

    # Create training cameras (8 views around the molecule)
    train_cameras = create_orbit_cameras(n_cameras=8, width=128, height_px=128)
    print(f"Created {len(train_cameras)} training cameras")
    return create_orbit_cameras, train_cameras


@app.cell
def _(device, torch):
    from mc3gs.render.splat_renderer import ReferenceSplatRenderer

    # Initialize renderer
    renderer = ReferenceSplatRenderer(device=device)

    def render_scene(scene, camera, sh_degree=0):
        """Render a scene from a camera viewpoint."""
        data = scene.gather()

        result = renderer.render(
            positions=data["positions"],
            covariances=data["covariances"],
            opacities=data["opacities"],
            sh_coeffs=data["sh_coeffs"],
            K=camera["K"],
            R=camera["R"],
            t=camera["t"],
            camera_center=camera["center"],
            width=camera["width"],
            height=camera["height"],
            sh_degree=sh_degree,
            background=torch.ones(3, device=data["positions"].device) * 0.9,  # Light gray bg
        )

        return result["image"]

    print("Renderer initialized")
    return ReferenceSplatRenderer, render_scene, renderer


@app.cell
def _(gt_scene, np, plt, render_scene, train_cameras, torch):
    # Render ground truth images (detached - no gradients needed)
    gt_images = []
    with torch.no_grad():
        for cam in train_cameras:
            img = render_scene(gt_scene, cam, sh_degree=0)
            gt_images.append(img.detach().clone())

    # Visualize ground truth views
    fig1, axes1 = plt.subplots(2, 4, figsize=(12, 6))
    for i, (ax, img) in enumerate(zip(axes1.flat, gt_images)):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(np.clip(img_np, 0, 1))
        ax.set_title(f"View {i}")
        ax.axis("off")

    plt.suptitle("Ground Truth Training Views", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return axes1, fig1, gt_images, i


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Initialize Learnable Scene

        Now we create a new molecule instance with **perturbed parameters**.
        The training will optimize these to match the ground truth.
        """
    )
    return


@app.cell
def _(C0, MoleculeInstance, Scene, device, template, torch):
    # Create learnable scene with perturbed initialization
    learn_instance = MoleculeInstance(
        template,
        sh_degree=0,
        init_position=torch.tensor([0.3, -0.2, 0.1]),  # Wrong position
        init_scale=0.8,  # Wrong scale
        init_opacity=0.7,  # Wrong opacity
    )

    # Perturb the colors
    with torch.no_grad():
        # Add noise to atom colors
        learn_instance.atom_sh_bank.sh_coeffs.data += torch.randn_like(
            learn_instance.atom_sh_bank.sh_coeffs
        ) * 0.5
        # Perturb bond color
        learn_instance.bond_sh_bank.sh_coeffs.data += torch.randn_like(
            learn_instance.bond_sh_bank.sh_coeffs
        ) * 0.3

    learn_scene = Scene()
    learn_scene.add_instance(learn_instance)
    learn_scene.to(device)

    print("Learnable scene created with perturbed parameters")
    print(f"  Initial position: {learn_instance.translation.data}")
    print(f"  Initial scale: {learn_instance.scale.item():.3f}")
    return learn_instance, learn_scene


@app.cell
def _(gt_images, learn_scene, np, plt, render_scene, train_cameras):
    # Compare initial state
    fig2, axes2 = plt.subplots(2, 4, figsize=(12, 6))

    for i2, ax2 in enumerate(axes2.flat):
        if i2 < 4:
            # Ground truth
            img = gt_images[i2].permute(1, 2, 0).detach().cpu().numpy()
            ax2.imshow(np.clip(img, 0, 1))
            ax2.set_title(f"GT View {i2}")
        else:
            # Initial prediction
            pred = render_scene(learn_scene, train_cameras[i2 - 4], sh_degree=0)
            img = pred.permute(1, 2, 0).detach().cpu().numpy()
            ax2.imshow(np.clip(img, 0, 1))
            ax2.set_title(f"Init View {i2 - 4}")
        ax2.axis("off")

    plt.suptitle("Ground Truth (top) vs Initial Prediction (bottom)", fontsize=14)
    plt.tight_layout()
    plt.show()
    return ax2, axes2, fig2, i2, img, pred


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Training Loop

        Now we train the learnable scene to match the ground truth views
        using photometric loss (L2 + SSIM).
        """
    )
    return


@app.cell
def _(F, device, gt_images, learn_scene, render_scene, torch, train_cameras):
    from mc3gs.train.losses import ssim_loss

    # Training configuration
    n_iterations = 200
    lr = 0.02

    # Optimizer - optimize all scene parameters
    optimizer = torch.optim.Adam(learn_scene.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=lr * 0.1
    )

    # Training history
    history = {"loss": [], "psnr": [], "iteration": []}

    print(f"Training for {n_iterations} iterations...")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        total_loss = 0.0
        total_psnr = 0.0

        # Iterate over all views
        for cam_idx, camera in enumerate(train_cameras):
            # Render prediction
            pred_img = render_scene(learn_scene, camera, sh_degree=0)

            # Get ground truth
            gt_img = gt_images[cam_idx]

            # L2 loss
            l2 = F.mse_loss(pred_img, gt_img)

            # SSIM loss
            ssim = ssim_loss(pred_img.unsqueeze(0), gt_img.unsqueeze(0))

            # Combined loss
            loss = l2 + 0.2 * ssim
            total_loss += loss

            # PSNR for monitoring
            mse = F.mse_loss(pred_img, gt_img)
            psnr = -10 * torch.log10(mse + 1e-8)
            total_psnr += psnr.item()

        # Average over views
        total_loss = total_loss / len(train_cameras)
        avg_psnr = total_psnr / len(train_cameras)

        # Backprop
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Record history
        if iteration % 10 == 0 or iteration == n_iterations - 1:
            history["loss"].append(total_loss.item())
            history["psnr"].append(avg_psnr)
            history["iteration"].append(iteration)

            if iteration % 50 == 0:
                print(f"  Iter {iteration:4d}: Loss={total_loss.item():.4f}, PSNR={avg_psnr:.2f}dB")

    print(f"\nTraining complete!")
    print(f"  Final Loss: {history['loss'][-1]:.4f}")
    print(f"  Final PSNR: {history['psnr'][-1]:.2f}dB")
    return (
        avg_psnr,
        cam_idx,
        camera,
        gt_img,
        history,
        iteration,
        l2,
        loss,
        lr,
        mse,
        n_iterations,
        optimizer,
        pred_img,
        psnr,
        scheduler,
        ssim,
        ssim_loss,
        total_loss,
        total_psnr,
    )


@app.cell
def _(history, np, plt):
    # Plot training curves
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))

    ax3a.plot(history["iteration"], history["loss"], "b-", linewidth=2)
    ax3a.set_xlabel("Iteration")
    ax3a.set_ylabel("Loss")
    ax3a.set_title("Training Loss")
    ax3a.grid(True, alpha=0.3)

    ax3b.plot(history["iteration"], history["psnr"], "g-", linewidth=2)
    ax3b.set_xlabel("Iteration")
    ax3b.set_ylabel("PSNR (dB)")
    ax3b.set_title("PSNR (higher is better)")
    ax3b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return ax3a, ax3b, fig3


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Results

        Let's compare the final optimized renders with the ground truth.
        """
    )
    return


@app.cell
def _(gt_images, learn_scene, np, plt, render_scene, train_cameras, torch):
    # Final comparison
    fig4, axes4 = plt.subplots(3, 4, figsize=(12, 9))

    with torch.no_grad():
        for i4 in range(4):
            # Ground truth
            gt_np = gt_images[i4].permute(1, 2, 0).cpu().numpy()
            axes4[0, i4].imshow(np.clip(gt_np, 0, 1))
            axes4[0, i4].set_title(f"GT View {i4}")
            axes4[0, i4].axis("off")

            # Optimized prediction
            pred4 = render_scene(learn_scene, train_cameras[i4], sh_degree=0)
            pred_np = pred4.permute(1, 2, 0).cpu().numpy()
            axes4[1, i4].imshow(np.clip(pred_np, 0, 1))
            axes4[1, i4].set_title(f"Optimized View {i4}")
            axes4[1, i4].axis("off")

            # Difference (amplified for visibility)
            diff = np.abs(gt_np - pred_np) * 5
            axes4[2, i4].imshow(np.clip(diff, 0, 1))
            axes4[2, i4].set_title(f"Diff x5")
            axes4[2, i4].axis("off")

    plt.suptitle("Ground Truth vs Optimized (with Difference)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return axes4, diff, fig4, gt_np, i4, pred4, pred_np


@app.cell
def _(gt_instance, learn_instance, mo, torch):
    # Compare learned parameters
    with torch.no_grad():
        gt_pos = gt_instance.translation.data
        learn_pos = learn_instance.translation.data
        gt_scale = gt_instance.scale.item()
        learn_scale = learn_instance.scale.item()

    mo.md(
        f"""
        ## Learned Parameters

        | Parameter | Ground Truth | Learned | Error |
        |-----------|--------------|---------|-------|
        | Position X | {gt_pos[0]:.3f} | {learn_pos[0]:.3f} | {abs(gt_pos[0] - learn_pos[0]):.3f} |
        | Position Y | {gt_pos[1]:.3f} | {learn_pos[1]:.3f} | {abs(gt_pos[1] - learn_pos[1]):.3f} |
        | Position Z | {gt_pos[2]:.3f} | {learn_pos[2]:.3f} | {abs(gt_pos[2] - learn_pos[2]):.3f} |
        | Scale | {gt_scale:.3f} | {learn_scale:.3f} | {abs(gt_scale - learn_scale):.3f} |

        The optimization successfully recovered the molecule's pose and appearance
        from the synthetic training views!
        """
    )
    return gt_pos, gt_scale, learn_pos, learn_scale


@app.cell
def _(mo):
    mo.md(
        """
        ## Conclusion

        This demo showed how MC-3GS can optimize molecule parameters from multi-view images:

        1. **Constrained optimization**: Only pose, scale, opacity, and colors are learned
        2. **Structure preservation**: The molecular template (atom positions, bond geometry) is fixed
        3. **Efficient training**: With strong priors from chemistry, optimization converges quickly

        ### Next Steps
        - Use real images from the NeRF synthetic dataset format
        - Add more molecules to the scene
        - Increase SH degree for view-dependent effects
        - Use CUDA backend for faster rendering
        """
    )
    return


if __name__ == "__main__":
    app.run()
