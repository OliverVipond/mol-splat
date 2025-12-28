"""MC-3GS Training Demo with Real Images from NeRF Dataset.

This marimo notebook demonstrates training MC-3GS using real multi-view
images from a NeRF synthetic dataset (transforms_train.json format).

We attempt to fit water molecules to the lego dataset as a demonstration
of the training pipeline with real image data.

To run: uv run marimo run notebooks/nerf_dataset_training_demo.py
"""

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # MC-3GS Training with NeRF Dataset Images

        This notebook demonstrates training **Molecule-Constrained Gaussian Splatting**
        using real multi-view images from a NeRF synthetic dataset.

        ## Overview
        1. Load camera poses and images from `transforms_train.json`
        2. Initialize a water molecule scene
        3. Train the scene to match the target views
        4. Visualize the optimization progress

        **Note:** This demo uses water molecules to fit to lego images - obviously
        this won't produce a perfect reconstruction! The goal is to demonstrate
        the training pipeline with real image data from NeRF datasets.

        **Computationally light setup:**
        - Downscaled images (0.25x = 200x200)
        - Only 8 training views (subset)
        - 100 training iterations
        """
    )
    return (mo,)


@app.cell
def _():
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return F, Image, Path, device, json, np, plt, torch


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. Load NeRF Dataset

        We load cameras and images from the `transforms_train.json` file.
        This is the standard format used by NeRF synthetic datasets.
        """
    )
    return


@app.cell
def _(Image, Path, device, json, np, torch):
    # Path to the dataset
    DATASET_PATH = Path("data/nerf_synthetic/lego")
    TRANSFORMS_FILE = DATASET_PATH / "transforms_train.json"

    # Load transforms
    with open(TRANSFORMS_FILE) as f:
        transforms_data = json.load(f)

    print(f"Loaded transforms from: {TRANSFORMS_FILE}")
    print(f"  Camera angle X: {transforms_data['camera_angle_x']:.4f} rad")
    print(f"  Number of frames: {len(transforms_data['frames'])}")

    # Parameters
    IMAGE_SCALE = 0.25  # Downscale for speed
    NUM_TRAIN_VIEWS = 8  # Use subset of views
    WHITE_BACKGROUND = True  # Blend alpha with white

    # Load a subset of cameras and images
    def load_nerf_data(transforms_data, dataset_path, scale=0.25, num_views=8, device="cpu"):
        """Load cameras and images from NeRF transforms.json format."""
        cameras = []
        images = []

        angle_x = transforms_data["camera_angle_x"]
        frames = transforms_data["frames"][:num_views]

        for frame in frames:
            # Load image
            file_path = frame["file_path"]
            img_path = dataset_path / f"{file_path}.png"

            img = Image.open(img_path)
            orig_w, orig_h = img.size

            # Resize
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Convert to numpy
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Handle alpha channel (blend with white background)
            if img_array.shape[-1] == 4:
                rgb = img_array[..., :3]
                alpha = img_array[..., 3:4]
                img_array = rgb * alpha + (1 - alpha)  # White background

            # To tensor [C, H, W]
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().to(device)
            images.append(img_tensor)

            # Compute camera intrinsics
            focal = 0.5 * new_w / np.tan(0.5 * angle_x)
            K = torch.tensor(
                [[focal, 0, new_w / 2], [0, focal, new_h / 2], [0, 0, 1]],
                dtype=torch.float32,
                device=device,
            )

            # Camera extrinsics from transform matrix
            c2w = np.array(frame["transform_matrix"])

            # Convert camera-to-world to world-to-camera
            # NeRF uses OpenGL convention (camera looks down -Z, Y up)
            # Our renderer uses OpenCV convention (camera looks down +Z, Y down)
            # Apply coordinate flip: flip Y and Z axes
            flip_yz = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            
            R_w2c = c2w[:3, :3].T  # World to camera rotation
            t_w2c = -c2w[:3, :3].T @ c2w[:3, 3]  # World to camera translation
            
            # Apply OpenGL to OpenCV conversion
            R = torch.tensor(flip_yz @ R_w2c, dtype=torch.float32, device=device)
            t = torch.tensor(flip_yz @ t_w2c, dtype=torch.float32, device=device)
            center = torch.tensor(c2w[:3, 3], dtype=torch.float32, device=device)

            cameras.append({
                "K": K,
                "R": R,
                "t": t,
                "center": center,
                "width": new_w,
                "height": new_h,
            })

        return cameras, images

    train_cameras, gt_images = load_nerf_data(
        transforms_data,
        DATASET_PATH,
        scale=IMAGE_SCALE,
        num_views=NUM_TRAIN_VIEWS,
        device=device,
    )

    print(f"\nLoaded {len(train_cameras)} training views")
    print(f"  Image size: {gt_images[0].shape[1]}x{gt_images[0].shape[2]}")
    return (
        DATASET_PATH,
        IMAGE_SCALE,
        NUM_TRAIN_VIEWS,
        TRANSFORMS_FILE,
        WHITE_BACKGROUND,
        gt_images,
        load_nerf_data,
        train_cameras,
        transforms_data,
    )


@app.cell
def _(gt_images, np, plt):
    # Visualize loaded ground truth images
    fig1, axes1 = plt.subplots(2, 4, figsize=(14, 7))

    for idx, ax in enumerate(axes1.flat):
        if idx < len(gt_images):
            img_np = gt_images[idx].permute(1, 2, 0).cpu().numpy()
            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(f"View {idx}")
        ax.axis("off")

    plt.suptitle("Ground Truth Training Views (from NeRF Lego Dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return axes1, fig1, idx


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. Create Molecule Template

        We create a water molecule template. Obviously water won't match
        the lego bulldozer, but this demonstrates the pipeline!

        For a real application, you would use a molecule that matches
        your scene (e.g., if imaging actual molecules).
        """
    )
    return


@app.cell
def _(device, torch):
    from mc3gs.chemistry.rdkit_templates import create_template_from_smiles
    from mc3gs.chemistry.typing import TypeVocabulary
    from mc3gs.model import MoleculeInstance, MoleculeTemplate, Scene

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

    # Scale the template to be visible in the scene
    # The lego scene is roughly centered at origin with radius ~4
    template = template.centered().scale(0.5)
    return (
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
        ## 3. Initialize Learnable Scene

        We create a scene with water molecules positioned in the scene.
        The optimization will adjust pose, scale, opacity, and colors
        to best match the target images.
        """
    )
    return


@app.cell
def _(MoleculeInstance, Scene, device, template, torch):
    # Create learnable scene with multiple water molecules
    learn_scene = Scene()

    # Add several water molecules at different positions
    positions = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.5, 0.0]),
        torch.tensor([-1.0, 0.0, 0.5]),
        torch.tensor([0.5, -0.5, -0.5]),
    ]

    instances = []
    for _i, _pos in enumerate(positions):
        instance = MoleculeInstance(
            template,
            sh_degree=0,
            init_position=_pos,
            init_scale=1.0,
            init_opacity=0.8,
        )
        instances.append(instance)
        learn_scene.add_instance(instance)

    learn_scene.to(device)

    print(f"Created scene with {len(instances)} water molecules")
    print(f"Total Gaussians: {learn_scene.total_gaussians}")
    return instance, instances, learn_scene, positions


@app.cell
def _(device, torch):
    from mc3gs.render.splat_renderer import ReferenceSplatRenderer

    # Initialize renderer
    renderer = ReferenceSplatRenderer(device=device)

    def render_scene(scene, camera, sh_degree=0, background_color=1.0):
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
            background=torch.ones(3, device=data["positions"].device) * background_color,
        )

        return result["image"]

    print("Renderer initialized")
    return ReferenceSplatRenderer, render_scene, renderer


@app.cell
def _(gt_images, learn_scene, np, plt, render_scene, torch, train_cameras):
    # Compare initial state with ground truth
    fig2, axes2 = plt.subplots(2, 4, figsize=(14, 7))

    with torch.no_grad():
        for i2 in range(4):
            # Ground truth
            gt_np = gt_images[i2].permute(1, 2, 0).cpu().numpy()
            axes2[0, i2].imshow(np.clip(gt_np, 0, 1))
            axes2[0, i2].set_title(f"GT View {i2}")
            axes2[0, i2].axis("off")

            # Initial prediction
            pred = render_scene(learn_scene, train_cameras[i2], sh_degree=0, background_color=1.0)
            pred_np = pred.permute(1, 2, 0).cpu().numpy()
            axes2[1, i2].imshow(np.clip(pred_np, 0, 1))
            axes2[1, i2].set_title(f"Init View {i2}")
            axes2[1, i2].axis("off")

    plt.suptitle("Ground Truth (top) vs Initial Prediction (bottom)", fontsize=14)
    plt.tight_layout()
    plt.show()
    return axes2, fig2, gt_np, i2, pred, pred_np


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Training Loop

        Now we train the learnable scene to match the ground truth views
        using photometric loss (L2 + SSIM).

        **Note:** Since water molecules can't represent a lego bulldozer,
        the optimization will just try to minimize the overall image error
        by adjusting colors and positions. This demonstrates the pipeline
        works with real images.
        """
    )
    return


@app.cell
def _(F, gt_images, learn_scene, render_scene, torch, train_cameras):
    from mc3gs.train.losses import ssim_loss

    # Training configuration
    n_iterations = 100
    lr = 0.05

    # Optimizer - optimize all scene parameters
    optimizer = torch.optim.Adam(learn_scene.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=lr * 0.1
    )

    # Training history
    history = {"loss": [], "psnr": [], "iteration": []}

    print(f"Training for {n_iterations} iterations on {len(train_cameras)} views...")
    print("(Note: Water molecules can't match lego, but this demonstrates the pipeline)\n")

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        total_loss = 0.0
        total_psnr = 0.0

        # Iterate over all views
        for cam_idx, camera in enumerate(train_cameras):
            # Render prediction (white background to match dataset)
            pred_img = render_scene(learn_scene, camera, sh_degree=0, background_color=1.0)

            # Get ground truth
            gt_img = gt_images[cam_idx]

            # L2 loss
            l2 = F.mse_loss(pred_img, gt_img)

            # SSIM loss (helps with structure)
            ssim = ssim_loss(pred_img.unsqueeze(0), gt_img.unsqueeze(0))

            # Combined loss
            loss = l2 + 0.2 * ssim
            total_loss += loss

            # PSNR for monitoring
            with torch.no_grad():
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
        if iteration % 5 == 0 or iteration == n_iterations - 1:
            history["loss"].append(total_loss.item())
            history["psnr"].append(avg_psnr)
            history["iteration"].append(iteration)

            if iteration % 20 == 0:
                print(f"  Iter {iteration:3d}: Loss={total_loss.item():.4f}, PSNR={avg_psnr:.2f}dB")

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
def _(history, plt):
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
        ## 5. Results

        Let's compare the final optimized renders with the ground truth.
        """
    )
    return


@app.cell
def _(gt_images, learn_scene, np, plt, render_scene, torch, train_cameras):
    # Final comparison
    fig4, axes4 = plt.subplots(3, 4, figsize=(14, 10))

    with torch.no_grad():
        for i4 in range(min(4, len(gt_images))):
            # Ground truth
            gt_np4 = gt_images[i4].permute(1, 2, 0).cpu().numpy()
            axes4[0, i4].imshow(np.clip(gt_np4, 0, 1))
            axes4[0, i4].set_title(f"GT View {i4}")
            axes4[0, i4].axis("off")

            # Optimized prediction
            pred4 = render_scene(learn_scene, train_cameras[i4], sh_degree=0, background_color=1.0)
            pred_np4 = pred4.permute(1, 2, 0).cpu().numpy()
            axes4[1, i4].imshow(np.clip(pred_np4, 0, 1))
            axes4[1, i4].set_title(f"Optimized View {i4}")
            axes4[1, i4].axis("off")

            # Difference (amplified for visibility)
            diff4 = np.abs(gt_np4 - pred_np4) * 3
            axes4[2, i4].imshow(np.clip(diff4, 0, 1))
            axes4[2, i4].set_title(f"Diff x3")
            axes4[2, i4].axis("off")

    plt.suptitle("Ground Truth vs Optimized (with Difference)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return axes4, diff4, fig4, gt_np4, i4, pred4, pred_np4


@app.cell
def _(instances, mo, torch):
    # Show learned parameters
    with torch.no_grad():
        params_table = "| Instance | Position | Scale | Avg Opacity |\n"
        params_table += "|----------|----------|-------|-------------|\n"

        for _idx5, _inst in enumerate(instances):
            _pos = _inst.translation.data
            _scale = _inst.scale.item()
            _opacity = _inst.opacity.mean().item()
            params_table += f"| {_idx5} | ({_pos[0]:.2f}, {_pos[1]:.2f}, {_pos[2]:.2f}) | {_scale:.3f} | {_opacity:.3f} |\n"

    mo.md(
        f"""
        ## Learned Parameters

        {params_table}

        The optimization adjusted the molecule positions, scales, and colors
        to minimize image reconstruction error.

        **Important:** Since water molecules cannot represent a lego bulldozer,
        the PSNR is low. In a real application, you would use molecules that
        match your actual scene (e.g., microscopy images of actual molecules).
        """
    )
    return (params_table,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 6. Save HTML Visualization

        Save an interactive HTML file displaying the fitted scene results.
        This can be opened in any browser to view the final renders.
        """
    )
    return


@app.cell
def _(Path, gt_images, history, learn_scene, np, render_scene, torch, train_cameras):
    import base64
    from datetime import datetime
    from io import BytesIO

    # Create output directory
    output_dir = Path("notebooks/example_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate rendered images as base64
    def tensor_to_base64(img_tensor):
        """Convert a [C, H, W] tensor to base64 PNG string."""
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_np)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Render all views
    rendered_views = []
    with torch.no_grad():
        for _cam_idx, _camera in enumerate(train_cameras):
            _gt = gt_images[_cam_idx]
            _pred = render_scene(learn_scene, _camera, sh_degree=0, background_color=1.0)
            rendered_views.append({
                "gt_b64": tensor_to_base64(_gt),
                "pred_b64": tensor_to_base64(_pred),
            })

    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MC-3GS Training Results - NeRF Dataset</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 2rem;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #00d4ff;
        }}
        .stat-label {{ color: #888; font-size: 0.9rem; }}
        .section-title {{
            font-size: 1.5rem;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}
        .views-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }}
        .view-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .view-header {{
            padding: 0.75rem 1rem;
            background: rgba(0,0,0,0.3);
            font-weight: 600;
        }}
        .view-images {{
            display: grid;
            grid-template-columns: 1fr 1fr;
        }}
        .view-images img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .img-label {{
            text-align: center;
            padding: 0.5rem;
            font-size: 0.8rem;
            color: #888;
            background: rgba(0,0,0,0.2);
        }}
        .footer {{
            text-align: center;
            margin-top: 3rem;
            color: #666;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 MC-3GS Training Results</h1>
        <p class="subtitle">Molecule-Constrained Gaussian Splatting on NeRF Lego Dataset</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(train_cameras)}</div>
                <div class="stat-label">Training Views</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{history["iteration"][-1] + 1}</div>
                <div class="stat-label">Iterations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{history["psnr"][-1]:.2f} dB</div>
                <div class="stat-label">Final PSNR</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{history["loss"][-1]:.4f}</div>
                <div class="stat-label">Final Loss</div>
            </div>
        </div>

        <h2 class="section-title">📸 Ground Truth vs Optimized Renders</h2>
        <div class="views-grid">
'''

    for _view_idx, _view_data in enumerate(rendered_views):
        html_content += f'''
            <div class="view-card">
                <div class="view-header">View {_view_idx}</div>
                <div class="view-images">
                    <div>
                        <img src="data:image/png;base64,{_view_data["gt_b64"]}" alt="Ground Truth {_view_idx}">
                        <div class="img-label">Ground Truth</div>
                    </div>
                    <div>
                        <img src="data:image/png;base64,{_view_data["pred_b64"]}" alt="Optimized {_view_idx}">
                        <div class="img-label">Optimized</div>
                    </div>
                </div>
            </div>
'''

    html_content += f'''
        </div>

        <div class="footer">
            <p>Generated by MC-3GS Training Demo • {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Note: Water molecules cannot fully represent the lego scene - this demonstrates the training pipeline.</p>
        </div>
    </div>
</body>
</html>
'''

    # Save HTML file
    html_path = output_dir / "nerf_training_results.html"
    with open(html_path, "w") as f:
        f.write(html_content)

    print(f"✓ Saved HTML visualization to: {html_path}")
    print(f"  Open in browser to view the results!")
    return BytesIO, base64, datetime, html_content, html_path, output_dir, rendered_views, tensor_to_base64


@app.cell
def _(mo):
    mo.md(
        """
        ## Conclusion

        This demo showed how to use MC-3GS with real images from NeRF datasets:

        1. **Data Loading**: Load cameras and images from `transforms_train.json`
        2. **Scene Setup**: Create molecule templates and initialize scene
        3. **Training**: Optimize molecule parameters using photometric loss
        4. **Evaluation**: Compare predictions with ground truth
        5. **Export**: Save HTML visualization for viewing in browser

        ### Output Files
        - `notebooks/example_outputs/nerf_training_results.html` - Interactive visualization

        ### For Real Applications
        - Use molecules that match your scene (e.g., actual molecular microscopy data)
        - Increase training iterations and views
        - Use higher SH degree for view-dependent effects
        - Use CUDA backend for faster rendering at higher resolutions
        """
    )
    return


if __name__ == "__main__":
    app.run()
