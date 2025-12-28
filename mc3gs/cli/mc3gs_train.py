"""CLI entrypoint for training MC-3GS models."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="mc3gs-train",
    help="Train Molecule-Constrained Gaussian Splatting models.",
)
console = Console()


@app.command()
def train(
    images: Annotated[
        Path,
        typer.Option("--images", "-i", help="Path to input images directory"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("outputs"),
    colmap: Annotated[
        Optional[Path],
        typer.Option("--colmap", "-c", help="Path to COLMAP sparse reconstruction"),
    ] = None,
    molecules: Annotated[
        Optional[Path],
        typer.Option("--molecules", "-m", help="Path to molecule templates (SMILES file or directory)"),
    ] = None,
    num_molecules: Annotated[
        int,
        typer.Option("--num-molecules", "-n", help="Number of molecule instances"),
    ] = 100,
    iterations: Annotated[
        int,
        typer.Option("--iterations", help="Number of training iterations"),
    ] = 30000,
    sh_degree: Annotated[
        int,
        typer.Option("--sh-degree", help="Maximum SH degree (0-4)"),
    ] = 3,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device to train on"),
    ] = "cuda",
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume from checkpoint"),
    ] = True,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", help="Path to configuration file (YAML/JSON)"),
    ] = None,
) -> None:
    """Train a MC-3GS model on a set of images."""
    import torch

    from mc3gs.chemistry.rdkit_templates import create_template_from_smiles
    from mc3gs.chemistry.typing import TypeVocabulary
    from mc3gs.config.schema import DataConfig, MC3GSConfig
    from mc3gs.data.cameras import CameraDataset
    from mc3gs.data.colmap import load_colmap_cameras, load_colmap_points
    from mc3gs.data.images import ImageDataset
    from mc3gs.model.molecule_instance import MoleculeInstance
    from mc3gs.model.scene import Scene
    from mc3gs.model.templates import MoleculeTemplate
    from mc3gs.train.trainer import Trainer

    console.print("[bold]MC-3GS Training[/bold]")
    console.print(f"Images: {images}")
    console.print(f"Output: {output}")

    # Validate inputs
    if not images.exists():
        console.print(f"[red]Error: Images directory not found: {images}[/red]")
        raise typer.Exit(1)

    # Build configuration
    if config is not None:
        # Load from file
        import json
        with open(config) as f:
            config_data = json.load(f)
        mc_config = MC3GSConfig(**config_data)
    else:
        mc_config = MC3GSConfig(
            project_name=output.stem,
            data=DataConfig(images_path=images),
        )

    # Override with CLI args
    mc_config.train.num_iterations = iterations
    mc_config.model.sh_degree = sh_degree
    mc_config.train.device = device

    # Load cameras
    if colmap is not None:
        console.print(f"Loading cameras from COLMAP: {colmap}")
        cameras = load_colmap_cameras(colmap, images)
        points, colors = load_colmap_points(colmap)
    else:
        console.print("[yellow]No COLMAP path provided. Looking for transforms.json...[/yellow]")
        transforms_path = images / "transforms.json"
        if transforms_path.exists():
            cameras = CameraDataset.from_transforms_json(transforms_path)
            points = None
            colors = None
        else:
            console.print("[red]Error: No camera poses found. Provide COLMAP or transforms.json[/red]")
            raise typer.Exit(1)

    console.print(f"Loaded {len(cameras)} cameras")

    # Load images
    dataset = ImageDataset(
        cameras,
        scale=mc_config.data.image_scale,
        white_background=mc_config.data.white_background,
        device=device,
    )

    # Create molecule templates
    vocab = TypeVocabulary.default(include_bonds=mc_config.chemistry.include_bonds)

    if molecules is not None:
        # Load templates from file/directory
        console.print(f"Loading molecule templates from: {molecules}")
        templates = _load_templates(molecules, vocab, device)
    else:
        # Use default molecules (benzene, water, methane for demo)
        console.print("[yellow]No molecules specified, using default set[/yellow]")
        templates = _create_default_templates(vocab, device)

    console.print(f"Created {len(templates)} molecule templates")

    # Initialize scene
    console.print("Initializing scene...")
    scene = Scene()

    # Place molecules
    if points is not None:
        # Sample positions from point cloud
        indices = torch.randperm(len(points))[:num_molecules]
        init_positions = torch.from_numpy(points[indices]).float().to(device)
        init_colors = torch.from_numpy(colors[indices]).float().to(device) if colors is not None else None
    else:
        # Random positions in a bounding box
        init_positions = torch.randn(num_molecules, 3, device=device) * 2

    # Create molecule instances
    for i in range(num_molecules):
        template = templates[i % len(templates)]
        instance = MoleculeInstance(
            template=template,
            sh_degree=sh_degree,
            init_position=init_positions[i],
            init_opacity=mc_config.model.init_opacity,
        )
        scene.add_instance(instance)

    console.print(f"Scene: {len(scene)} molecules, {scene.total_gaussians} Gaussians")

    # Create trainer
    trainer = Trainer(
        scene=scene,
        dataset=dataset,
        config=mc_config,
        output_dir=output,
    )

    # Train
    try:
        metrics = trainer.train(resume=resume)
        console.print("[bold green]Training complete![/bold green]")
        console.print(f"Final PSNR: {metrics.get('psnr', 0):.2f}")
        console.print(f"Final Loss: {metrics.get('total', 0):.4f}")
    except KeyboardInterrupt:
        console.print("[yellow]Training interrupted[/yellow]")

    # Save final scene
    trainer.save()
    console.print(f"Scene saved to: {output}")


def _load_templates(path: Path, vocab, device: str) -> list:
    """Load molecule templates from file or directory."""
    from mc3gs.chemistry.rdkit_templates import create_template_from_smiles
    from mc3gs.model.templates import MoleculeTemplate

    templates = []

    if path.is_file():
        # Assume SMILES file (one per line)
        with open(path) as f:
            for line in f:
                smiles = line.strip()
                if smiles and not smiles.startswith("#"):
                    try:
                        template_dict = create_template_from_smiles(smiles, vocab)
                        template = MoleculeTemplate.from_chemistry_template(
                            template_dict, vocab, name=smiles, device=device
                        )
                        templates.append(template)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to load {smiles}: {e}[/yellow]")
    elif path.is_dir():
        # Load .pt files
        for pt_file in path.glob("*.pt"):
            template = MoleculeTemplate.load(pt_file, device=device)
            templates.append(template)

    if not templates:
        raise ValueError(f"No valid templates found in {path}")

    return templates


def _create_default_templates(vocab, device: str) -> list:
    """Create default molecule templates for demo."""
    from mc3gs.chemistry.rdkit_templates import create_template_from_smiles
    from mc3gs.model.templates import MoleculeTemplate

    default_smiles = [
        ("benzene", "c1ccccc1"),
        ("water", "O"),
        ("methane", "C"),
        ("ethanol", "CCO"),
        ("caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ]

    templates = []
    for name, smiles in default_smiles:
        try:
            template_dict = create_template_from_smiles(smiles, vocab)
            template = MoleculeTemplate.from_chemistry_template(
                template_dict, vocab, name=name, device=device
            )
            templates.append(template)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to create {name}: {e}[/yellow]")

    return templates


@app.command()
def resume(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file or directory"),
    ],
    iterations: Annotated[
        Optional[int],
        typer.Option("--iterations", help="Additional iterations to train"),
    ] = None,
) -> None:
    """Resume training from a checkpoint."""
    from mc3gs.train.checkpoints import load_checkpoint

    console.print(f"Resuming from: {checkpoint}")

    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoints" / "checkpoint_best.pt"

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    data = load_checkpoint(checkpoint)
    console.print(f"Loaded checkpoint from iteration {data.iteration}")

    # Re-run train command with the loaded config
    # This is a simplified version - full implementation would
    # properly restore the scene and optimizer


@app.command()
def validate(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file"),
    ],
    images: Annotated[
        Path,
        typer.Option("--images", "-i", help="Path to validation images"),
    ],
) -> None:
    """Validate a trained model on a set of images."""
    console.print(f"Validating {checkpoint} on {images}")
    # Implementation would load the checkpoint and run evaluation


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
