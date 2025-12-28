"""CLI entrypoint for exporting MC-3GS models."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="mc3gs-export",
    help="Export trained MC-3GS models to various formats.",
)
console = Console()


@app.command()
def gaussians(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output PLY file path"),
    ] = Path("gaussians.ply"),
    sh_degree: Annotated[
        int,
        typer.Option("--sh-degree", help="Maximum SH degree to export"),
    ] = 3,
) -> None:
    """Export Gaussians to PLY format for standard viewers."""
    from mc3gs.export.export_gaussians import export_gaussians_ply
    from mc3gs.train.checkpoints import load_checkpoint, restore_scene_from_checkpoint

    console.print(f"Loading checkpoint: {checkpoint}")

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    data = load_checkpoint(checkpoint, device="cpu")
    scene = restore_scene_from_checkpoint(data, device="cpu")

    console.print(f"Exporting {scene.total_gaussians} Gaussians to {output}")

    export_gaussians_ply(scene, output, include_sh=True, max_sh_degree=sh_degree)

    console.print(f"[green]Exported to {output}[/green]")


@app.command()
def scene(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSON file path"),
    ] = Path("scene.json"),
    include_gaussians: Annotated[
        bool,
        typer.Option("--include-gaussians/--no-gaussians", help="Include Gaussian data"),
    ] = False,
) -> None:
    """Export scene as JSON with molecule transforms and color schemes."""
    from mc3gs.export.export_gaussians import export_scene_json
    from mc3gs.train.checkpoints import load_checkpoint, restore_scene_from_checkpoint

    console.print(f"Loading checkpoint: {checkpoint}")

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    data = load_checkpoint(checkpoint, device="cpu")
    scene = restore_scene_from_checkpoint(data, device="cpu")

    console.print(f"Exporting scene with {len(scene)} molecules to {output}")

    export_scene_json(scene, output, include_gaussians=include_gaussians)

    console.print(f"[green]Exported to {output}[/green]")


@app.command()
def blender(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("blender_export"),
    include_script: Annotated[
        bool,
        typer.Option("--script/--no-script", help="Include Blender import script"),
    ] = True,
) -> None:
    """Export scene for Blender geometry nodes."""
    from mc3gs.export.export_blender import export_to_blender
    from mc3gs.train.checkpoints import load_checkpoint, restore_scene_from_checkpoint

    console.print(f"Loading checkpoint: {checkpoint}")

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    data = load_checkpoint(checkpoint, device="cpu")
    scene = restore_scene_from_checkpoint(data, device="cpu")

    console.print(f"Exporting to Blender format: {output}")

    export_to_blender(scene, output, include_script=include_script)

    console.print(f"[green]Exported to {output}/[/green]")
    if include_script:
        console.print("Run import_molecules.py in Blender to import the scene")


@app.command()
def threejs(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("threejs_export"),
    include_viewer: Annotated[
        bool,
        typer.Option("--viewer/--no-viewer", help="Include HTML viewer"),
    ] = True,
) -> None:
    """Export scene for Three.js web viewer."""
    from mc3gs.export.export_threejs import export_to_threejs
    from mc3gs.train.checkpoints import load_checkpoint, restore_scene_from_checkpoint

    console.print(f"Loading checkpoint: {checkpoint}")

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    data = load_checkpoint(checkpoint, device="cpu")
    scene = restore_scene_from_checkpoint(data, device="cpu")

    console.print(f"Exporting to Three.js format: {output}")

    export_to_threejs(scene, output, include_viewer=include_viewer)

    console.print(f"[green]Exported to {output}/[/green]")
    if include_viewer:
        console.print(f"Open {output}/index.html in a browser to view")


@app.command()
def colors(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output CSV file path"),
    ] = Path("colors.csv"),
) -> None:
    """Export per-molecule, per-type colors to CSV."""
    from mc3gs.export.export_gaussians import export_colors_csv
    from mc3gs.train.checkpoints import load_checkpoint, restore_scene_from_checkpoint

    console.print(f"Loading checkpoint: {checkpoint}")

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    data = load_checkpoint(checkpoint, device="cpu")
    scene = restore_scene_from_checkpoint(data, device="cpu")

    console.print(f"Exporting colors to {output}")

    export_colors_csv(scene, output)

    console.print(f"[green]Exported to {output}[/green]")


@app.command()
def all(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint file"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("export"),
) -> None:
    """Export scene to all supported formats."""
    from mc3gs.export.export_blender import export_to_blender
    from mc3gs.export.export_gaussians import (
        export_colors_csv,
        export_gaussians_ply,
        export_scene_json,
    )
    from mc3gs.export.export_threejs import export_to_threejs
    from mc3gs.train.checkpoints import load_checkpoint, restore_scene_from_checkpoint

    console.print(f"Loading checkpoint: {checkpoint}")

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)

    data = load_checkpoint(checkpoint, device="cpu")
    scene = restore_scene_from_checkpoint(data, device="cpu")

    console.print(f"Exporting scene with {len(scene)} molecules, {scene.total_gaussians} Gaussians")

    # Export all formats
    console.print("  → Gaussians PLY...")
    export_gaussians_ply(scene, output / "gaussians.ply")

    console.print("  → Scene JSON...")
    export_scene_json(scene, output / "scene.json")

    console.print("  → Colors CSV...")
    export_colors_csv(scene, output / "colors.csv")

    console.print("  → Blender...")
    export_to_blender(scene, output / "blender")

    console.print("  → Three.js...")
    export_to_threejs(scene, output / "threejs")

    console.print(f"[green]All exports complete in {output}/[/green]")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
