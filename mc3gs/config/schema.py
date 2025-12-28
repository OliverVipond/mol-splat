"""Configuration schema using Pydantic models."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CameraConfig(BaseModel):
    """Camera-related configuration."""

    source: Literal["colmap", "provided", "synthetic"] = Field(
        default="colmap",
        description="Source of camera poses",
    )
    colmap_path: Path | None = Field(
        default=None,
        description="Path to COLMAP sparse reconstruction",
    )
    scale_factor: float = Field(
        default=1.0,
        gt=0.0,
        description="Scale factor for camera positions",
    )


class DataConfig(BaseModel):
    """Data loading configuration."""

    images_path: Path = Field(
        description="Path to directory containing input images",
    )
    cameras: CameraConfig = Field(default_factory=CameraConfig)
    image_scale: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Scale factor for images (for memory efficiency)",
    )
    white_background: bool = Field(
        default=False,
        description="Use white background instead of black",
    )


class ChemistryConfig(BaseModel):
    """Chemistry and molecule template configuration."""

    atom_types: list[str] = Field(
        default_factory=lambda: [
            "C_sp3",
            "C_sp2",
            "C_sp2_arom",
            "N",
            "N_arom",
            "O",
            "S",
            "F",
            "Cl",
            "Br",
            "P",
            "H",
        ],
        description="Vocabulary of atom types",
    )
    bond_types: list[str] = Field(
        default_factory=lambda: ["single", "double", "triple", "aromatic"],
        description="Vocabulary of bond types",
    )
    include_bonds: bool = Field(
        default=True,
        description="Include bond Gaussians in templates",
    )
    gaussians_per_atom: int = Field(
        default=1,
        ge=1,
        description="Number of Gaussians per atom slot",
    )
    use_vdw_radii: bool = Field(
        default=True,
        description="Use Van der Waals radii for atom covariances",
    )


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    sh_degree: int = Field(
        default=3,
        ge=0,
        le=4,
        description="Maximum spherical harmonics degree",
    )
    init_opacity: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Initial opacity for Gaussians",
    )
    min_opacity: float = Field(
        default=0.005,
        gt=0.0,
        lt=1.0,
        description="Minimum opacity threshold for pruning",
    )
    enable_scale: bool = Field(
        default=True,
        description="Enable per-molecule scale parameter",
    )
    init_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Initial scale for molecules",
    )


class RenderConfig(BaseModel):
    """Rendering configuration."""

    backend: Literal["reference", "cuda"] = Field(
        default="reference",
        description="Rendering backend to use",
    )
    tile_size: int = Field(
        default=16,
        ge=1,
        description="Tile size for tiled rasterization",
    )
    max_gaussians_per_tile: int = Field(
        default=256,
        ge=1,
        description="Maximum Gaussians per tile",
    )
    depth_sort: bool = Field(
        default=True,
        description="Enable depth sorting for compositing",
    )


class OptimConfig(BaseModel):
    """Optimization configuration."""

    lr_position: float = Field(default=1.6e-4, gt=0.0)
    lr_rotation: float = Field(default=1e-3, gt=0.0)
    lr_scale: float = Field(default=5e-3, gt=0.0)
    lr_opacity: float = Field(default=5e-2, gt=0.0)
    lr_sh: float = Field(default=2.5e-3, gt=0.0)

    lr_decay_steps: int = Field(default=30000, gt=0)
    lr_decay_rate: float = Field(default=0.01, gt=0.0, le=1.0)

    weight_decay: float = Field(default=0.0, ge=0.0)
    eps: float = Field(default=1e-15, gt=0.0)


class TrainConfig(BaseModel):
    """Training configuration."""

    num_iterations: int = Field(default=30000, gt=0)
    batch_size: int = Field(default=1, ge=1)

    # Loss weights
    lambda_l2: float = Field(default=0.8, ge=0.0)
    lambda_ssim: float = Field(default=0.2, ge=0.0)
    lambda_opacity_reg: float = Field(default=1e-4, ge=0.0)
    lambda_scale_reg: float = Field(default=1e-4, ge=0.0)

    # Training stages
    warmup_iterations: int = Field(default=500, ge=0)
    sh_degree_increase_interval: int = Field(default=1000, gt=0)

    # Checkpointing
    checkpoint_interval: int = Field(default=5000, gt=0)
    log_interval: int = Field(default=100, gt=0)
    val_interval: int = Field(default=1000, gt=0)

    # Device
    device: str = Field(default="cuda")

    optim: OptimConfig = Field(default_factory=OptimConfig)


class ExportConfig(BaseModel):
    """Export configuration."""

    output_dir: Path = Field(default=Path("outputs"))
    export_ply: bool = Field(default=True)
    export_json: bool = Field(default=True)
    export_blender: bool = Field(default=False)
    export_threejs: bool = Field(default=False)


class MC3GSConfig(BaseModel):
    """Root configuration for MC-3GS."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(default="mc3gs_project")
    seed: int = Field(default=42)

    data: DataConfig
    chemistry: ChemistryConfig = Field(default_factory=ChemistryConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
