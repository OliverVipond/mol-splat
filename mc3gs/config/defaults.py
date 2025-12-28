"""Default configuration factory."""

from pathlib import Path

from mc3gs.config.schema import DataConfig, MC3GSConfig


def get_default_config(images_path: str | Path = "data/images") -> MC3GSConfig:
    """Create a default MC-3GS configuration.

    Args:
        images_path: Path to the input images directory.

    Returns:
        A fully populated MC3GSConfig with sensible defaults.
    """
    return MC3GSConfig(
        data=DataConfig(images_path=Path(images_path)),
    )


def get_debug_config(images_path: str | Path = "data/images") -> MC3GSConfig:
    """Create a minimal configuration for debugging.

    Args:
        images_path: Path to the input images directory.

    Returns:
        A configuration suitable for quick debugging runs.
    """
    config = get_default_config(images_path)
    config.model.sh_degree = 0
    config.train.num_iterations = 100
    config.train.log_interval = 10
    config.train.val_interval = 50
    config.train.checkpoint_interval = 50
    config.render.backend = "reference"
    return config
