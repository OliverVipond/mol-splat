"""Checkpoint saving and loading utilities."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from mc3gs.config.schema import MC3GSConfig
from mc3gs.model.scene import Scene


@dataclass
class CheckpointData:
    """Data container for checkpoint contents."""

    iteration: int
    scene_state: dict
    optimizer_state: dict | None
    config: MC3GSConfig
    metrics: dict[str, float]
    timestamp: str


def save_checkpoint(
    path: Path | str,
    iteration: int,
    scene: Scene,
    optimizer_state: dict | None,
    config: MC3GSConfig,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Output path.
        iteration: Current training iteration.
        scene: Scene to save.
        optimizer_state: Optimizer state dict (optional).
        config: Training configuration.
        metrics: Optional metrics dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "iteration": iteration,
        "scene_state": scene.state_dict(),
        "optimizer_state": optimizer_state,
        "config": config.model_dump(),
        "metrics": metrics or {},
        "timestamp": datetime.now().isoformat(),
        "templates": [
            {
                "p_local": inst.template.p_local.cpu(),
                "cov_local": inst.template.cov_local.cpu(),
                "type_id": inst.template.type_id.cpu(),
                "vocab": inst.template.type_vocab,
                "name": inst.template.name,
                "metadata": inst.template.metadata,
            }
            for inst in scene.instances
        ],
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path | str,
    device: str = "cuda",
) -> CheckpointData:
    """Load a training checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to load tensors to.

    Returns:
        CheckpointData with loaded contents.
    """
    checkpoint = torch.load(path, map_location=device)

    config = MC3GSConfig(**checkpoint["config"])

    return CheckpointData(
        iteration=checkpoint["iteration"],
        scene_state=checkpoint["scene_state"],
        optimizer_state=checkpoint.get("optimizer_state"),
        config=config,
        metrics=checkpoint.get("metrics", {}),
        timestamp=checkpoint.get("timestamp", ""),
    )


def restore_scene_from_checkpoint(
    checkpoint: CheckpointData,
    device: str = "cuda",
) -> Scene:
    """Restore a scene from checkpoint data.

    Args:
        checkpoint: Loaded checkpoint data.
        device: Device to load to.

    Returns:
        Restored Scene.
    """
    from mc3gs.model.molecule_instance import MoleculeInstance
    from mc3gs.model.templates import MoleculeTemplate

    # This is a simplified restore - in practice you'd need
    # the original templates stored in the checkpoint
    scene = Scene()

    # Load state dict (this requires the scene structure to match)
    scene.load_state_dict(checkpoint.scene_state)
    scene.to(device)

    return scene


class Checkpointer:
    """Manages checkpoint saving and loading during training."""

    def __init__(
        self,
        output_dir: Path | str,
        max_checkpoints: int = 5,
        save_best: bool = True,
    ) -> None:
        """Initialize checkpointer.

        Args:
            output_dir: Directory for checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
            save_best: Whether to save best checkpoint separately.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best

        self.checkpoints: list[Path] = []
        self.best_metric: float = float("inf")
        self.best_path: Path | None = None

    def save(
        self,
        iteration: int,
        scene: Scene,
        optimizer_state: dict | None,
        config: MC3GSConfig,
        metrics: dict[str, float] | None = None,
        metric_key: str = "total_loss",
    ) -> Path:
        """Save checkpoint and manage checkpoint history.

        Args:
            iteration: Current iteration.
            scene: Scene to save.
            optimizer_state: Optimizer state.
            config: Configuration.
            metrics: Metrics dictionary.
            metric_key: Key for determining best checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        # Save current checkpoint
        path = self.output_dir / f"checkpoint_{iteration:06d}.pt"
        save_checkpoint(path, iteration, scene, optimizer_state, config, metrics)
        self.checkpoints.append(path)

        # Save best checkpoint
        if self.save_best and metrics and metric_key in metrics:
            current_metric = metrics[metric_key]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_path = self.output_dir / "checkpoint_best.pt"
                save_checkpoint(
                    self.best_path, iteration, scene, optimizer_state, config, metrics
                )

        # Cleanup old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if old_path.exists() and old_path != self.best_path:
                old_path.unlink()

        return path

    def load_latest(self, device: str = "cuda") -> CheckpointData | None:
        """Load the most recent checkpoint.

        Args:
            device: Device to load to.

        Returns:
            Checkpoint data or None if no checkpoints exist.
        """
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None

        # Exclude 'best' checkpoint from iteration checkpoints
        iteration_checkpoints = [
            p for p in checkpoints if "best" not in p.stem
        ]

        if not iteration_checkpoints:
            return None

        return load_checkpoint(iteration_checkpoints[-1], device)

    def load_best(self, device: str = "cuda") -> CheckpointData | None:
        """Load the best checkpoint.

        Args:
            device: Device to load to.

        Returns:
            Checkpoint data or None if no best checkpoint exists.
        """
        best_path = self.output_dir / "checkpoint_best.pt"
        if not best_path.exists():
            return None
        return load_checkpoint(best_path, device)

    def load_iteration(
        self,
        iteration: int,
        device: str = "cuda",
    ) -> CheckpointData | None:
        """Load checkpoint from a specific iteration.

        Args:
            iteration: Target iteration.
            device: Device to load to.

        Returns:
            Checkpoint data or None if not found.
        """
        path = self.output_dir / f"checkpoint_{iteration:06d}.pt"
        if not path.exists():
            return None
        return load_checkpoint(path, device)
