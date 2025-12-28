"""Optimization utilities for MC-3GS training."""

from typing import Any

import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LRScheduler

from mc3gs.config.schema import OptimConfig
from mc3gs.model.scene import Scene


def build_optimizer(
    scene: Scene,
    config: OptimConfig,
) -> Optimizer:
    """Build optimizer with parameter groups for different learning rates.

    Different parameter types (position, rotation, scale, opacity, SH)
    often benefit from different learning rates.

    Args:
        scene: Scene containing molecule instances.
        config: Optimization configuration.

    Returns:
        Configured optimizer.
    """
    param_groups = []

    for instance in scene.instances:
        # Position (translation)
        param_groups.append({
            "params": [instance.translation],
            "lr": config.lr_position,
            "name": "position",
        })

        # Rotation
        param_groups.append({
            "params": [instance.rotation],
            "lr": config.lr_rotation,
            "name": "rotation",
        })

        # Scale
        param_groups.append({
            "params": [instance.log_scale],
            "lr": config.lr_scale,
            "name": "scale",
        })

        # Opacity
        param_groups.append({
            "params": [instance.logit_opacity],
            "lr": config.lr_opacity,
            "name": "opacity",
        })

        # SH coefficients
        param_groups.append({
            "params": [instance.sh_bank.sh_coeffs],
            "lr": config.lr_sh,
            "name": "sh",
        })

    return Adam(
        param_groups,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: Optimizer,
    config: OptimConfig,
) -> LRScheduler:
    """Build learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        config: Optimization configuration.

    Returns:
        Learning rate scheduler.
    """
    # Compute gamma for exponential decay
    # lr(step) = lr_init * gamma^step
    # lr(final) = lr_init * decay_rate
    # gamma = decay_rate^(1/num_steps)
    gamma = config.lr_decay_rate ** (1 / config.lr_decay_steps)

    return ExponentialLR(optimizer, gamma=gamma)


class ParameterGroupOptimizer:
    """Optimizer wrapper with per-group learning rate control.

    Allows dynamic adjustment of learning rates for different
    parameter groups during training.
    """

    def __init__(
        self,
        scene: Scene,
        config: OptimConfig,
    ) -> None:
        """Initialize optimizer.

        Args:
            scene: Scene to optimize.
            config: Optimization configuration.
        """
        self.scene = scene
        self.config = config
        self.optimizer = build_optimizer(scene, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Track parameter group names
        self.group_names: list[str] = []
        for pg in self.optimizer.param_groups:
            self.group_names.append(pg.get("name", "unknown"))

    def step(self) -> None:
        """Perform optimization step."""
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()

    def scheduler_step(self) -> None:
        """Update learning rates."""
        self.scheduler.step()

    def get_learning_rates(self) -> dict[str, float]:
        """Get current learning rates by group name.

        Returns:
            Dictionary of group name -> learning rate.
        """
        lrs = {}
        for name, pg in zip(self.group_names, self.optimizer.param_groups):
            lrs[name] = pg["lr"]
        return lrs

    def set_learning_rate(self, group_name: str, lr: float) -> None:
        """Set learning rate for a specific group.

        Args:
            group_name: Name of the parameter group.
            lr: New learning rate.
        """
        for name, pg in zip(self.group_names, self.optimizer.param_groups):
            if name == group_name:
                pg["lr"] = lr

    def freeze_group(self, group_name: str) -> None:
        """Freeze a parameter group (set lr to 0).

        Args:
            group_name: Name of the parameter group to freeze.
        """
        self.set_learning_rate(group_name, 0.0)

    def unfreeze_group(self, group_name: str, lr: float | None = None) -> None:
        """Unfreeze a parameter group.

        Args:
            group_name: Name of the parameter group.
            lr: Learning rate (uses config default if None).
        """
        if lr is None:
            lr_map = {
                "position": self.config.lr_position,
                "rotation": self.config.lr_rotation,
                "scale": self.config.lr_scale,
                "opacity": self.config.lr_opacity,
                "sh": self.config.lr_sh,
            }
            lr = lr_map.get(group_name, 1e-3)
        self.set_learning_rate(group_name, lr)

    def state_dict(self) -> dict[str, Any]:
        """Get optimizer state dict."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


def compute_gradient_norms(scene: Scene) -> dict[str, float]:
    """Compute gradient norms for monitoring.

    Args:
        scene: Scene with computed gradients.

    Returns:
        Dictionary of parameter type -> gradient norm.
    """
    norms: dict[str, list[float]] = {
        "position": [],
        "rotation": [],
        "scale": [],
        "opacity": [],
        "sh": [],
    }

    for instance in scene.instances:
        if instance.translation.grad is not None:
            norms["position"].append(instance.translation.grad.norm().item())
        if instance.rotation.grad is not None:
            norms["rotation"].append(instance.rotation.grad.norm().item())
        if instance.log_scale.grad is not None:
            norms["scale"].append(instance.log_scale.grad.norm().item())
        if instance.logit_opacity.grad is not None:
            norms["opacity"].append(instance.logit_opacity.grad.norm().item())
        if instance.sh_bank.sh_coeffs.grad is not None:
            norms["sh"].append(instance.sh_bank.sh_coeffs.grad.norm().item())

    # Average norms
    return {k: sum(v) / len(v) if v else 0.0 for k, v in norms.items()}
