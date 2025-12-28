"""Main training loop for MC-3GS."""

from pathlib import Path
from typing import Callable

import torch
from rich.console import Console
from rich.progress import Progress, TaskID
from torch import Tensor
from torch.utils.data import DataLoader

from mc3gs.config.schema import MC3GSConfig
from mc3gs.data.images import ImageDataset
from mc3gs.model.scene import Scene
from mc3gs.render.backend import RenderBackend, get_backend
from mc3gs.train.checkpoints import Checkpointer
from mc3gs.train.losses import psnr, total_loss
from mc3gs.train.optim import ParameterGroupOptimizer, compute_gradient_norms

console = Console()


class Trainer:
    """Main training class for MC-3GS.

    Handles the training loop, logging, checkpointing, and
    progressive training stages.
    """

    def __init__(
        self,
        scene: Scene,
        dataset: ImageDataset,
        config: MC3GSConfig,
        output_dir: Path | str | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            scene: Scene to train.
            dataset: Training dataset.
            config: Training configuration.
            output_dir: Directory for outputs.
        """
        self.scene = scene
        self.dataset = dataset
        self.config = config
        self.device = config.train.device

        # Move scene to device
        self.scene.to(self.device)

        # Setup output directory
        if output_dir is None:
            output_dir = Path("outputs") / config.project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup renderer
        self.renderer: RenderBackend = get_backend(
            config.render.backend,
            self.device,
        )

        # Setup optimizer
        self.optimizer = ParameterGroupOptimizer(scene, config.train.optim)

        # Setup checkpointer
        self.checkpointer = Checkpointer(
            self.output_dir / "checkpoints",
            max_checkpoints=5,
        )

        # Training state
        self.iteration = 0
        self.active_sh_degree = 0
        self.max_sh_degree = config.model.sh_degree

        # Metrics history
        self.metrics_history: list[dict[str, float]] = []

        # Callbacks
        self.callbacks: list[Callable[[int, dict], None]] = []

    def _render_view(
        self,
        camera: dict[str, Tensor],
    ) -> Tensor:
        """Render a single view.

        Args:
            camera: Camera parameters dict.

        Returns:
            Rendered image [3, H, W].
        """
        # Gather all Gaussians
        data = self.scene.gather()

        result = self.renderer.render(
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
            sh_degree=self.active_sh_degree,
        )

        return result["image"]

    def _compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all losses.

        Args:
            pred: Predicted image [3, H, W].
            target: Target image [3, H, W].

        Returns:
            Dictionary of losses.
        """
        # Photometric loss
        losses = total_loss(
            pred,
            target,
            lambda_l2=self.config.train.lambda_l2,
            lambda_ssim=self.config.train.lambda_ssim,
        )

        # Regularization losses
        reg_losses = self.scene.regularization_loss(
            pose_weight=1e-4,
            opacity_weight=self.config.train.lambda_opacity_reg,
            sh_weight=1e-4,
        )

        losses["reg_pose"] = reg_losses["pose"]
        losses["reg_opacity"] = reg_losses["opacity"]
        losses["reg_sh"] = reg_losses["sh"]
        losses["reg_total"] = reg_losses["total"]

        # Total loss including regularization
        losses["total"] = losses["total"] + losses["reg_total"]

        # Metrics
        losses["psnr"] = psnr(pred, target)

        return losses

    def _training_step(self, image: Tensor, camera: dict[str, Tensor]) -> dict[str, float]:
        """Perform a single training step.

        Args:
            image: Target image [3, H, W].
            camera: Camera parameters.

        Returns:
            Dictionary of loss values.
        """
        self.optimizer.zero_grad()

        # Forward pass
        pred = self._render_view(camera)

        # Compute loss
        losses = self._compute_loss(pred, image)

        # Backward pass
        losses["total"].backward()

        # Optimizer step
        self.optimizer.step()

        # Convert to floats
        return {k: v.item() for k, v in losses.items()}

    def _update_sh_degree(self) -> None:
        """Progressively increase SH degree during training."""
        interval = self.config.train.sh_degree_increase_interval

        if self.active_sh_degree < self.max_sh_degree:
            if self.iteration > 0 and self.iteration % interval == 0:
                self.active_sh_degree += 1
                console.print(
                    f"[blue]Increased SH degree to {self.active_sh_degree}[/blue]"
                )

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metric values.
        """
        if self.iteration % self.config.train.log_interval == 0:
            lr = self.optimizer.get_learning_rates()
            console.print(
                f"[{self.iteration:6d}] "
                f"loss: {metrics['total']:.4f} "
                f"psnr: {metrics['psnr']:.2f} "
                f"l2: {metrics['l2']:.4f} "
                f"ssim: {metrics['ssim']:.4f} "
                f"lr_pos: {lr['position']:.2e}"
            )

    def train(
        self,
        num_iterations: int | None = None,
        resume: bool = True,
    ) -> dict[str, float]:
        """Run training loop.

        Args:
            num_iterations: Number of iterations (uses config if None).
            resume: Whether to resume from checkpoint.

        Returns:
            Final metrics dictionary.
        """
        if num_iterations is None:
            num_iterations = self.config.train.num_iterations

        # Try to resume
        if resume:
            checkpoint = self.checkpointer.load_latest(self.device)
            if checkpoint is not None:
                self.scene.load_state_dict(checkpoint.scene_state)
                if checkpoint.optimizer_state:
                    self.optimizer.load_state_dict(checkpoint.optimizer_state)
                self.iteration = checkpoint.iteration
                console.print(f"[green]Resumed from iteration {self.iteration}[/green]")

        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

        console.print(f"[bold]Starting training for {num_iterations} iterations[/bold]")
        console.print(f"Scene: {len(self.scene)} molecules, {self.scene.total_gaussians} Gaussians")

        final_metrics: dict[str, float] = {}

        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Training...",
                total=num_iterations - self.iteration,
            )

            while self.iteration < num_iterations:
                for image, camera in dataloader:
                    if self.iteration >= num_iterations:
                        break

                    # Move to device
                    image = image.squeeze(0).to(self.device)
                    camera_dict = camera.to_tensors(self.device)

                    # Training step
                    metrics = self._training_step(image, camera_dict)

                    # Update SH degree
                    self._update_sh_degree()

                    # Learning rate scheduling
                    self.optimizer.scheduler_step()

                    # Logging
                    self._log_metrics(metrics)

                    # Checkpointing
                    if self.iteration % self.config.train.checkpoint_interval == 0:
                        self.checkpointer.save(
                            self.iteration,
                            self.scene,
                            self.optimizer.state_dict(),
                            self.config,
                            metrics,
                        )

                    # Callbacks
                    for callback in self.callbacks:
                        callback(self.iteration, metrics)

                    self.iteration += 1
                    final_metrics = metrics
                    progress.update(task, advance=1)

        # Save final checkpoint
        self.checkpointer.save(
            self.iteration,
            self.scene,
            self.optimizer.state_dict(),
            self.config,
            final_metrics,
        )

        console.print("[bold green]Training complete![/bold green]")
        return final_metrics

    def evaluate(self, dataset: ImageDataset | None = None) -> dict[str, float]:
        """Evaluate scene on a dataset.

        Args:
            dataset: Evaluation dataset (uses training dataset if None).

        Returns:
            Average metrics over the dataset.
        """
        if dataset is None:
            dataset = self.dataset

        self.scene.eval()
        total_metrics: dict[str, float] = {}
        count = 0

        with torch.no_grad():
            for image, camera in dataset:
                image = image.to(self.device)
                camera_dict = camera.to_tensors(self.device)

                pred = self._render_view(camera_dict)
                metrics = self._compute_loss(pred, image)

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v.item()
                count += 1

        self.scene.train()

        return {k: v / count for k, v in total_metrics.items()}

    def add_callback(self, callback: Callable[[int, dict], None]) -> None:
        """Add a training callback.

        Args:
            callback: Function called after each iteration with (iteration, metrics).
        """
        self.callbacks.append(callback)

    def save(self, path: Path | str | None = None) -> Path:
        """Save the trained scene.

        Args:
            path: Output path.

        Returns:
            Path to saved file.
        """
        if path is None:
            path = self.output_dir / "scene_final.pt"
        else:
            path = Path(path)

        self.scene.save(str(path))
        return path
