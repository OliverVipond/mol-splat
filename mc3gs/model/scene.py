"""Scene containing multiple molecule instances."""

from collections.abc import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from mc3gs.model.molecule_instance import MoleculeInstance
from mc3gs.model.templates import MoleculeTemplate


class Scene(nn.Module):
    """Scene composed of molecule instances.

    The scene manages multiple molecule instances, each with its own
    pose, opacity, and per-type color scheme. It provides methods for
    gathering all Gaussians for rendering.
    """

    def __init__(self) -> None:
        """Initialize empty scene."""
        super().__init__()
        self.instances = nn.ModuleList()

    def add_instance(self, instance: MoleculeInstance) -> int:
        """Add a molecule instance to the scene.

        Args:
            instance: MoleculeInstance to add.

        Returns:
            Index of the added instance.
        """
        idx = len(self.instances)
        self.instances.append(instance)
        return idx

    def add_instances(self, instances: list[MoleculeInstance]) -> list[int]:
        """Add multiple instances to the scene.

        Args:
            instances: List of MoleculeInstance objects.

        Returns:
            List of indices for the added instances.
        """
        return [self.add_instance(inst) for inst in instances]

    def remove_instance(self, idx: int) -> MoleculeInstance:
        """Remove an instance by index.

        Args:
            idx: Index of instance to remove.

        Returns:
            The removed instance.
        """
        instance = self.instances[idx]
        del self.instances[idx]
        return instance

    def __len__(self) -> int:
        """Number of instances in the scene."""
        return len(self.instances)

    def __getitem__(self, idx: int) -> MoleculeInstance:
        """Get instance by index."""
        return self.instances[idx]

    def __iter__(self) -> Iterator[MoleculeInstance]:
        """Iterate over instances."""
        return iter(self.instances)

    @property
    def total_gaussians(self) -> int:
        """Total number of Gaussians across all instances."""
        return sum(inst.template.num_gaussians for inst in self.instances)

    @property
    def device(self) -> torch.device:
        """Device of the scene parameters."""
        if len(self.instances) > 0:
            return self.instances[0].translation.device
        return torch.device("cpu")

    def gather(self) -> dict[str, Tensor]:
        """Gather all Gaussians from all instances.

        Returns:
            Dictionary with:
                - positions: [N_total, 3]
                - covariances: [N_total, 3, 3]
                - opacities: [N_total]
                - type_ids: [N_total]
                - mol_ids: [N_total] - which molecule each Gaussian belongs to
                - sh_coeffs: [N_total, B, 3]
        """
        if len(self.instances) == 0:
            device = torch.device("cpu")
            return {
                "positions": torch.empty(0, 3, device=device),
                "covariances": torch.empty(0, 3, 3, device=device),
                "opacities": torch.empty(0, device=device),
                "type_ids": torch.empty(0, dtype=torch.long, device=device),
                "mol_ids": torch.empty(0, dtype=torch.long, device=device),
                "sh_coeffs": torch.empty(0, 0, 3, device=device),
            }

        positions = []
        covariances = []
        opacities = []
        type_ids = []
        mol_ids = []
        sh_coeffs = []

        for mol_id, instance in enumerate(self.instances):
            gaussians = instance.world_gaussians()
            n = instance.template.num_gaussians

            positions.append(gaussians["positions"])
            covariances.append(gaussians["covariances"])
            opacities.append(gaussians["opacities"])
            type_ids.append(gaussians["type_ids"])
            mol_ids.append(
                torch.full((n,), mol_id, dtype=torch.long, device=self.device)
            )
            sh_coeffs.append(gaussians["sh_coeffs"])

        return {
            "positions": torch.cat(positions, dim=0),
            "covariances": torch.cat(covariances, dim=0),
            "opacities": torch.cat(opacities, dim=0),
            "type_ids": torch.cat(type_ids, dim=0),
            "mol_ids": torch.cat(mol_ids, dim=0),
            "sh_coeffs": torch.cat(sh_coeffs, dim=0),
        }

    def gather_for_render(
        self,
        camera_center: Tensor,
    ) -> dict[str, Tensor]:
        """Gather Gaussians and compute view directions for rendering.

        Args:
            camera_center: Camera center in world coordinates [3].

        Returns:
            Dictionary with gathered data plus view_dirs [N, 3].
        """
        data = self.gather()

        # Compute view directions (from Gaussian to camera)
        positions = data["positions"]
        view_dirs = camera_center - positions
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
        data["view_dirs"] = view_dirs

        return data

    def regularization_loss(
        self,
        pose_weight: float = 1e-4,
        opacity_weight: float = 1e-4,
        sh_weight: float = 1e-4,
    ) -> dict[str, Tensor]:
        """Compute regularization losses for all instances.

        Args:
            pose_weight: Weight for pose regularization.
            opacity_weight: Weight for opacity sparsity.
            sh_weight: Weight for SH smoothness.

        Returns:
            Dictionary with individual and total losses.
        """
        pose_loss = torch.tensor(0.0, device=self.device)
        opacity_loss = torch.tensor(0.0, device=self.device)
        sh_loss = torch.tensor(0.0, device=self.device)

        for instance in self.instances:
            pose_loss = pose_loss + instance.pose_regularization(pose_weight)
            opacity_loss = opacity_loss + instance.opacity_regularization(opacity_weight)
            sh_loss = sh_loss + instance.sh_regularization(sh_weight)

        total = pose_loss + opacity_loss + sh_loss

        return {
            "pose": pose_loss,
            "opacity": opacity_loss,
            "sh": sh_loss,
            "total": total,
        }

    def prune_instances(self, min_opacity: float = 0.01) -> int:
        """Remove instances with low total opacity contribution.

        Args:
            min_opacity: Minimum mean opacity to keep an instance.

        Returns:
            Number of pruned instances.
        """
        to_keep = []
        for instance in self.instances:
            mean_opacity = instance.opacity.mean().item()
            if mean_opacity >= min_opacity:
                to_keep.append(instance)

        pruned = len(self.instances) - len(to_keep)
        self.instances = nn.ModuleList(to_keep)
        return pruned

    @classmethod
    def from_templates(
        cls,
        templates: list[MoleculeTemplate],
        positions: Tensor,
        rotations: Tensor | None = None,
        sh_degree: int = 3,
        init_opacity: float = 0.5,
    ) -> "Scene":
        """Create scene from templates and positions.

        Args:
            templates: List of templates (can repeat).
            positions: Initial positions [M, 3] for M instances.
            rotations: Initial rotations [M, 3] (axis-angle). Default: identity.
            sh_degree: SH degree for color.
            init_opacity: Initial opacity.

        Returns:
            Scene with M molecule instances.
        """
        scene = cls()
        m = positions.shape[0]

        if rotations is None:
            rotations = torch.zeros(m, 3, device=positions.device)

        for i in range(m):
            template = templates[i % len(templates)]
            instance = MoleculeInstance(
                template=template,
                sh_degree=sh_degree,
                init_position=positions[i],
                init_rotation=rotations[i],
                init_opacity=init_opacity,
            )
            scene.add_instance(instance)

        return scene

    @classmethod
    def from_point_cloud(
        cls,
        template: MoleculeTemplate,
        points: Tensor,
        colors: Tensor | None = None,
        sh_degree: int = 3,
        init_opacity: float = 0.5,
    ) -> "Scene":
        """Create scene by placing molecules at point cloud positions.

        Args:
            template: Template to instance.
            points: Point positions [N, 3].
            colors: Optional initial colors [N, 3].
            sh_degree: SH degree.
            init_opacity: Initial opacity.

        Returns:
            Scene with N molecule instances.
        """
        scene = cls()

        for i in range(points.shape[0]):
            instance = MoleculeInstance(
                template=template,
                sh_degree=sh_degree,
                init_position=points[i],
                init_opacity=init_opacity,
            )

            # Initialize colors from point cloud if provided
            if colors is not None:
                instance.sh_bank.set_from_colors(
                    colors[i].expand(template.type_vocab.num_types, -1)
                )

            scene.add_instance(instance)

        return scene

    def save(self, path: str) -> None:
        """Save scene state.

        Args:
            path: Output path.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_instances": len(self.instances),
                "templates": [inst.template for inst in self.instances],
            },
            path,
        )

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f"instances={len(self.instances)}, total_gaussians={self.total_gaussians}"
