"""Molecule template data structure."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mc3gs.chemistry.typing import TypeVocabulary


@dataclass
class MoleculeTemplate:
    """Template geometry for a molecule in local coordinates.

    A template defines the local Gaussian splat positions, covariances,
    and type assignments for a molecule. Multiple instances can share
    the same template while having different poses and colors.

    Attributes:
        p_local: Local positions of Gaussian centers [N, 3].
        cov_local: Local covariance matrices [N, 3, 3].
        type_id: Type ID for each Gaussian [N].
        type_vocab: Vocabulary mapping type IDs to labels.
        name: Optional name for the template.
        metadata: Optional additional metadata.
    """

    p_local: Tensor  # [N, 3]
    cov_local: Tensor  # [N, 3, 3]
    type_id: Tensor  # [N] LongTensor
    type_vocab: TypeVocabulary
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate template data."""
        n = self.p_local.shape[0]
        assert self.p_local.shape == (n, 3), f"p_local shape: {self.p_local.shape}"
        assert self.cov_local.shape == (n, 3, 3), f"cov_local shape: {self.cov_local.shape}"
        assert self.type_id.shape == (n,), f"type_id shape: {self.type_id.shape}"

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians in the template."""
        return self.p_local.shape[0]

    @property
    def num_types(self) -> int:
        """Number of unique types in this template."""
        return len(self.type_id.unique())

    @property
    def unique_types(self) -> Tensor:
        """Unique type IDs in this template."""
        return self.type_id.unique()

    @property
    def device(self) -> torch.device:
        """Device of the template tensors."""
        return self.p_local.device

    def to(self, device: str | torch.device) -> "MoleculeTemplate":
        """Move template to specified device.

        Args:
            device: Target device.

        Returns:
            New MoleculeTemplate on the specified device.
        """
        return MoleculeTemplate(
            p_local=self.p_local.to(device),
            cov_local=self.cov_local.to(device),
            type_id=self.type_id.to(device),
            type_vocab=self.type_vocab,
            name=self.name,
            metadata=self.metadata,
        )

    def get_type_mask(self, type_id: int) -> Tensor:
        """Get boolean mask for Gaussians of a specific type.

        Args:
            type_id: Type ID to match.

        Returns:
            Boolean tensor [N].
        """
        return self.type_id == type_id

    def get_type_indices(self, type_id: int) -> Tensor:
        """Get indices of Gaussians with a specific type.

        Args:
            type_id: Type ID to match.

        Returns:
            Long tensor of indices.
        """
        return torch.where(self.type_id == type_id)[0]

    def is_bond_mask(self) -> Tensor:
        """Get boolean mask indicating which Gaussians are bonds.

        Returns:
            Boolean tensor [N] where True = bond, False = atom.
        """
        return torch.tensor(
            [self.type_vocab.is_bond(tid.item()) for tid in self.type_id],
            dtype=torch.bool,
            device=self.device,
        )

    def center(self) -> Tensor:
        """Compute centroid of template positions.

        Returns:
            Centroid [3].
        """
        return self.p_local.mean(dim=0)

    def centered(self) -> "MoleculeTemplate":
        """Return template with positions centered at origin.

        Returns:
            New centered MoleculeTemplate.
        """
        centroid = self.center()
        return MoleculeTemplate(
            p_local=self.p_local - centroid,
            cov_local=self.cov_local,
            type_id=self.type_id,
            type_vocab=self.type_vocab,
            name=self.name,
            metadata=self.metadata,
        )

    def scale(self, factor: float) -> "MoleculeTemplate":
        """Return scaled template.

        Args:
            factor: Scale factor.

        Returns:
            New scaled MoleculeTemplate.
        """
        return MoleculeTemplate(
            p_local=self.p_local * factor,
            cov_local=self.cov_local * (factor**2),
            type_id=self.type_id,
            type_vocab=self.type_vocab,
            name=self.name,
            metadata=self.metadata,
        )

    @classmethod
    def from_arrays(
        cls,
        positions: np.ndarray,
        covariances: np.ndarray,
        type_ids: np.ndarray,
        vocab: TypeVocabulary,
        name: str = "",
        device: str = "cpu",
    ) -> "MoleculeTemplate":
        """Create template from numpy arrays.

        Args:
            positions: Positions [N, 3].
            covariances: Covariances [N, 3, 3].
            type_ids: Type IDs [N].
            vocab: Type vocabulary.
            name: Template name.
            device: Target device.

        Returns:
            MoleculeTemplate instance.
        """
        return cls(
            p_local=torch.from_numpy(positions).float().to(device),
            cov_local=torch.from_numpy(covariances).float().to(device),
            type_id=torch.from_numpy(type_ids).long().to(device),
            type_vocab=vocab,
            name=name,
        )

    @classmethod
    def from_chemistry_template(
        cls,
        template_dict: dict,
        vocab: TypeVocabulary,
        name: str = "",
        device: str = "cpu",
    ) -> "MoleculeTemplate":
        """Create template from chemistry module output.

        Args:
            template_dict: Dictionary from create_template_from_smiles.
            vocab: Type vocabulary.
            name: Template name.
            device: Target device.

        Returns:
            MoleculeTemplate instance.
        """
        metadata = {
            k: v
            for k, v in template_dict.items()
            if k not in ("positions", "covariances", "type_ids", "labels")
        }
        return cls(
            p_local=torch.from_numpy(template_dict["positions"]).float().to(device),
            cov_local=torch.from_numpy(template_dict["covariances"]).float().to(device),
            type_id=torch.from_numpy(template_dict["type_ids"]).long().to(device),
            type_vocab=vocab,
            name=name or template_dict.get("smiles", ""),
            metadata=metadata,
        )

    def save(self, path: Path | str) -> None:
        """Save template to disk.

        Args:
            path: Output path (.pt).
        """
        path = Path(path)
        torch.save(
            {
                "p_local": self.p_local.cpu(),
                "cov_local": self.cov_local.cpu(),
                "type_id": self.type_id.cpu(),
                "vocab": self.type_vocab,
                "name": self.name,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str, device: str = "cpu") -> "MoleculeTemplate":
        """Load template from disk.

        Args:
            path: Path to template file.
            device: Target device.

        Returns:
            MoleculeTemplate instance.
        """
        data = torch.load(path, map_location=device)
        return cls(
            p_local=data["p_local"].to(device),
            cov_local=data["cov_local"].to(device),
            type_id=data["type_id"].to(device),
            type_vocab=data["vocab"],
            name=data.get("name", ""),
            metadata=data.get("metadata", {}),
        )
