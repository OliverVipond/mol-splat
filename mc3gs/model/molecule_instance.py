"""Molecule instance with learnable pose and appearance."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from mc3gs.model.constraints import AtomSHBank, BondSHBank
from mc3gs.model.templates import MoleculeTemplate


def axis_angle_to_rotation_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle representation to rotation matrix.

    Args:
        axis_angle: Axis-angle vector [3].

    Returns:
        Rotation matrix [3, 3].
    """
    angle = torch.norm(axis_angle)
    if angle < 1e-8:
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

    axis = axis_angle / angle
    K = torch.tensor(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ],
        device=axis_angle.device,
        dtype=axis_angle.dtype,
    )

    # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
    R = (
        torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * (K @ K)
    )
    return R


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [4] as (w, x, y, z).

    Returns:
        Rotation matrix [3, 3].
    """
    q = q / torch.norm(q)  # Normalize
    w, x, y, z = q

    return torch.tensor(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ],
        device=q.device,
        dtype=q.dtype,
    )


class MoleculeInstance(nn.Module):
    """A molecule instance with learnable pose, opacity, and colors.

    Each instance applies a rigid transformation to a template and
    maintains:
    - Per-atom-type SH coefficients (atoms of same type share color)
    - Single SH coefficient for all bonds (all bonds same color per molecule)

    Learnable parameters:
    - translation: [3] position in world space
    - rotation: [3] or [4] orientation (axis-angle or quaternion)
    - log_scale: scalar uniform scale
    - logit_opacity: [N] per-Gaussian opacity
    - atom_sh_bank: per-atom-type SH coefficients
    - bond_sh_bank: single SH coefficient for all bonds

    Fixed (from template):
    - Local Gaussian positions
    - Local Gaussian covariances (shape is fixed!)

    Attributes:
        template: The molecule template (geometry and types).
        atom_sh_bank: Per-atom-type SH coefficient bank.
        bond_sh_bank: Single SH coefficient for all bonds.
    """

    def __init__(
        self,
        template: MoleculeTemplate,
        sh_degree: int = 3,
        init_position: Tensor | None = None,
        init_rotation: Tensor | None = None,
        init_scale: float = 1.0,
        init_opacity: float = 0.5,
        use_quaternion: bool = False,
        enable_scale: bool = True,
    ) -> None:
        """Initialize molecule instance.

        Args:
            template: Molecule template with local geometry.
            sh_degree: Maximum spherical harmonics degree.
            init_position: Initial position [3]. Default: origin.
            init_rotation: Initial rotation (axis-angle [3] or quaternion [4]).
            init_scale: Initial uniform scale factor.
            init_opacity: Initial opacity for all Gaussians.
            use_quaternion: Use quaternion instead of axis-angle for rotation.
            enable_scale: Enable learnable scale parameter.
        """
        super().__init__()

        self.template = template
        self.use_quaternion = use_quaternion
        self.enable_scale = enable_scale
        self.sh_degree = sh_degree

        device = template.device

        # Translation
        if init_position is None:
            init_position = torch.zeros(3, device=device)
        self.translation = nn.Parameter(init_position.clone())

        # Rotation (axis-angle or quaternion)
        if use_quaternion:
            if init_rotation is None:
                init_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            self.rotation = nn.Parameter(init_rotation.clone())
        else:
            if init_rotation is None:
                init_rotation = torch.zeros(3, device=device)
            self.rotation = nn.Parameter(init_rotation.clone())

        # Scale (log parameterization for positivity)
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_scale), device=device)
        )

        # Per-Gaussian opacity (logit parameterization)
        n = template.num_gaussians
        init_logit = math.log(init_opacity / (1 - init_opacity))
        self.logit_opacity = nn.Parameter(torch.full((n,), init_logit, device=device))

        # Separate SH banks for atoms and bonds
        num_atom_types = template.type_vocab.num_atom_types
        self.atom_sh_bank = AtomSHBank(num_atom_types, sh_degree)

        # Single color for all bonds in this molecule
        self.bond_sh_bank = BondSHBank(sh_degree)

        # Precompute atom/bond masks from template
        self._is_bond = template.is_bond_mask()

    @property
    def scale(self) -> Tensor:
        """Current scale factor."""
        if self.enable_scale:
            return torch.exp(self.log_scale)
        return torch.ones(1, device=self.log_scale.device)

    @property
    def opacity(self) -> Tensor:
        """Current opacity values [N]."""
        return torch.sigmoid(self.logit_opacity)

    @property
    def rotation_matrix(self) -> Tensor:
        """Current rotation matrix [3, 3]."""
        if self.use_quaternion:
            return quaternion_to_rotation_matrix(self.rotation)
        return axis_angle_to_rotation_matrix(self.rotation)

    def world_positions(self) -> Tensor:
        """Compute world-space Gaussian positions.

        Returns:
            Positions [N, 3] in world coordinates.
        """
        R = self.rotation_matrix
        scale = self.scale
        # μ = R @ (ρ * p_local) + t
        return (scale * self.template.p_local) @ R.T + self.translation

    def world_covariances(self) -> Tensor:
        """Compute world-space Gaussian covariances.

        Returns:
            Covariance matrices [N, 3, 3] in world coordinates.
        """
        R = self.rotation_matrix
        scale_sq = self.scale**2

        # Σ_world = R @ (ρ² * Σ_local) @ R^T
        cov_scaled = scale_sq * self.template.cov_local
        return torch.einsum("ij,njk,lk->nil", R, cov_scaled, R)

    def get_sh_coeffs(self) -> Tensor:
        """Get SH coefficients for all Gaussians.

        Atoms get their SH from atom_sh_bank (shared by atom type).
        Bonds get their SH from bond_sh_bank (single color for all bonds).

        Returns:
            SH coefficients [N, B, 3] where B = (degree+1)^2.
        """
        n = self.template.num_gaussians
        num_sh = self.atom_sh_bank.num_sh_coeffs
        device = self.template.p_local.device

        sh_coeffs = torch.zeros(n, num_sh, 3, device=device)

        # Atoms: look up by type
        atom_mask = ~self._is_bond
        if atom_mask.any():
            atom_type_ids = self.template.type_id[atom_mask]
            sh_coeffs[atom_mask] = self.atom_sh_bank.get_sh_coeffs(atom_type_ids)

        # Bonds: all get the same single color
        if self._is_bond.any():
            num_bonds = self._is_bond.sum().item()
            sh_coeffs[self._is_bond] = self.bond_sh_bank.get_sh_coeffs(num_bonds)

        return sh_coeffs

    def world_gaussians(self) -> dict[str, Tensor]:
        """Get all world-space Gaussian parameters.

        Returns:
            Dictionary with:
                - positions: [N, 3]
                - covariances: [N, 3, 3]
                - opacities: [N]
                - type_ids: [N]
                - is_bond: [N] boolean mask
                - sh_coeffs: [N, B, 3]
        """
        return {
            "positions": self.world_positions(),
            "covariances": self.world_covariances(),
            "opacities": self.opacity,
            "type_ids": self.template.type_id,
            "is_bond": self._is_bond,
            "sh_coeffs": self.get_sh_coeffs(),
        }

    def pose_regularization(self, weight: float = 1e-4) -> Tensor:
        """Compute regularization loss on pose parameters.

        Args:
            weight: Regularization weight.

        Returns:
            Scalar regularization loss.
        """
        loss = torch.tensor(0.0, device=self.translation.device)

        # Penalize rotation magnitude
        if self.use_quaternion:
            # Penalize deviation from identity quaternion
            identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.rotation.device)
            loss = loss + weight * ((self.rotation - identity) ** 2).sum()
        else:
            loss = loss + weight * (self.rotation**2).sum()

        # Penalize extreme scales
        if self.enable_scale:
            loss = loss + weight * (self.log_scale**2)

        return loss

    def opacity_regularization(self, weight: float = 1e-4) -> Tensor:
        """Compute sparsity regularization on opacity.

        Args:
            weight: Regularization weight.

        Returns:
            Scalar regularization loss.
        """
        return weight * self.opacity.sum()

    def prune_threshold(self) -> float:
        """Get the minimum opacity for this instance."""
        return self.opacity.min().item()

    def total_contribution(self) -> Tensor:
        """Compute total opacity contribution of this instance."""
        return self.opacity.sum()

    def clone(self) -> "MoleculeInstance":
        """Create a copy of this instance with the same template.

        Returns:
            New MoleculeInstance with cloned parameters.
        """
        instance = MoleculeInstance(
            template=self.template,
            sh_degree=self.sh_degree,
            init_position=self.translation.data.clone(),
            init_rotation=self.rotation.data.clone(),
            init_scale=self.scale.item(),
            init_opacity=0.5,  # Will be overwritten
            use_quaternion=self.use_quaternion,
            enable_scale=self.enable_scale,
        )
        instance.logit_opacity.data = self.logit_opacity.data.clone()
        instance.atom_sh_bank.sh_coeffs.data = self.atom_sh_bank.sh_coeffs.data.clone()
        instance.bond_sh_bank.sh_coeffs.data = self.bond_sh_bank.sh_coeffs.data.clone()
        return instance

    def sh_regularization(self, weight: float = 1e-4) -> Tensor:
        """Compute regularization loss on SH coefficients.

        Args:
            weight: Regularization weight.

        Returns:
            Scalar regularization loss.
        """
        atom_reg = self.atom_sh_bank.regularization_loss(weight)
        bond_reg = self.bond_sh_bank.regularization_loss(weight)
        return atom_reg + bond_reg

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"template={self.template.name}, "
            f"n_gaussians={self.template.num_gaussians}, "
            f"n_atoms={int((~self._is_bond).sum())}, "
            f"n_bonds={int(self._is_bond.sum())}, "
            f"sh_degree={self.sh_degree}"
        )
