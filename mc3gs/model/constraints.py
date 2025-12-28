"""Constraint definitions for molecule-level parameter sharing.

Constraints in MC-3GS:
1. Covariance is fixed from template (not learnable)
2. Per-molecule learnable: orientation, scale, single bond color
3. Atoms share SH coefficients by atom type (e.g., all carbons same color)
4. Bonds share a single SH coefficient per molecule (all bonds same color)
"""

from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor


class ColorConstraint(Protocol):
    """Protocol for color constraints."""

    def get_sh_coeffs(self, type_ids: Tensor, is_bond: Tensor | None = None) -> Tensor:
        """Get SH coefficients for given type IDs.

        Args:
            type_ids: Type IDs [N].
            is_bond: Boolean mask indicating which are bonds [N].

        Returns:
            SH coefficients [N, B, 3] where B = (L+1)^2.
        """
        ...


class AtomSHBank(nn.Module):
    """Spherical harmonics coefficient bank for atoms with per-type sharing.

    All atoms of the same type (e.g., all carbons) share the same
    SH coefficients within a molecule.

    Attributes:
        num_atom_types: Number of unique atom types.
        sh_degree: Maximum SH degree.
        num_sh_coeffs: Number of SH basis functions = (degree+1)^2.
    """

    def __init__(
        self,
        num_atom_types: int,
        sh_degree: int = 3,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize atom SH coefficient bank.

        Args:
            num_atom_types: Number of unique atom types.
            sh_degree: Maximum spherical harmonics degree (0-4).
            init_scale: Scale for random initialization.
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        # SH coefficients: [T, B, 3] where T = num_atom_types, B = num_sh_coeffs
        sh_coeffs = torch.zeros(num_atom_types, self.num_sh_coeffs, 3)

        # DC component (index 0) gets non-zero init
        sh_coeffs[:, 0, :] = torch.randn(num_atom_types, 3) * init_scale + 0.5

        # Higher degrees get smaller init
        if self.num_sh_coeffs > 1:
            sh_coeffs[:, 1:, :] = torch.randn(
                num_atom_types, self.num_sh_coeffs - 1, 3
            ) * (init_scale * 0.1)

        self.sh_coeffs = nn.Parameter(sh_coeffs)

    def get_sh_coeffs(self, atom_type_ids: Tensor) -> Tensor:
        """Get SH coefficients for given atom type IDs.

        Args:
            atom_type_ids: Atom type IDs [N].

        Returns:
            SH coefficients [N, B, 3].
        """
        return self.sh_coeffs[atom_type_ids]

    def set_from_colors(self, colors: Tensor) -> None:
        """Initialize DC component from RGB colors.

        Args:
            colors: RGB colors [T, 3] in range [0, 1].
        """
        C0 = 0.28209479177387814
        with torch.no_grad():
            self.sh_coeffs[:, 0, :] = colors / C0

    def regularization_loss(self, weight: float = 1e-4) -> Tensor:
        """Compute regularization loss on SH coefficients."""
        if self.num_sh_coeffs > 1:
            higher_order = self.sh_coeffs[:, 1:, :]
            return weight * (higher_order**2).mean()
        return torch.tensor(0.0, device=self.sh_coeffs.device)


class BondSHBank(nn.Module):
    """Single SH coefficient for ALL bonds in a molecule.

    All bonds within a molecule share the same color, but different
    molecules can have different bond colors.

    Attributes:
        sh_degree: Maximum SH degree.
        num_sh_coeffs: Number of SH basis functions = (degree+1)^2.
    """

    def __init__(
        self,
        sh_degree: int = 3,
        init_scale: float = 0.1,
        init_color: tuple[float, float, float] = (0.3, 0.3, 0.3),
    ) -> None:
        """Initialize bond SH coefficient bank.

        Args:
            sh_degree: Maximum spherical harmonics degree (0-4).
            init_scale: Scale for random initialization.
            init_color: Initial bond color (gray by default).
        """
        super().__init__()

        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        # Single SH coefficient for all bonds: [1, B, 3]
        sh_coeffs = torch.zeros(1, self.num_sh_coeffs, 3)

        # Initialize DC from init_color
        C0 = 0.28209479177387814
        sh_coeffs[0, 0, :] = torch.tensor(init_color) / C0

        # Higher degrees get small random init
        if self.num_sh_coeffs > 1:
            sh_coeffs[:, 1:, :] = torch.randn(1, self.num_sh_coeffs - 1, 3) * (
                init_scale * 0.1
            )

        self.sh_coeffs = nn.Parameter(sh_coeffs)

    def get_sh_coeffs(self, num_bonds: int) -> Tensor:
        """Get SH coefficients for all bonds (same color).

        Args:
            num_bonds: Number of bonds.

        Returns:
            SH coefficients [num_bonds, B, 3] (all identical).
        """
        return self.sh_coeffs.expand(num_bonds, -1, -1)

    def set_from_color(self, color: Tensor) -> None:
        """Set bond color from RGB.

        Args:
            color: RGB color [3] in range [0, 1].
        """
        C0 = 0.28209479177387814
        with torch.no_grad():
            self.sh_coeffs[0, 0, :] = color / C0

    def regularization_loss(self, weight: float = 1e-4) -> Tensor:
        """Compute regularization loss on SH coefficients."""
        if self.num_sh_coeffs > 1:
            higher_order = self.sh_coeffs[:, 1:, :]
            return weight * (higher_order**2).mean()
        return torch.tensor(0.0, device=self.sh_coeffs.device)


# Keep SharedSHBank for backward compatibility but mark as deprecated
class SharedSHBank(nn.Module):
    """Spherical harmonics coefficient bank with per-type sharing.

    DEPRECATED: Use AtomSHBank + BondSHBank instead for proper constraints.
    
    All Gaussians of the same type within a molecule share the same
    SH coefficients. This is the key constraint in MC-3GS.

    Attributes:
        num_types: Number of unique types.
        sh_degree: Maximum SH degree.
        num_sh_coeffs: Number of SH basis functions = (degree+1)^2.
    """

    def __init__(
        self,
        num_types: int,
        sh_degree: int = 3,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize SH coefficient bank.

        Args:
            num_types: Number of unique types (atoms + bonds).
            sh_degree: Maximum spherical harmonics degree (0-4).
            init_scale: Scale for random initialization.
        """
        super().__init__()

        self.num_types = num_types
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        # SH coefficients: [T, B, 3] where T = num_types, B = num_sh_coeffs
        # Initialize DC component (degree 0) with small random values
        # Higher degrees start near zero
        sh_coeffs = torch.zeros(num_types, self.num_sh_coeffs, 3)

        # DC component (index 0) gets non-zero init
        sh_coeffs[:, 0, :] = torch.randn(num_types, 3) * init_scale + 0.5

        # Higher degrees get smaller init
        if self.num_sh_coeffs > 1:
            sh_coeffs[:, 1:, :] = torch.randn(
                num_types, self.num_sh_coeffs - 1, 3
            ) * (init_scale * 0.1)

        self.sh_coeffs = nn.Parameter(sh_coeffs)

    def get_sh_coeffs(self, type_ids: Tensor) -> Tensor:
        """Get SH coefficients for given type IDs.

        Args:
            type_ids: Type IDs [N].

        Returns:
            SH coefficients [N, B, 3].
        """
        return self.sh_coeffs[type_ids]

    def get_active_degree(self, current_degree: int) -> Tensor:
        """Get SH coefficients up to a specific degree.

        Useful for progressive training where we start with low
        degrees and increase over time.

        Args:
            current_degree: Maximum degree to include.

        Returns:
            SH coefficients [T, B', 3] where B' = (current_degree+1)^2.
        """
        num_active = (current_degree + 1) ** 2
        return self.sh_coeffs[:, :num_active, :]

    def set_from_colors(self, colors: Tensor) -> None:
        """Initialize DC component from RGB colors.

        Args:
            colors: RGB colors [T, 3] in range [0, 1].
        """
        # Convert RGB to SH DC coefficient
        # For SH, DC = color * C0 where C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        with torch.no_grad():
            self.sh_coeffs[:, 0, :] = colors / C0

    def regularization_loss(self, weight: float = 1e-4) -> Tensor:
        """Compute regularization loss on SH coefficients.

        Penalizes high-frequency components to encourage smooth colors.

        Args:
            weight: Regularization weight.

        Returns:
            Scalar regularization loss.
        """
        # L2 on higher-degree coefficients
        if self.num_sh_coeffs > 1:
            higher_order = self.sh_coeffs[:, 1:, :]
            return weight * (higher_order**2).mean()
        return torch.tensor(0.0, device=self.sh_coeffs.device)


class IndependentSHBank(nn.Module):
    """Per-Gaussian SH coefficients without type sharing.

    This is used for comparison or when constraints are disabled.
    Each Gaussian has its own independent SH coefficients.
    """

    def __init__(
        self,
        num_gaussians: int,
        sh_degree: int = 3,
        init_scale: float = 0.1,
    ) -> None:
        """Initialize independent SH coefficient bank.

        Args:
            num_gaussians: Number of Gaussians.
            sh_degree: Maximum spherical harmonics degree.
            init_scale: Scale for random initialization.
        """
        super().__init__()

        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        # Each Gaussian has its own SH coefficients
        sh_coeffs = torch.zeros(num_gaussians, self.num_sh_coeffs, 3)
        sh_coeffs[:, 0, :] = torch.randn(num_gaussians, 3) * init_scale + 0.5

        if self.num_sh_coeffs > 1:
            sh_coeffs[:, 1:, :] = torch.randn(
                num_gaussians, self.num_sh_coeffs - 1, 3
            ) * (init_scale * 0.1)

        self.sh_coeffs = nn.Parameter(sh_coeffs)

    def get_sh_coeffs(self, indices: Tensor) -> Tensor:
        """Get SH coefficients for given Gaussian indices.

        Args:
            indices: Gaussian indices [N].

        Returns:
            SH coefficients [N, B, 3].
        """
        return self.sh_coeffs[indices]
