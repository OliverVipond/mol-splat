"""Spherical harmonics utilities for view-dependent color."""

import math

import torch
from torch import Tensor

# Pre-computed SH basis constants
C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
C1 = 0.4886025119029199   # sqrt(3) / (2 * sqrt(pi))
C2 = [
    1.0925484305920792,   # sqrt(15) / (2 * sqrt(pi))
    -1.0925484305920792,
    0.31539156525252005,  # sqrt(5) / (4 * sqrt(pi))
    -1.0925484305920792,
    0.5462742152960396,   # sqrt(15) / (4 * sqrt(pi))
]
C3 = [
    -0.5900435899266435,  # sqrt(35/(2*pi)) / 4
    2.890611442640554,    # sqrt(105/pi) / 2
    -0.4570457994644658,  # sqrt(21/(2*pi)) / 4
    0.3731763325901154,   # sqrt(7/pi) / 4
    -0.4570457994644658,
    1.445305721320277,    # sqrt(105/pi) / 4
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,   # 3 * sqrt(35/pi) / 4
    -1.7701307697799304,  # 3 * sqrt(35/(2*pi)) / 4
    0.9461746957575601,   # 3 * sqrt(5/pi) / 4
    -0.6690465435572892,  # 3 * sqrt(5/(2*pi)) / 4
    0.10578554691520431,  # 3 / (16 * sqrt(pi))
    -0.6690465435572892,
    0.47308734787878004,  # 3 * sqrt(5/pi) / 8
    -1.7701307697799304,
    0.6258357354491761,   # 3 * sqrt(35/pi) / 16
]


def eval_sh_basis(directions: Tensor, degree: int) -> Tensor:
    """Evaluate spherical harmonics basis functions.

    Args:
        directions: Unit direction vectors [N, 3].
        degree: Maximum SH degree (0-4).

    Returns:
        SH basis values [N, (degree+1)^2].
    """
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    result = [torch.ones_like(x) * C0]

    if degree >= 1:
        result.extend([
            C1 * y,
            C1 * z,
            C1 * x,
        ])

    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        result.extend([
            C2[0] * xy,
            C2[1] * yz,
            C2[2] * (3 * zz - 1),
            C2[3] * xz,
            C2[4] * (xx - yy),
        ])

    if degree >= 3:
        result.extend([
            C3[0] * y * (3 * xx - yy),
            C3[1] * xy * z,
            C3[2] * y * (5 * zz - 1),
            C3[3] * z * (5 * zz - 3),
            C3[4] * x * (5 * zz - 1),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3 * yy),
        ])

    if degree >= 4:
        result.extend([
            C4[0] * xy * (xx - yy),
            C4[1] * yz * (3 * xx - yy),
            C4[2] * xy * (7 * zz - 1),
            C4[3] * yz * (7 * zz - 3),
            C4[4] * (35 * zz * zz - 30 * zz + 3),
            C4[5] * xz * (7 * zz - 3),
            C4[6] * (xx - yy) * (7 * zz - 1),
            C4[7] * xz * (xx - 3 * yy),
            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)),
        ])

    return torch.stack(result, dim=-1)


def shade_sh(
    sh_coeffs: Tensor,
    directions: Tensor,
    active_degree: int | None = None,
) -> Tensor:
    """Evaluate view-dependent color from SH coefficients.

    Args:
        sh_coeffs: SH coefficients [N, B, 3] or [B, 3].
        directions: Unit view directions [N, 3] or [3].
        active_degree: Optional max degree to use (for progressive training).

    Returns:
        RGB colors [N, 3] or [3].
    """
    # Determine full degree from coefficient count
    full_num_coeffs = sh_coeffs.shape[-2]
    full_degree = int(math.sqrt(full_num_coeffs)) - 1

    if active_degree is None:
        active_degree = full_degree

    active_num_coeffs = (active_degree + 1) ** 2

    # Compute basis
    basis = eval_sh_basis(directions, active_degree)  # [N, B'] or [B']

    # Use only active coefficients
    coeffs = sh_coeffs[..., :active_num_coeffs, :]  # [N, B', 3] or [B', 3]

    # Compute RGB: sum over basis functions
    if coeffs.dim() == 2:
        # Single set of coefficients
        rgb = torch.einsum("b,bc->c", basis, coeffs)
    else:
        # Per-Gaussian coefficients
        rgb = torch.einsum("nb,nbc->nc", basis, coeffs)

    return rgb


def rgb_to_sh_dc(rgb: Tensor) -> Tensor:
    """Convert RGB color to SH DC (degree 0) coefficient.

    Args:
        rgb: RGB colors [..., 3] in range [0, 1].

    Returns:
        SH DC coefficients [..., 3].
    """
    return rgb / C0


def sh_dc_to_rgb(sh_dc: Tensor) -> Tensor:
    """Convert SH DC coefficient to RGB color.

    Args:
        sh_dc: SH DC coefficients [..., 3].

    Returns:
        RGB colors [..., 3] (may need clamping to [0, 1]).
    """
    return sh_dc * C0


def initialize_sh_from_color(
    color: Tensor,
    num_types: int,
    degree: int,
) -> Tensor:
    """Initialize SH coefficients from a base color.

    Args:
        color: Base RGB color [3] or [num_types, 3].
        num_types: Number of types.
        degree: SH degree.

    Returns:
        SH coefficients [num_types, (degree+1)^2, 3].
    """
    num_coeffs = (degree + 1) ** 2

    # Initialize with zeros
    sh = torch.zeros(num_types, num_coeffs, 3, device=color.device, dtype=color.dtype)

    # Set DC component from color
    if color.dim() == 1:
        sh[:, 0, :] = rgb_to_sh_dc(color).unsqueeze(0).expand(num_types, -1)
    else:
        sh[:, 0, :] = rgb_to_sh_dc(color)

    return sh
