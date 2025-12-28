"""Projection utilities for 3D Gaussians to 2D."""

import torch
from torch import Tensor


def project_points(
    points: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Project 3D points to 2D pixel coordinates.

    Args:
        points: World-space points [N, 3].
        K: Intrinsic matrix [3, 3].
        R: Rotation matrix (world-to-camera) [3, 3].
        t: Translation vector (world-to-camera) [3].

    Returns:
        Tuple of:
            - uv: 2D pixel coordinates [N, 2].
            - depth: Depth values [N].
    """
    # Transform to camera space
    points_cam = (points @ R.T) + t  # [N, 3]

    # Project to image plane
    points_proj = points_cam @ K.T  # [N, 3]

    # Perspective divide
    depth = points_proj[:, 2]
    uv = points_proj[:, :2] / (depth.unsqueeze(-1) + 1e-8)

    return uv, depth


def projection_jacobian(
    points_cam: Tensor,
    K: Tensor,
) -> Tensor:
    """Compute Jacobian of projection at camera-space points.

    The Jacobian J relates 3D displacements to 2D displacements:
    du = J @ dX (where X is in camera coordinates).

    Args:
        points_cam: Camera-space points [N, 3].
        K: Intrinsic matrix [3, 3].

    Returns:
        Jacobian matrices [N, 2, 3].
    """
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]

    z_sq = z * z + 1e-8

    # Jacobian of (fx*x/z, fy*y/z) w.r.t. (x, y, z)
    # d(u)/d(x) = fx/z, d(u)/d(y) = 0, d(u)/d(z) = -fx*x/z^2
    # d(v)/d(x) = 0, d(v)/d(y) = fy/z, d(v)/d(z) = -fy*y/z^2

    J = torch.zeros(points_cam.shape[0], 2, 3, device=points_cam.device, dtype=points_cam.dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / z_sq
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / z_sq

    return J


def project_gaussians(
    positions: Tensor,
    covariances: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project 3D Gaussians to 2D.

    Args:
        positions: 3D Gaussian means [N, 3].
        covariances: 3D covariance matrices [N, 3, 3].
        K: Intrinsic matrix [3, 3].
        R: Rotation matrix [3, 3].
        t: Translation vector [3].
        eps: Small value for numerical stability.

    Returns:
        Tuple of:
            - uv: 2D means [N, 2].
            - cov2d: 2D covariance matrices [N, 2, 2].
            - depth: Depth values [N].
            - valid: Boolean mask for valid projections [N].
    """
    # Transform to camera space
    positions_cam = (positions @ R.T) + t  # [N, 3]
    covariances_cam = torch.einsum("ij,njk,lk->nil", R, covariances, R)

    # Compute depth
    depth = positions_cam[:, 2]
    valid = depth > eps

    # Project means
    uv, _ = project_points(positions, K, R, t)

    # Compute Jacobian at each point
    J = projection_jacobian(positions_cam, K)  # [N, 2, 3]

    # Project covariance: Σ_2D = J @ Σ_cam @ J^T
    cov2d = torch.einsum("nij,njk,nlk->nil", J, covariances_cam, J)

    # Add small regularization for numerical stability
    cov2d = cov2d + eps * torch.eye(2, device=cov2d.device, dtype=cov2d.dtype)

    return uv, cov2d, depth, valid


def compute_gaussian_2d_extent(
    cov2d: Tensor,
    opacity: Tensor,
    threshold: float = 1.0 / 255.0,
) -> Tensor:
    """Compute the extent (radius) of 2D Gaussians for culling.

    Computes the radius at which the Gaussian contribution falls
    below a threshold.

    Args:
        cov2d: 2D covariance matrices [N, 2, 2].
        opacity: Opacity values [N].
        threshold: Minimum visible contribution.

    Returns:
        Radius values [N].
    """
    # Maximum eigenvalue gives the extent
    # For a 2x2 matrix, eigenvalues can be computed analytically
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 0]
    d = cov2d[:, 1, 1]

    # Trace and determinant
    trace = a + d
    det = a * d - b * c

    # Eigenvalues: λ = (trace ± sqrt(trace^2 - 4*det)) / 2
    discriminant = torch.sqrt(torch.clamp(trace**2 - 4 * det, min=0))
    lambda_max = (trace + discriminant) / 2

    # Standard deviation along major axis
    sigma_max = torch.sqrt(torch.clamp(lambda_max, min=1e-8))

    # Number of sigmas needed to reach threshold
    # opacity * exp(-0.5 * r^2 / sigma^2) = threshold
    # r = sigma * sqrt(-2 * log(threshold / opacity))
    log_ratio = torch.log(torch.clamp(opacity / threshold, min=1e-8))
    num_sigmas = torch.sqrt(torch.clamp(2 * log_ratio, min=0))

    return sigma_max * num_sigmas


def get_tile_bins(
    uv: Tensor,
    radii: Tensor,
    tile_size: int,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor]:
    """Compute tile ranges for each Gaussian.

    Args:
        uv: 2D centers [N, 2].
        radii: Gaussian radii [N].
        tile_size: Size of tiles in pixels.
        image_width: Image width.
        image_height: Image height.

    Returns:
        Tuple of:
            - tile_min: Minimum tile indices [N, 2].
            - tile_max: Maximum tile indices [N, 2].
    """
    num_tiles_x = (image_width + tile_size - 1) // tile_size
    num_tiles_y = (image_height + tile_size - 1) // tile_size

    # Compute bounding box in pixels
    min_xy = uv - radii.unsqueeze(-1)
    max_xy = uv + radii.unsqueeze(-1)

    # Convert to tile indices
    tile_min = torch.floor(min_xy / tile_size).long()
    tile_max = torch.floor(max_xy / tile_size).long() + 1

    # Clamp to valid range
    tile_min = torch.clamp(tile_min, min=0)
    tile_max[:, 0] = torch.clamp(tile_max[:, 0], max=num_tiles_x)
    tile_max[:, 1] = torch.clamp(tile_max[:, 1], max=num_tiles_y)

    return tile_min, tile_max
