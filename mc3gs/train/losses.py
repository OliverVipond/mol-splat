"""Loss functions for MC-3GS training."""

import torch
import torch.nn.functional as F
from torch import Tensor


def rgb_l2_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Compute L2 (MSE) loss between predicted and target images.

    Args:
        pred: Predicted image [C, H, W] or [B, C, H, W].
        target: Target image [C, H, W] or [B, C, H, W].
        mask: Optional mask [H, W] or [B, H, W].

    Returns:
        Scalar loss value.
    """
    diff_sq = (pred - target) ** 2

    if mask is not None:
        if diff_sq.dim() == 3:
            mask = mask.unsqueeze(0)  # [1, H, W]
        elif diff_sq.dim() == 4:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        diff_sq = diff_sq * mask
        return diff_sq.sum() / (mask.sum() * pred.shape[-3] + 1e-8)

    return diff_sq.mean()


def rgb_l1_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Compute L1 loss between predicted and target images.

    Args:
        pred: Predicted image [C, H, W] or [B, C, H, W].
        target: Target image [C, H, W] or [B, C, H, W].
        mask: Optional mask [H, W] or [B, H, W].

    Returns:
        Scalar loss value.
    """
    diff = torch.abs(pred - target)

    if mask is not None:
        if diff.dim() == 3:
            mask = mask.unsqueeze(0)
        elif diff.dim() == 4:
            mask = mask.unsqueeze(1)
        diff = diff * mask
        return diff.sum() / (mask.sum() * pred.shape[-3] + 1e-8)

    return diff.mean()


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> Tensor:
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    return g / g.sum()


def _create_ssim_window(size: int, channels: int, device: torch.device) -> Tensor:
    """Create SSIM window (2D Gaussian kernel)."""
    _1D_window = _gaussian_kernel(size, 1.5, device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()
    window = _2D_window.expand(channels, 1, size, size).contiguous()
    return window


def ssim_loss(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> Tensor:
    """Compute SSIM loss (1 - SSIM).

    Args:
        pred: Predicted image [C, H, W] or [B, C, H, W].
        target: Target image [C, H, W] or [B, C, H, W].
        window_size: Size of the Gaussian window.
        size_average: If True, average over all elements.

    Returns:
        Scalar loss value (1 - SSIM).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    _, channels, height, width = pred.shape

    # Create window
    window = _create_ssim_window(window_size, channels, pred.device)

    # Constants for numerical stability
    C1 = 0.01**2
    C2 = 0.03**2

    # Compute means
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute variances
    sigma1_sq = F.conv2d(pred**2, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target**2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return 1 - ssim_map.mean()
    return 1 - ssim_map


def compute_ssim(
    pred: Tensor,
    target: Tensor,
    window_size: int = 11,
) -> Tensor:
    """Compute SSIM value (not loss).

    Args:
        pred: Predicted image.
        target: Target image.
        window_size: Size of the Gaussian window.

    Returns:
        SSIM value (higher is better).
    """
    return 1 - ssim_loss(pred, target, window_size)


def psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> Tensor:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image.
        target: Target image.
        max_val: Maximum value of the signal.

    Returns:
        PSNR in dB.
    """
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(max_val**2 / (mse + 1e-10))


def total_loss(
    pred: Tensor,
    target: Tensor,
    lambda_l2: float = 0.8,
    lambda_ssim: float = 0.2,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute total photometric loss.

    Args:
        pred: Predicted image [C, H, W] or [B, C, H, W].
        target: Target image [C, H, W] or [B, C, H, W].
        lambda_l2: Weight for L2 loss.
        lambda_ssim: Weight for SSIM loss.
        mask: Optional mask.

    Returns:
        Dictionary with individual and total losses.
    """
    l2 = rgb_l2_loss(pred, target, mask)
    ssim = ssim_loss(pred, target)

    total = lambda_l2 * l2 + lambda_ssim * ssim

    return {
        "l2": l2,
        "ssim": ssim,
        "total": total,
    }


def opacity_sparsity_loss(opacities: Tensor, target_sparsity: float = 0.5) -> Tensor:
    """Encourage sparse opacity distribution.

    Args:
        opacities: Opacity values [N].
        target_sparsity: Target mean opacity.

    Returns:
        Sparsity loss.
    """
    return torch.abs(opacities.mean() - target_sparsity)


def scale_regularization_loss(
    scales: Tensor,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
) -> Tensor:
    """Penalize scales outside a reasonable range.

    Args:
        scales: Scale values [M].
        min_scale: Minimum allowed scale.
        max_scale: Maximum allowed scale.

    Returns:
        Scale regularization loss.
    """
    below_min = F.relu(min_scale - scales)
    above_max = F.relu(scales - max_scale)
    return (below_min**2 + above_max**2).mean()
