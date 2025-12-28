"""Rendering backend abstraction."""

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import Tensor


class RenderBackend(Protocol):
    """Protocol for rendering backends."""

    def render(
        self,
        positions: Tensor,
        covariances: Tensor,
        opacities: Tensor,
        sh_coeffs: Tensor,
        K: Tensor,
        R: Tensor,
        t: Tensor,
        camera_center: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        background: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Render Gaussians to an image.

        Args:
            positions: 3D positions [N, 3].
            covariances: 3D covariances [N, 3, 3].
            opacities: Opacity values [N].
            sh_coeffs: SH coefficients [N, B, 3].
            K: Intrinsic matrix [3, 3].
            R: Rotation matrix [3, 3].
            t: Translation vector [3].
            camera_center: Camera center in world coords [3].
            width: Image width.
            height: Image height.
            sh_degree: Active SH degree.
            background: Background color [3]. Default: black.

        Returns:
            Dictionary with:
                - image: Rendered image [3, H, W].
                - depth: Depth map [H, W] (optional).
                - alpha: Alpha map [H, W] (optional).
        """
        ...


class BaseBackend(ABC):
    """Base class for rendering backends."""

    def __init__(self, device: str = "cuda") -> None:
        """Initialize backend.

        Args:
            device: Target device.
        """
        self.device = device

    @abstractmethod
    def render(
        self,
        positions: Tensor,
        covariances: Tensor,
        opacities: Tensor,
        sh_coeffs: Tensor,
        K: Tensor,
        R: Tensor,
        t: Tensor,
        camera_center: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        background: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Render Gaussians to an image."""
        ...

    def check_inputs(
        self,
        positions: Tensor,
        covariances: Tensor,
        opacities: Tensor,
    ) -> None:
        """Validate input tensor shapes.

        Args:
            positions: Should be [N, 3].
            covariances: Should be [N, 3, 3].
            opacities: Should be [N].
        """
        n = positions.shape[0]
        assert positions.shape == (n, 3), f"positions: {positions.shape}"
        assert covariances.shape == (n, 3, 3), f"covariances: {covariances.shape}"
        assert opacities.shape == (n,), f"opacities: {opacities.shape}"


def get_backend(name: str = "reference", device: str = "cuda") -> RenderBackend:
    """Get a rendering backend by name.

    Args:
        name: Backend name ("reference" or "cuda").
        device: Target device.

    Returns:
        RenderBackend instance.
    """
    if name == "reference":
        from mc3gs.render.splat_renderer import ReferenceSplatRenderer
        return ReferenceSplatRenderer(device=device)
    elif name == "cuda":
        try:
            from mc3gs.render.cuda_backend import CUDASplatRenderer
            return CUDASplatRenderer(device=device)
        except ImportError:
            raise ImportError(
                "CUDA backend requires diff-gaussian-rasterization package. "
                "Install with: pip install 'mc3gs[cuda]'"
            )
    else:
        raise ValueError(f"Unknown backend: {name}")
