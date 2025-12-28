"""Image loading and dataset utilities."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from mc3gs.data.cameras import Camera


def load_image(
    path: Path,
    scale: float = 1.0,
    white_background: bool = False,
) -> np.ndarray:
    """Load and preprocess an image.

    Args:
        path: Path to image file.
        scale: Scale factor for resizing (0 < scale <= 1).
        white_background: If True, blend alpha with white background.

    Returns:
        Image as numpy array with shape (H, W, 3), normalized to [0, 1].
    """
    img = Image.open(path)

    # Handle scaling
    if scale != 1.0:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to numpy
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Handle alpha channel
    if img_array.ndim == 2:
        # Grayscale -> RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        # RGBA -> RGB with background blending
        rgb = img_array[..., :3]
        alpha = img_array[..., 3:4]
        if white_background:
            background = np.ones_like(rgb)
        else:
            background = np.zeros_like(rgb)
        img_array = rgb * alpha + background * (1 - alpha)
    elif img_array.shape[-1] != 3:
        raise ValueError(f"Unexpected number of channels: {img_array.shape[-1]}")

    return img_array


def image_to_tensor(img: np.ndarray, device: str = "cpu") -> Tensor:
    """Convert image array to PyTorch tensor.

    Args:
        img: Image array with shape (H, W, 3).
        device: Target device.

    Returns:
        Tensor with shape (3, H, W).
    """
    return torch.from_numpy(img).permute(2, 0, 1).float().to(device)


def tensor_to_image(tensor: Tensor) -> np.ndarray:
    """Convert PyTorch tensor to image array.

    Args:
        tensor: Tensor with shape (3, H, W) or (H, W, 3).

    Returns:
        Image array with shape (H, W, 3), values in [0, 1].
    """
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    return tensor.detach().cpu().numpy().clip(0, 1)


class ImageDataset(Dataset[tuple[Tensor, Camera]]):
    """Dataset of images with associated cameras."""

    def __init__(
        self,
        cameras: Sequence[Camera],
        scale: float = 1.0,
        white_background: bool = False,
        preload: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize image dataset.

        Args:
            cameras: Sequence of Camera instances with image_path set.
            scale: Scale factor for images.
            white_background: Use white background for alpha blending.
            preload: If True, preload all images into memory.
            device: Target device for tensors.
        """
        self.cameras = list(cameras)
        self.scale = scale
        self.white_background = white_background
        self.device = device

        # Validate that all cameras have image paths
        for i, cam in enumerate(self.cameras):
            if cam.image_path is None:
                raise ValueError(f"Camera {i} has no image path")
            if not cam.image_path.exists():
                raise FileNotFoundError(f"Image not found: {cam.image_path}")

        # Preload images if requested
        self._images: list[Tensor] | None = None
        if preload:
            self._images = [self._load_image(i) for i in range(len(self.cameras))]

    def _load_image(self, idx: int) -> Tensor:
        """Load a single image by index."""
        cam = self.cameras[idx]
        assert cam.image_path is not None
        img = load_image(cam.image_path, self.scale, self.white_background)
        return image_to_tensor(img, self.device)

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.cameras)

    def __getitem__(self, idx: int) -> tuple[Tensor, Camera]:
        """Get image and camera by index.

        Returns:
            Tuple of (image tensor [3, H, W], camera).
        """
        if self._images is not None:
            image = self._images[idx]
        else:
            image = self._load_image(idx)
        return image, self.cameras[idx]

    @classmethod
    def from_directory(
        cls,
        images_dir: Path,
        cameras: Sequence[Camera],
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        scale: float = 1.0,
        white_background: bool = False,
        preload: bool = False,
        device: str = "cpu",
    ) -> "ImageDataset":
        """Create dataset from a directory of images.

        Args:
            images_dir: Directory containing images.
            cameras: Cameras corresponding to images (matched by order).
            extensions: Valid image extensions.
            scale: Scale factor.
            white_background: Use white background.
            preload: Preload images.
            device: Target device.

        Returns:
            ImageDataset instance.
        """
        # Find all images
        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in extensions
        )

        if len(image_paths) != len(cameras):
            raise ValueError(
                f"Number of images ({len(image_paths)}) != "
                f"number of cameras ({len(cameras)})"
            )

        # Update cameras with image paths
        updated_cameras = []
        for cam, img_path in zip(cameras, image_paths, strict=True):
            updated_cam = Camera(
                K=cam.K,
                R=cam.R,
                t=cam.t,
                width=cam.width,
                height=cam.height,
                image_path=img_path,
            )
            updated_cameras.append(updated_cam)

        return cls(
            updated_cameras,
            scale=scale,
            white_background=white_background,
            preload=preload,
            device=device,
        )
