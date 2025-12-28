"""Camera data structures and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class Camera:
    """Camera with intrinsics and extrinsics.

    Attributes:
        K: Intrinsic matrix (3x3).
        R: Rotation matrix world-to-camera (3x3).
        t: Translation vector world-to-camera (3,).
        width: Image width in pixels.
        height: Image height in pixels.
        image_path: Optional path to associated image.
    """

    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    width: int
    height: int
    image_path: Path | None = None

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates."""
        return -self.R.T @ self.t

    @property
    def projection_matrix(self) -> np.ndarray:
        """Full projection matrix P = K @ [R | t]."""
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt

    def to_tensors(self, device: str = "cpu") -> dict[str, Tensor]:
        """Convert camera parameters to PyTorch tensors.

        Args:
            device: Target device for tensors.

        Returns:
            Dictionary containing K, R, t, center as tensors.
        """
        return {
            "K": torch.from_numpy(self.K).float().to(device),
            "R": torch.from_numpy(self.R).float().to(device),
            "t": torch.from_numpy(self.t).float().to(device),
            "center": torch.from_numpy(self.center).float().to(device),
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_opencv(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        rvec: np.ndarray,
        tvec: np.ndarray,
        width: int,
        height: int,
        image_path: Path | None = None,
    ) -> "Camera":
        """Create camera from OpenCV-style parameters.

        Args:
            fx: Focal length x.
            fy: Focal length y.
            cx: Principal point x.
            cy: Principal point y.
            rvec: Rodrigues rotation vector (3,).
            tvec: Translation vector (3,).
            width: Image width.
            height: Image height.
            image_path: Optional path to image.

        Returns:
            Camera instance.
        """
        import cv2

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        return cls(
            K=K,
            R=R,
            t=tvec.flatten(),
            width=width,
            height=height,
            image_path=image_path,
        )


class CameraDataset(Dataset[Camera]):
    """Dataset of cameras for batched processing."""

    def __init__(self, cameras: list[Camera]) -> None:
        """Initialize camera dataset.

        Args:
            cameras: List of Camera instances.
        """
        self.cameras = cameras

    def __len__(self) -> int:
        """Return number of cameras."""
        return len(self.cameras)

    def __getitem__(self, idx: int) -> Camera:
        """Get camera by index."""
        return self.cameras[idx]

    def __iter__(self) -> Iterator[Camera]:
        """Iterate over cameras."""
        return iter(self.cameras)

    @classmethod
    def from_transforms_json(cls, path: Path) -> "CameraDataset":
        """Load cameras from transforms.json (NeRF format).

        Args:
            path: Path to transforms.json file.

        Returns:
            CameraDataset instance.
        """
        import json

        with open(path) as f:
            data = json.load(f)

        cameras = []
        # Extract camera intrinsics
        if "camera_angle_x" in data:
            # NeRF synthetic format
            angle_x = data["camera_angle_x"]
            # Will need image dimensions to compute focal length
            # For now, use placeholder
            fl_x = None
        elif "fl_x" in data:
            fl_x = data["fl_x"]
            fl_y = data.get("fl_y", fl_x)
            cx = data.get("cx", data.get("w", 800) / 2)
            cy = data.get("cy", data.get("h", 800) / 2)
        else:
            raise ValueError("Unsupported transforms.json format")

        for frame in data["frames"]:
            c2w = np.array(frame["transform_matrix"])

            # Convert camera-to-world to world-to-camera
            R = c2w[:3, :3].T
            t = -R @ c2w[:3, 3]

            # Get per-frame intrinsics if available
            if fl_x is None:
                w = frame.get("w", 800)
                h = frame.get("h", 800)
                focal = 0.5 * w / np.tan(0.5 * angle_x)
                K = np.array(
                    [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64
                )
            else:
                w = data.get("w", 800)
                h = data.get("h", 800)
                K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float64)

            image_path = None
            if "file_path" in frame:
                image_path = path.parent / frame["file_path"]
                if not image_path.suffix:
                    # Try common extensions
                    for ext in [".png", ".jpg", ".jpeg"]:
                        if image_path.with_suffix(ext).exists():
                            image_path = image_path.with_suffix(ext)
                            break

            cameras.append(
                Camera(K=K, R=R, t=t, width=w, height=h, image_path=image_path)
            )

        return cls(cameras)
