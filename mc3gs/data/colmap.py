"""COLMAP data loading utilities."""

import struct
from collections.abc import Generator
from pathlib import Path

import numpy as np

from mc3gs.data.cameras import Camera, CameraDataset


def _read_next_bytes(
    fid,
    num_bytes: int,
    format_char_sequence: str,
    endian_character: str = "<",
) -> tuple:
    """Read and unpack bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def _parse_cameras_binary(path: Path) -> dict[int, dict]:
    """Parse COLMAP cameras.bin file."""
    cameras = {}

    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            camera_properties = _read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]

            # Read model parameters
            num_params = {
                0: 3,  # SIMPLE_PINHOLE
                1: 4,  # PINHOLE
                2: 4,  # SIMPLE_RADIAL
                3: 5,  # RADIAL
                4: 8,  # OPENCV
                5: 12,  # OPENCV_FISHEYE
                6: 12,  # FULL_OPENCV
            }.get(model_id, 4)

            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }

    return cameras


def _parse_images_binary(path: Path) -> dict[int, dict]:
    """Parse COLMAP images.bin file."""
    images = {}

    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            binary_image_properties = _read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qw, qx, qy, qz = binary_image_properties[1:5]
            tx, ty, tz = binary_image_properties[5:8]
            camera_id = binary_image_properties[8]

            # Read image name
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]

            # Read 2D points (skip them)
            num_points2D = _read_next_bytes(fid, 8, "Q")[0]
            _ = _read_next_bytes(fid, 24 * num_points2D, "ddi" * num_points2D)

            images[image_id] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
                "camera_id": camera_id,
                "name": image_name,
            }

    return images


def _parse_points3D_binary(path: Path) -> Generator[dict, None, None]:
    """Parse COLMAP points3D.bin file as generator."""
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):
            binary_point = _read_next_bytes(fid, 43, "QdddBBBd")
            point_id = binary_point[0]
            xyz = np.array(binary_point[1:4])
            rgb = np.array(binary_point[4:7])
            error = binary_point[7]

            # Read track (skip)
            track_length = _read_next_bytes(fid, 8, "Q")[0]
            _ = _read_next_bytes(fid, 8 * track_length, "ii" * track_length)

            yield {
                "id": point_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
            }


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )


def _build_intrinsic_matrix(camera_data: dict) -> np.ndarray:
    """Build intrinsic matrix from COLMAP camera parameters."""
    model_id = camera_data["model_id"]
    params = camera_data["params"]
    width = camera_data["width"]
    height = camera_data["height"]

    if model_id == 0:  # SIMPLE_PINHOLE
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model_id == 1:  # PINHOLE
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    elif model_id in (2, 3, 4, 5, 6):  # Models with distortion
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
        # Note: distortion parameters are ignored for now
    else:
        # Fallback: assume first param is focal length
        fx = fy = params[0]
        cx, cy = width / 2, height / 2

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_colmap_cameras(
    sparse_path: Path,
    images_dir: Path | None = None,
    scale_factor: float = 1.0,
) -> CameraDataset:
    """Load cameras from COLMAP sparse reconstruction.

    Args:
        sparse_path: Path to COLMAP sparse directory (containing cameras.bin, images.bin).
        images_dir: Optional path to images directory. If None, uses sparse_path parent.
        scale_factor: Scale factor for camera positions.

    Returns:
        CameraDataset with loaded cameras.
    """
    cameras_bin = sparse_path / "cameras.bin"
    images_bin = sparse_path / "images.bin"

    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(
            f"COLMAP files not found in {sparse_path}. "
            "Expected cameras.bin and images.bin"
        )

    if images_dir is None:
        images_dir = sparse_path.parent / "images"

    # Parse binary files
    camera_data = _parse_cameras_binary(cameras_bin)
    image_data = _parse_images_binary(images_bin)

    cameras = []
    for image_id in sorted(image_data.keys()):
        img = image_data[image_id]
        cam = camera_data[img["camera_id"]]

        # Build intrinsic matrix
        K = _build_intrinsic_matrix(cam)

        # Build extrinsics
        R = _qvec_to_rotmat(img["qvec"])
        t = img["tvec"] * scale_factor

        # Find image path
        image_path = images_dir / img["name"]

        cameras.append(
            Camera(
                K=K,
                R=R,
                t=t,
                width=cam["width"],
                height=cam["height"],
                image_path=image_path if image_path.exists() else None,
            )
        )

    return CameraDataset(cameras)


def load_colmap_points(
    sparse_path: Path,
    scale_factor: float = 1.0,
    max_error: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray]:
    """Load 3D points from COLMAP sparse reconstruction.

    Args:
        sparse_path: Path to COLMAP sparse directory.
        scale_factor: Scale factor for point positions.
        max_error: Maximum reprojection error to include a point.

    Returns:
        Tuple of (positions [N, 3], colors [N, 3]).
    """
    points_bin = sparse_path / "points3D.bin"

    if not points_bin.exists():
        raise FileNotFoundError(f"points3D.bin not found in {sparse_path}")

    positions = []
    colors = []

    for point in _parse_points3D_binary(points_bin):
        if point["error"] <= max_error:
            positions.append(point["xyz"] * scale_factor)
            colors.append(point["rgb"] / 255.0)

    return np.array(positions, dtype=np.float32), np.array(colors, dtype=np.float32)
