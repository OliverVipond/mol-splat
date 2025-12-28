"""Export utilities for Gaussian splat data."""

import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from mc3gs.model.scene import Scene
from mc3gs.render.sh import C0


def export_gaussians_ply(
    scene: Scene,
    path: Path | str,
    include_sh: bool = True,
    max_sh_degree: int = 3,
) -> None:
    """Export scene Gaussians to PLY format.

    This format is compatible with standard Gaussian splatting viewers.

    Args:
        scene: Scene to export.
        path: Output PLY file path.
        include_sh: Whether to include SH coefficients.
        max_sh_degree: Maximum SH degree to export.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Gather all Gaussians
    with torch.no_grad():
        data = scene.gather()

        positions = data["positions"].cpu().numpy()
        covariances = data["covariances"].cpu().numpy()
        opacities = data["opacities"].cpu().numpy()
        sh_coeffs = data["sh_coeffs"].cpu().numpy()

    n = positions.shape[0]

    # Decompose covariance into scale and rotation
    # Σ = R @ S @ S @ R^T where S is diagonal
    scales = np.zeros((n, 3), dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)  # quaternion

    for i in range(n):
        cov = covariances[i]
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        scales[i] = np.sqrt(eigenvalues)

        # Convert rotation matrix to quaternion
        R = eigenvectors
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1

        # Matrix to quaternion
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        rotations[i] = [w, x, y, z]

    # Build PLY data
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),  # unused but standard
        ("ny", "f4"),
        ("nz", "f4"),
    ]

    # Add SH coefficients for DC term (as RGB)
    dtype.extend([
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ])

    # Add rest of SH if requested
    if include_sh:
        num_sh = (max_sh_degree + 1) ** 2 - 1  # Exclude DC
        for i in range(num_sh):
            for c in range(3):
                dtype.append((f"f_rest_{i * 3 + c}", "f4"))

    # Add opacity and scale/rotation
    dtype.extend([
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ])

    # Create structured array
    elements = np.empty(n, dtype=dtype)

    elements["x"] = positions[:, 0]
    elements["y"] = positions[:, 1]
    elements["z"] = positions[:, 2]
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0

    # DC term
    elements["f_dc_0"] = sh_coeffs[:, 0, 0]
    elements["f_dc_1"] = sh_coeffs[:, 0, 1]
    elements["f_dc_2"] = sh_coeffs[:, 0, 2]

    # Rest of SH
    if include_sh:
        num_sh = min((max_sh_degree + 1) ** 2 - 1, sh_coeffs.shape[1] - 1)
        for i in range(num_sh):
            for c in range(3):
                elements[f"f_rest_{i * 3 + c}"] = sh_coeffs[:, i + 1, c]

    # Opacity (store as logit for compatibility)
    elements["opacity"] = np.log(opacities / (1 - opacities + 1e-8))

    # Scale (store as log)
    elements["scale_0"] = np.log(scales[:, 0])
    elements["scale_1"] = np.log(scales[:, 1])
    elements["scale_2"] = np.log(scales[:, 2])

    # Rotation (quaternion)
    elements["rot_0"] = rotations[:, 0]
    elements["rot_1"] = rotations[:, 1]
    elements["rot_2"] = rotations[:, 2]
    elements["rot_3"] = rotations[:, 3]

    # Write PLY
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(str(path))


def export_scene_json(
    scene: Scene,
    path: Path | str,
    include_gaussians: bool = False,
) -> None:
    """Export scene as JSON with molecule transforms and color schemes.

    This format is useful for custom viewers and analysis.

    Args:
        scene: Scene to export.
        path: Output JSON file path.
        include_gaussians: Whether to include full Gaussian data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    molecules = []

    with torch.no_grad():
        for idx, instance in enumerate(scene.instances):
            mol_data = {
                "id": idx,
                "template_name": instance.template.name,
                "num_gaussians": instance.template.num_gaussians,
                "transform": {
                    "translation": instance.translation.cpu().tolist(),
                    "rotation": instance.rotation.cpu().tolist(),
                    "scale": instance.scale.item(),
                },
                "color_scheme": {},
            }

            # Extract per-type colors (DC component)
            sh_coeffs = instance.sh_bank.sh_coeffs.cpu()
            for type_id in instance.template.unique_types.cpu().tolist():
                dc = sh_coeffs[type_id, 0, :]
                rgb = (dc * C0).clamp(0, 1).tolist()
                label = instance.template.type_vocab.get_label(type_id)
                mol_data["color_scheme"][label] = {
                    "type_id": type_id,
                    "rgb": rgb,
                    "sh_dc": dc.tolist(),
                }

            if include_gaussians:
                gaussians = instance.world_gaussians()
                mol_data["gaussians"] = {
                    "positions": gaussians["positions"].cpu().tolist(),
                    "opacities": gaussians["opacities"].cpu().tolist(),
                }

            molecules.append(mol_data)

    scene_data = {
        "num_molecules": len(scene),
        "total_gaussians": scene.total_gaussians,
        "molecules": molecules,
    }

    with open(path, "w") as f:
        json.dump(scene_data, f, indent=2)


def export_colors_csv(
    scene: Scene,
    path: Path | str,
) -> None:
    """Export per-molecule, per-type colors to CSV.

    Args:
        scene: Scene to export.
        path: Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append("molecule_id,type_label,type_id,r,g,b")

    with torch.no_grad():
        for mol_id, instance in enumerate(scene.instances):
            sh_coeffs = instance.sh_bank.sh_coeffs.cpu()
            for type_id in instance.template.unique_types.cpu().tolist():
                dc = sh_coeffs[type_id, 0, :]
                rgb = (dc * C0).clamp(0, 1)
                label = instance.template.type_vocab.get_label(type_id)
                rows.append(
                    f"{mol_id},{label},{type_id},"
                    f"{rgb[0].item():.4f},{rgb[1].item():.4f},{rgb[2].item():.4f}"
                )

    with open(path, "w") as f:
        f.write("\n".join(rows))
