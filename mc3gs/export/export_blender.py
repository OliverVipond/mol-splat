"""Export utilities for Blender integration."""

import json
from pathlib import Path

import torch

from mc3gs.model.scene import Scene
from mc3gs.render.sh import C0


def export_to_blender(
    scene: Scene,
    path: Path | str,
    include_script: bool = True,
) -> None:
    """Export scene data for Blender geometry nodes.

    Creates JSON data and optionally a Python script for importing
    the scene into Blender.

    Args:
        scene: Scene to export.
        path: Output directory path.
        include_script: Whether to include Blender import script.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    molecules = []

    with torch.no_grad():
        for idx, instance in enumerate(scene.instances):
            # Get transform
            translation = instance.translation.cpu().numpy()
            rotation = instance.rotation.cpu().numpy()
            scale = instance.scale.item()

            # Get colors
            colors = {}
            sh_coeffs = instance.sh_bank.sh_coeffs.cpu()
            for type_id in instance.template.unique_types.cpu().tolist():
                dc = sh_coeffs[type_id, 0, :]
                rgb = (dc * C0).clamp(0, 1).tolist()
                label = instance.template.type_vocab.get_label(type_id)
                colors[label] = rgb

            molecules.append({
                "id": idx,
                "template": instance.template.name,
                "translation": translation.tolist(),
                "rotation": rotation.tolist(),  # axis-angle
                "scale": scale,
                "colors": colors,
            })

    # Save transforms
    data = {
        "molecules": molecules,
        "num_molecules": len(molecules),
    }

    with open(path / "molecules.json", "w") as f:
        json.dump(data, f, indent=2)

    # Generate Blender script
    if include_script:
        script = _generate_blender_script()
        with open(path / "import_molecules.py", "w") as f:
            f.write(script)


def _generate_blender_script() -> str:
    """Generate Blender Python script for importing molecules."""
    return '''"""
Blender script for importing MC-3GS molecule instances.

Usage:
1. Open Blender
2. Run this script from the Text Editor
3. Point to the molecules.json file when prompted
"""

import bpy
import json
import math
from mathlib import Vector, Matrix

def axis_angle_to_euler(axis_angle):
    """Convert axis-angle to Euler angles."""
    angle = (axis_angle[0]**2 + axis_angle[1]**2 + axis_angle[2]**2)**0.5
    if angle < 1e-8:
        return (0, 0, 0)
    axis = [a / angle for a in axis_angle]
    # Create rotation matrix and convert to Euler
    # Simplified - for production use mathutils
    return (axis_angle[0], axis_angle[1], axis_angle[2])


def create_atom_material(name, color):
    """Create a material for an atom type."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.3
    return mat


def import_molecules(json_path, molecule_asset=None):
    """Import molecules from JSON file.
    
    Args:
        json_path: Path to molecules.json
        molecule_asset: Optional mesh to instance (creates spheres if None)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create collection for molecules
    collection = bpy.data.collections.new("MC3GS_Molecules")
    bpy.context.scene.collection.children.link(collection)
    
    for mol in data['molecules']:
        # Create empty for molecule instance
        empty = bpy.data.objects.new(f"Molecule_{mol['id']}", None)
        empty.empty_display_type = 'ARROWS'
        empty.empty_display_size = 0.5
        
        # Set transform
        empty.location = mol['translation']
        empty.rotation_euler = axis_angle_to_euler(mol['rotation'])
        empty.scale = (mol['scale'], mol['scale'], mol['scale'])
        
        collection.objects.link(empty)
        
        # Store color data as custom properties
        for atom_type, color in mol['colors'].items():
            empty[f"color_{atom_type}"] = color
    
    print(f"Imported {len(data['molecules'])} molecules")


# Run if called directly
if __name__ == "__main__":
    # You can modify this path
    json_path = "molecules.json"
    import_molecules(json_path)
'''


def export_blender_template(
    template_dict: dict,
    path: Path | str,
) -> None:
    """Export a molecule template for Blender.

    Args:
        template_dict: Template data from chemistry module.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {
        "positions": template_dict["positions"].tolist(),
        "type_ids": template_dict["type_ids"].tolist(),
        "labels": template_dict["labels"],
        "num_gaussians": len(template_dict["positions"]),
    }

    # Add bond connectivity if available
    if "bonds" in template_dict:
        data["bonds"] = template_dict["bonds"]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
