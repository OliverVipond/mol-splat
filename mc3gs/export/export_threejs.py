"""Export utilities for Three.js web viewer."""

import json
from pathlib import Path

import numpy as np
import torch

from mc3gs.model.scene import Scene
from mc3gs.render.sh import C0


def export_to_threejs(
    scene: Scene,
    path: Path | str,
    include_viewer: bool = True,
) -> None:
    """Export scene for Three.js web viewer.

    Creates JSON data and optionally a minimal HTML viewer.

    Args:
        scene: Scene to export.
        path: Output directory path.
        include_viewer: Whether to include HTML viewer.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Gather all Gaussians
    with torch.no_grad():
        data = scene.gather()

        positions = data["positions"].cpu().numpy()
        opacities = data["opacities"].cpu().numpy()
        sh_coeffs = data["sh_coeffs"].cpu().numpy()
        mol_ids = data["mol_ids"].cpu().numpy()
        type_ids = data["type_ids"].cpu().numpy()

    # Convert SH DC to RGB colors
    colors = (sh_coeffs[:, 0, :] * C0).clip(0, 1)

    # Create compact binary format for positions and colors
    # Position: float32 x 3 = 12 bytes
    # Color: uint8 x 3 = 3 bytes
    # Opacity: float32 = 4 bytes
    # Total: 19 bytes per Gaussian

    n = positions.shape[0]

    # Save as JSON for simplicity (binary would be more efficient)
    gaussians_data = {
        "count": n,
        "positions": positions.astype(np.float32).tolist(),
        "colors": (colors * 255).astype(np.uint8).tolist(),
        "opacities": opacities.astype(np.float32).tolist(),
        "molecule_ids": mol_ids.astype(np.int32).tolist(),
        "type_ids": type_ids.astype(np.int32).tolist(),
    }

    with open(path / "gaussians.json", "w") as f:
        json.dump(gaussians_data, f)

    # Export molecule metadata
    molecules_data = {
        "count": len(scene),
        "molecules": [],
    }

    for idx, instance in enumerate(scene.instances):
        mol = {
            "id": idx,
            "template": instance.template.name,
            "num_gaussians": instance.template.num_gaussians,
            "translation": instance.translation.cpu().tolist(),
            "scale": instance.scale.item(),
        }
        molecules_data["molecules"].append(mol)

    with open(path / "molecules.json", "w") as f:
        json.dump(molecules_data, f, indent=2)

    # Generate HTML viewer
    if include_viewer:
        html = _generate_threejs_viewer()
        with open(path / "index.html", "w") as f:
            f.write(html)


def _generate_threejs_viewer() -> str:
    """Generate a minimal Three.js Gaussian splat viewer."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MC-3GS Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; }
        canvas { display: block; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: monospace;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div id="info">
        <div>MC-3GS Gaussian Splat Viewer</div>
        <div id="stats">Loading...</div>
    </div>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        let scene, camera, renderer, controls;
        let gaussians = null;

        init();
        loadData();

        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            // Camera
            camera = new THREE.PerspectiveCamera(
                75,
                window.innerWidth / window.innerHeight,
                0.1,
                1000
            );
            camera.position.set(5, 5, 5);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.body.appendChild(renderer.domElement);

            // Controls
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 10);
            scene.add(directionalLight);

            // Handle resize
            window.addEventListener('resize', onWindowResize);

            // Start animation loop
            animate();
        }

        async function loadData() {
            try {
                const response = await fetch('gaussians.json');
                const data = await response.json();
                
                createGaussianPoints(data);
                
                document.getElementById('stats').textContent = 
                    `Gaussians: ${data.count}`;
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('stats').textContent = 'Error loading data';
            }
        }

        function createGaussianPoints(data) {
            const geometry = new THREE.BufferGeometry();
            
            // Positions
            const positions = new Float32Array(data.count * 3);
            for (let i = 0; i < data.count; i++) {
                positions[i * 3] = data.positions[i][0];
                positions[i * 3 + 1] = data.positions[i][1];
                positions[i * 3 + 2] = data.positions[i][2];
            }
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

            // Colors
            const colors = new Float32Array(data.count * 3);
            for (let i = 0; i < data.count; i++) {
                colors[i * 3] = data.colors[i][0] / 255;
                colors[i * 3 + 1] = data.colors[i][1] / 255;
                colors[i * 3 + 2] = data.colors[i][2] / 255;
            }
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            // Material
            const material = new THREE.PointsMaterial({
                size: 0.1,
                vertexColors: true,
                transparent: true,
                opacity: 0.8,
                sizeAttenuation: true,
            });

            // Points
            gaussians = new THREE.Points(geometry, material);
            scene.add(gaussians);

            // Center camera on point cloud
            geometry.computeBoundingSphere();
            const center = geometry.boundingSphere.center;
            const radius = geometry.boundingSphere.radius;
            
            camera.position.set(
                center.x + radius * 2,
                center.y + radius * 2,
                center.z + radius * 2
            );
            controls.target.copy(center);
            controls.update();
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
    </script>
</body>
</html>
'''


def export_instanced_threejs(
    scene: Scene,
    path: Path | str,
) -> None:
    """Export scene using Three.js instancing for molecules.

    More efficient for scenes with many molecule instances.

    Args:
        scene: Scene to export.
        path: Output directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Group molecules by template
    template_groups: dict[str, list[dict]] = {}

    with torch.no_grad():
        for idx, instance in enumerate(scene.instances):
            template_name = instance.template.name or f"template_{idx}"

            if template_name not in template_groups:
                # Store template geometry
                template_groups[template_name] = {
                    "geometry": {
                        "positions": instance.template.p_local.cpu().numpy().tolist(),
                        "type_ids": instance.template.type_id.cpu().numpy().tolist(),
                    },
                    "instances": [],
                }

            # Store instance transform
            template_groups[template_name]["instances"].append({
                "translation": instance.translation.cpu().numpy().tolist(),
                "rotation": instance.rotation.cpu().numpy().tolist(),
                "scale": instance.scale.item(),
                "colors": _extract_colors(instance),
            })

    with open(path / "instanced_scene.json", "w") as f:
        json.dump(template_groups, f, indent=2)


def _extract_colors(instance) -> dict:
    """Extract per-type colors from an instance."""
    colors = {}
    sh_coeffs = instance.sh_bank.sh_coeffs.cpu()

    for type_id in instance.template.unique_types.cpu().tolist():
        dc = sh_coeffs[type_id, 0, :]
        rgb = (dc * C0).clamp(0, 1).tolist()
        label = instance.template.type_vocab.get_label(type_id)
        colors[label] = rgb

    return colors
