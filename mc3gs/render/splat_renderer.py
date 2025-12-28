"""Reference (pure PyTorch) Gaussian splat renderer.

This is a slow but readable implementation for validation and debugging.
For production use, prefer the CUDA backend.
"""

import torch
from torch import Tensor

from mc3gs.render.backend import BaseBackend
from mc3gs.render.projection import project_gaussians
from mc3gs.render.sh import shade_sh


class ReferenceSplatRenderer(BaseBackend):
    """Pure PyTorch reference implementation of Gaussian splatting.

    This renderer is intentionally simple and slow, prioritizing
    correctness and readability over performance.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initialize reference renderer.

        Args:
            device: Target device.
        """
        super().__init__(device)

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
        """Render Gaussians using front-to-back alpha compositing.

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
            Dictionary with rendered image and alpha.
        """
        self.check_inputs(positions, covariances, opacities)

        if background is None:
            background = torch.zeros(3, device=self.device)
        else:
            background = background.to(self.device)

        n = positions.shape[0]
        if n == 0:
            image = background.view(3, 1, 1).expand(3, height, width).clone()
            alpha = torch.zeros(height, width, device=self.device)
            return {"image": image, "alpha": alpha}

        # Project to 2D
        uv, cov2d, depth, valid = project_gaussians(
            positions, covariances, K, R, t
        )

        # Filter invalid Gaussians
        valid_indices = torch.where(valid)[0]
        if len(valid_indices) == 0:
            image = background.view(3, 1, 1).expand(3, height, width).clone()
            alpha = torch.zeros(height, width, device=self.device)
            return {"image": image, "alpha": alpha}

        # Sort by depth (front to back)
        valid_depth = depth[valid_indices]
        sort_order = torch.argsort(valid_depth)
        sorted_indices = valid_indices[sort_order]

        # Get sorted data
        uv = uv[sorted_indices]
        cov2d = cov2d[sorted_indices]
        opacity = opacities[sorted_indices]
        sh = sh_coeffs[sorted_indices]

        # Compute view directions for color
        view_dirs = camera_center - positions[sorted_indices]
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)

        # Evaluate colors from SH
        colors = shade_sh(sh, view_dirs, sh_degree)  # [N_valid, 3]
        colors = torch.clamp(colors, 0, 1)

        # Precompute inverse covariances
        cov2d_inv = torch.linalg.inv(cov2d)  # [N_valid, 2, 2]

        # Create pixel grid
        y_coords = torch.arange(height, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(width, device=self.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        pixels = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        # Initialize accumulators
        image = torch.zeros(height, width, 3, device=self.device)
        transmittance = torch.ones(height, width, device=self.device)

        # Front-to-back compositing
        for i in range(len(sorted_indices)):
            if transmittance.max() < 1e-4:
                break  # Early termination

            # Compute Gaussian weight at all pixels
            diff = pixels - uv[i]  # [H, W, 2]
            cov_inv = cov2d_inv[i]  # [2, 2]

            # Mahalanobis distance: (diff @ cov_inv @ diff^T)
            quad_form = torch.einsum("hwi,ij,hwj->hw", diff, cov_inv, diff)
            weight = torch.exp(-0.5 * quad_form)

            # Alpha contribution
            alpha_i = opacity[i] * weight

            # Accumulate color
            color_i = colors[i]  # [3]
            contribution = transmittance * alpha_i

            image = image + contribution.unsqueeze(-1) * color_i

            # Update transmittance
            transmittance = transmittance * (1 - alpha_i)

        # Add background
        image = image + transmittance.unsqueeze(-1) * background

        # Convert to [C, H, W]
        image = image.permute(2, 0, 1)
        alpha = 1 - transmittance

        return {"image": image, "alpha": alpha}

    def render_tiled(
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
        tile_size: int = 16,
        background: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Tiled rendering for better memory efficiency.

        Args:
            tile_size: Size of tiles in pixels.
            Other args: Same as render().

        Returns:
            Same as render().
        """
        self.check_inputs(positions, covariances, opacities)

        if background is None:
            background = torch.zeros(3, device=self.device)

        # Project all Gaussians once
        uv, cov2d, depth, valid = project_gaussians(
            positions, covariances, K, R, t
        )

        # Compute radii for culling
        from mc3gs.render.projection import compute_gaussian_2d_extent
        radii = compute_gaussian_2d_extent(cov2d, opacities)

        # Sort by depth
        valid_indices = torch.where(valid)[0]
        if len(valid_indices) == 0:
            image = background.view(3, 1, 1).expand(3, height, width).clone()
            alpha = torch.zeros(height, width, device=self.device)
            return {"image": image, "alpha": alpha}

        valid_depth = depth[valid_indices]
        sort_order = torch.argsort(valid_depth)
        sorted_indices = valid_indices[sort_order]

        # Precompute sorted data
        uv_sorted = uv[sorted_indices]
        cov2d_sorted = cov2d[sorted_indices]
        opacity_sorted = opacities[sorted_indices]
        sh_sorted = sh_coeffs[sorted_indices]
        radii_sorted = radii[sorted_indices]

        # View directions
        view_dirs = camera_center - positions[sorted_indices]
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)
        colors_sorted = torch.clamp(shade_sh(sh_sorted, view_dirs, sh_degree), 0, 1)

        # Inverse covariances
        cov2d_inv_sorted = torch.linalg.inv(cov2d_sorted)

        # Allocate output
        image = torch.zeros(height, width, 3, device=self.device)
        transmittance = torch.ones(height, width, device=self.device)

        # Process tiles
        num_tiles_y = (height + tile_size - 1) // tile_size
        num_tiles_x = (width + tile_size - 1) // tile_size

        for ty in range(num_tiles_y):
            for tx in range(num_tiles_x):
                # Tile bounds
                y0 = ty * tile_size
                y1 = min(y0 + tile_size, height)
                x0 = tx * tile_size
                x1 = min(x0 + tile_size, width)

                tile_center = torch.tensor(
                    [(x0 + x1) / 2, (y0 + y1) / 2],
                    device=self.device,
                )
                tile_radius = tile_size * 0.707  # sqrt(2) / 2

                # Find Gaussians that overlap this tile
                dist_to_tile = torch.norm(uv_sorted - tile_center, dim=-1)
                in_tile = dist_to_tile < (radii_sorted + tile_radius)
                tile_indices = torch.where(in_tile)[0]

                if len(tile_indices) == 0:
                    continue

                # Create pixel grid for this tile
                y_coords = torch.arange(y0, y1, device=self.device, dtype=torch.float32)
                x_coords = torch.arange(x0, x1, device=self.device, dtype=torch.float32)
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
                pixels = torch.stack([grid_x, grid_y], dim=-1)  # [TH, TW, 2]

                tile_h, tile_w = y1 - y0, x1 - x0

                # Process Gaussians for this tile
                for idx in tile_indices:
                    diff = pixels - uv_sorted[idx]
                    cov_inv = cov2d_inv_sorted[idx]
                    quad_form = torch.einsum("hwi,ij,hwj->hw", diff, cov_inv, diff)
                    weight = torch.exp(-0.5 * quad_form)

                    alpha_i = opacity_sorted[idx] * weight
                    color_i = colors_sorted[idx]

                    tile_trans = transmittance[y0:y1, x0:x1]
                    contribution = tile_trans * alpha_i

                    image[y0:y1, x0:x1] += contribution.unsqueeze(-1) * color_i
                    transmittance[y0:y1, x0:x1] *= (1 - alpha_i)

        # Add background
        image = image + transmittance.unsqueeze(-1) * background

        # Convert to [C, H, W]
        image = image.permute(2, 0, 1)
        alpha = 1 - transmittance

        return {"image": image, "alpha": alpha}
