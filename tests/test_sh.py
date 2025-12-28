"""Tests for spherical harmonics module."""

import math

import pytest
import torch

from mc3gs.render.sh import (
    C0,
    eval_sh_basis,
    initialize_sh_from_color,
    rgb_to_sh_dc,
    sh_dc_to_rgb,
    shade_sh,
)


class TestSHBasis:
    """Tests for SH basis function evaluation."""

    def test_basis_shape_degree_0(self):
        """Test SH basis shape for degree 0."""
        directions = torch.randn(10, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        basis = eval_sh_basis(directions, degree=0)

        assert basis.shape == (10, 1)

    def test_basis_shape_degree_3(self):
        """Test SH basis shape for degree 3."""
        directions = torch.randn(10, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        basis = eval_sh_basis(directions, degree=3)

        # (3+1)^2 = 16 basis functions
        assert basis.shape == (10, 16)

    def test_dc_component_is_constant(self):
        """Test that degree 0 (DC) component is constant."""
        directions = torch.randn(100, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        basis = eval_sh_basis(directions, degree=0)

        # All DC values should be equal to C0
        assert torch.allclose(basis[:, 0], torch.full((100,), C0), atol=1e-6)


class TestShading:
    """Tests for SH shading."""

    def test_dc_only_shading(self):
        """Test shading with only DC component gives constant color."""
        # Create DC-only coefficients
        sh_coeffs = torch.zeros(10, 1, 3)
        sh_coeffs[:, 0, :] = torch.tensor([1.0, 0.5, 0.0])  # Orange-ish

        # Different directions
        directions = torch.randn(10, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        colors = shade_sh(sh_coeffs, directions, active_degree=0)

        # All colors should be the same (view-independent for DC only)
        expected = torch.tensor([1.0, 0.5, 0.0]) * C0
        for i in range(10):
            assert torch.allclose(colors[i], expected, atol=1e-5)

    def test_shading_output_shape(self):
        """Test shading output shape."""
        sh_coeffs = torch.randn(5, 16, 3)  # 5 Gaussians, degree 3
        directions = torch.randn(5, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        colors = shade_sh(sh_coeffs, directions, active_degree=3)

        assert colors.shape == (5, 3)


class TestColorConversion:
    """Tests for RGB <-> SH DC conversion."""

    def test_rgb_to_sh_roundtrip(self):
        """Test RGB -> SH DC -> RGB roundtrip."""
        rgb = torch.tensor([0.5, 0.7, 0.3])

        sh_dc = rgb_to_sh_dc(rgb)
        rgb_back = sh_dc_to_rgb(sh_dc)

        assert torch.allclose(rgb, rgb_back, atol=1e-6)

    def test_initialize_sh_from_color(self):
        """Test SH initialization from color."""
        color = torch.tensor([1.0, 0.0, 0.0])  # Red
        num_types = 5
        degree = 2

        sh = initialize_sh_from_color(color, num_types, degree)

        # Shape: [num_types, (degree+1)^2, 3]
        assert sh.shape == (5, 9, 3)

        # DC should be set
        assert sh[:, 0, :].abs().sum() > 0

        # Higher degrees should be zero
        assert sh[:, 1:, :].abs().sum() == 0
