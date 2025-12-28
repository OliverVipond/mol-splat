"""Tests for projection module."""

import pytest
import torch

from mc3gs.render.projection import (
    compute_gaussian_2d_extent,
    project_gaussians,
    project_points,
    projection_jacobian,
)


@pytest.fixture
def camera_params():
    """Create simple camera parameters."""
    K = torch.tensor([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    R = torch.eye(3)
    t = torch.tensor([0.0, 0.0, 5.0])
    return K, R, t


class TestProjectPoints:
    """Tests for point projection."""

    def test_point_at_optical_axis(self, camera_params):
        """Test projection of point on optical axis."""
        K, R, t = camera_params
        points = torch.tensor([[0.0, 0.0, 5.0]])  # At camera center in world

        uv, depth = project_points(points, K, R, t)

        # Should project to principal point
        assert torch.allclose(uv[0], torch.tensor([320.0, 240.0]), atol=1e-3)
        assert torch.allclose(depth[0], torch.tensor(10.0), atol=1e-3)

    def test_multiple_points(self, camera_params):
        """Test projection of multiple points."""
        K, R, t = camera_params
        points = torch.randn(10, 3)
        points[:, 2] += 10  # Ensure positive depth

        uv, depth = project_points(points, K, R, t)

        assert uv.shape == (10, 2)
        assert depth.shape == (10,)
        assert (depth > 0).all()


class TestProjectionJacobian:
    """Tests for projection Jacobian computation."""

    def test_jacobian_shape(self, camera_params):
        """Test Jacobian output shape."""
        K, R, t = camera_params
        points_cam = torch.randn(5, 3)
        points_cam[:, 2] = points_cam[:, 2].abs() + 1  # Positive z

        J = projection_jacobian(points_cam, K)

        assert J.shape == (5, 2, 3)

    def test_jacobian_finite_difference(self, camera_params):
        """Test Jacobian against finite differences."""
        K, R, t = camera_params
        point_cam = torch.tensor([[1.0, 0.5, 3.0]])

        J = projection_jacobian(point_cam, K)

        # Finite difference approximation
        eps = 1e-5
        J_fd = torch.zeros(1, 2, 3)

        for i in range(3):
            p_plus = point_cam.clone()
            p_plus[0, i] += eps
            p_minus = point_cam.clone()
            p_minus[0, i] -= eps

            uv_plus = (K @ p_plus.T).T
            uv_plus = uv_plus[:, :2] / uv_plus[:, 2:3]

            uv_minus = (K @ p_minus.T).T
            uv_minus = uv_minus[:, :2] / uv_minus[:, 2:3]

            J_fd[0, :, i] = (uv_plus - uv_minus).squeeze() / (2 * eps)

        # Use relative tolerance for finite difference comparison
        assert torch.allclose(J, J_fd, rtol=0.01, atol=1.0)


class TestProjectGaussians:
    """Tests for 3D to 2D Gaussian projection."""

    def test_output_shapes(self, camera_params):
        """Test projected Gaussian output shapes."""
        K, R, t = camera_params

        positions = torch.randn(10, 3)
        positions[:, 2] += 10  # Ensure positive depth

        covariances = torch.eye(3).unsqueeze(0).expand(10, -1, -1).clone()

        uv, cov2d, depth, valid = project_gaussians(positions, covariances, K, R, t)

        assert uv.shape == (10, 2)
        assert cov2d.shape == (10, 2, 2)
        assert depth.shape == (10,)
        assert valid.shape == (10,)

    def test_cov2d_is_positive_definite(self, camera_params):
        """Test that projected covariances are positive definite."""
        K, R, t = camera_params

        positions = torch.randn(10, 3)
        positions[:, 2] += 10

        # Random positive definite covariances
        L = torch.randn(10, 3, 3)
        covariances = torch.bmm(L, L.transpose(-1, -2)) + 0.01 * torch.eye(3)

        uv, cov2d, depth, valid = project_gaussians(positions, covariances, K, R, t)

        # Check eigenvalues are positive
        eigenvalues = torch.linalg.eigvalsh(cov2d)
        assert (eigenvalues > 0).all()


class TestGaussianExtent:
    """Tests for 2D Gaussian extent computation."""

    def test_extent_increases_with_opacity(self):
        """Test that extent increases with opacity."""
        cov2d = torch.eye(2).unsqueeze(0).expand(2, -1, -1).clone()
        opacities = torch.tensor([0.1, 0.9])

        radii = compute_gaussian_2d_extent(cov2d, opacities)

        assert radii[1] > radii[0]

    def test_extent_increases_with_variance(self):
        """Test that extent increases with variance."""
        cov2d = torch.stack([
            torch.eye(2),
            torch.eye(2) * 4,
        ])
        opacities = torch.tensor([0.5, 0.5])

        radii = compute_gaussian_2d_extent(cov2d, opacities)

        assert radii[1] > radii[0]
