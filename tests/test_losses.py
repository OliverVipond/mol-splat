"""Tests for loss functions."""

import pytest
import torch

from mc3gs.train.losses import compute_ssim, psnr, rgb_l1_loss, rgb_l2_loss, ssim_loss, total_loss


class TestRGBLoss:
    """Tests for RGB loss functions."""

    def test_l2_loss_zero_for_identical(self):
        """Test L2 loss is zero for identical images."""
        img = torch.rand(3, 64, 64)
        loss = rgb_l2_loss(img, img)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_l2_loss_positive(self):
        """Test L2 loss is positive for different images."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)

        loss = rgb_l2_loss(img1, img2)

        assert loss > 0

    def test_l1_loss_zero_for_identical(self):
        """Test L1 loss is zero for identical images."""
        img = torch.rand(3, 64, 64)
        loss = rgb_l1_loss(img, img)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_masked_loss(self):
        """Test masked loss computation."""
        img1 = torch.zeros(3, 64, 64)
        img2 = torch.ones(3, 64, 64)

        # Half mask
        mask = torch.zeros(64, 64)
        mask[:32, :] = 1

        loss = rgb_l2_loss(img1, img2, mask=mask)

        # Loss should be non-zero only in masked region
        assert loss > 0


class TestSSIMLoss:
    """Tests for SSIM loss function."""

    def test_ssim_zero_for_identical(self):
        """Test SSIM loss is near zero for identical images."""
        img = torch.rand(3, 64, 64)
        loss = ssim_loss(img, img)

        assert loss < 0.01

    def test_ssim_positive_for_different(self):
        """Test SSIM loss is positive for different images."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)

        loss = ssim_loss(img1, img2)

        assert loss > 0

    def test_ssim_bounded(self):
        """Test SSIM loss is in [0, 1]."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)

        loss = ssim_loss(img1, img2)

        assert 0 <= loss <= 1


class TestPSNR:
    """Tests for PSNR computation."""

    def test_psnr_high_for_identical(self):
        """Test PSNR is very high for identical images."""
        img = torch.rand(3, 64, 64)
        result = psnr(img, img)

        assert result > 50  # Very high PSNR for identical images

    def test_psnr_low_for_different(self):
        """Test PSNR is lower for different images."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)

        result = psnr(img1, img2)

        assert result < 30


class TestTotalLoss:
    """Tests for combined loss function."""

    def test_total_loss_contains_components(self):
        """Test total loss returns all components."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)

        losses = total_loss(img1, img2)

        assert "l2" in losses
        assert "ssim" in losses
        assert "total" in losses

    def test_total_loss_weighted_sum(self):
        """Test total loss is weighted sum of components."""
        img1 = torch.rand(3, 64, 64)
        img2 = torch.rand(3, 64, 64)

        lambda_l2, lambda_ssim = 0.7, 0.3
        losses = total_loss(img1, img2, lambda_l2=lambda_l2, lambda_ssim=lambda_ssim)

        expected_total = lambda_l2 * losses["l2"] + lambda_ssim * losses["ssim"]
        assert torch.allclose(losses["total"], expected_total, atol=1e-5)
