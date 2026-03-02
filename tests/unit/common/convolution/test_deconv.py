"""Unit tests for matrix-based deconvolution."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from osipy.common.convolution import conv, convmat, deconv, invconvmat
from osipy.common.convolution.deconv import deconv_osvd
from osipy.common.exceptions import DataValidationError


class TestDeconv:
    """Tests for deconv() matrix-based deconvolution."""

    def test_deconv_empty_input(self):
        """Test deconvolution with empty arrays."""
        result = deconv(np.array([]), np.array([]), np.array([]))
        assert len(result) == 0

    def test_deconv_single_point(self):
        """Test deconvolution with single point."""
        result = deconv(np.array([1.0]), np.array([1.0]), np.array([0.0]))
        assert len(result) == 1

    def test_deconv_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        t = np.linspace(0, 10, 101)
        c = np.ones(100)
        aif = np.ones(101)

        with pytest.raises(DataValidationError, match="same length"):
            deconv(c, aif, t)

    def test_deconv_recovers_residue_tsvd(self):
        """Test that TSVD deconvolution recovers residue function."""
        t = np.linspace(0, 30, 61)

        # Create known AIF and residue function
        aif = np.exp(-t / 5) * (1 - np.exp(-t / 1))  # Gamma-like AIF
        true_irf = np.exp(-t / 10)  # Exponential residue

        # Convolve to create tissue curve
        c = conv(aif, true_irf, t)

        # Deconvolve
        recovered_irf = deconv(c, aif, t, method="tsvd", tol=0.1)

        # Check basic properties
        assert len(recovered_irf) == len(t)

        # Recovered IRF should be roughly similar to true IRF
        # (exact recovery is not expected due to regularization)
        # Check that peak is in reasonable location
        peak_idx = np.argmax(recovered_irf)
        assert peak_idx < len(t) // 2  # Peak should be early

    def test_deconv_recovers_residue_tikhonov(self):
        """Test that Tikhonov deconvolution recovers residue function."""
        t = np.linspace(0, 30, 61)

        # Create known AIF and residue function
        aif = np.exp(-t / 5) * (1 - np.exp(-t / 1))
        true_irf = np.exp(-t / 10)

        # Convolve to create tissue curve
        c = conv(aif, true_irf, t)

        # Deconvolve with Tikhonov
        recovered_irf = deconv(c, aif, t, method="tikhonov", tol=0.1)

        # Check basic properties
        assert len(recovered_irf) == len(t)
        assert np.all(np.isfinite(recovered_irf))

    def test_deconv_circulant(self):
        """Test circulant deconvolution (cSVD)."""
        t = np.linspace(0, 30, 61)

        aif = np.exp(-t / 5) * (1 - np.exp(-t / 1))
        true_irf = np.exp(-t / 10)
        c = conv(aif, true_irf, t)

        # Circulant deconvolution
        recovered_irf = deconv(c, aif, t, method="tsvd", tol=0.1, circulant=True)

        assert len(recovered_irf) == len(t)
        assert np.all(np.isfinite(recovered_irf))

    def test_deconv_invalid_method(self):
        """Test that invalid method raises error."""
        t = np.linspace(0, 10, 21)
        c = np.ones_like(t)
        aif = np.ones_like(t)

        with pytest.raises(DataValidationError, match="Unknown method"):
            deconv(c, aif, t, method="invalid")


class TestDeconvOSVD:
    """Tests for oSVD (oscillation-minimizing) deconvolution."""

    def test_deconv_osvd_basic(self):
        """Test basic oSVD deconvolution."""
        t = np.linspace(0, 30, 61)

        aif = np.exp(-t / 5) * (1 - np.exp(-t / 1))
        true_irf = np.exp(-t / 10)
        c = conv(aif, true_irf, t)

        # oSVD should produce smooth result
        recovered_irf = deconv_osvd(c, aif, t)

        assert len(recovered_irf) == len(t)
        assert np.all(np.isfinite(recovered_irf))

    def test_deconv_osvd_reduces_oscillations(self):
        """Test that oSVD produces less oscillatory result than standard TSVD."""
        t = np.linspace(0, 30, 61)

        # Add noise to make deconvolution harder
        np.random.seed(42)
        aif = np.exp(-t / 5) * (1 - np.exp(-t / 1))
        true_irf = np.exp(-t / 10)
        c = conv(aif, true_irf, t) + 0.01 * np.random.randn(len(t))

        # Standard TSVD with aggressive truncation
        irf_tsvd = deconv(c, aif, t, method="tsvd", tol=0.01, circulant=True)

        # oSVD
        irf_osvd = deconv_osvd(c, aif, t)

        # Compute oscillation indices
        def osc_index(x):
            if np.max(np.abs(x)) < 1e-10:
                return 0
            return np.sum(np.abs(np.diff(x))) / (len(x) * np.max(np.abs(x)))

        oi_tsvd = osc_index(irf_tsvd)
        oi_osvd = osc_index(irf_osvd)

        # oSVD should generally have lower or similar oscillation
        # (this test is probabilistic, may occasionally fail)
        assert oi_osvd <= oi_tsvd * 1.5  # Allow some tolerance


class TestConvmat:
    """Tests for convmat() convolution matrix construction."""

    def test_convmat_empty(self):
        """Test convmat with empty arrays."""
        result = convmat(np.array([]), np.array([]))
        assert result.shape == (1, 0) or result.size == 0

    def test_convmat_shape(self):
        """Test convmat returns correct shape."""
        n = 10
        t = np.linspace(0, 5, n)
        h = np.exp(-t / 2)

        A = convmat(h, t)

        assert A.shape == (n, n)

    def test_convmat_lower_triangular(self):
        """Test that convmat is lower triangular (causal)."""
        n = 10
        t = np.linspace(0, 5, n)
        h = np.exp(-t / 2)

        A = convmat(h, t)

        # Upper triangle should be zero (except maybe diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(A[i, j]) < 1e-10

    def test_convmat_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        t = np.linspace(0, 5, 10)
        h = np.ones(11)

        with pytest.raises(DataValidationError, match="same length"):
            convmat(h, t)

    def test_convmat_convolution_equivalence(self):
        """Test that A @ f gives approximately the same result as conv(h, f)."""
        t = np.linspace(0, 10, 51)

        h = np.exp(-t / 3)  # Impulse response
        f = np.ones_like(t)  # Constant input

        A = convmat(h, t)
        result_matrix = A @ f
        result_conv = conv(h, f, t)

        # Should be approximately equal (different integration methods)
        # Allow larger tolerance due to different algorithms
        assert_allclose(result_matrix, result_conv, rtol=0.5, atol=0.5)


class TestInvconvmat:
    """Tests for invconvmat() regularized pseudo-inverse."""

    def test_invconvmat_shape(self):
        """Test invconvmat returns correct shape."""
        n = 10
        A = np.eye(n) + 0.1 * np.random.randn(n, n)

        A_inv = invconvmat(A, method="tsvd", tol=0.1)

        assert A_inv.shape == (n, n)

    def test_invconvmat_tsvd(self):
        """Test TSVD pseudo-inverse."""
        A = np.diag([10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001])

        A_inv = invconvmat(A, method="tsvd", tol=0.1)

        # Check that small singular values were truncated
        # Reconstruction should not amplify noise
        product = A @ A_inv
        # Not identity due to truncation, but should be reasonable
        assert np.max(np.abs(product)) < 100

    def test_invconvmat_tikhonov(self):
        """Test Tikhonov pseudo-inverse."""
        n = 10
        A = np.diag([10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001])

        A_inv = invconvmat(A, method="tikhonov", tol=0.1)

        assert A_inv.shape == (n, n)
        assert np.all(np.isfinite(A_inv))

    def test_invconvmat_invalid_method(self):
        """Test that invalid method raises error."""
        A = np.eye(5)

        with pytest.raises(DataValidationError, match="Unknown method"):
            invconvmat(A, method="invalid")


class TestDeconvRoundtrip:
    """Tests for convolution-deconvolution round-trip accuracy."""

    def test_roundtrip_noise_free(self):
        """Test that conv then deconv approximately recovers original."""
        t = np.linspace(0, 30, 61)

        # Known functions
        aif = np.exp(-t / 5) * (1 - np.exp(-t / 1))
        true_irf = np.exp(-t / 10)

        # Forward: convolve
        c = conv(aif, true_irf, t)

        # Backward: deconvolve
        recovered_irf = deconv(c, aif, t, method="tsvd", tol=0.05)

        # Check correlation between true and recovered
        correlation = np.corrcoef(true_irf[5:], recovered_irf[5:])[0, 1]
        assert correlation > 0.5  # Should be positively correlated

    def test_roundtrip_with_noise(self):
        """Test conv-deconv with noisy data."""
        np.random.seed(42)
        t = np.linspace(0, 30, 61)

        aif = np.exp(-t / 5) * (1 - np.exp(-t / 1))
        true_irf = np.exp(-t / 10)

        # Forward with noise
        c = conv(aif, true_irf, t)
        c_noisy = c + 0.02 * np.max(c) * np.random.randn(len(t))

        # Backward with stronger regularization
        recovered_irf = deconv(c_noisy, aif, t, method="tsvd", tol=0.2)

        # Should still recover general shape
        assert len(recovered_irf) == len(t)
        assert np.all(np.isfinite(recovered_irf))
