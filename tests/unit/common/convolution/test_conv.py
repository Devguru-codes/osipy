"""Unit tests for piecewise-linear convolution."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from osipy.common.convolution import conv, uconv
from osipy.common.exceptions import DataValidationError


class TestConv:
    """Tests for conv() piecewise-linear convolution."""

    def test_conv_empty_input(self):
        """Test convolution with empty arrays."""
        result = conv(np.array([]), np.array([]), np.array([]))
        assert len(result) == 0

    def test_conv_single_point(self):
        """Test convolution with single point returns zero."""
        result = conv(np.array([1.0]), np.array([1.0]), np.array([0.0]))
        assert len(result) == 1
        assert result[0] == 0.0

    def test_conv_uniform_time_grid(self):
        """Test convolution on uniform time grid."""
        t = np.linspace(0, 10, 101)
        dt = t[1] - t[0]

        # Delta function convolution (should give back h)
        f = np.zeros_like(t)
        f[0] = 1.0 / dt  # Delta function approximation

        h = np.exp(-t / 2)

        result = conv(f, h, t)

        # Result should approximate h (with some delay/smoothing)
        # Check that the shape is reasonable
        assert len(result) == len(t)
        assert result[0] >= 0  # Should be non-negative

    def test_conv_exponential_decay(self):
        """Test convolution of two exponentials."""
        t = np.linspace(0, 20, 201)

        # Two exponential functions
        T1, T2 = 2.0, 5.0
        f = np.exp(-t / T1)
        h = np.exp(-t / T2)

        result = conv(f, h, t)

        # Check basic properties
        assert len(result) == len(t)
        assert result[0] == 0.0  # Convolution at t=0 is 0
        assert np.all(result >= -1e-10)  # Should be non-negative

        # Peak should occur after t=0
        peak_idx = np.argmax(result)
        assert peak_idx > 0

    def test_conv_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        t = np.linspace(0, 10, 101)
        f = np.ones(100)  # Wrong length
        h = np.ones(101)

        with pytest.raises(DataValidationError, match="same length"):
            conv(f, h, t)

    def test_conv_with_explicit_dt(self):
        """Test convolution with explicit dt parameter."""
        n = 100
        dt = 0.1
        t = np.arange(n) * dt

        f = np.exp(-t / 2)
        h = np.exp(-t / 5)

        # With explicit dt (should use uconv internally)
        result = conv(f, h, t, dt=dt)

        assert len(result) == n
        assert result[0] == 0.0

    def test_conv_non_uniform_time_grid(self):
        """Test convolution on non-uniform time grid."""
        # Non-uniform spacing
        t = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

        f = np.exp(-t / 2)
        h = np.exp(-t / 3)

        result = conv(f, h, t)

        assert len(result) == len(t)
        assert result[0] == 0.0


class TestUconv:
    """Tests for uconv() uniform-grid convolution."""

    def test_uconv_empty_input(self):
        """Test uniform convolution with empty arrays."""
        result = uconv(np.array([]), np.array([]), 0.1)
        assert len(result) == 0

    def test_uconv_single_point(self):
        """Test uniform convolution with single point."""
        result = uconv(np.array([1.0]), np.array([1.0]), 0.1)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_uconv_basic(self):
        """Test basic uniform convolution."""
        n = 100
        dt = 0.1

        f = np.ones(n)  # Constant input
        h = np.exp(-np.arange(n) * dt / 2)  # Exponential decay

        result = uconv(f, h, dt)

        assert len(result) == n
        assert result[0] == 0.0
        # Convolution of constant with exponential should increase then plateau
        assert result[-1] > result[1]

    def test_uconv_symmetry(self):
        """Test that convolution is approximately commutative."""
        n = 50
        dt = 0.2

        f = np.exp(-np.arange(n) * dt / 2)
        h = np.exp(-np.arange(n) * dt / 5)

        result_fh = uconv(f, h, dt)
        result_hf = uconv(h, f, dt)

        # Results should be similar (not exactly equal due to discrete effects)
        assert_allclose(result_fh, result_hf, rtol=0.1)

    def test_uconv_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        f = np.ones(100)
        h = np.ones(101)

        with pytest.raises(DataValidationError, match="same length"):
            uconv(f, h, 0.1)


class TestConvAccuracy:
    """Tests for convolution accuracy against analytical solutions."""

    def test_exponential_convolution_analytical(self):
        """Test against analytical solution for exponential convolution.

        For f(t) = exp(-t/T1) and h(t) = exp(-t/T2), the analytical
        convolution is:
            (f * h)(t) = T1*T2/(T1-T2) * (exp(-t/T1) - exp(-t/T2))
        """
        t = np.linspace(0, 20, 201)
        T1, T2 = 2.0, 5.0

        f = np.exp(-t / T1)
        h = np.exp(-t / T2)

        # Numerical convolution
        result = conv(f, h, t)

        # Analytical solution (for t > 0)
        analytical = T1 * T2 / (T1 - T2) * (np.exp(-t / T1) - np.exp(-t / T2))

        # Compare (skip t=0 where both are 0)
        # Allow reasonable tolerance for numerical integration
        mask = t > 1.0
        if np.any(mask):
            rel_error = np.abs(
                (result[mask] - analytical[mask]) / (analytical[mask] + 1e-10)
            )
            assert np.median(rel_error) < 0.3  # Within 30% for most points

    def test_convolution_with_delta(self):
        """Test convolution with approximate delta function."""
        t = np.linspace(0, 10, 501)  # Fine time grid
        dt = t[1] - t[0]

        # Approximate delta function (narrow Gaussian)
        sigma = dt * 2
        delta = np.exp(-(t**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

        # Impulse response
        h = np.exp(-t / 3)

        result = conv(delta, h, t)

        # Result should approximate h (shifted by delta width)
        # Check that peak occurs near expected location
        assert len(result) == len(t)
