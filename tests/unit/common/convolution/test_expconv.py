"""Unit tests for exponential convolution functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from osipy.common.convolution import biexpconv, expconv, nexpconv
from osipy.common.exceptions import DataValidationError


class TestExpconv:
    """Tests for expconv() recursive exponential convolution."""

    def test_expconv_empty_input(self):
        """Test expconv with empty arrays."""
        result = expconv(np.array([]), 1.0, np.array([]))
        assert len(result) == 0

    def test_expconv_zero_time_constant(self):
        """Test expconv with zero time constant returns zeros."""
        t = np.linspace(0, 10, 101)
        f = np.ones_like(t)

        result = expconv(f, 0.0, t)

        assert len(result) == len(t)
        assert_allclose(result, np.zeros_like(t))

    def test_expconv_negative_time_constant(self):
        """Test expconv with negative time constant returns zeros."""
        t = np.linspace(0, 10, 101)
        f = np.ones_like(t)

        result = expconv(f, -1.0, t)

        assert_allclose(result, np.zeros_like(t))

    def test_expconv_constant_input_analytical(self):
        """Test expconv of constant input against analytical solution.

        For f(t) = C (constant), the analytical convolution with exp(-t/T) is:
            (f * h)(t) = C * T * (1 - exp(-t/T))
        """
        t = np.linspace(0, 20, 201)
        T = 5.0
        C = 2.0

        f = np.full_like(t, C)

        result = expconv(f, T, t)

        # Analytical solution
        analytical = C * T * (1 - np.exp(-t / T))

        # Compare (skip very early points)
        mask = t > 0.5
        assert_allclose(result[mask], analytical[mask], rtol=0.1)

    def test_expconv_exponential_input(self):
        """Test expconv with exponential input."""
        t = np.linspace(0, 20, 201)
        T1 = 2.0  # Input time constant
        T2 = 5.0  # Convolution time constant

        f = np.exp(-t / T1)

        result = expconv(f, T2, t)

        # Check basic properties
        assert len(result) == len(t)
        assert result[0] == 0.0  # Starts at zero
        assert np.all(result >= -1e-10)  # Non-negative

        # Should have a peak then decay
        peak_idx = np.argmax(result)
        assert peak_idx > 0
        assert peak_idx < len(t) - 1

    def test_expconv_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        t = np.linspace(0, 10, 101)
        f = np.ones(100)  # Wrong length

        with pytest.raises(DataValidationError, match="same length"):
            expconv(f, 1.0, t)

    def test_expconv_large_time_constant(self):
        """Test expconv with large time constant approaches integral."""
        t = np.linspace(0, 10, 101)
        T = 1000.0  # Very large T

        f = np.ones_like(t)

        result = expconv(f, T, t)

        # For T >> t, convolution ≈ T * integral(f) ≈ T * t * mean(f)
        # Actually: integral_0^t f(u) du for constant f = t
        # So result ≈ t (approximately, for large T)
        # Check that result grows roughly linearly
        assert result[-1] > result[len(t) // 2]


class TestBiexpconv:
    """Tests for biexpconv() bi-exponential convolution."""

    def test_biexpconv_empty_input(self):
        """Test biexpconv with empty arrays."""
        result = biexpconv(np.array([]), 1.0, 2.0, np.array([]))
        assert len(result) == 0

    def test_biexpconv_zero_time_constants(self):
        """Test biexpconv with zero time constants."""
        t = np.linspace(0, 10, 101)
        f = np.ones_like(t)

        result = biexpconv(f, 0.0, 1.0, t)
        assert_allclose(result, np.zeros_like(t))

    def test_biexpconv_equal_time_constants(self):
        """Test biexpconv with equal time constants (limiting case)."""
        t = np.linspace(0, 10, 101)
        T = 5.0

        f = np.ones_like(t)

        # Should handle T1 == T2 gracefully
        result = biexpconv(f, T, T, t)

        # Should return finite values
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))

    def test_biexpconv_basic(self):
        """Test basic biexpconv computation."""
        t = np.linspace(0, 20, 201)
        T1, T2 = 2.0, 5.0

        f = np.ones_like(t)

        result = biexpconv(f, T1, T2, t)

        # Check basic properties
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))

    def test_biexpconv_symmetry(self):
        """Test that biexpconv is symmetric in T1, T2."""
        t = np.linspace(0, 20, 201)
        T1, T2 = 2.0, 5.0

        f = np.exp(-t / 3)

        result_12 = biexpconv(f, T1, T2, t)
        result_21 = biexpconv(f, T2, T1, t)

        # biexpconv(f, T1, T2) = (E1 - E2) / (T1 - T2)
        # biexpconv(f, T2, T1) = (E2 - E1) / (T2 - T1) = (E1 - E2) / (T1 - T2)
        # They should be equal (not negatives)
        assert_allclose(result_12, result_21, rtol=1e-10)


class TestNexpconv:
    """Tests for nexpconv() n-exponential convolution."""

    def test_nexpconv_empty_input(self):
        """Test nexpconv with empty arrays."""
        result = nexpconv(np.array([]), 1.0, 3, np.array([]))
        assert len(result) == 0

    def test_nexpconv_n_equals_1(self):
        """Test that n=1 is equivalent to expconv."""
        t = np.linspace(0, 20, 201)
        T = 5.0

        f = np.exp(-t / 2)

        result_nexp = nexpconv(f, T, 1, t)
        result_exp = expconv(f, T, t)

        assert_allclose(result_nexp, result_exp, rtol=1e-10)

    def test_nexpconv_invalid_parameters(self):
        """Test nexpconv with invalid parameters."""
        t = np.linspace(0, 10, 101)
        f = np.ones_like(t)

        # Zero time constant
        result = nexpconv(f, 0.0, 3, t)
        assert_allclose(result, np.zeros_like(t))

        # n < 1
        result = nexpconv(f, 1.0, 0, t)
        assert_allclose(result, np.zeros_like(t))

    def test_nexpconv_n_equals_2(self):
        """Test nexpconv with n=2 (gamma variate with shape=2)."""
        t = np.linspace(0, 30, 301)
        T = 5.0

        f = np.ones_like(t)

        result = nexpconv(f, T, 2, t)

        # Check basic properties
        assert len(result) == len(t)
        assert result[0] == 0.0
        assert np.all(np.isfinite(result))

        # n=2 gamma variate should have delayed peak
        peak_idx = np.argmax(result)
        assert peak_idx > 0

    def test_nexpconv_large_n(self):
        """Test nexpconv with large n (uses Gaussian approximation)."""
        t = np.linspace(0, 100, 501)
        T = 2.0
        n = 25  # Large n triggers Gaussian approximation

        # Use a decaying input so the convolution result has a clear peak
        f = np.exp(-t / 10)

        result = nexpconv(f, T, n, t)

        # Check basic properties
        assert len(result) == len(t)
        assert np.all(np.isfinite(result))

        # For decaying input, the result should have a peak
        # somewhere after t=0 due to the delayed gamma variate
        peak_idx = np.argmax(result)
        assert peak_idx > 0  # Peak is not at t=0
        assert result[0] == 0.0 or result[0] < result[peak_idx]  # Increases from start


class TestExpconvFlouri:
    """Tests verifying Flouri et al. (2016) formula implementation."""

    def test_expconv_linear_input(self):
        """Test expconv with linearly increasing input.

        For f(t) = a*t, the analytical convolution with exp(-t/T) is:
            (f * h)(t) = a * T * (t - T * (1 - exp(-t/T)))
        """
        t = np.linspace(0, 20, 201)
        T = 5.0
        a = 0.5  # Slope

        f = a * t

        result = expconv(f, T, t)

        # Analytical solution
        analytical = a * T * (t - T * (1 - np.exp(-t / T)))

        # Compare
        mask = t > 1.0
        assert_allclose(result[mask], analytical[mask], rtol=0.15)

    def test_expconv_preserves_integral(self):
        """Test that convolution integral is preserved.

        integral(f * h) = integral(f) * integral(h)
        For h = exp(-t/T), integral_0^inf = T
        """
        t = np.linspace(0, 50, 501)  # Long enough for decay
        T = 5.0

        f = np.exp(-t / 2)
        f_integral = 2.0  # integral_0^inf exp(-t/2) dt

        result = expconv(f, T, t)

        # Numerical integral of result
        dt = t[1] - t[0]
        result_integral = np.trapezoid(result, dx=dt)

        # Expected: f_integral * T
        expected_integral = f_integral * T

        assert abs(result_integral - expected_integral) / expected_integral < 0.2
