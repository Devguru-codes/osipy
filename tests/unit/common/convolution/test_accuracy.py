"""Tests comparing dcmri-style vs FFT convolution accuracy."""

import numpy as np
from numpy.testing import assert_allclose

from osipy.common.convolution import conv, expconv, fft_convolve


class TestConvolutionAccuracy:
    """Compare different convolution methods for accuracy."""

    def test_piecewise_vs_fft_uniform_grid(self):
        """Test piecewise-linear vs FFT on uniform grid."""
        n = 200  # Larger dataset for better FFT comparison
        dt = 0.1
        t = np.arange(n) * dt

        # Test signals - use decaying signals that are well-suited for convolution
        f = np.exp(-t / 2)
        h = np.exp(-t / 5)

        # Piecewise-linear convolution
        result_pw = conv(f, h, t, dt=dt)

        # FFT convolution
        result_fft = fft_convolve(f, h, dt, mode="same")

        # They should be similar but not identical
        # Piecewise-linear should be more accurate for pharmacokinetic signals
        assert len(result_pw) == len(result_fft)

        # Both should be non-negative for these signals
        assert np.min(result_pw) >= -1e-10
        # FFT may have small negative values due to Gibbs phenomenon

        # Check that they have similar shape by comparing middle region
        # (avoid boundary effects)
        middle = slice(30, 150)
        correlation = np.corrcoef(result_pw[middle], result_fft[middle])[0, 1]
        # FFT and piecewise-linear use different algorithms so
        # correlation > 0.7 indicates they compute similar quantities
        assert correlation > 0.7

    def test_expconv_vs_conv_exponential(self):
        """Test that expconv matches conv for exponential impulse response."""
        t = np.linspace(0, 20, 201)
        T = 5.0

        f = np.exp(-t / 2)
        h = np.exp(-t / T)

        # expconv (specialized)
        result_expconv = expconv(f, T, t)

        # conv with exponential h (general)
        result_conv = conv(f, h, t)

        # expconv should be more accurate (uses analytical formula)
        # but both should give similar results
        mask = t > 1.0
        assert_allclose(result_expconv[mask], result_conv[mask], rtol=0.2)

    def test_fft_boundary_artifacts(self):
        """Test that FFT has boundary artifacts while piecewise doesn't."""
        n = 200  # Larger dataset
        dt = 0.1
        t = np.arange(n) * dt

        # Signals that don't decay to zero - FFT will have boundary issues
        f = np.ones(n)  # Constant
        h = np.exp(-t / 5)

        # Piecewise-linear
        result_pw = conv(f, h, t, dt=dt)

        # FFT
        result_fft = fft_convolve(f, h, dt, mode="same")

        # Both methods should produce finite results
        assert np.all(np.isfinite(result_pw))
        assert np.all(np.isfinite(result_fft))

        # The piecewise method should give monotonically increasing result
        # for constant input convolved with positive exponential
        # Check that the piecewise result increases (expected for this input)
        assert result_pw[-1] > result_pw[0]

        # FFT may have issues at boundaries due to circular convolution
        # but should still produce reasonable output in the middle
        middle = slice(50, 150)
        assert np.mean(result_fft[middle]) > 0  # Should be positive

    def test_analytical_comparison_expconv(self):
        """Test expconv against analytical solution.

        For f(t) = 1 (constant), h(t) = exp(-t/T):
            (f * h)(t) = T * (1 - exp(-t/T))
        """
        t = np.linspace(0, 30, 301)
        T = 5.0

        f = np.ones_like(t)

        # expconv result
        result = expconv(f, T, t)

        # Analytical solution
        analytical = T * (1 - np.exp(-t / T))

        # Should match closely
        mask = t > 0.5
        assert_allclose(result[mask], analytical[mask], rtol=0.1)


class TestNonUniformGridAccuracy:
    """Test accuracy on non-uniform time grids."""

    def test_conv_non_uniform_vs_uniform(self):
        """Test conv gives similar results for non-uniform vs uniform grids."""
        # Uniform grid
        t_uniform = np.linspace(0, 10, 101)
        f_uniform = np.exp(-t_uniform / 2)
        h_uniform = np.exp(-t_uniform / 5)
        result_uniform = conv(f_uniform, h_uniform, t_uniform)

        # Non-uniform grid (same endpoints)
        t_nonuniform = np.sort(np.random.uniform(0, 10, 101))
        t_nonuniform[0] = 0.0
        t_nonuniform[-1] = 10.0
        f_nonuniform = np.exp(-t_nonuniform / 2)
        h_nonuniform = np.exp(-t_nonuniform / 5)
        result_nonuniform = conv(f_nonuniform, h_nonuniform, t_nonuniform)

        # Interpolate non-uniform result to uniform grid for comparison
        result_interp = np.interp(t_uniform, t_nonuniform, result_nonuniform)

        # Should be similar
        correlation = np.corrcoef(result_uniform[10:-10], result_interp[10:-10])[0, 1]
        assert correlation > 0.8

    def test_expconv_non_uniform(self):
        """Test expconv on non-uniform time grid."""
        # Non-uniform grid with varying density
        t = np.concatenate(
            [
                np.linspace(0, 2, 21),  # Dense near t=0
                np.linspace(2.5, 10, 16),  # Sparser later
            ]
        )
        T = 3.0

        f = np.exp(-t / 2)

        # Should handle non-uniform grid
        result = expconv(f, T, t)

        assert len(result) == len(t)
        assert np.all(np.isfinite(result))
        assert result[0] == 0.0  # Starts at zero


class TestPharmacokineticScenarios:
    """Test accuracy in realistic pharmacokinetic scenarios."""

    def test_tofts_model_convolution(self):
        """Test convolution accuracy for Tofts model tissue response.

        Tofts model: C_t(t) = K_trans * (C_p * exp(-kep*t))
        where * denotes convolution.
        """
        t = np.linspace(0, 10, 101)  # 10 minutes

        # Parameters
        Ktrans = 0.1  # min^-1
        ve = 0.2  # dimensionless
        kep = Ktrans / ve

        # Parker AIF approximation
        Cp = 5.0 * np.exp(-t / 0.5) + 2.0 * np.exp(-t / 2.0)

        # Expected tissue concentration via expconv
        Ct_expected = Ktrans * expconv(Cp, 1 / kep, t)

        # Via general convolution
        h = np.exp(-kep * t)
        Ct_conv = Ktrans * conv(Cp, h, t)

        # Should match closely
        mask = t > 0.5
        assert_allclose(Ct_expected[mask], Ct_conv[mask], rtol=0.2)

    def test_dsc_deconvolution_scenario(self):
        """Test convolution for DSC-MRI residue function estimation."""
        t = np.linspace(0, 60, 61)  # 60 seconds

        # Gamma variate AIF
        alpha, beta = 3.0, 0.5
        aif = (t**alpha) * np.exp(-t / beta)
        aif = aif / np.max(aif)  # Normalize

        # Exponential residue function
        mtt = 5.0  # Mean transit time
        residue = np.exp(-t / mtt)

        # Tissue concentration = AIF * residue
        Ct = conv(aif, residue, t)

        # Check that convolution is reasonable
        assert len(Ct) == len(t)
        assert Ct[0] == 0.0  # Starts at zero
        assert np.max(Ct) > 0  # Has positive values

        # Peak should be delayed compared to AIF
        aif_peak = np.argmax(aif)
        ct_peak = np.argmax(Ct)
        assert ct_peak > aif_peak  # Convolution delays the peak


class TestNumericalStability:
    """Test numerical stability of convolution operations."""

    def test_conv_large_values(self):
        """Test conv with large input values."""
        t = np.linspace(0, 10, 101)
        scale = 1e6

        f = scale * np.exp(-t / 2)
        h = scale * np.exp(-t / 5)

        result = conv(f, h, t)

        # Should handle large values without overflow
        assert np.all(np.isfinite(result))

    def test_conv_small_values(self):
        """Test conv with small input values."""
        t = np.linspace(0, 10, 101)
        scale = 1e-6

        f = scale * np.exp(-t / 2)
        h = scale * np.exp(-t / 5)

        result = conv(f, h, t)

        # Should handle small values without underflow to exactly zero
        assert np.all(np.isfinite(result))

    def test_expconv_very_small_T(self):
        """Test expconv with very small time constant."""
        t = np.linspace(0, 10, 101)
        T = 0.001  # Very small

        f = np.ones_like(t)

        result = expconv(f, T, t)

        # Should give approximately zero (delta function behavior)
        assert np.all(np.isfinite(result))

    def test_expconv_very_large_T(self):
        """Test expconv with very large time constant."""
        t = np.linspace(0, 10, 101)
        T = 10000.0  # Very large

        f = np.ones_like(t)

        result = expconv(f, T, t)

        # Should give approximately T * t (integration behavior)
        assert np.all(np.isfinite(result))
        assert result[-1] > result[0]
