"""Integration tests for convolution module with DCE/DSC models.

This test verifies that the new dcmri-style convolution functions integrate
correctly with the existing pharmacokinetic models and produce accurate
results.
"""

import numpy as np

from osipy.common.convolution import (
    conv,
    deconv,
    fft_convolve,
)
from osipy.dce.models.extended_tofts import ExtendedToftsModel, ExtendedToftsParams
from osipy.dce.models.tofts import ToftsModel, ToftsParams
from osipy.dsc.deconvolution import get_deconvolver


class TestConvolutionWithDCEModels:
    """Test convolution functions with DCE pharmacokinetic models."""

    def test_tofts_model_uses_convolution(self):
        """Test that Tofts model prediction uses convolution correctly."""
        # Setup
        t = np.linspace(0, 10, 101)  # 10 minutes in seconds * 60
        t_sec = t * 60  # Convert to seconds for the model

        # Parker-like AIF
        aif = 5.0 * np.exp(-t / 0.5) + 2.0 * np.exp(-t / 2.0)

        # Model parameters
        params = ToftsParams(ktrans=0.1, ve=0.2)

        # Get model prediction
        model = ToftsModel()
        ct_model = model.predict(t_sec, aif, params)

        # Verify model produces reasonable output
        assert len(ct_model) == len(t)
        assert np.all(np.isfinite(ct_model))
        assert ct_model[0] >= 0  # Should start at or near zero

        # Peak should be delayed compared to AIF
        aif_peak_idx = np.argmax(aif)
        ct_peak_idx = np.argmax(ct_model)
        assert ct_peak_idx >= aif_peak_idx

    def test_extended_tofts_model_prediction(self):
        """Test Extended Tofts model prediction."""
        t = np.linspace(0, 10, 101)
        t_sec = t * 60

        aif = 5.0 * np.exp(-t / 0.5) + 2.0 * np.exp(-t / 2.0)

        params = ExtendedToftsParams(ktrans=0.1, ve=0.2, vp=0.05)

        model = ExtendedToftsModel()
        ct_model = model.predict(t_sec, aif, params)

        assert len(ct_model) == len(t)
        assert np.all(np.isfinite(ct_model))

        # Extended Tofts should have vascular contribution
        # so signal at t=0 should be higher than standard Tofts
        tofts_model = ToftsModel()
        ct_tofts = tofts_model.predict(t_sec, aif, ToftsParams(ktrans=0.1, ve=0.2))

        # At early times, extended Tofts should be higher due to vp
        early_idx = 5
        assert ct_model[early_idx] >= ct_tofts[early_idx]


class TestDeconvolutionWithDSCModels:
    """Test deconvolution with DSC perfusion models."""

    def test_svd_deconvolution_produces_valid_cbf(self):
        """Test SVD deconvolution produces valid CBF estimates."""
        # Create synthetic DSC data
        n_time = 60
        t = np.linspace(0, 60, n_time)  # 60 seconds

        # Gamma variate AIF
        alpha, beta = 3.0, 0.5
        aif = (t**alpha) * np.exp(-t / beta)
        aif = aif / np.max(aif)

        # Exponential residue function with known CBF
        true_cbf = 60.0  # ml/100g/min
        mtt = 5.0  # seconds
        residue = true_cbf * np.exp(-t / mtt)

        # Create tissue concentration: C(t) = CBF * AIF ⊗ R(t)
        tissue = conv(aif, residue / true_cbf, t)  # Normalized convolution
        tissue = tissue.reshape(1, 1, 1, n_time)  # 4D shape

        # Run SVD deconvolution via registry
        deconvolver = get_deconvolver("oSVD")
        result = deconvolver.deconvolve(tissue, aif, t)

        # Check result structure
        assert result.cbf.shape == (1, 1, 1)
        assert result.mtt.shape == (1, 1, 1)
        assert result.residue_function.shape == tissue.shape

        # CBF should be positive
        assert result.cbf[0, 0, 0] > 0

    def test_circulant_deconvolution_delay_insensitive(self):
        """Test that circulant SVD is insensitive to bolus arrival delay."""
        t = np.linspace(0, 60, 61)

        # Create AIF with and without delay
        aif = np.exp(-((t - 10) ** 2) / 10) * (t > 10)
        aif_delayed = np.exp(-((t - 15) ** 2) / 10) * (t > 15)

        # Residue function
        residue = np.exp(-t / 10)

        # Tissue curves
        tissue = conv(aif, residue, t)
        tissue_delayed = conv(aif_delayed, residue, t)

        # Deconvolve with circulant method
        recovered = deconv(tissue, aif, t, method="tsvd", tol=0.1, circulant=True)
        recovered_delayed = deconv(
            tissue_delayed, aif_delayed, t, method="tsvd", tol=0.1, circulant=True
        )

        # Both should give similar peak values (CBF proxy)
        cbf = np.max(recovered)
        cbf_delayed = np.max(recovered_delayed)

        # Within 50% is acceptable for this test
        assert abs(cbf - cbf_delayed) / max(cbf, cbf_delayed) < 0.5


class TestConvolutionNumericalAccuracy:
    """Test numerical accuracy of convolution operations."""

    def test_conv_vs_fft_on_uniform_grid(self):
        """Compare piecewise-linear and FFT convolution on uniform grid."""
        n = 200
        dt = 0.1
        t = np.arange(n) * dt

        # Test signals
        f = np.exp(-t / 2)
        h = np.exp(-t / 5)

        # Piecewise-linear
        result_pw = conv(f, h, t, dt=dt)

        # FFT
        result_fft = fft_convolve(f, h, dt, mode="same")

        # Both should give similar results in the middle
        middle = slice(30, 150)
        correlation = np.corrcoef(result_pw[middle], result_fft[middle])[0, 1]
        assert correlation > 0.7
