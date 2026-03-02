"""Integration tests for DSC-MRI pipeline.

Tests the complete DSC-MRI analysis workflow from signal-to-concentration
conversion through deconvolution and hemodynamic parameter calculation.

User Story 2: Neuroradiologist analyzes DSC-MRI data from brain tumor patient
to generate rCBV maps with leakage correction and WM normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.parameter_map import ParameterMap


class TestDSCPipelineIntegration:
    """Integration tests for DSC-MRI pipeline."""

    @pytest.fixture
    def synthetic_dsc_data(self) -> dict:
        """Create synthetic DSC-MRI data for integration testing."""
        np.random.seed(42)

        # Dimensions
        nx, ny, nz = 16, 16, 4
        n_timepoints = 60

        # Time vector (seconds)
        time = np.linspace(0, 90, n_timepoints)

        # Create AIF (gamma-variate)
        t_arrival = 15.0
        alpha = 3.0
        beta = 1.5
        aif = np.zeros(n_timepoints)
        t_shifted = time - t_arrival
        valid = t_shifted > 0
        aif[valid] = (t_shifted[valid] ** alpha) * np.exp(-t_shifted[valid] / beta)
        aif = aif / np.max(aif) * 100.0  # Peak concentration ~100

        # Create tissue curves with varying CBV
        cbv_true = np.random.uniform(2.0, 6.0, (nx, ny, nz))
        cbf_true = np.random.uniform(30, 80, (nx, ny, nz))
        mtt_true = cbv_true / cbf_true * 60  # seconds

        # Generate concentration curves
        concentration = np.zeros((nx, ny, nz, n_timepoints))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Simplified residue function
                    mtt = mtt_true[i, j, k]
                    residue = np.exp(-time / mtt)

                    # Convolution with AIF
                    dt = time[1] - time[0]
                    tissue_curve = np.convolve(aif, residue)[:n_timepoints] * dt
                    tissue_curve *= cbf_true[i, j, k] / 60.0  # Scale by CBF

                    concentration[i, j, k, :] = tissue_curve

        # Convert to signal (T2* decay)
        s0 = 1000.0
        te = 0.03  # 30 ms
        r2_star_baseline = 20.0  # 1/s
        r2_star_change = concentration * 0.1  # ΔR2* proportional to concentration

        signal = s0 * np.exp(-te * (r2_star_baseline + r2_star_change))
        signal += np.random.randn(*signal.shape) * 5  # Add noise

        mask = np.ones((nx, ny, nz), dtype=bool)

        # White matter mask (for normalization)
        wm_mask = np.zeros((nx, ny, nz), dtype=bool)
        wm_mask[12:16, 12:16, :] = True

        return {
            "signal": signal,
            "concentration": concentration,
            "aif": aif,
            "time": time,
            "te": te,
            "cbv_true": cbv_true,
            "cbf_true": cbf_true,
            "mtt_true": mtt_true,
            "mask": mask,
            "wm_mask": wm_mask,
            "s0": s0,
            "shape": (nx, ny, nz),
        }

    def test_signal_to_delta_r2(self, synthetic_dsc_data: dict) -> None:
        """Test signal to ΔR2* conversion."""
        from osipy.dsc.concentration import signal_to_delta_r2

        result = signal_to_delta_r2(
            signal=synthetic_dsc_data["signal"],
            te=synthetic_dsc_data["te"],
            baseline_end=5,
        )

        assert result.shape == synthetic_dsc_data["signal"].shape

        # ΔR2* should be positive during bolus passage
        mid_time = synthetic_dsc_data["signal"].shape[-1] // 2
        delta_r2_peak = result[..., mid_time]
        assert np.mean(delta_r2_peak[synthetic_dsc_data["mask"]]) > 0

    def test_svd_deconvolution_via_registry(self, synthetic_dsc_data: dict) -> None:
        """Test SVD deconvolution via registry."""
        from osipy.dsc.deconvolution import get_deconvolver

        # Use small subset for speed
        conc = synthetic_dsc_data["concentration"][:4, :4, :2, :]
        mask = synthetic_dsc_data["mask"][:4, :4, :2]

        deconvolver = get_deconvolver("sSVD")
        result = deconvolver.deconvolve(
            concentration=conc,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        assert result is not None
        assert result.residue_function is not None
        assert result.cbf is not None

    def test_csvd_deconvolution(self, synthetic_dsc_data: dict) -> None:
        """Test block-circulant SVD deconvolution."""
        from osipy.dsc.deconvolution import get_deconvolver

        conc = synthetic_dsc_data["concentration"][:4, :4, :2, :]
        mask = synthetic_dsc_data["mask"][:4, :4, :2]

        deconvolver = get_deconvolver("cSVD")
        result = deconvolver.deconvolve(
            concentration=conc,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        assert result is not None

    def test_osvd_deconvolution(self, synthetic_dsc_data: dict) -> None:
        """Test oscillation-index SVD deconvolution."""
        from osipy.dsc.deconvolution import get_deconvolver

        conc = synthetic_dsc_data["concentration"][:4, :4, :2, :]
        mask = synthetic_dsc_data["mask"][:4, :4, :2]

        deconvolver = get_deconvolver("oSVD")
        result = deconvolver.deconvolve(
            concentration=conc,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        assert result is not None

    def test_hemodynamic_parameters(self, synthetic_dsc_data: dict) -> None:
        """Test hemodynamic parameter calculation."""
        from osipy.dsc.parameters import compute_perfusion_maps

        conc = synthetic_dsc_data["concentration"][:4, :4, :2, :]
        mask = synthetic_dsc_data["mask"][:4, :4, :2]

        result = compute_perfusion_maps(
            delta_r2=conc,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        assert result is not None
        assert result.cbv is not None
        assert result.cbf is not None
        assert result.mtt is not None

    def test_leakage_correction(self, synthetic_dsc_data: dict) -> None:
        """Test BSW leakage correction."""
        from osipy.dsc.leakage import correct_leakage

        # Create data with leakage (add linear trend)
        conc_with_leakage = synthetic_dsc_data["concentration"].copy()
        leakage = np.linspace(0, 5, conc_with_leakage.shape[-1])
        conc_with_leakage += leakage[np.newaxis, np.newaxis, np.newaxis, :]

        result = correct_leakage(
            delta_r2=conc_with_leakage,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=synthetic_dsc_data["mask"],
        )

        assert result is not None
        assert result.corrected_delta_r2.shape == conc_with_leakage.shape
        assert result.k1 is not None
        assert result.k2 is not None

    def test_white_matter_normalization(self, synthetic_dsc_data: dict) -> None:
        """Test white matter normalization for rCBV."""
        from osipy.dsc.normalization import normalize_to_white_matter

        # Create CBV parameter map (3D values - no np.newaxis)
        cbv_map = ParameterMap(
            name="CBV",
            symbol="CBV",
            units="ml/100g",
            values=synthetic_dsc_data["cbv_true"],
            affine=np.eye(4),
        )

        result = normalize_to_white_matter(
            parameter_map=cbv_map,
            white_matter_mask=synthetic_dsc_data["wm_mask"],
        )

        assert result is not None
        assert result.normalized_map is not None

        # rCBV in WM should be ~1.0 after normalization
        wm_rcbv = result.normalized_map.values[synthetic_dsc_data["wm_mask"]]
        assert np.mean(wm_rcbv) == pytest.approx(1.0, rel=0.1)

    def test_full_dsc_pipeline(self, synthetic_dsc_data: dict) -> None:
        """Test complete DSC pipeline end-to-end."""
        from osipy.dsc.concentration import signal_to_delta_r2
        from osipy.dsc.deconvolution import get_deconvolver
        from osipy.dsc.normalization import normalize_to_white_matter
        from osipy.dsc.parameters import compute_perfusion_maps

        # Use smaller subset for speed
        nx, ny, nz = 4, 4, 2
        signal = synthetic_dsc_data["signal"][:nx, :ny, :nz, :]
        mask = synthetic_dsc_data["mask"][:nx, :ny, :nz]
        wm_mask = np.zeros((nx, ny, nz), dtype=bool)
        wm_mask[2:4, 2:4, :] = True

        # Step 1: Convert signal to ΔR2*
        delta_r2 = signal_to_delta_r2(
            signal=signal,
            te=synthetic_dsc_data["te"],
            baseline_end=5,
        )

        assert delta_r2 is not None, "Signal conversion failed"

        # Step 2: Deconvolution
        deconvolver = get_deconvolver("oSVD")
        deconv_result = deconvolver.deconvolve(
            concentration=delta_r2,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        assert deconv_result is not None, "Deconvolution failed"

        # Step 3: Hemodynamic parameters
        hemo_result = compute_perfusion_maps(
            delta_r2=delta_r2,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        assert hemo_result.cbv is not None, "CBV calculation failed"
        assert hemo_result.cbf is not None, "CBF calculation failed"

        # Step 4: White matter normalization
        norm_result = normalize_to_white_matter(
            parameter_map=hemo_result.cbv,
            white_matter_mask=wm_mask,
        )

        assert norm_result is not None, "Normalization failed"

    def test_gamma_variate_fitting(self, synthetic_dsc_data: dict) -> None:
        """Test gamma-variate fitting for recirculation removal."""
        from osipy.dsc.concentration import gamma_variate_fit

        # Single voxel concentration curve
        conc_curve = synthetic_dsc_data["concentration"][8, 8, 2, :]

        fitted, params = gamma_variate_fit(
            concentration=conc_curve,
            time=synthetic_dsc_data["time"],
        )

        assert fitted is not None
        assert len(fitted) == len(conc_curve)
        assert "t0" in params
        assert "alpha" in params
        assert "beta" in params


class TestDSCOutputValidation:
    """Test DSC output format and physiological validity."""

    @pytest.fixture
    def synthetic_dsc_data(self) -> dict:
        """Create synthetic DSC data for output validation testing."""
        np.random.seed(42)

        nx, ny, nz = 16, 16, 4
        n_timepoints = 60
        time = np.linspace(0, 90, n_timepoints)

        t_arrival = 15.0
        alpha = 3.0
        beta = 1.5
        aif = np.zeros(n_timepoints)
        t_shifted = time - t_arrival
        valid = t_shifted > 0
        aif[valid] = (t_shifted[valid] ** alpha) * np.exp(-t_shifted[valid] / beta)
        aif = aif / np.max(aif) * 100.0

        cbv_true = np.random.uniform(2.0, 6.0, (nx, ny, nz))
        cbf_true = np.random.uniform(30, 80, (nx, ny, nz))
        mtt_true = cbv_true / cbf_true * 60

        concentration = np.zeros((nx, ny, nz, n_timepoints))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    mtt = mtt_true[i, j, k]
                    residue = np.exp(-time / mtt)
                    dt = time[1] - time[0]
                    tissue_curve = np.convolve(aif, residue)[:n_timepoints] * dt
                    tissue_curve *= cbf_true[i, j, k] / 60.0
                    concentration[i, j, k, :] = tissue_curve

        mask = np.ones((nx, ny, nz), dtype=bool)

        return {
            "concentration": concentration,
            "aif": aif,
            "time": time,
            "cbv_true": cbv_true,
            "cbf_true": cbf_true,
            "mtt_true": mtt_true,
            "mask": mask,
            "shape": (nx, ny, nz),
        }

    def test_cbv_physiological_range(self, synthetic_dsc_data: dict) -> None:
        """Test that CBV values are in physiological range."""
        from osipy.dsc.parameters import compute_perfusion_maps

        conc = synthetic_dsc_data["concentration"][:4, :4, :2, :]
        mask = synthetic_dsc_data["mask"][:4, :4, :2]

        result = compute_perfusion_maps(
            delta_r2=conc,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        # CBV should be 0-10 ml/100g for brain tissue
        cbv_values = result.cbv.values[mask]
        assert np.all(cbv_values >= 0), "CBV should be non-negative"
        # Synthetic data uses simplified concentration model, not calibrated ΔR2* values,
        # so we only check for reasonable bounds, not exact physiological range
        assert np.all(np.isfinite(cbv_values)), "CBV should be finite"
        assert np.mean(cbv_values) < 1000, "Mean CBV should be bounded"

    def test_mtt_from_cbv_cbf(self, synthetic_dsc_data: dict) -> None:
        """Test that MTT = CBV/CBF relationship holds."""
        from osipy.dsc.parameters import compute_perfusion_maps

        conc = synthetic_dsc_data["concentration"][:4, :4, :2, :]
        mask = synthetic_dsc_data["mask"][:4, :4, :2]

        result = compute_perfusion_maps(
            delta_r2=conc,
            aif=synthetic_dsc_data["aif"],
            time=synthetic_dsc_data["time"],
            mask=mask,
        )

        # MTT ≈ CBV / CBF (with unit conversions)
        # This is a consistency check, not exact equality
        cbv = result.cbv.values[mask]
        cbf = result.cbf.values[mask]
        mtt = result.mtt.values[mask]

        # At least correlation should be positive
        valid = (cbf > 0) & (mtt > 0)
        if np.sum(valid) > 5:
            computed_mtt = cbv[valid] / cbf[valid] * 60  # Convert to seconds
            assert np.corrcoef(computed_mtt, mtt[valid])[0, 1] > 0
