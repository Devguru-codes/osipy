"""Integration tests for DCE-MRI pipeline.

Tests the complete DCE-MRI analysis workflow from data loading
through T1 mapping, signal-to-concentration conversion, model fitting,
and parameter map generation.

User Story 1: Clinical researcher loads DCE-MRI dataset, computes T1 maps
from VFA acquisitions, converts signal to contrast agent concentration,
fits Extended Tofts model using population AIF, and exports quantitative
parameter maps (Ktrans, ve, vp).
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.common.parameter_map import ParameterMap


class TestDCEPipelineIntegration:
    """Integration tests for DCE-MRI pipeline."""

    @pytest.fixture
    def synthetic_dce_data(self) -> dict:
        """Create synthetic DCE-MRI data for integration testing."""
        np.random.seed(42)

        # Dimensions
        nx, ny, nz = 16, 16, 4
        n_timepoints = 60

        # Time vector (seconds)
        time = np.linspace(0, 360, n_timepoints)  # 6 minutes

        # Create synthetic T1 map
        t1_map = np.random.uniform(800, 1500, (nx, ny, nz))  # ms

        # Acquisition parameters
        tr = 5.0  # ms

        # Create synthetic VFA data for T1 mapping
        flip_angles = np.array([2, 5, 10, 15, 20])
        vfa_data = np.zeros((nx, ny, nz, len(flip_angles)))
        for i, fa in enumerate(flip_angles):
            fa_rad = np.radians(fa)
            e1 = np.exp(-tr / t1_map)
            m0 = 1000 * np.ones((nx, ny, nz))
            vfa_data[..., i] = (
                m0 * np.sin(fa_rad) * (1 - e1) / (1 - np.cos(fa_rad) * e1)
            )
            vfa_data[..., i] += np.random.randn(nx, ny, nz) * 10

        # Create synthetic DCE time series
        # Ground truth parameters
        ktrans_true = np.random.uniform(0.01, 0.3, (nx, ny, nz))
        ve_true = np.random.uniform(0.1, 0.5, (nx, ny, nz))
        vp_true = np.random.uniform(0.01, 0.1, (nx, ny, nz))

        # Simple Parker AIF approximation
        from osipy.common.aif.population import parker_aif_curve

        aif = parker_aif_curve(time)

        # Generate signal using Extended Tofts model approximation
        dce_signal = np.zeros((nx, ny, nz, n_timepoints))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Simplified convolution for testing
                    kt = ktrans_true[i, j, k]
                    ve = ve_true[i, j, k]
                    vp_val = vp_true[i, j, k]

                    # Tissue concentration (simplified)
                    tissue_conc = np.zeros(n_timepoints)
                    dt = time[1] - time[0]
                    for t_idx in range(1, n_timepoints):
                        # Simple convolution
                        impulse_response = kt * np.exp(
                            -kt / ve * (time[t_idx] - time[:t_idx])
                        )
                        tissue_conc[t_idx] = (
                            np.sum(aif[:t_idx] * impulse_response[::-1]) * dt
                        )

                    # Add vascular term
                    total_conc = tissue_conc + vp_val * aif

                    # Convert to signal (simplified)
                    r1 = 4.5  # relaxivity
                    s0 = 1000
                    t1_0 = t1_map[i, j, k]
                    signal = s0 * (1 + r1 * total_conc * t1_0 / 1000)
                    dce_signal[i, j, k, :] = signal + np.random.randn(n_timepoints) * 5

        mask = np.ones((nx, ny, nz), dtype=bool)

        return {
            "vfa_data": vfa_data,
            "flip_angles": flip_angles,
            "dce_signal": dce_signal,
            "time": time,
            "t1_true": t1_map,
            "ktrans_true": ktrans_true,
            "ve_true": ve_true,
            "vp_true": vp_true,
            "mask": mask,
            "tr": tr,
            "shape": (nx, ny, nz),
        }

    def test_t1_mapping_from_vfa(self, synthetic_dce_data: dict) -> None:
        """Test T1 mapping from VFA data."""
        from osipy.dce.t1_mapping import compute_t1_vfa

        result = compute_t1_vfa(
            signal=synthetic_dce_data["vfa_data"],
            flip_angles=synthetic_dce_data["flip_angles"],
            tr=synthetic_dce_data["tr"],
            mask=synthetic_dce_data["mask"],
        )

        assert result.t1_map is not None
        assert result.t1_map.values.shape[:3] == synthetic_dce_data["shape"]

        # T1 values should be in reasonable range (filter out NaN from failed fits)
        t1_values = result.t1_map.values[synthetic_dce_data["mask"]]
        valid_t1 = t1_values[np.isfinite(t1_values)]
        assert len(valid_t1) > 0, "At least some T1 values should be valid"
        assert np.all(valid_t1 > 0), "Valid T1 should be positive"
        assert np.mean(valid_t1) > 500, "Mean T1 should be > 500ms"
        assert np.mean(valid_t1) < 3000, "Mean T1 should be < 3000ms"

    def test_signal_to_concentration(self, synthetic_dce_data: dict) -> None:
        """Test signal to concentration conversion."""
        from osipy.common.types import DCEAcquisitionParams
        from osipy.dce.concentration import signal_to_concentration

        # Create T1 parameter map (3D, no newaxis needed)
        t1_map = ParameterMap(
            name="T1",
            symbol="T1",
            units="ms",
            values=synthetic_dce_data["t1_true"],
            affine=np.eye(4),
        )

        # Create acquisition params
        acquisition_params = DCEAcquisitionParams(
            tr=synthetic_dce_data["tr"],
            flip_angles=[15.0],
            relaxivity=4.5,
            baseline_frames=5,
        )

        result = signal_to_concentration(
            signal=synthetic_dce_data["dce_signal"],
            t1_map=t1_map,
            acquisition_params=acquisition_params,
        )

        assert result is not None
        assert result.shape == synthetic_dce_data["dce_signal"].shape

    def test_parker_aif_generation(self) -> None:
        """Test Parker AIF generation."""
        from osipy.common.aif.population import ParkerAIF

        time = np.linspace(0, 360, 60)

        aif = ParkerAIF()
        aif_values = aif.get_concentration(time)

        assert len(aif_values) == len(time)
        assert np.all(aif_values >= 0), "AIF should be non-negative"

        # AIF should have a peak
        peak_idx = np.argmax(aif_values)
        assert peak_idx > 0, "Peak should not be at t=0"
        assert peak_idx < len(time) - 1, "Peak should not be at end"

    def test_tofts_model_fitting(self, synthetic_dce_data: dict) -> None:
        """Test Tofts model fitting."""
        from osipy.common.aif.population import ParkerAIF
        from osipy.dce.fitting import fit_model

        # Use a small subset for speed
        concentration = synthetic_dce_data["dce_signal"][:4, :4, :2, :]
        mask = synthetic_dce_data["mask"][:4, :4, :2]

        aif = ParkerAIF()
        aif_values = aif.get_concentration(synthetic_dce_data["time"])

        result = fit_model(
            "tofts",
            concentration=concentration,
            aif=aif_values,
            time=synthetic_dce_data["time"],
            mask=mask,
        )

        assert result is not None
        assert "Ktrans" in result.parameter_maps

    def test_extended_tofts_model_fitting(self, synthetic_dce_data: dict) -> None:
        """Test Extended Tofts model fitting."""
        from osipy.common.aif.population import ParkerAIF
        from osipy.dce.fitting import fit_model

        # Use a small subset for speed
        concentration = synthetic_dce_data["dce_signal"][:4, :4, :2, :]
        mask = synthetic_dce_data["mask"][:4, :4, :2]

        aif = ParkerAIF()
        aif_values = aif.get_concentration(synthetic_dce_data["time"])

        result = fit_model(
            "extended_tofts",
            concentration=concentration,
            aif=aif_values,
            time=synthetic_dce_data["time"],
            mask=mask,
        )

        assert result is not None
        assert "Ktrans" in result.parameter_maps
        assert "vp" in result.parameter_maps

    def test_full_dce_pipeline(self, synthetic_dce_data: dict) -> None:
        """Test complete DCE pipeline end-to-end."""
        from osipy.common.aif.population import ParkerAIF
        from osipy.common.types import DCEAcquisitionParams
        from osipy.dce.concentration import signal_to_concentration
        from osipy.dce.fitting import fit_model
        from osipy.dce.t1_mapping import compute_t1_vfa

        # Use smaller subset for integration test speed
        nx, ny, nz = 4, 4, 2

        vfa_data = synthetic_dce_data["vfa_data"][:nx, :ny, :nz, :]
        dce_signal = synthetic_dce_data["dce_signal"][:nx, :ny, :nz, :]
        mask = synthetic_dce_data["mask"][:nx, :ny, :nz]

        # Step 1: T1 mapping
        t1_result = compute_t1_vfa(
            signal=vfa_data,
            flip_angles=synthetic_dce_data["flip_angles"],
            tr=synthetic_dce_data["tr"],
            mask=mask,
        )

        assert t1_result.t1_map is not None, "T1 mapping failed"

        # Step 2: Signal to concentration
        acquisition_params = DCEAcquisitionParams(
            tr=synthetic_dce_data["tr"],
            flip_angles=[15.0],
            relaxivity=4.5,
            baseline_frames=5,
        )

        concentration = signal_to_concentration(
            signal=dce_signal,
            t1_map=t1_result.t1_map,
            acquisition_params=acquisition_params,
        )

        assert concentration is not None, "Concentration conversion failed"

        # Step 3: Get AIF
        aif = ParkerAIF()
        aif_values = aif.get_concentration(synthetic_dce_data["time"])

        assert len(aif_values) == len(synthetic_dce_data["time"]), (
            "AIF generation failed"
        )

        # Step 4: Model fitting
        result = fit_model(
            "tofts",
            concentration=concentration,
            aif=aif_values,
            time=synthetic_dce_data["time"],
            mask=mask,
        )

        assert result is not None, "Model fitting failed"
        assert "Ktrans" in result.parameter_maps

    def test_quality_mask_generation(self, synthetic_dce_data: dict) -> None:
        """Test that quality mask is generated for failed fits."""
        from osipy.common.aif.population import ParkerAIF
        from osipy.dce.fitting import fit_model

        # Create data with some problematic voxels
        concentration = synthetic_dce_data["dce_signal"][:4, :4, :2, :].copy()
        concentration[0, 0, 0, :] = 0  # Zero signal
        concentration[1, 1, 0, :] = np.nan  # NaN values

        mask = synthetic_dce_data["mask"][:4, :4, :2]

        aif = ParkerAIF()
        aif_values = aif.get_concentration(synthetic_dce_data["time"])

        result = fit_model(
            "tofts",
            concentration=concentration,
            aif=aif_values,
            time=synthetic_dce_data["time"],
            mask=mask,
        )

        # Should handle bad voxels gracefully
        assert result is not None
        # Quality mask should indicate failed voxels
        assert result.quality_mask is not None


class TestDCEOutputValidation:
    """Test DCE output format and units."""

    def test_parameter_maps_have_affine(self) -> None:
        """Test that parameter maps include affine transformation."""
        # ParameterMap requires affine
        pm = ParameterMap(
            name="test",
            symbol="t",
            units="au",
            values=np.zeros((5, 5, 3)),
            affine=np.eye(4),
        )

        assert pm.affine is not None
        assert pm.affine.shape == (4, 4)
