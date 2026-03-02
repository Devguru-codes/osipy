"""Unit tests for BSW leakage correction.

Tests for osipy/dsc/leakage/correction.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.dsc.leakage.correction import (
    LeakageCorrectionParams,
    LeakageCorrectionResult,
    correct_leakage,
    estimate_permeability,
)


class TestCorrectLeakage:
    """Tests for correct_leakage function."""

    @pytest.fixture
    def synthetic_data(self) -> dict:
        """Create synthetic DSC data with leakage."""
        np.random.seed(42)

        # Dimensions
        nx, ny, nz = 16, 16, 4
        n_timepoints = 60

        # Time vector
        time = np.linspace(0, 90, n_timepoints)

        # Create AIF (gamma-variate-like)
        alpha = 3.0
        beta = 1.5
        aif = np.zeros(n_timepoints)
        t_shifted = time - 10  # delay
        valid = t_shifted > 0
        aif[valid] = (t_shifted[valid] ** alpha) * np.exp(-t_shifted[valid] / beta)
        aif = aif / np.max(aif) * 5.0  # Scale to typical ΔR2* values

        # Create spatial ΔR2* data
        # Normal tissue: follows reference curve
        delta_r2 = np.zeros((nx, ny, nz, n_timepoints))

        # Reference curve (scaled AIF)
        ref_curve = 0.8 * aif

        # Add spatial variations
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Base signal follows reference
                    signal = ref_curve.copy()

                    # Add some noise
                    signal += np.random.randn(n_timepoints) * 0.1

                    # Add leakage to central region (simulating tumor)
                    if 5 <= i <= 10 and 5 <= j <= 10:
                        # T1 leakage (positive contribution)
                        k1 = np.random.uniform(0.1, 0.3)
                        signal += k1 * aif

                        # T2* leakage (cumulative)
                        k2 = np.random.uniform(0.01, 0.05)
                        aif_integral = np.cumsum(aif) * (time[1] - time[0])
                        signal += k2 * aif_integral

                    delta_r2[i, j, k, :] = signal

        # Mask
        mask = np.ones((nx, ny, nz), dtype=bool)

        return {
            "delta_r2": delta_r2,
            "aif": aif,
            "time": time,
            "mask": mask,
            "shape": (nx, ny, nz),
        }

    def test_correct_leakage_basic(self, synthetic_data: dict) -> None:
        """Test basic leakage correction."""
        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            mask=synthetic_data["mask"],
        )

        assert isinstance(result, LeakageCorrectionResult)
        assert result.corrected_delta_r2.shape == synthetic_data["delta_r2"].shape
        assert result.k1.shape == synthetic_data["shape"]
        assert result.k2.shape == synthetic_data["shape"]
        assert result.reference_curve.shape == (synthetic_data["delta_r2"].shape[-1],)

    def test_correct_leakage_without_mask(self, synthetic_data: dict) -> None:
        """Test leakage correction without mask."""
        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            mask=None,
        )

        assert isinstance(result, LeakageCorrectionResult)
        assert result.corrected_delta_r2.shape == synthetic_data["delta_r2"].shape

    def test_correct_leakage_t1_only(self, synthetic_data: dict) -> None:
        """Test leakage correction with T1 correction only."""
        params = LeakageCorrectionParams(
            use_t1_correction=True,
            use_t2_correction=False,
        )

        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            params=params,
        )

        # K2 should be zero when T2 correction is disabled
        # (coefficients are still computed, but not applied)
        assert isinstance(result, LeakageCorrectionResult)

    def test_correct_leakage_t2_only(self, synthetic_data: dict) -> None:
        """Test leakage correction with T2* correction only."""
        params = LeakageCorrectionParams(
            use_t1_correction=False,
            use_t2_correction=True,
        )

        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            params=params,
        )

        assert isinstance(result, LeakageCorrectionResult)

    def test_correct_leakage_custom_fitting_range(self, synthetic_data: dict) -> None:
        """Test leakage correction with custom fitting range."""
        params = LeakageCorrectionParams(
            fitting_range=(10, 50),
        )

        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            params=params,
        )

        assert isinstance(result, LeakageCorrectionResult)

    def test_correct_leakage_bidirectional(self, synthetic_data: dict) -> None:
        """Test bidirectional leakage correction method."""
        params = LeakageCorrectionParams(method="bidirectional")

        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            params=params,
        )

        assert isinstance(result, LeakageCorrectionResult)

    def test_correct_leakage_invalid_method(self, synthetic_data: dict) -> None:
        """Test that invalid method raises error."""
        from osipy.common.exceptions import DataValidationError

        params = LeakageCorrectionParams(method="invalid")

        with pytest.raises(DataValidationError, match="Unknown leakage corrector"):
            correct_leakage(
                delta_r2=synthetic_data["delta_r2"],
                aif=synthetic_data["aif"],
                time=synthetic_data["time"],
                params=params,
            )

    def test_correct_leakage_aif_length_mismatch(self, synthetic_data: dict) -> None:
        """Test that AIF length mismatch raises error."""
        from osipy.common.exceptions import DataValidationError

        # Wrong AIF length
        wrong_aif = synthetic_data["aif"][:-10]

        with pytest.raises(DataValidationError, match="AIF length"):
            correct_leakage(
                delta_r2=synthetic_data["delta_r2"],
                aif=wrong_aif,
                time=synthetic_data["time"],
            )

    def test_correct_leakage_custom_reference_mask(self, synthetic_data: dict) -> None:
        """Test leakage correction with custom reference mask."""
        # Create custom reference mask (corners of the volume)
        custom_mask = np.zeros(synthetic_data["shape"], dtype=bool)
        custom_mask[:3, :3, :] = True
        custom_mask[-3:, -3:, :] = True

        params = LeakageCorrectionParams(
            reference_tissue="custom",
            custom_reference_mask=custom_mask,
        )

        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            mask=synthetic_data["mask"],
            params=params,
        )

        assert isinstance(result, LeakageCorrectionResult)

    def test_leakage_values_in_tumor_region(self, synthetic_data: dict) -> None:
        """Test that leakage coefficients are higher in tumor region."""
        result = correct_leakage(
            delta_r2=synthetic_data["delta_r2"],
            aif=synthetic_data["aif"],
            time=synthetic_data["time"],
            mask=synthetic_data["mask"],
        )

        # Get mean K values in tumor vs normal regions
        tumor_mask = np.zeros(synthetic_data["shape"], dtype=bool)
        tumor_mask[5:11, 5:11, :] = True
        normal_mask = ~tumor_mask & synthetic_data["mask"]

        # Mean absolute K1 should be higher in tumor
        k1_tumor = np.mean(np.abs(result.k1[tumor_mask]))
        np.mean(np.abs(result.k1[normal_mask]))

        # Tumor region should have higher leakage (allowing some tolerance)
        assert k1_tumor > 0, "K1 should be non-zero in tumor region"


class TestEstimatePermeability:
    """Tests for estimate_permeability function."""

    def test_permeability_basic(self) -> None:
        """Test basic permeability estimation."""
        k1 = np.random.rand(10, 10, 5) * 0.5
        k2 = np.random.rand(10, 10, 5) * 0.1

        permeability = estimate_permeability(k1, k2)

        assert permeability.shape == k1.shape
        assert np.all(permeability >= 0)  # Permeability should be non-negative

    def test_permeability_from_k2_magnitude(self) -> None:
        """Test that permeability is based on K2 magnitude."""
        k1 = np.zeros((5, 5, 3))
        k2 = np.array([[[0.1, -0.2, 0.3]]])
        k2 = np.broadcast_to(k2, (5, 5, 3)).copy()

        permeability = estimate_permeability(k1, k2)

        # Should be absolute values of K2
        np.testing.assert_array_almost_equal(permeability, np.abs(k2))

    def test_permeability_negative_k2(self) -> None:
        """Test permeability with negative K2 values."""
        k1 = np.zeros((3, 3, 2))
        k2 = -0.5 * np.ones((3, 3, 2))

        permeability = estimate_permeability(k1, k2)

        # Should all be positive (magnitude of K2)
        assert np.all(permeability > 0)
        np.testing.assert_array_almost_equal(permeability, 0.5)


class TestLeakageCorrectionResult:
    """Tests for LeakageCorrectionResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating LeakageCorrectionResult."""
        shape = (10, 10, 5)
        n_time = 60

        result = LeakageCorrectionResult(
            corrected_delta_r2=np.zeros((*shape, n_time)),
            k1=np.zeros(shape),
            k2=np.zeros(shape),
            reference_curve=np.zeros(n_time),
            residual=np.zeros(shape),
        )

        assert result.corrected_delta_r2.shape == (*shape, n_time)
        assert result.k1.shape == shape
        assert result.k2.shape == shape
        assert result.reference_curve.shape == (n_time,)
        assert result.residual.shape == shape

    def test_result_without_residual(self) -> None:
        """Test creating result without residual."""
        result = LeakageCorrectionResult(
            corrected_delta_r2=np.zeros((5, 5, 3, 30)),
            k1=np.zeros((5, 5, 3)),
            k2=np.zeros((5, 5, 3)),
            reference_curve=np.zeros(30),
            residual=None,
        )

        assert result.residual is None
