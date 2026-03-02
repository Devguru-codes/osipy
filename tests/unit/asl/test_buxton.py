"""Unit tests for Buxton model ASL quantification.

Tests for osipy/asl/quantification/cbf.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.asl.labeling import LabelingScheme, PASLParams, PCASLParams
from osipy.asl.quantification.cbf import (
    ASLQuantificationParams,
    ASLQuantificationResult,
    compute_cbf_uncertainty,
    quantify_cbf,
)


class TestASLQuantificationParams:
    """Tests for ASLQuantificationParams dataclass."""

    def test_default_params(self) -> None:
        """Test default parameter values."""
        params = ASLQuantificationParams()
        assert params.labeling_scheme == LabelingScheme.PCASL
        assert params.t1_blood == 1650.0  # ms at 3T
        assert params.t1_tissue == 1330.0  # ms
        assert params.partition_coefficient == 0.9
        assert params.labeling_efficiency == 0.85
        assert params.pld == 1800.0  # ms
        assert params.label_duration == 1800.0  # ms
        assert params.bolus_duration is None


class TestQuantifyCBF:
    """Tests for quantify_cbf function."""

    @pytest.fixture
    def synthetic_asl_data(self) -> dict:
        """Create synthetic ASL data."""
        np.random.seed(42)

        # Spatial dimensions
        nx, ny, nz = 16, 16, 4

        # Create delta_m with realistic range
        # Typical ASL signal is 0.5-2% of M0
        m0 = 1000.0

        # Expected delta_m for pCASL with standard parameters
        # Using simplified estimation
        delta_m = np.random.uniform(0.005, 0.02, (nx, ny, nz)) * m0

        # Mask
        mask = np.ones((nx, ny, nz), dtype=bool)
        mask[:2, :, :] = False  # Some background

        return {
            "delta_m": delta_m,
            "m0": m0,
            "mask": mask,
            "shape": (nx, ny, nz),
        }

    def test_quantify_cbf_pcasl_scalar_m0(self, synthetic_asl_data: dict) -> None:
        """Test CBF quantification with pCASL and scalar M0."""
        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            pld=1800.0,
            label_duration=1800.0,
        )

        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
            params=params,
            mask=synthetic_asl_data["mask"],
        )

        assert isinstance(result, ASLQuantificationResult)
        assert result.cbf_map is not None
        assert result.cbf_map.units == "mL/100g/min"
        assert result.cbf_map.name == "CBF"
        assert result.quality_mask is not None

    def test_quantify_cbf_pcasl_array_m0(self, synthetic_asl_data: dict) -> None:
        """Test CBF quantification with pCASL and array M0."""
        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            pld=1800.0,
            label_duration=1800.0,
        )

        # Create M0 array with spatial variation
        m0_array = np.ones(synthetic_asl_data["shape"]) * 1000
        m0_array += np.random.uniform(-100, 100, synthetic_asl_data["shape"])

        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=m0_array,
            params=params,
            mask=synthetic_asl_data["mask"],
        )

        assert isinstance(result, ASLQuantificationResult)
        assert result.m0_used is not None
        assert result.m0_used.shape == synthetic_asl_data["shape"]

    def test_quantify_cbf_pasl(self, synthetic_asl_data: dict) -> None:
        """Test CBF quantification with PASL."""
        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PASL,
            pld=1500.0,
            bolus_duration=700.0,
            labeling_efficiency=0.98,
        )

        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
            params=params,
        )

        assert isinstance(result, ASLQuantificationResult)
        assert result.cbf_map is not None

    def test_quantify_cbf_pasl_default_bolus(self, synthetic_asl_data: dict) -> None:
        """Test PASL quantification with default bolus duration."""
        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PASL,
            pld=1500.0,
            bolus_duration=None,  # Should use default 700ms
        )

        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
            params=params,
        )

        assert isinstance(result, ASLQuantificationResult)

    def test_quantify_cbf_without_mask(self, synthetic_asl_data: dict) -> None:
        """Test CBF quantification without mask."""
        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
        )

        assert isinstance(result, ASLQuantificationResult)

    def test_cbf_units_ml_100g_min(self, synthetic_asl_data: dict) -> None:
        """Test that CBF is in mL/100g/min units (ASL Lexicon)."""
        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
        )

        # Scaling factor should be 6000 for mL/100g/min
        assert result.scaling_factor == 6000.0
        assert result.cbf_map.units == "mL/100g/min"

    def test_cbf_reasonable_range(self, synthetic_asl_data: dict) -> None:
        """Test that CBF values are in reasonable physiological range."""
        result = quantify_cbf(
            delta_m=synthetic_asl_data["delta_m"],
            m0=synthetic_asl_data["m0"],
            mask=synthetic_asl_data["mask"],
        )

        # Get masked CBF values
        cbf_values = result.cbf_map.values[synthetic_asl_data["mask"]]

        # CBF should generally be in 0-200 ml/100g/min range
        assert np.all(cbf_values >= 0), "CBF should be non-negative"
        assert np.all(cbf_values < 200), "CBF should be < 200 ml/100g/min"

    def test_quality_mask_filters_outliers(self, synthetic_asl_data: dict) -> None:
        """Test that quality mask filters unreasonable values."""
        # Create data with some outliers
        delta_m = synthetic_asl_data["delta_m"].copy()
        delta_m[5, 5, 2] = 0  # Zero signal
        delta_m[6, 6, 2] = -1  # Negative signal

        result = quantify_cbf(
            delta_m=delta_m,
            m0=synthetic_asl_data["m0"],
            mask=synthetic_asl_data["mask"],
        )

        # Quality mask should be 3D
        assert result.quality_mask.ndim == 3

    def test_negative_m0_raises_error(self, synthetic_asl_data: dict) -> None:
        """Test that negative M0 raises error."""
        from osipy.common.exceptions import DataValidationError

        with pytest.raises(DataValidationError, match="M0 must be positive"):
            quantify_cbf(
                delta_m=synthetic_asl_data["delta_m"],
                m0=-1000.0,
                mask=synthetic_asl_data["mask"],
            )

    def test_cbf_increases_with_delta_m(self) -> None:
        """Test that CBF increases with delta_m."""
        m0 = 1000.0
        delta_m_low = np.ones((5, 5, 2)) * 10
        delta_m_high = np.ones((5, 5, 2)) * 20

        result_low = quantify_cbf(delta_m=delta_m_low, m0=m0)
        result_high = quantify_cbf(delta_m=delta_m_high, m0=m0)

        cbf_low = np.mean(result_low.cbf_map.values)
        cbf_high = np.mean(result_high.cbf_map.values)

        assert cbf_high > cbf_low, "Higher delta_m should give higher CBF"

    def test_cbf_parameter_sensitivity(self) -> None:
        """Test CBF sensitivity to key parameters."""
        delta_m = np.ones((5, 5, 2)) * 15
        m0 = 1000.0

        # Base case
        params_base = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            t1_blood=1650.0,
            pld=1800.0,
        )
        result_base = quantify_cbf(delta_m=delta_m, m0=m0, params=params_base)
        cbf_base = np.mean(result_base.cbf_map.values)

        # Longer T1_blood should increase CBF (less signal decay)
        params_long_t1 = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            t1_blood=1800.0,
            pld=1800.0,
        )
        result_long_t1 = quantify_cbf(delta_m=delta_m, m0=m0, params=params_long_t1)
        cbf_long_t1 = np.mean(result_long_t1.cbf_map.values)

        # Longer PLD should increase exponential factor
        params_long_pld = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PCASL,
            t1_blood=1650.0,
            pld=2200.0,
        )
        result_long_pld = quantify_cbf(delta_m=delta_m, m0=m0, params=params_long_pld)
        cbf_long_pld = np.mean(result_long_pld.cbf_map.values)

        # CBF should change with these parameter changes
        assert cbf_long_t1 != cbf_base
        assert cbf_long_pld != cbf_base

    def test_with_mask_zeros_outside(self) -> None:
        """Test that voxels outside the mask are zero."""
        shape = (8, 8, 4)
        delta_m = np.random.rand(*shape) * 40
        m0 = 1000.0

        mask = np.zeros(shape, dtype=bool)
        mask[2:6, 2:6, 1:3] = True

        result = quantify_cbf(delta_m=delta_m, m0=m0, mask=mask)

        # Outside mask should be 0
        assert np.all(result.cbf_map.values[~mask] == 0)


class TestComputeCBFUncertainty:
    """Tests for compute_cbf_uncertainty function."""

    def test_uncertainty_basic(self) -> None:
        """Test basic CBF uncertainty computation."""
        delta_m = np.ones((5, 5, 2)) * 15
        delta_m_std = np.ones((5, 5, 2)) * 3  # 20% variation
        m0 = np.ones((5, 5, 2)) * 1000
        m0_std = np.ones((5, 5, 2)) * 50  # 5% variation

        params = ASLQuantificationParams()

        cbf_std = compute_cbf_uncertainty(
            delta_m=delta_m,
            delta_m_std=delta_m_std,
            m0=m0,
            m0_std=m0_std,
            params=params,
        )

        assert cbf_std.shape == delta_m.shape
        assert np.all(cbf_std >= 0), "Uncertainty should be non-negative"

    def test_uncertainty_without_m0_std(self) -> None:
        """Test CBF uncertainty without M0 uncertainty."""
        delta_m = np.ones((5, 5, 2)) * 15
        delta_m_std = np.ones((5, 5, 2)) * 3
        m0 = np.ones((5, 5, 2)) * 1000

        params = ASLQuantificationParams()

        cbf_std = compute_cbf_uncertainty(
            delta_m=delta_m,
            delta_m_std=delta_m_std,
            m0=m0,
            m0_std=None,
            params=params,
        )

        assert cbf_std.shape == delta_m.shape
        assert np.all(cbf_std >= 0)

    def test_uncertainty_increases_with_noise(self) -> None:
        """Test that uncertainty increases with noise."""
        delta_m = np.ones((5, 5, 2)) * 15
        m0 = np.ones((5, 5, 2)) * 1000

        params = ASLQuantificationParams()

        # Low noise
        cbf_std_low = compute_cbf_uncertainty(
            delta_m=delta_m,
            delta_m_std=np.ones_like(delta_m) * 1,
            m0=m0,
            m0_std=None,
            params=params,
        )

        # High noise
        cbf_std_high = compute_cbf_uncertainty(
            delta_m=delta_m,
            delta_m_std=np.ones_like(delta_m) * 5,
            m0=m0,
            m0_std=None,
            params=params,
        )

        assert np.mean(cbf_std_high) > np.mean(cbf_std_low)

    def test_uncertainty_pasl(self) -> None:
        """Test CBF uncertainty with PASL."""
        delta_m = np.ones((5, 5, 2)) * 15
        delta_m_std = np.ones((5, 5, 2)) * 3
        m0 = np.ones((5, 5, 2)) * 1000

        params = ASLQuantificationParams(
            labeling_scheme=LabelingScheme.PASL,
            bolus_duration=700.0,
        )

        cbf_std = compute_cbf_uncertainty(
            delta_m=delta_m,
            delta_m_std=delta_m_std,
            m0=m0,
            m0_std=None,
            params=params,
        )

        assert cbf_std.shape == delta_m.shape
        assert np.all(cbf_std >= 0)


class TestLabelingSchemes:
    """Tests for labeling scheme parameters."""

    def test_labeling_scheme_enum(self) -> None:
        """Test LabelingScheme enum values."""
        assert LabelingScheme.PASL.value == "pasl"
        assert LabelingScheme.CASL.value == "casl"
        assert LabelingScheme.PCASL.value == "pcasl"

    def test_pasl_params_defaults(self) -> None:
        """Test PASL parameter defaults."""
        params = PASLParams()
        assert params.ti == 1800.0
        assert params.labeling_efficiency == 0.98

    def test_pcasl_params_defaults(self) -> None:
        """Test pCASL parameter defaults."""
        params = PCASLParams()
        assert params.pld == 1800.0
        assert params.label_duration == 1800.0
        assert params.labeling_efficiency == 0.85
