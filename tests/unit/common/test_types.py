"""Unit tests for osipy.common.types module."""

import numpy as np

from osipy.common.types import (
    AcquisitionParams,
    ASLAcquisitionParams,
    DCEAcquisitionParams,
    DSCAcquisitionParams,
    IVIMAcquisitionParams,
    LabelingType,
)


class TestAcquisitionParams:
    """Tests for AcquisitionParams base class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        params = AcquisitionParams()
        assert params.tr is None
        assert params.te is None
        assert params.field_strength is None


class TestDCEAcquisitionParams:
    """Tests for DCEAcquisitionParams."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        params = DCEAcquisitionParams()
        assert params.flip_angles == []
        assert params.baseline_frames == 5
        assert params.temporal_resolution == 1.0
        assert params.relaxivity == 4.5


class TestDSCAcquisitionParams:
    """Tests for DSCAcquisitionParams."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        params = DSCAcquisitionParams()
        assert params.baseline_frames == 10


class TestASLAcquisitionParams:
    """Tests for ASLAcquisitionParams."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        params = ASLAcquisitionParams()
        assert params.labeling_type == LabelingType.PCASL
        assert params.pld == 1800.0
        assert params.labeling_duration == 1800.0
        assert params.background_suppression is False
        assert params.bs_efficiency == 1.0
        assert params.m0_scale is None

    def test_with_multi_pld(self) -> None:
        """Test multi-PLD initialization."""
        params = ASLAcquisitionParams(
            pld=[200, 500, 1000, 1500, 2000],
            labeling_type=LabelingType.PCASL,
        )
        assert params.pld == [200, 500, 1000, 1500, 2000]


class TestIVIMAcquisitionParams:
    """Tests for IVIMAcquisitionParams."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        params = IVIMAcquisitionParams()
        np.testing.assert_array_equal(
            params.b_values,
            np.array(
                [0, 10, 20, 40, 80, 110, 140, 170, 200, 300, 400, 500, 600, 700, 800]
            ),
        )

    def test_with_custom_bvalues(self) -> None:
        """Test with custom b-values."""
        b_vals = np.array([0, 10, 20, 50, 100, 200, 400, 800])
        params = IVIMAcquisitionParams(b_values=b_vals)
        np.testing.assert_array_equal(params.b_values, b_vals)
