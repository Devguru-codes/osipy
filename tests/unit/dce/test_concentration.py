"""Unit tests for DCE signal-to-concentration conversion.

Tests for signal intensity to contrast agent concentration conversion.
"""

import numpy as np
import pytest

from osipy.common.exceptions import DataValidationError
from osipy.common.parameter_map import ParameterMap
from osipy.common.types import DCEAcquisitionParams


class TestDCEAcquisitionParams:
    """Tests for DCE acquisition parameters."""

    def test_default_values(self) -> None:
        """Test default parameter values."""
        params = DCEAcquisitionParams()
        assert params.tr is None  # TR is optional
        assert params.flip_angles == []
        assert params.relaxivity == 4.5

    def test_custom_values(self) -> None:
        """Test custom parameter values."""
        params = DCEAcquisitionParams(
            tr=3.5,
            flip_angles=[10.0],
            relaxivity=3.9,
            te=1.5,
        )
        assert params.tr == 3.5
        assert params.flip_angles == [10.0]
        assert params.relaxivity == 3.9
        assert params.te == 1.5


class TestParameterMapUsage:
    """Tests for ParameterMap usage in DCE context."""

    def test_create_t1_map(self) -> None:
        """Test creating a T1 parameter map."""
        values = np.full((4, 4, 2), 1000.0)
        affine = np.eye(4)

        t1_map = ParameterMap(
            name="T1",
            symbol="T1",
            units="ms",
            values=values,
            affine=affine,
            model_name="VFA",
            fitting_method="linear",
        )

        assert t1_map.name == "T1"
        assert t1_map.units == "ms"
        assert t1_map.shape == (4, 4, 2)
        assert np.all(t1_map.values == 1000.0)

    def test_parameter_map_statistics(self) -> None:
        """Test ParameterMap statistics computation."""
        rng = np.random.default_rng(42)
        values = rng.uniform(0.1, 0.5, size=(8, 8, 4))
        affine = np.eye(4)

        ktrans_map = ParameterMap(
            name="Ktrans",
            symbol="Ktrans",
            units="1/min",
            values=values,
            affine=affine,
        )

        stats = ktrans_map.statistics()
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert 0.1 < stats["mean"] < 0.5

    def test_parameter_map_validation(self) -> None:
        """Test ParameterMap validation."""
        # Should fail with non-3D values
        with pytest.raises(DataValidationError, match="Values must be 3D"):
            ParameterMap(
                name="Test",
                symbol="T",
                units="",
                values=np.ones((4, 4)),  # 2D instead of 3D
                affine=np.eye(4),
            )

        # Should fail with wrong affine shape
        with pytest.raises(DataValidationError, match="Affine must be 4x4"):
            ParameterMap(
                name="Test",
                symbol="T",
                units="",
                values=np.ones((4, 4, 2)),
                affine=np.eye(3),  # Wrong shape
            )
