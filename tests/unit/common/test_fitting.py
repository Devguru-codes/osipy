"""Unit tests for osipy.common.fitting module."""

import numpy as np
import pytest

from osipy.common.fitting.result import FittingResult


class TestFittingResult:
    """Tests for FittingResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> FittingResult:
        """Create sample fitting result."""
        return FittingResult(
            parameters={"Ktrans": 0.15, "ve": 0.3, "vp": 0.02},
            uncertainties={"Ktrans": 0.01, "ve": 0.05, "vp": 0.002},
            residuals=np.random.randn(30) * 0.01,
            r_squared=0.95,
            converged=True,
            n_iterations=15,
            termination_reason="converged",
            model_name="ExtendedTofts",
            initial_guess={"Ktrans": 0.1, "ve": 0.2, "vp": 0.01},
            bounds={"Ktrans": (0, 5), "ve": (0, 1), "vp": (0, 0.2)},
        )

    def test_is_valid_good_fit(self, sample_result: FittingResult) -> None:
        """Test is_valid for good fit."""
        assert sample_result.is_valid is True

    def test_is_valid_unconverged(self) -> None:
        """Test is_valid for unconverged fit."""
        result = FittingResult(
            parameters={"Ktrans": 0.15},
            residuals=np.zeros(10),
            r_squared=0.8,
            converged=False,
            n_iterations=1000,
            termination_reason="max_iter",
            model_name="Tofts",
        )
        assert result.is_valid is False

    def test_is_valid_poor_r_squared(self) -> None:
        """Test is_valid for poor R² fit."""
        result = FittingResult(
            parameters={"Ktrans": 0.15},
            residuals=np.zeros(10),
            r_squared=0.3,
            converged=True,
            n_iterations=10,
            termination_reason="converged",
            model_name="Tofts",
        )
        assert result.is_valid is False

    def test_rmse(self) -> None:
        """Test RMSE calculation."""
        residuals = np.array([0.1, -0.1, 0.1, -0.1])
        result = FittingResult(
            parameters={"Ktrans": 0.15},
            residuals=residuals,
            r_squared=0.9,
            converged=True,
            n_iterations=10,
            termination_reason="converged",
            model_name="Tofts",
        )
        assert result.rmse == pytest.approx(0.1)

    def test_rmse_empty_residuals(self) -> None:
        """Test RMSE with empty residuals."""
        result = FittingResult(
            parameters={"Ktrans": 0.15},
            residuals=np.array([]),
            r_squared=0.9,
            converged=True,
            n_iterations=10,
            termination_reason="converged",
            model_name="Tofts",
        )
        assert np.isnan(result.rmse)

    def test_n_parameters(self, sample_result: FittingResult) -> None:
        """Test n_parameters property."""
        assert sample_result.n_parameters == 3

    def test_get_parameter(self, sample_result: FittingResult) -> None:
        """Test get_parameter method."""
        assert sample_result.get_parameter("Ktrans") == 0.15
        assert sample_result.get_parameter("ve") == 0.3
        assert np.isnan(sample_result.get_parameter("nonexistent"))
        assert sample_result.get_parameter("nonexistent", 0.0) == 0.0

    def test_get_uncertainty(self, sample_result: FittingResult) -> None:
        """Test get_uncertainty method."""
        assert sample_result.get_uncertainty("Ktrans") == 0.01
        assert np.isnan(sample_result.get_uncertainty("nonexistent"))

    def test_get_uncertainty_no_uncertainties(self) -> None:
        """Test get_uncertainty when uncertainties is None."""
        result = FittingResult(
            parameters={"Ktrans": 0.15},
            uncertainties=None,
            residuals=np.zeros(10),
            r_squared=0.9,
            converged=True,
            n_iterations=10,
            termination_reason="converged",
            model_name="Tofts",
        )
        assert np.isnan(result.get_uncertainty("Ktrans"))
