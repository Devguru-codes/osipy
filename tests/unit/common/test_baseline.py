"""Unit tests for osipy.common.signal.baseline module."""

import numpy as np
import pytest

from osipy.common.exceptions import DataValidationError
from osipy.common.signal.baseline import baseline_correction, estimate_baseline_std


class TestBaselineCorrection:
    """Tests for baseline_correction function."""

    @pytest.fixture
    def sample_data(self) -> np.ndarray:
        """Create sample time-series data with offset."""
        # Create data with known baseline offset of 100
        np.random.seed(42)
        data = np.random.rand(10, 10, 5, 30) + 100.0
        return data

    def test_mean_correction(self, sample_data: np.ndarray) -> None:
        """Test mean baseline correction."""
        corrected = baseline_correction(sample_data, baseline_frames=5, method="mean")

        # Baseline should be approximately zero after correction
        baseline_mean = np.mean(corrected[..., :5])
        assert abs(baseline_mean) < 0.1

    def test_median_correction(self, sample_data: np.ndarray) -> None:
        """Test median baseline correction."""
        corrected = baseline_correction(sample_data, baseline_frames=5, method="median")

        # Baseline should be approximately zero after correction
        baseline_median = np.median(corrected[..., :5])
        assert abs(baseline_median) < 0.1

    def test_normalize_correction(self, sample_data: np.ndarray) -> None:
        """Test normalization baseline correction."""
        corrected = baseline_correction(
            sample_data, baseline_frames=5, method="normalize"
        )

        # Baseline should be approximately 1 after normalization
        baseline_mean = np.mean(corrected[..., :5])
        assert abs(baseline_mean - 1.0) < 0.01

    def test_percent_correction(self, sample_data: np.ndarray) -> None:
        """Test percent change baseline correction."""
        corrected = baseline_correction(
            sample_data, baseline_frames=5, method="percent"
        )

        # Baseline should be approximately 0% change
        baseline_mean = np.mean(corrected[..., :5])
        assert abs(baseline_mean) < 1.0  # Less than 1% deviation

    def test_shape_preserved(self, sample_data: np.ndarray) -> None:
        """Test that output shape matches input shape."""
        corrected = baseline_correction(sample_data, baseline_frames=5)
        assert corrected.shape == sample_data.shape

    def test_invalid_baseline_frames(self, sample_data: np.ndarray) -> None:
        """Test that invalid baseline_frames raises ValueError."""
        with pytest.raises(DataValidationError, match="positive"):
            baseline_correction(sample_data, baseline_frames=0)

    def test_baseline_frames_exceeds_data(self, sample_data: np.ndarray) -> None:
        """Test that baseline_frames > n_timepoints raises ValueError."""
        with pytest.raises(DataValidationError, match="exceeds"):
            baseline_correction(sample_data, baseline_frames=100)

    def test_invalid_method(self, sample_data: np.ndarray) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(DataValidationError, match="Unknown method"):
            baseline_correction(sample_data, baseline_frames=5, method="invalid")


class TestEstimateBaselineStd:
    """Tests for estimate_baseline_std function."""

    def test_known_noise(self) -> None:
        """Test noise estimation with known noise level."""
        np.random.seed(42)
        noise_level = 0.5
        data = np.random.randn(10, 10, 5, 30) * noise_level

        estimated_std = estimate_baseline_std(data, baseline_frames=10)

        # Estimated std should be close to true noise level
        assert np.mean(estimated_std) == pytest.approx(noise_level, rel=0.2)

    def test_output_shape(self) -> None:
        """Test output shape."""
        data = np.random.rand(10, 10, 5, 30)
        std = estimate_baseline_std(data, baseline_frames=5)

        # Output should be spatial dimensions only
        assert std.shape == (10, 10, 5)

    def test_invalid_baseline_frames(self) -> None:
        """Test that baseline_frames <= 1 raises ValueError."""
        data = np.random.rand(10, 10, 5, 30)
        with pytest.raises(DataValidationError, match="must be > 1"):
            estimate_baseline_std(data, baseline_frames=1)
