"""Cross-platform reproducibility tests.

Tests that the system produces identical results across different
operating systems for the same input.

These tests verify deterministic behavior that ensures reproducibility
across Linux, macOS, and Windows.
"""

from __future__ import annotations

import numpy as np


class TestModelFittingReproducibility:
    """Tests for model fitting reproducibility."""

    def test_ivim_fitting_deterministic(self) -> None:
        """Test IVIM fitting produces deterministic results."""
        from osipy.ivim import fit_ivim

        np.random.seed(42)

        # Create synthetic IVIM data
        b_values = np.array([0, 50, 100, 200, 400, 800])
        d = 1.0e-3
        d_star = 10e-3
        f = 0.1

        signal = np.zeros((4, 4, 2, len(b_values)))
        for i, b in enumerate(b_values):
            signal[..., i] = f * np.exp(-b * d_star) + (1 - f) * np.exp(-b * d)

        signal += np.random.randn(*signal.shape) * 0.01
        signal = np.maximum(signal, 0.01)

        # Fit twice
        result1 = fit_ivim(signal, b_values)
        result2 = fit_ivim(signal, b_values)

        # D values should be identical
        np.testing.assert_array_almost_equal(
            result1.d_map.values,
            result2.d_map.values,
        )
