"""Integration tests for automated perfusion analysis pipeline.

Tests for osipy/pipeline/ module.
"""

from __future__ import annotations

import numpy as np

from osipy.common.types import Modality
from osipy.pipeline import (
    IVIMPipeline,
    IVIMPipelineConfig,
    PipelineResult,
    run_analysis,
)


class TestRunAnalysis:
    """Tests for unified run_analysis function."""

    def test_run_analysis_returns_pipeline_result(self) -> None:
        """Test that run_analysis returns PipelineResult."""
        # Create minimal IVIM data (simplest case)
        data = np.random.rand(8, 8, 4, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        result = run_analysis(
            data,
            modality=Modality.IVIM,
            b_values=b_values,
        )

        assert isinstance(result, PipelineResult)
        assert result.modality == Modality.IVIM

    def test_run_analysis_string_modality(self) -> None:
        """Test run_analysis accepts string modality."""
        data = np.random.rand(8, 8, 4, 6)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        result = run_analysis(
            data,
            modality="ivim",
            b_values=b_values,
        )

        assert result.modality == Modality.IVIM


class TestIVIMPipelineIntegration:
    """Integration tests for IVIM pipeline."""

    def test_ivim_pipeline_full_run(self) -> None:
        """Test full IVIM pipeline execution."""
        # Create synthetic IVIM data
        shape = (8, 8, 4)
        b_values = np.array([0, 50, 100, 200, 400, 800])
        n_bvalues = len(b_values)

        # Known parameters
        d = 1.0e-3  # mm^2/s
        d_star = 10e-3  # mm^2/s
        f = 0.1

        # Generate bi-exponential signal
        signal = np.zeros((*shape, n_bvalues))
        for i, b in enumerate(b_values):
            signal[..., i] = f * np.exp(-b * d_star) + (1 - f) * np.exp(-b * d)

        # Add noise
        signal += np.random.randn(*signal.shape) * 0.01
        signal = np.maximum(signal, 0.01)

        config = IVIMPipelineConfig()
        pipeline = IVIMPipeline(config)
        result = pipeline.run(signal, b_values=b_values)

        assert result is not None
        assert hasattr(result, "fit_result")


class TestPipelineMemoryEfficiency:
    """Tests for memory-efficient processing."""

    def test_pipeline_handles_large_data(self) -> None:
        """Test pipeline can handle larger datasets without memory issues."""
        # Create moderately large data
        data = np.random.rand(32, 32, 8, 6).astype(np.float32)
        b_values = np.array([0, 50, 100, 200, 400, 800])

        config = IVIMPipelineConfig()
        pipeline = IVIMPipeline(config)

        # Should complete without memory error
        result = pipeline.run(data, b_values=b_values)
        assert result is not None


class TestAutomatedPipelineHeadless:
    """Tests for headless/automated pipeline execution."""

    def test_headless_execution_no_display(self) -> None:
        """Test pipeline runs without display requirements."""
        import os

        # Ensure no display is required
        original_display = os.environ.get("DISPLAY")
        try:
            os.environ.pop("DISPLAY", None)

            # Run minimal analysis
            data = np.random.rand(4, 4, 2, 4)
            b_values = np.array([0, 100, 400, 800])

            result = run_analysis(data, modality="ivim", b_values=b_values)
            assert result is not None

        finally:
            if original_display:
                os.environ["DISPLAY"] = original_display

    def test_pipeline_deterministic_results(self) -> None:
        """Test pipeline produces deterministic results."""
        np.random.seed(42)
        data = np.random.rand(4, 4, 2, 4)
        b_values = np.array([0, 100, 400, 800])

        # Run twice with same seed
        np.random.seed(42)
        result1 = run_analysis(data.copy(), modality="ivim", b_values=b_values)

        np.random.seed(42)
        result2 = run_analysis(data.copy(), modality="ivim", b_values=b_values)

        # Results should be identical
        for key in result1.parameter_maps:
            if key in result2.parameter_maps:
                np.testing.assert_array_almost_equal(
                    result1.parameter_maps[key].values,
                    result2.parameter_maps[key].values,
                )
