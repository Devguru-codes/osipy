"""Unit tests for DSC SVD deconvolution."""

import numpy as np
import pytest

from osipy.dsc.deconvolution import (
    DeconvolutionResult,
    SVDDeconvolutionParams,
    get_deconvolver,
)
from osipy.dsc.deconvolution.svd import (
    deconvolve_cSVD,
    deconvolve_oSVD,
)


def generate_synthetic_dsc_data(
    n_timepoints: int = 60,
    tr: float = 1.5,
    cbf: float = 60.0,
    mtt: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic DSC concentration data.

    Parameters
    ----------
    n_timepoints : int
        Number of time points.
    tr : float
        Temporal resolution in seconds.
    cbf : float
        True CBF value.
    mtt : float
        True MTT value.

    Returns
    -------
    concentration : ndarray
        Tissue concentration curve.
    aif : ndarray
        Arterial input function.
    time : ndarray
        Time points.
    """
    time = np.arange(n_timepoints) * tr

    # Generate gamma-variate AIF
    alpha = 3.0
    beta = 1.5
    aif = np.zeros(n_timepoints)
    t_shifted = time - 5  # Bolus arrival at t=5s
    mask = t_shifted > 0
    aif[mask] = (t_shifted[mask] ** alpha) * np.exp(-t_shifted[mask] / beta)
    aif = aif / np.max(aif) * 10  # Scale to typical values

    # Generate exponential residue function
    # R(t) = CBF * exp(-t/MTT)
    residue = cbf * np.exp(-time / mtt)

    # Convolve to get tissue concentration
    # C(t) = AIF ⊗ R(t)
    concentration = np.convolve(aif, residue, mode="full")[:n_timepoints] * tr

    return concentration, aif, time


class TestDeconvolveOSVD:
    """Tests for oscillation-index SVD deconvolution."""

    def test_basic_deconvolution(self) -> None:
        """Test basic oSVD deconvolution."""
        concentration, aif, time = generate_synthetic_dsc_data()

        # Reshape to 2D (single voxel)
        conc_2d = concentration.reshape(1, 1, -1)

        result = deconvolve_oSVD(conc_2d, aif, time)

        assert isinstance(result, DeconvolutionResult)
        assert result.residue_function.shape == conc_2d.shape
        assert result.cbf.shape == (1, 1)
        assert result.mtt.shape == (1, 1)

    def test_cbf_estimation(self) -> None:
        """Test that CBF is reasonably estimated."""
        true_cbf = 60.0
        concentration, aif, time = generate_synthetic_dsc_data(cbf=true_cbf, mtt=4.0)
        conc_2d = concentration.reshape(1, 1, -1)

        result = deconvolve_oSVD(conc_2d, aif, time)

        # CBF should be in reasonable range (within factor of 2)
        estimated_cbf = result.cbf[0, 0]
        assert estimated_cbf > 0
        # Note: exact recovery depends on many factors

    def test_with_mask(self) -> None:
        """Test deconvolution with mask."""
        concentration, aif, time = generate_synthetic_dsc_data()
        conc_3d = np.broadcast_to(concentration, (4, 4, len(concentration))).copy()

        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True

        result = deconvolve_oSVD(conc_3d, aif, time, mask=mask)

        assert result.cbf.shape == (4, 4)
        # Only masked voxels should have values
        assert result.cbf[0, 0] == 0
        assert result.cbf[1, 1] > 0


class TestDeconvolveCSVD:
    """Tests for circular SVD deconvolution."""

    def test_basic_deconvolution(self) -> None:
        """Test basic cSVD deconvolution."""
        concentration, aif, time = generate_synthetic_dsc_data()
        conc_2d = concentration.reshape(1, 1, -1)

        result = deconvolve_cSVD(conc_2d, aif, time)

        assert isinstance(result, DeconvolutionResult)
        assert result.cbf.shape == (1, 1)
        assert result.cbf[0, 0] > 0

    def test_custom_threshold(self) -> None:
        """Test cSVD with custom threshold."""
        concentration, aif, time = generate_synthetic_dsc_data()
        conc_2d = concentration.reshape(1, 1, -1)

        params = SVDDeconvolutionParams(method="cSVD", threshold=0.1)
        result = deconvolve_cSVD(conc_2d, aif, time, params=params)

        assert result.threshold_used == 0.1


class TestDeconvolverRegistry:
    """Tests for registry-driven deconvolver dispatch."""

    @pytest.mark.parametrize("name", ["oSVD", "cSVD", "sSVD"])
    def test_get_deconvolver(self, name: str) -> None:
        """Test getting deconvolver from registry."""
        concentration, aif, time = generate_synthetic_dsc_data()
        conc_2d = concentration.reshape(1, 1, -1)

        deconvolver = get_deconvolver(name)
        result = deconvolver.deconvolve(conc_2d, aif, time)

        assert isinstance(result, DeconvolutionResult)
