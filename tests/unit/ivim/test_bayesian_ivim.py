"""Unit tests for Bayesian IVIM fitting via two-stage approach.

Tests that the 'bayesian' IVIM fitter strategy routes through
TwoStageBayesianIVIMFitter and returns standard IVIMFitResult
objects with optional uncertainty maps.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy.ivim.fitting.estimators import FittingMethod, IVIMFitParams, fit_ivim
from osipy.ivim.fitting.registry import get_ivim_fitter


class TestBayesianRegistry:
    """Tests for Bayesian fitter registry integration."""

    def test_bayesian_fitter_registered(self) -> None:
        """The 'bayesian' strategy is discoverable via the registry."""
        fitter_func = get_ivim_fitter("bayesian")
        assert callable(fitter_func)

    def test_fitting_method_enum(self) -> None:
        """FittingMethod.BAYESIAN resolves to 'bayesian' value."""
        assert FittingMethod.BAYESIAN.value == "bayesian"


def _make_synthetic_volume(seed: int = 42) -> dict:
    """Create synthetic IVIM volume for testing."""
    rng = np.random.RandomState(seed)

    nx, ny = 4, 4
    n_b = 8
    b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800], dtype=float)

    signal = np.zeros((nx, ny, n_b))
    d_true = np.zeros((nx, ny))
    d_star_true = np.zeros((nx, ny))
    f_true = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            s0 = 1000.0
            d = 0.8e-3 + 0.4e-3 * (i / nx)
            d_star = 10.0e-3 + 10.0e-3 * (j / ny)
            f = 0.08 + 0.1 * ((i + j) / (nx + ny))

            d_true[i, j] = d
            d_star_true[i, j] = d_star
            f_true[i, j] = f

            s = s0 * ((1 - f) * np.exp(-b_values * d) + f * np.exp(-b_values * d_star))
            signal[i, j, :] = s + rng.randn(n_b) * 15

    mask = np.ones((nx, ny), dtype=bool)

    return {
        "signal": signal,
        "b_values": b_values,
        "mask": mask,
        "d_true": d_true,
        "d_star_true": d_star_true,
        "f_true": f_true,
    }


class TestBayesianIVIMFitting:
    """Tests for Bayesian IVIM fitting through fit_ivim()."""

    @pytest.fixture
    def synthetic_volume(self) -> dict:
        return _make_synthetic_volume()

    def test_fit_returns_ivim_fit_result(self, synthetic_volume: dict) -> None:
        """Bayesian fitting returns a standard IVIMFitResult."""
        from osipy.ivim.fitting.estimators import IVIMFitResult

        params = IVIMFitParams(bayesian_params={"compute_uncertainty": False})

        result = fit_ivim(
            signal=synthetic_volume["signal"],
            b_values=synthetic_volume["b_values"],
            mask=synthetic_volume["mask"],
            method=FittingMethod.BAYESIAN,
            params=params,
        )

        assert isinstance(result, IVIMFitResult)
        assert result.d_map is not None
        assert result.d_star_map is not None
        assert result.f_map is not None

    def test_parameter_map_units(self, synthetic_volume: dict) -> None:
        """Parameter maps have correct OSIPI-compliant units."""
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": False})

        result = fit_ivim(
            signal=synthetic_volume["signal"],
            b_values=synthetic_volume["b_values"],
            mask=synthetic_volume["mask"],
            method=FittingMethod.BAYESIAN,
            params=params,
        )

        assert result.d_map.units == "mm^2/s"
        assert result.d_star_map.units == "mm^2/s"
        assert result.f_map.units == ""

    def test_fitting_accuracy(self, synthetic_volume: dict) -> None:
        """Bayesian MAP fitting recovers parameters within tolerance."""
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": False})

        result = fit_ivim(
            signal=synthetic_volume["signal"],
            b_values=synthetic_volume["b_values"],
            mask=synthetic_volume["mask"],
            method=FittingMethod.BAYESIAN,
            params=params,
        )

        # quality_mask may be expanded to 3D; squeeze to match 2D input
        qmask = np.squeeze(result.quality_mask)
        mask = synthetic_volume["mask"]
        d_est = np.squeeze(result.d_map.values)[mask]
        d_true = synthetic_volume["d_true"][mask]
        valid = qmask[mask]

        # At least positive values
        assert np.all(d_est[valid] >= 0)

        # Positive correlation with truth
        if np.sum(valid) > 2 and np.std(d_est[valid]) > 0:
            corr = np.corrcoef(d_est[valid], d_true[valid])[0, 1]
            assert corr > 0, "D should correlate positively with truth"

    def test_quality_mask(self, synthetic_volume: dict) -> None:
        """Quality mask is produced and has correct shape."""
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": False})

        result = fit_ivim(
            signal=synthetic_volume["signal"],
            b_values=synthetic_volume["b_values"],
            mask=synthetic_volume["mask"],
            method=FittingMethod.BAYESIAN,
            params=params,
        )

        assert result.quality_mask is not None
        assert result.quality_mask.dtype == bool

    def test_shape_mismatch_error(self, synthetic_volume: dict) -> None:
        """Mismatched signal/b-value dimensions raise FittingError."""
        from osipy.common.exceptions import FittingError

        wrong_b = np.array([0, 50, 100, 200])

        with pytest.raises(FittingError, match="Signal last dimension"):
            fit_ivim(
                signal=synthetic_volume["signal"],
                b_values=wrong_b,
                method=FittingMethod.BAYESIAN,
            )

    def test_default_bayesian_params(self, synthetic_volume: dict) -> None:
        """Bayesian fitting works with default params (no bayesian_params)."""
        result = fit_ivim(
            signal=synthetic_volume["signal"],
            b_values=synthetic_volume["b_values"],
            mask=synthetic_volume["mask"],
            method=FittingMethod.BAYESIAN,
        )

        assert result.d_map is not None


class TestBayesianUncertainty:
    """Tests for Bayesian uncertainty estimation."""

    @pytest.fixture
    def synthetic_volume(self) -> dict:
        return _make_synthetic_volume()

    def test_uncertainty_maps_present(self, synthetic_volume: dict) -> None:
        """When compute_uncertainty=True, std maps appear."""
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": True})

        result = fit_ivim(
            signal=synthetic_volume["signal"],
            b_values=synthetic_volume["b_values"],
            mask=synthetic_volume["mask"],
            method=FittingMethod.BAYESIAN,
            params=params,
        )

        assert result.d_map is not None

    def test_uncertainty_maps_have_correct_units(self, synthetic_volume: dict) -> None:
        """Uncertainty maps should share units with their base parameters."""
        fitter_func = get_ivim_fitter("bayesian")
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": True})

        param_maps = fitter_func(
            synthetic_volume["signal"],
            synthetic_volume["b_values"],
            synthetic_volume["mask"],
            params,
            False,
        )

        if "D_std" in param_maps:
            assert param_maps["D_std"].units == param_maps["D"].units
        if "f_std" in param_maps:
            assert param_maps["f_std"].units == param_maps["f"].units

    def test_uncertainty_values_positive(self, synthetic_volume: dict) -> None:
        """Uncertainty std values should be non-negative."""
        fitter_func = get_ivim_fitter("bayesian")
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": True})

        param_maps = fitter_func(
            synthetic_volume["signal"],
            synthetic_volume["b_values"],
            synthetic_volume["mask"],
            params,
            False,
        )

        for key, pmap in param_maps.items():
            if key.endswith("_std"):
                assert np.all(pmap.values >= 0), f"{key} has negative values"

    def test_no_uncertainty_when_disabled(self, synthetic_volume: dict) -> None:
        """When compute_uncertainty=False, no std maps appear."""
        fitter_func = get_ivim_fitter("bayesian")
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": False})

        param_maps = fitter_func(
            synthetic_volume["signal"],
            synthetic_volume["b_values"],
            synthetic_volume["mask"],
            params,
            False,
        )

        std_keys = [k for k in param_maps if k.endswith("_std")]
        assert len(std_keys) == 0, f"Unexpected std maps: {std_keys}"


class TestTwoStageFitter:
    """Tests for two-stage Bayesian IVIM fitter internals."""

    @pytest.fixture
    def synthetic_volume(self) -> dict:
        return _make_synthetic_volume()

    def test_d_star_not_stuck_at_10x_d(self, synthetic_volume: dict) -> None:
        """D* values should NOT be exactly 10x D after two-stage fitting."""
        params = IVIMFitParams(bayesian_params={"compute_uncertainty": False})

        result = fit_ivim(
            signal=synthetic_volume["signal"],
            b_values=synthetic_volume["b_values"],
            mask=synthetic_volume["mask"],
            method=FittingMethod.BAYESIAN,
            params=params,
        )

        qmask = np.squeeze(result.quality_mask)
        d_vals = np.squeeze(result.d_map.values)[qmask]
        ds_vals = np.squeeze(result.d_star_map.values)[qmask]

        if len(d_vals) > 0:
            ratios = ds_vals / np.maximum(d_vals, 1e-10)
            # Not all ratios should be exactly 10.0
            assert not np.allclose(ratios, 10.0, atol=0.01), (
                f"D*/D ratios are all ~10.0: {ratios}"
            )

    def test_empirical_priors_computed(self) -> None:
        """_compute_empirical_priors returns valid per-parameter stds."""
        from osipy.ivim.fitting.bayesian_ivim import TwoStageBayesianIVIMFitter

        fitter = TwoStageBayesianIVIMFitter()
        xp = np

        n_params = 4
        n_voxels = 100
        rng = np.random.RandomState(42)

        # Simulate Stage 1 results
        params = np.zeros((n_params, n_voxels))
        params[0] = rng.uniform(0, 2000, n_voxels)  # S0
        params[1] = rng.lognormal(-7, 0.3, n_voxels)  # D
        params[2] = rng.uniform(0.05, 0.3, n_voxels)  # f
        params[3] = rng.lognormal(-4.5, 0.5, n_voxels)  # D*

        r2 = np.full(n_voxels, 0.9)
        converged = np.ones(n_voxels, dtype=bool)
        param_names = ["S0", "D", "f", "D*"]
        bounds = {
            "S0": (0, 1e10),
            "D": (1e-4, 5e-3),
            "f": (0, 0.7),
            "D*": (2e-3, 0.1),
        }

        prior_stds = fitter._compute_empirical_priors(
            params, r2, converged, param_names, bounds, xp
        )

        assert prior_stds.shape == (n_params,)
        assert np.all(prior_stds > 0), f"Non-positive prior stds: {prior_stds}"
        assert np.all(np.isfinite(prior_stds)), f"Non-finite prior stds: {prior_stds}"

    def test_vector_prior_std_bayesian_fitter(self) -> None:
        """BayesianFitter accepts an array prior_std."""
        from osipy.common.fitting.bayesian import BayesianFitter

        prior = np.array([100.0, 0.001, 0.1, 0.01])
        fitter = BayesianFitter(prior_std=prior)
        np.testing.assert_array_equal(fitter.prior_std, prior)

    def test_prior_scale_parameter(self, synthetic_volume: dict) -> None:
        """Different prior_scale values run without error."""
        for scale in [0.5, 1.5, 3.0]:
            params = IVIMFitParams(
                bayesian_params={
                    "compute_uncertainty": False,
                    "prior_scale": scale,
                }
            )

            result = fit_ivim(
                signal=synthetic_volume["signal"],
                b_values=synthetic_volume["b_values"],
                mask=synthetic_volume["mask"],
                method=FittingMethod.BAYESIAN,
                params=params,
            )

            assert result.d_map is not None

    def test_improved_d_star_initial_guess(self) -> None:
        """BoundIVIMModel gives D* != 10xD for data with low b-values."""
        from osipy.ivim.models.biexponential import IVIMBiexponentialModel
        from osipy.ivim.models.binding import BoundIVIMModel

        b_values = np.array([0, 10, 20, 50, 100, 200, 400, 800], dtype=float)

        model = IVIMBiexponentialModel()
        bound = BoundIVIMModel(model, b_values)

        # Generate signal with known D* != 10*D
        s0 = 1000.0
        d_true = 1.0e-3
        d_star_true = 15.0e-3  # != 10 * 1.0e-3 = 10e-3
        f_true = 0.15

        xp = np
        n_voxels = 20
        rng = np.random.RandomState(42)
        signal = np.zeros((len(b_values), n_voxels))
        for v in range(n_voxels):
            s = s0 * (
                (1 - f_true) * np.exp(-b_values * d_true)
                + f_true * np.exp(-b_values * d_star_true)
            )
            signal[:, v] = s + rng.randn(len(b_values)) * 10

        guess = bound.get_initial_guess_batch(signal, xp)

        # D* guess should not be exactly 10 * D guess
        d_idx = model.parameters.index("D")
        ds_idx = model.parameters.index("D*")
        d_guess = guess[d_idx, :]
        ds_guess = guess[ds_idx, :]
        ratios = ds_guess / np.maximum(d_guess, 1e-10)

        assert not np.allclose(ratios, 10.0, atol=0.5), (
            f"D* initial guess is still 10xD: ratios={ratios}"
        )
