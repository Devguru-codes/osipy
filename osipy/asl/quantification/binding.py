"""ASL binding adapter for the shared fitting infrastructure.

``BoundASLModel`` wraps the Buxton pCASL model with PLDs, M0, and
labeling parameters, producing a ``FittableModel`` that the shared
fitter can use without knowing about ASL-specific context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.models.fittable import BaseBoundModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from osipy.asl.quantification.base import BaseASLModel
    from osipy.asl.quantification.multi_pld import MultiPLDParams


class BoundASLModel(BaseBoundModel):
    """ASL model with PLDs and labeling parameters bound.

    Wraps the Buxton pCASL model so the fitter only sees
    ``predict_array_batch(free_params) -> output``.

    The model always predicts with unit M0 (M0=1). Callers must
    normalize observed delta-M by M0 before fitting so that the
    fitter works on M0-normalized data. This avoids per-voxel
    side-channels that are incompatible with active-voxel tracking
    in the LM fitter.

    Parameters
    ----------
    params : MultiPLDParams
        ASL quantification parameters (PLDs, label duration, T1s, etc.).
    fixed : dict[str, float] | None
        Parameters to fix at constant values during fitting.
    model : BaseASLModel | None
        ASL signal model to use. If ``None``, defaults to
        ``BuxtonMultiPLDModel``.
    """

    def __init__(
        self,
        params: MultiPLDParams,
        fixed: dict[str, float] | None = None,
        model: BaseASLModel | None = None,
    ) -> None:
        if model is None:
            from osipy.asl.quantification.multi_pld import BuxtonMultiPLDModel

            model = BuxtonMultiPLDModel()
        super().__init__(model, fixed)

        self._params = params
        # Convert PLDs and timing to seconds
        self._plds_s = params.plds / 1000.0
        self._tau_s = params.label_duration / 1000.0
        self._t1b_s = params.t1_blood / 1000.0
        self._t1t_s = params.t1_tissue / 1000.0
        self._alpha = params.labeling_efficiency
        self._lam = params.partition_coefficient

    def ensure_device(self, xp: Any) -> None:
        """Transfer PLD array to the target device."""
        super().ensure_device(xp)
        self._plds_s = xp.asarray(self._plds_s)

    def predict_array_batch(
        self, free_params_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        """Predict M0-normalized delta-M for a batch of voxels.

        Parameters
        ----------
        free_params_batch : NDArray
            Free parameter values, shape ``(n_free, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Predicted delta-M/M0, shape ``(n_plds, n_voxels)``.
        """
        from osipy.asl.quantification.multi_pld import _buxton_model_pcasl_batch

        full_params = self._expand_params(free_params_batch, xp)
        all_names = self._model.parameters

        cbf_idx = all_names.index("CBF")
        att_idx = all_names.index("ATT")
        cbf = full_params[cbf_idx, :]
        att = full_params[att_idx, :]

        n_voxels = free_params_batch.shape[1]
        m0 = xp.ones(n_voxels, dtype=free_params_batch.dtype)

        return _buxton_model_pcasl_batch(
            xp.asarray(self._plds_s),
            cbf,
            att,
            m0,
            self._tau_s,
            self._t1b_s,
            self._t1t_s,
            self._alpha,
            self._lam,
            xp,
        )

    def get_initial_guess_batch(
        self, observed_batch: NDArray[np.floating[Any]], xp: Any
    ) -> NDArray[np.floating[Any]]:
        """Get initial parameter guesses from signal shape.

        Parameters
        ----------
        observed_batch : NDArray
            Observed data, shape ``(n_plds, n_voxels)``.
        xp : module
            Array module.

        Returns
        -------
        NDArray
            Initial guesses, shape ``(n_free, n_voxels)``.
        """
        _n_plds, n_voxels = observed_batch.shape
        all_names = self._model.parameters

        full_guess = xp.zeros((len(all_names), n_voxels), dtype=observed_batch.dtype)

        # CBF initial: 60 mL/100g/min (typical gray matter)
        cbf_idx = all_names.index("CBF")
        full_guess[cbf_idx, :] = 60.0

        # ATT initial: estimate from signal peak timing
        att_idx = all_names.index("ATT")
        peak_idx = xp.argmax(observed_batch, axis=0)  # (n_voxels,)
        plds_s = xp.asarray(self._plds_s)
        att_init = plds_s[peak_idx]
        att_init = xp.clip(att_init, 0.3, 3.0)
        full_guess[att_idx, :] = att_init

        if not self._fixed:
            return full_guess

        # Filter to free params only
        free_guess = xp.zeros((self._n_free, n_voxels), dtype=full_guess.dtype)
        for free_idx, all_idx in enumerate(self._free_indices):
            free_guess[free_idx, :] = full_guess[all_idx, :]
        return free_guess
