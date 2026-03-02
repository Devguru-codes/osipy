"""Base class for ASL signal models.

All ASL models (single-PLD quantification and multi-PLD ATT estimation)
inherit from ``BaseASLModel``, which extends ``BaseSignalModel`` with
ASL-specific properties: ``labeling_type`` and ``quantify()``.

References
----------
.. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from osipy.common.models.base import BaseSignalModel

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class BaseASLModel(BaseSignalModel):
    """Abstract base class for ASL signal models.

    Unifies single-PLD quantification models and multi-PLD ATT
    estimation models under ``BaseSignalModel``. Each model implements:

    - ``parameters``, ``parameter_units``, ``get_bounds()`` — from BaseSignalModel
    - ``labeling_type`` — ASL-specific ('pcasl', 'pasl', 'casl')
    - ``quantify()`` — optional analytical inverse (closed-form CBF)

    Models that support analytical quantification override ``quantify()``
    to compute CBF directly from delta-M and M0. Multi-PLD models that
    require iterative fitting leave ``quantify()`` as ``None`` and use
    the ``BoundASLModel`` + fitter pathway instead.

    References
    ----------
    .. [1] OSIPI ASL Lexicon, https://osipi.github.io/ASL-Lexicon/
    .. [2] Suzuki Y et al. MRM 2024;91(5):1743-1760. doi:10.1002/mrm.29815
    """

    @property
    @abstractmethod
    def labeling_type(self) -> str:
        """Return labeling type (e.g., 'pcasl', 'pasl', 'casl')."""
        ...

    def quantify(
        self,
        delta_m: "NDArray[np.floating[Any]]",
        m0: "NDArray[np.floating[Any]]",
        params: Any,
    ) -> "NDArray[np.floating[Any]] | None":
        """Analytical quantification shortcut.

        Override for models with closed-form CBF equations (single-PLD).
        Returns ``None`` if this model requires iterative fitting.

        Parameters
        ----------
        delta_m : NDArray
            Control-label difference signal.
        m0 : NDArray
            M0 calibration values.
        params : Any
            Quantification parameters.

        Returns
        -------
        NDArray | None
            CBF values in mL/100g/min, or ``None`` if iterative fitting required.
        """
        return None
