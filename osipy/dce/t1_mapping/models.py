"""Signal models for T1 mapping.

Provides ``SPGRSignalModel`` (VFA) and ``LookLockerSignalModel`` as
``BaseSignalModel`` subclasses, enabling registry-driven dispatch and
integration with the shared fitting infrastructure.

References
----------
.. [1] Deoni SCL et al. MRM 2003;49(3):515-526.
.. [2] Look DC, Locker DR. Rev Sci Instrum 1970;41:250-251.
"""

from __future__ import annotations

from osipy.common.models.base import BaseSignalModel


class SPGRSignalModel(BaseSignalModel):
    """SPGR signal model for VFA T1 mapping.

    S(alpha) = M0 * sin(alpha) * (1 - E1) / (1 - E1 * cos(alpha))

    where E1 = exp(-TR / T1).

    References
    ----------
    .. [1] Deoni SCL et al. MRM 2003;49(3):515-526.
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "SPGR"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Deoni SCL et al. MRM 2003;49(3):515-526."

    @property
    def parameters(self) -> list[str]:
        """Return list of parameter names."""
        return ["T1", "M0"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return mapping of parameter names to units."""
        return {"T1": "ms", "M0": "a.u."}

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds."""
        return {"T1": (1.0, 10000.0), "M0": (0.0, 1e10)}


class LookLockerSignalModel(BaseSignalModel):
    """Look-Locker inversion recovery signal model.

    S(TI) = A - B * exp(-TI / T1*)

    Model parameters: T1_star (apparent T1 in ms), A (steady-state signal),
    B (signal amplitude).

    References
    ----------
    .. [1] Look DC, Locker DR. Rev Sci Instrum 1970;41:250-251.
    .. [2] Deichmann R, Haase A. J Magn Reson 1992;96:608-612.
    """

    @property
    def name(self) -> str:
        """Return model name."""
        return "Look-Locker"

    @property
    def reference(self) -> str:
        """Return primary literature citation."""
        return "Look DC, Locker DR. Rev Sci Instrum 1970;41:250-251."

    @property
    def parameters(self) -> list[str]:
        """Return list of parameter names."""
        return ["T1_star", "A", "B"]

    @property
    def parameter_units(self) -> dict[str, str]:
        """Return mapping of parameter names to units."""
        return {"T1_star": "ms", "A": "a.u.", "B": "a.u."}

    def get_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds."""
        return {
            "T1_star": (1.0, 10000.0),
            "A": (-1e10, 1e10),
            "B": (0.0, 1e10),
        }
