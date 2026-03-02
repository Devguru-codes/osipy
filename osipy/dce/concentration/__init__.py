"""Signal-to-concentration conversion for DCE-MRI (OSIPI: P.SC1.001).

This module provides functions for converting signal intensity
to indicator concentration (OSIPI: Q.IC1.001).

References
----------
.. [1] OSIPI CAPLEX, https://osipi.github.io/OSIPI_CAPLEX/
"""

from osipy.common.types import DCEAcquisitionParams
from osipy.dce.concentration.registry import (
    get_concentration_model,
    list_concentration_models,
    register_concentration_model,
)
from osipy.dce.concentration.signal_to_conc import signal_to_concentration

__all__ = [
    "DCEAcquisitionParams",
    "get_concentration_model",
    "list_concentration_models",
    "register_concentration_model",
    "signal_to_concentration",
]
