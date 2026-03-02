"""Backward-compatibility shim for convolution operations.

All convolution functions now live in ``osipy.common.convolution``.
This module re-exports them for backward compatibility.

.. deprecated::
    Import from ``osipy.common.convolution`` instead.
"""

from osipy.common.convolution.deconv import deconvolve_svd, deconvolve_svd_batch
from osipy.common.convolution.fft import convolve_aif, convolve_aif_batch

__all__ = [
    "convolve_aif",
    "convolve_aif_batch",
    "deconvolve_svd",
    "deconvolve_svd_batch",
]
