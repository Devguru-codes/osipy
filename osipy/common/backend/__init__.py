"""GPU/CPU backend abstraction layer for osipy.

This module provides CPU/GPU agnostic array operations using the array module
pattern. When CuPy is available and GPU is enabled, operations automatically
execute on GPU. Otherwise, they fall back to NumPy on CPU.

The design follows Constitution Principle VII (Hardware Abstraction):
- All numerical algorithms are CPU/GPU agnostic
- GPU acceleration is optional (CuPy not required)
- GPU and CPU produce numerically equivalent results
- Automatic fallback when GPU memory is exhausted

Example
-------
>>> from osipy.common.backend import get_array_module, to_gpu, to_numpy
>>> import numpy as np
>>> data = np.random.randn(100, 100)
>>> # Automatically use GPU if available
>>> xp = get_array_module(data)
>>> result = xp.sum(data)
>>> # Explicitly transfer to GPU
>>> gpu_data = to_gpu(data)  # Returns CuPy array if available, else NumPy

References
----------
.. [1] CuPy Documentation: https://docs.cupy.dev/en/stable/
.. [2] OSIPI Code Collection: https://osipi.org/
"""

from osipy.common.backend.array_module import (
    get_array_module,
    to_gpu,
    to_numpy,
)
from osipy.common.backend.config import (
    GPUConfig,
    get_backend,
    is_gpu_available,
    set_backend,
)

# Check GPU availability at import time
GPU_AVAILABLE = is_gpu_available()

__all__ = [
    # Status flag
    "GPU_AVAILABLE",
    # Configuration
    "GPUConfig",
    # Array module utilities
    "get_array_module",
    "get_backend",
    "is_gpu_available",
    "set_backend",
    "to_gpu",
    "to_numpy",
]
