"""Arterial delay utilities for DCE-MRI.

Provides time-shifting of arterial input functions to model
bolus arrival delays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def shift_aif(
    aif: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    delay: float | NDArray[np.floating[Any]],
    xp: Any = None,
) -> NDArray[np.floating[Any]]:
    """Shift AIF by a time delay using linear interpolation.

    Implements aif_shifted[i] = aif(t[i] - delay), with zero for t - delay < 0.
    GPU/CPU agnostic via xp array module.

    Parameters
    ----------
    aif : NDArray[np.floating]
        Arterial input function values, shape (n_time,).
    time : NDArray[np.floating]
        Time points in seconds, shape (n_time,).
    delay : float or NDArray[np.floating]
        Delay in seconds. Scalar returns shape ``(n_time,)``.
        Array of shape ``(n_voxels,)`` returns ``(n_time, n_voxels)``.
    xp : module, optional
        Array module. Auto-detected if None.

    Returns
    -------
    NDArray[np.floating]
        Shifted AIF values.
    """
    if xp is None:
        xp = get_array_module(aif, time)

    # Scalar delay — use xp.interp directly
    if isinstance(delay, (int, float)):
        shifted_time = time - delay
        return xp.interp(shifted_time, time, aif, left=0.0, right=aif[-1])

    delay = xp.asarray(delay, dtype=time.dtype)
    if delay.ndim == 0:
        shifted_time = time - float(delay)
        return xp.interp(shifted_time, time, aif, left=0.0, right=aif[-1])

    # Array of delays (n_voxels,) — vectorized interpolation
    # shifted_times[i, v] = time[i] - delay[v]
    shifted_times = time[:, xp.newaxis] - delay[xp.newaxis, :]  # (n_time, n_voxels)

    # Vectorized searchsorted + linear interpolation
    flat = shifted_times.ravel()
    idx = xp.searchsorted(time, flat) - 1
    idx = xp.clip(idx, 0, len(time) - 2)

    t0 = time[idx]
    t1 = time[idx + 1]
    dt = t1 - t0
    dt = xp.where(dt > 0, dt, xp.ones_like(dt))
    frac = (flat - t0) / dt
    frac = xp.clip(frac, 0.0, 1.0)

    result = aif[idx] * (1.0 - frac) + aif[idx + 1] * frac
    result = xp.where(flat > time[0], result, 0.0)

    return result.reshape(shifted_times.shape)
