"""Baseline correction for time-series data.

This module provides functions for baseline estimation and correction
in perfusion time-series data.

GPU/CPU agnostic using the xp array module pattern.

"""

from typing import TYPE_CHECKING, Any

from osipy.common.backend.array_module import get_array_module
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def baseline_correction(
    data: "NDArray[np.floating[Any]]",
    baseline_frames: int = 5,
    method: str = "mean",
) -> "NDArray[np.floating[Any]]":
    """Correct baseline in time-series data.

    Estimates baseline from initial frames and subtracts or normalizes
    the signal relative to baseline.

    GPU/CPU agnostic - operates on same device as input data.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-series data. Last dimension is time.
    baseline_frames : int, default=5
        Number of initial frames for baseline estimation.
    method : str, default="mean"
        Correction method:
        - "mean": Subtract mean baseline (additive correction)
        - "median": Subtract median baseline
        - "normalize": Divide by mean baseline (multiplicative)
        - "percent": Convert to percent change from baseline

    Returns
    -------
    NDArray[np.floating]
        Baseline-corrected data with same shape as input.

    Raises
    ------
    ValueError
        If baseline_frames exceeds data length or invalid method.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.signal.baseline import baseline_correction
    >>> data = np.random.rand(64, 64, 20, 30) + 100  # Signal with offset
    >>> corrected = baseline_correction(data, baseline_frames=5)
    >>> print(corrected.shape)
    (64, 64, 20, 30)
    """
    xp = get_array_module(data)

    if baseline_frames <= 0:
        msg = "baseline_frames must be positive"
        raise DataValidationError(msg)

    n_timepoints = data.shape[-1]
    if baseline_frames > n_timepoints:
        msg = (
            f"baseline_frames ({baseline_frames}) exceeds "
            f"number of time points ({n_timepoints})"
        )
        raise DataValidationError(msg)

    # Extract baseline data
    baseline_data = data[..., :baseline_frames]

    # Compute baseline estimate
    if method == "mean":
        baseline = xp.mean(baseline_data, axis=-1, keepdims=True)
        return data - baseline
    elif method == "median":
        baseline = xp.median(baseline_data, axis=-1, keepdims=True)
        return data - baseline
    elif method == "normalize":
        baseline = xp.mean(baseline_data, axis=-1, keepdims=True)
        # Avoid division by zero
        baseline = xp.where(baseline == 0, 1.0, baseline)
        return data / baseline
    elif method == "percent":
        baseline = xp.mean(baseline_data, axis=-1, keepdims=True)
        # Avoid division by zero
        baseline = xp.where(baseline == 0, 1.0, baseline)
        return 100.0 * (data - baseline) / baseline
    else:
        msg = (
            f"Unknown method: {method}. Use 'mean', 'median', 'normalize', or 'percent'"
        )
        raise DataValidationError(msg)


def estimate_baseline_std(
    data: "NDArray[np.floating[Any]]",
    baseline_frames: int = 5,
) -> "NDArray[np.floating[Any]]":
    """Estimate noise standard deviation from baseline frames.

    GPU/CPU agnostic - operates on same device as input data.

    Parameters
    ----------
    data : NDArray[np.floating]
        Time-series data. Last dimension is time.
    baseline_frames : int, default=5
        Number of initial frames for noise estimation.

    Returns
    -------
    NDArray[np.floating]
        Standard deviation estimate for each voxel.
        Shape is data.shape[:-1].

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.signal.baseline import estimate_baseline_std
    >>> data = np.random.randn(64, 64, 20, 30)
    >>> noise_std = estimate_baseline_std(data, baseline_frames=5)
    >>> print(noise_std.shape)
    (64, 64, 20)
    """
    xp = get_array_module(data)

    if baseline_frames <= 1:
        msg = "baseline_frames must be > 1 for std estimation"
        raise DataValidationError(msg)

    n_timepoints = data.shape[-1]
    if baseline_frames > n_timepoints:
        msg = (
            f"baseline_frames ({baseline_frames}) exceeds time points ({n_timepoints})"
        )
        raise DataValidationError(msg)

    baseline_data = data[..., :baseline_frames]
    return xp.std(baseline_data, axis=-1)
