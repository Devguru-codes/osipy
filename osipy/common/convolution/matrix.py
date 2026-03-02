"""Convolution matrix construction and inversion.

This module provides functions for building convolution matrices and
computing their regularized pseudo-inverses for deconvolution operations.

References
----------
    - dcmri (https://github.com/dcmri/dcmri) utils.convmat(), utils.invconvmat()
    - Wu et al. (2003). Tracer arrival timing-insensitive technique.
      Magn Reson Med. 50:164-174.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from osipy.common.backend import get_array_module
from osipy.common.exceptions import DataValidationError

if TYPE_CHECKING:
    from numpy.typing import NDArray


def convmat(
    h: NDArray[np.floating],
    t: NDArray[np.floating],
    *,
    order: Literal[1, 2] = 1,
) -> NDArray[np.floating]:
    """Construct a convolution matrix from an impulse response.

    Creates a lower triangular matrix A such that the convolution
    f * h can be computed as A @ h when f is known, or the deconvolution
    can be computed by solving A @ x = y for x.

    Parameters
    ----------
    h : ndarray
        Impulse response function (e.g., AIF). Shape: (n_times,).
    t : ndarray
        Time points in seconds. Shape: (n_times,). Must be monotonically
        increasing.
    order : {1, 2}, default 1
        Integration order:
        - 1: First-order (trapezoidal) integration
        - 2: Second-order (Simpson's rule) integration

    Returns
    -------
    ndarray
        Convolution matrix. Shape: (n_times, n_times).
        Lower triangular Toeplitz-like structure.

    Notes
    -----
    For uniform time grids, the matrix is Toeplitz (constant diagonals).
    For non-uniform grids, the matrix accounts for varying time steps.

    The matrix is constructed such that:
        (f * h)[i] = sum_j A[i,j] * f[j]

    Using first-order (trapezoidal) integration:
        A[i,j] = dt[j] * (h[i-j] + h[i-j-1]) / 2  for j < i
        A[i,i] = 0

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import convmat
    >>> t = np.linspace(0, 10, 11)
    >>> aif = np.exp(-t / 3)
    >>> A = convmat(aif, t)
    >>> # Convolution: result = A @ signal

    References
    ----------
    .. [1] dcmri library: https://github.com/dcmri/dcmri
    """
    xp = get_array_module(h)

    h = xp.asarray(h)
    t = xp.asarray(t)

    n = len(t)
    if len(h) != n:
        raise DataValidationError("h and t must have the same length")

    if n == 0:
        return xp.array([[]], dtype=h.dtype)

    # Compute time steps
    dt = xp.diff(t)
    dt = xp.concatenate([xp.array([0.0]), dt])

    # Initialize convolution matrix
    A = xp.zeros((n, n), dtype=h.dtype)

    if order == 1:
        # First-order (trapezoidal) integration
        for i in range(1, n):
            for j in range(i):
                # Index into h: h[i-j] and h[i-j-1]
                idx = i - j
                if idx < n and idx >= 1:
                    # Trapezoidal weight
                    A[i, j] = dt[j + 1] * (h[idx] + h[idx - 1]) / 2.0
                elif idx == 0:
                    A[i, j] = dt[j + 1] * h[0] / 2.0
    else:
        # Second-order (Simpson's rule) integration
        for i in range(1, n):
            for j in range(i):
                idx = i - j
                if idx < n and idx >= 1:
                    # Simpson's rule weight (approximate for non-uniform grids)
                    if j == 0 or j == i - 1:
                        weight = dt[j + 1] / 3.0
                    elif j % 2 == 1:
                        weight = 4.0 * dt[j + 1] / 3.0
                    else:
                        weight = 2.0 * dt[j + 1] / 3.0
                    A[i, j] = weight * h[idx]

    return A


def invconvmat(
    A: NDArray[np.floating],
    *,
    method: Literal["tsvd", "tikhonov"] = "tsvd",
    tol: float = 0.1,
    lambda_reg: float | None = None,
) -> NDArray[np.floating]:
    """Compute regularized pseudo-inverse of a convolution matrix.

    Uses SVD decomposition with regularization to compute a stable
    inverse of the convolution matrix for deconvolution operations.

    Parameters
    ----------
    A : ndarray
        Convolution matrix. Shape: (n, n) or (m, n).
    method : {"tsvd", "tikhonov"}, default "tsvd"
        Regularization method:
        - "tsvd": Truncated SVD. Singular values below tol * max(s) are
          set to zero.
        - "tikhonov": Tikhonov regularization. Singular values are damped
          as s / (s^2 + lambda^2).
    tol : float, default 0.1
        For TSVD: Relative tolerance for singular value truncation.
        Singular values s < tol * max(s) are set to zero.
        For Tikhonov: Used to compute lambda if not provided.
    lambda_reg : float, optional
        Tikhonov regularization parameter. If not provided, computed
        as lambda = tol * max(singular_values).

    Returns
    -------
    ndarray
        Regularized pseudo-inverse. Shape: (n, m) where A is (m, n).

    Notes
    -----
    TSVD (Truncated Singular Value Decomposition):
        A_inv = V @ diag(1/s_truncated) @ U.T
        where s_truncated sets small singular values to 0.

    Tikhonov regularization:
        A_inv = V @ diag(s / (s^2 + lambda^2)) @ U.T
        This damps rather than truncates small singular values.

    The choice of regularization affects noise amplification in
    deconvolution. TSVD is simpler but can introduce ringing artifacts.
    Tikhonov provides smoother results but may over-smooth.

    Examples
    --------
    >>> import numpy as np
    >>> from osipy.common.convolution import convmat, invconvmat
    >>> t = np.linspace(0, 10, 51)
    >>> aif = np.exp(-t / 3)
    >>> A = convmat(aif, t)
    >>> A_inv = invconvmat(A, method="tsvd", tol=0.1)
    >>> # Deconvolution: irf = A_inv @ signal

    References
    ----------
    .. [1] dcmri library: https://github.com/dcmri/dcmri
    .. [2] Hansen PC (1998). Rank-Deficient and Discrete Ill-Posed Problems.
    """
    xp = get_array_module(A)

    A = xp.asarray(A)

    # SVD decomposition
    # Handle both numpy and cupy
    try:
        U, s, Vt = xp.linalg.svd(A, full_matrices=False)
    except AttributeError:
        # Fallback for older numpy versions
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        U = xp.asarray(U)
        s = xp.asarray(s)
        Vt = xp.asarray(Vt)

    s_max = xp.max(s) if len(s) > 0 else 1.0

    if method == "tsvd":
        # Truncated SVD: set small singular values to 0
        threshold = float(tol * s_max)
        s_inv = xp.where(s > threshold, 1.0 / s, 0.0)

    elif method == "tikhonov":
        # Tikhonov regularization
        if lambda_reg is None:
            lambda_reg = float(tol * s_max)
        s_inv = s / (s**2 + lambda_reg**2)

    else:
        raise DataValidationError(
            f"Unknown method: {method}. Use 'tsvd' or 'tikhonov'."
        )

    # Compute pseudo-inverse: V @ diag(s_inv) @ U.T
    # Note: Vt is V transposed from SVD
    A_inv = (Vt.T * s_inv) @ U.T

    return A_inv


def circulant_convmat(
    h: NDArray[np.floating],
    t: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Construct a block-circulant convolution matrix.

    Creates a circulant matrix for delay-insensitive deconvolution
    methods (cSVD/oSVD). The circulant structure allows the AIF to
    wrap around, making the deconvolution insensitive to tracer
    arrival time differences.

    Parameters
    ----------
    h : ndarray
        Impulse response function (e.g., AIF). Shape: (n_times,).
    t : ndarray
        Time points in seconds. Shape: (n_times,).

    Returns
    -------
    ndarray
        Block-circulant convolution matrix. Shape: (n_times, n_times).

    Notes
    -----
    The circulant matrix has the form:
        C[i,j] = h[(i-j) mod n]

    This is equivalent to circular convolution rather than linear
    convolution. The benefit is that the deconvolution becomes
    insensitive to shifts between the AIF and tissue curve.

    References
    ----------
    .. [1] Wu et al. (2003). Tracer arrival timing-insensitive technique.
           Magn Reson Med. 50:164-174.
    """
    xp = get_array_module(h)

    h = xp.asarray(h)
    t = xp.asarray(t)

    n = len(h)
    if n == 0:
        return xp.array([[]], dtype=h.dtype)

    # Compute time step (assume uniform for circulant)
    dt = (t[-1] - t[0]) / (n - 1) if n > 1 else 1.0

    # Build circulant matrix
    C = xp.zeros((n, n), dtype=h.dtype)
    for i in range(n):
        for j in range(n):
            idx = (i - j) % n
            C[i, j] = h[idx] * dt

    return C
