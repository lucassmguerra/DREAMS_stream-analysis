# -*- coding: utf-8 -*-
"""
Velocity dispersion along a stream.

The public entry point normalizes shapes and dispatches to one of two workers, a
numba-jitted one and a NumPy one. Both compute the same quantity, the total 3D
dispersion ``sqrt(var_x + var_y + var_z)`` of the masked particles, optionally
after removing a polynomial trend in phi1 from each velocity component.

The two workers are not bitwise equivalent to each other, and never were. The
numba worker solves the polynomial fit by normal equations on a Vandermonde-Gram
matrix with ``np.linalg.solve``. The NumPy worker uses ``scipy.linalg.lstsq`` on
the Vandermonde matrix directly. Same model, different conditioning, so they
agree to fit tolerance and not to the last bit. Pin ``use_numba`` when you need
reproducibility across machines.

Supports the ``vel.std.tot_km_s`` column of the metrics table in
*No Stream Left Unscathed* (Arora et al.).
"""
from __future__ import annotations

import numpy as np

from ._numba import NUMBA_AVAILABLE, njit, prange

__all__ = ["compute_velocity_dispersion"]


def compute_velocity_dispersion(
    xv: np.ndarray,
    mask: np.ndarray | None = None,
    detrend: bool = False,
    phi1: np.ndarray | None = None,
    poly_degree: int = 5,
    use_numba: bool = True,
) -> float | np.ndarray:
    """
    Compute 3D velocity dispersion σ_t per stream, with optional detrending.

    Parameters
    ----------
    xv : np.ndarray
        Phase-space coordinates. Velocities are in columns 3, 4, 5.
        Accepted shapes:
          - (N, 6)          single stream
          - (S, N, 6)       S streams, each with N points
        Each row is [x, y, z, vx, vy, vz]. Positions in kpc, velocities in km/s.
    mask : np.ndarray or None, optional
        Boolean mask of valid particles. Accepted shapes:
          - (N,)            single-stream mask (will be converted to (1, N))
          - (S, N)          per-stream masks
        If None, all particles are considered valid.
    detrend : bool, optional
        If True, detrend velocities as a function of `phi1`.
    phi1 : np.ndarray or None, optional
        Coordinate used for detrending. Accepted shapes:
          - (N,)            single-stream phi1 (converted to (1, N))
          - (S, N)          per-stream phi1
        If `detrend=True`, `phi1` must be provided.
    poly_degree : int, optional
        Degree of polynomial used for detrending (default: 5).
    use_numba : bool, optional
        whether to turn off numba based code.

    Returns
    -------
    sigma_t : float or np.ndarray
        Velocity dispersion per stream. If a single-stream input was supplied
        (xv shape (N,6)), a scalar or 1D array for that stream is returned
        (i.e. the wrapper returns `sigma_t[0]` to match original single-stream API).
        For multi-stream input the function returns an array of shape (S,).

    Raises
    ------
    ValueError
        If `detrend=True` and `phi1` is None, if `xv` is not 2D or 3D, if its
        last dimension is not 6, or if `mask` or `phi1` cannot be normalized to
        shape (S, N).

    Notes
    -----
    With numba installed and ``use_numba=True``, calling with ``detrend=False``
    passes ``phi1=None`` into a jitted function that indexes ``phi1`` inside a
    branch numba cannot prune at compile time, and compilation fails with a
    ``TypingError``. Pass ``use_numba=False`` for the undetrended case. Reported
    in FINDINGS.md, not fixed, because fixing it would change what the function
    computes for some inputs.

    Bit-for-bit results depend on `use_numba`, because the two backends solve the
    polynomial fit differently. See the module docstring.
    """
    if detrend and phi1 is None:
        raise ValueError("phi1 must be provided when detrend=True.")

    xv = np.asarray(xv)

    # Normalize xv to shape (S, N, 6)
    if xv.ndim == 2:
        # Single stream case: (N, 6) -> (1, N, 6)
        xv = xv[None, ...]
        was_single = True
    elif xv.ndim == 3:
        was_single = False
    else:
        raise ValueError(f"xv must be 2D (N, 6) or 3D (S, N, 6); got shape {xv.shape}")

    S, N, D = xv.shape
    if D != 6:
        raise ValueError(f"Last dimension of xv must be 6 (x,y,z,vx,vy,vz); got {D}")

    # -------------------------
    # Normalize / validate mask
    # -------------------------
    if mask is None:
        mask = np.ones((S, N), dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.ndim == 1:
            if mask.shape[0] != N:
                raise ValueError(f"If mask is 1D it must have length N={N}; got {mask.shape[0]}")
            mask = mask[None, :]  # shape (1, N)
        elif mask.ndim == 2:
            if mask.shape[1] != N:
                raise ValueError(f"mask.shape[1] must equal N={N}; got {mask.shape}")
        else:
            raise ValueError(f"mask must be 1D (N,) or 2D (S,N); got ndim={mask.ndim}")
        # Broadcast single-row masks to all streams if needed
        if mask.shape[0] == 1 and S > 1:
            mask = np.repeat(mask, S, axis=0)
        if mask.shape != (S, N):
            raise ValueError(f"mask must have shape (S, N) after normalization; got {mask.shape}")
        # Ensure boolean dtype
        mask = mask.astype(bool)

    # -------------------------
    # Normalize / validate phi1
    # -------------------------
    if phi1 is not None:
        phi1 = np.asarray(phi1)
        if phi1.ndim == 1:
            if phi1.shape[0] != N:
                raise ValueError(f"If phi1 is 1D it must have length N={N}; got {phi1.shape[0]}")
            phi1 = phi1[None, :]  # make (1, N)
        elif phi1.ndim == 2:
            if phi1.shape[1] != N:
                raise ValueError(f"phi1.shape[1] must equal N={N}; got {phi1.shape}")
        else:
            raise ValueError(f"phi1 must be 1D (N,) or 2D (S,N); got ndim={phi1.ndim}")

        # Broadcast single-row phi1 to all streams if needed
        if phi1.shape[0] == 1 and S > 1:
            phi1 = np.repeat(phi1, S, axis=0)
        if phi1.shape[0] not in (1, S):
            raise ValueError(
                f"phi1 must have shape (S, N) or (1, N) after normalization; got {phi1.shape}"
            )

    # Call the (numba-accelerated) worker function
    if NUMBA_AVAILABLE and use_numba:
        sigma_t = _compute_vel_disp_numba(xv, mask, detrend, phi1, poly_degree)
    else:
        sigma_t = _compute_vel_disp_numpy(xv, mask, detrend, phi1, poly_degree)

    # If original input was single-stream, unwrap the first entry to preserve API
    return sigma_t[0] if was_single else sigma_t


def _compute_vel_disp_numpy(
    xv: np.ndarray,
    mask: np.ndarray,
    detrend: bool,
    phi1: None | np.ndarray,
    poly_degree: int,
) -> np.ndarray:
    """
    Compute per-stream 3D velocity dispersion (NumPy fallback).

    Parameters
    ----------
    xv : ndarray, shape (S, N, 6)
        Phase-space arrays `[x,y,z,vx,vy,vz]`.
    mask : ndarray, shape (S, N)
        Boolean mask of valid points (will be coerced to bool).
    detrend : bool
        If True, subtract a polynomial trend in `phi1` before computing dispersion.
    phi1 : ndarray or None, shape (S, N)
        Coordinate used for detrending; required if `detrend=True`.
    poly_degree : int
        Polynomial degree for detrending.

    Returns
    -------
    sigma_t : ndarray, shape (S,)
        3D velocity dispersion per stream (sqrt(var_x + var_y + var_z)).
        NaN for a stream whose mask selects nothing. Note that the numba worker
        returns 0.0 in that case instead.
    """

    # Prefer SciPy's LAPACK-backed lstsq if available
    try:
        import scipy.linalg as sla

        _lstsq = sla.lstsq
    except Exception:
        _lstsq = np.linalg.lstsq

    # Basic assertions (wrapper should normally ensure shapes, but be defensive)
    assert xv.ndim == 3 and xv.shape[2] == 6, "xv must have shape (S, N, 6)"
    S, N, D = xv.shape
    mask = np.asarray(mask).astype(bool)
    sigma_t = np.full((S,), np.nan, dtype=float)

    mean = np.mean
    sqrt = np.sqrt

    for si in range(S):
        valid = mask[si]
        if not np.any(valid):
            sigma_t[si] = np.nan
            continue

        v = xv[si, valid, 3:6]  # (M, 3)
        M = v.shape[0]

        if detrend:
            if phi1 is None:
                raise ValueError("phi1 must be provided when detrend=True.")
            phi_si = phi1[si, valid]

            if M > poly_degree:
                A = np.vander(phi_si, poly_degree + 1)  # (M, P)
                # _lstsq signature: SciPy returns (coeffs, residues, rank, s); NumPy returns (coeffs, residuals, rank, s)
                coeffs = (
                    _lstsq(A, v, rcond=None)[0] if _lstsq is np.linalg.lstsq else _lstsq(A, v)[0]
                )
                trend = A.dot(coeffs)  # (M, 3)
                vres = v - trend
            else:
                # Not enough points: fall back to subtracting mean
                vres = v - mean(v, axis=0)
        else:
            vres = v - mean(v, axis=0)

        var_components = mean(vres**2, axis=0)  # population variance (ddof=0)
        sigma_t[si] = sqrt(var_components.sum())

    return sigma_t


@njit(parallel=True)
def _compute_vel_disp_numba(
    xv,
    mask,
    detrend,
    phi1,
    poly_degree,
):
    """
    Numba-accelerated internal function if available for computing total velocity dispersion.

    Parameters
    ----------
    xv : np.ndarray,
        Phase-space coordinates. Velocities are in columns 3, 4, 5.
        - Shape (N, 6): N points, each row [x, y, z, vx, vy, vz]
        - Shape (S, N, 6): M objects with N points each
        Positions in kiloparsecs (kpc), velocities in km/s.
    mask : np.ndarray, shape (N) or (S, N)
        Boolean mask for valid particles.
    detrend : bool
        Whether to detrend velocities as a function of phi1.
    phi1 : np.ndarray, shape (N) or (S, N)
        Phi1 coordinates for detrending.
    poly_degree : int
        Polynomial degree to fit for detrending.

    Returns
    -------
    sigma_t : np.ndarray, shape float or (S,)
        Velocity dispersion per stream. 0.0 for a stream whose mask selects
        nothing. Note that the NumPy worker returns NaN in that case instead.

    Notes
    -----
    numba types the whole body regardless of the runtime value of `detrend`, so
    `phi1` must be an array even when `detrend` is False. Passing None fails to
    compile. See the note on the public wrapper.
    """

    S, N, _ = xv.shape
    sigma_t = np.zeros(S)

    for i in prange(S):
        count = 0

        # Allocate max-size buffers
        x = np.empty(N)
        vx = np.empty(N)
        vy = np.empty(N)
        vz = np.empty(N)
        phi = np.empty(N)

        # Gather valid particle xv
        for j in range(N):
            if mask[i, j]:
                x[count] = xv[i, j, 0]
                vx[count] = xv[i, j, 3]
                vy[count] = xv[i, j, 4]
                vz[count] = xv[i, j, 5]
                if detrend:
                    phi[count] = phi1[i, j]
                count += 1

        if count == 0:
            sigma_t[i] = 0.0
            continue

        if detrend:
            # Polynomial fit via least-squares normal equations.
            # Builds the Gram matrix A[p,q] = sum_k x_k^(p+q) and the moment
            # vector B[p] = sum_k y_k x_k^p directly, then solves A c = B. This
            # avoids forming and factorizing the full Vandermonde matrix, which
            # numba has no lstsq for, at the cost of squaring the condition
            # number. It is why this backend and the NumPy one do not agree bit
            # for bit.
            def polyfit_fit(xvals, yvals, deg, n):
                A = np.zeros((deg + 1, deg + 1))
                B = np.zeros(deg + 1)

                for p in range(deg + 1):
                    for q in range(deg + 1):
                        for k in range(n):
                            A[p, q] += xvals[k] ** (p + q)
                    for k in range(n):
                        B[p] += yvals[k] * xvals[k] ** p
                return np.linalg.solve(A, B)

            # Evaluate polynomial via Horner's method
            def eval_poly(xval, coeffs):
                result = 0.0
                for i in range(len(coeffs) - 1, -1, -1):
                    result = result * xval + coeffs[i]
                return result

            # Fit and subtract trend for each velocity component
            cx = polyfit_fit(phi, vx, poly_degree, count)
            cy = polyfit_fit(phi, vy, poly_degree, count)
            cz = polyfit_fit(phi, vz, poly_degree, count)

            for j in range(count):
                vx[j] -= eval_poly(phi[j], cx)
                vy[j] -= eval_poly(phi[j], cy)
                vz[j] -= eval_poly(phi[j], cz)
        else:
            # Subtract mean velocity
            vx_mean = vy_mean = vz_mean = 0.0
            for j in range(count):
                vx_mean += vx[j]
                vy_mean += vy[j]
                vz_mean += vz[j]
            vx_mean /= count
            vy_mean /= count
            vz_mean /= count
            for j in range(count):
                vx[j] -= vx_mean
                vy[j] -= vy_mean
                vz[j] -= vz_mean

        # Compute variance and combine into σ_t
        varx = vary = varz = 0.0
        for j in range(count):
            varx += vx[j] * vx[j]
            vary += vy[j] * vy[j]
            varz += vz[j] * vz[j]

        sigma_x = np.sqrt(varx / count)
        sigma_y = np.sqrt(vary / count)
        sigma_z = np.sqrt(varz / count)
        sigma_t[i] = np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)

    return sigma_t
