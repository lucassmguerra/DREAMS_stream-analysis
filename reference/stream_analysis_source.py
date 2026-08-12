# -*- coding: utf-8 -*-
"""
Stream-analysis
===========

Utility functions for analyzing Nbody GC streams.

Docstring style
---------------
All docstrings follow the NumPy/SciPy convention
(see https://numpydoc.readthedocs.io/).

Notes
-----
- This file is intended as a library of helper functions, **not** a standalone script.
- Type hints follow PEP 484/PEP 604, with postponed evaluation enabled via
  ``from __future__ import annotations``.

Author: Arpit Arora
Created: Sept 28, 2025
"""
# Standard library
from __future__ import annotations
from typing import Any, Callable 
__docformat__ = "numpy"  # optional, descriptive only

# Third-party
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# scipy methods used in the analysis
from scipy.stats import gaussian_kde, kstest, wasserstein_distance, combine_pvalues
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from scipy.signal import welch, detrend

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range
    def njit(nopython=True, cache=True, parallel=True): # Dummy decorator
        def decorator(func):
            return func
        return decorator
    print("Numba not found. Running with pure NumPy/SciPy (might be slower for some operations).")

def compute_orbit_properties(
    orbits: np.ndarray,
    central_mass: float | None = None,
    mu: float | None = None,
    G: float = 4.3009e-6
) -> pd.DataFrame:
    """
    Compute orbital diagnostics for an ensemble of orbits.

    The function expects `orbits` as a 1D array-like of length `n_orbits` where each
    element contains a two-item sequence: `(times, coords)`. `times` is a 1D array
    of shape (n_steps,) (assumed uniformly sampled and identical length for all
    orbits). `coords` is a 2D array of shape (n_steps, 6) containing
    (x, y, z, vx, vy, vz) for each time step.

    The routine returns a DataFrame with per-orbit scalar diagnostics:
    - minima and maxima of radius from detected pericenter/apocenter extrema,
    - time-averaged radius,
    - number of detected pericenter passages,
    - an FFT-based dominant radial period estimate,
    - a simple geometric eccentricity from r_a/r_p,
    - an optional osculating eccentricity computed from instantaneous state
      vectors given a gravitational parameter `mu` (or `central_mass`).

    Parameters
    ----------
    orbits : array-like, shape (n_orbits,)
        Each element must be a length-2 sequence [times, coords]:
        - times : array-like, shape (n_steps,)
            Uniformly sampled time array (same length and sampling for all orbits).
        - coords : array-like, shape (n_steps, 6)
            Phase-space coordinates per timestep in the order (x, y, z, vx, vy, vz).
    central_mass : float or None, optional
        Central mass in solar masses (Msun). Used to form `mu = G * central_mass`
        when `mu` is not provided. Units assumed to be compatible with the default
        gravitational constant: r in kpc, v in km/s, mass in Msun.
    mu : float or None, optional
        Gravitational parameter (mu = G * M) supplied directly. If provided it
        takes precedence over `central_mass`. Units must be consistent with the
        units of `coords`.
    G : float, optional
        Gravitational constant used when `central_mass` is provided. Default value
        corresponds to G = 4.30091727003628e-6 kpc (km/s)^2 / Msun.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed 0..(n_orbits-1) with the following columns (all floats
        unless noted otherwise):

        - min_pericenter_dist : float
            Smallest detected local minimum of radius R(t) (pericenter) across the
            sampled time series. NaN if no pericenter detected.

        - max_apocenter_dist : float
            Largest detected local maximum of radius R(t) (apocenter) across the
            sampled time series. NaN if no apocenter detected.

        - mean_dist : float
            Time-average radius <R(t)>.

        - n_pericenters : int
            Count of detected pericenter passages based on simple three-point
            local-minimum detection.

        - period : float
            Period estimate (same time units as `times`) obtained from the dominant
            frequency in the FFT of the detrended radius time series. NaN for
            undefined / DC-only signals.

        - eccentricity_geo : float
            Geometric eccentricity defined as (r_a - r_p) / (r_a + r_p) using the
            detected extrema. NaN if either extrema are missing.

        - eccentricity_osculating : float
            Robust central estimate (median over time) of the instantaneous
            osculating eccentricity computed from the eccentricity vector
            e = (1/mu) (v x h) - r_hat, where h = r x v and r_hat is the radial
            unit vector. Returned as NaN if neither `mu` nor `central_mass` is
            provided or if numerical issues occur.

    Notes
    -----
    - The extrema detection uses a local three-point comparison (R(t) vs R(t±Δt));
      this is simple and efficient but sensitive to sampling. Pre-filtering or a
      stricter peak-finding routine may be required for noisy / undersampled data.
    - The FFT-based period requires a uniform sampling interval `dt`; the routine
      uses the first orbit's `times` to infer `dt`.
    - The osculating eccentricity assumes a point-mass central potential. For
      extended or time-dependent potentials the osculating e(t) has only local meaning.
    - Units must be consistent when using `central_mass` or `mu`. The default G
      assumes r in kpc, v in km/s, mass in Msun. If your units differ, pass `mu`
      in compatible units instead of `central_mass`.

    Examples
    --------
    >>> # orbits is an array-like where each element is [times, coords]
    >>> df = compute_orbit_properties(orbits, central_mass=1e12)
    >>> df[['min_pericenter_dist', 'max_apocenter_dist', 'eccentricity_geo']]

    """
    # Stack inputs into arrays
    times  = np.vstack(orbits[:, 0])[:, ::-1]      # shape (n_orbits, n_steps)  now increasing
    coords = np.stack(orbits[:, 1])[:, ::-1, :]    # shape (n_orbits, n_steps, 6)
    r = np.linalg.norm(coords[..., :3], axis=2)    # (n_orbits, n_steps)
    dt = times[0, 1] - times[0, 0]

    # detect extrema
    r_prev = r[:, :-2]
    r_mid  = r[:, 1:-1]
    r_next = r[:, 2:]
    peri_mask = (r_mid < r_prev) & (r_mid < r_next)
    apo_mask  = (r_mid > r_prev) & (r_mid > r_next)

    n_peri = peri_mask.sum(axis=1)
    min_peri = np.where(peri_mask, r_mid, np.inf).min(axis=1)
    min_peri[np.isinf(min_peri)] = np.nan
    max_apo = np.where(apo_mask, r_mid, -np.inf).max(axis=1)
    max_apo[np.isneginf(max_apo)] = np.nan
    mean_r = r.mean(axis=1)

    # FFT period estimation
    r_detrended = detrend(r, axis=1, type='linear')
    Rf = np.fft.rfft(r_detrended, axis=1)
    amp = np.abs(Rf)
    amp[:, 0] = 0
    peak_idx = amp.argmax(axis=1)
    freqs = np.fft.rfftfreq(r.shape[1], d=dt)
    # protect against zero-frequency
    with np.errstate(divide='ignore', invalid='ignore'):
        period = 1.0 / freqs[peak_idx]
        period[~np.isfinite(period)] = np.nan

    # geometric eccentricity from pericenter/apocenter
    ecc_geo = np.full_like(mean_r, np.nan, dtype=float)
    valid = np.isfinite(min_peri) & np.isfinite(max_apo) & ((max_apo + min_peri) > 0)
    ecc_geo[valid] = (max_apo[valid] - min_peri[valid]) / (max_apo[valid] + min_peri[valid])

    # optional osculating eccentricity (median over times)
    ecc_osc = np.full_like(mean_r, np.nan, dtype=float)
    mu_val = None
    if mu is not None:
        mu_val = mu
    elif central_mass is not None:
        mu_val = G * central_mass

    if mu_val is not None:
        # compute eccentricity vector: e = (1/mu) * (v x h) - r_hat, h = r x v
        r_vec = coords[..., :3]    # (n_orbits, n_steps, 3)
        v_vec = coords[..., 3:]    # (n_orbits, n_steps, 3)
        # h = r x v
        h = np.cross(r_vec, v_vec, axis=2)
        vxh = np.cross(v_vec, h, axis=2)
        r_norm = np.linalg.norm(r_vec, axis=2)
        # protect division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            r_hat = np.nan_to_num(r_vec / r_norm[..., None])
        e_vec = (vxh / mu_val[..., None]) - r_hat if np.ndim(mu_val) else (vxh / mu_val) - r_hat
        e_mag = np.linalg.norm(e_vec, axis=2)  # (n_orbits, n_steps)
        # take a robust central value (median)
        ecc_osc = np.median(e_mag, axis=1)
        ecc_osc[~np.isfinite(ecc_osc)] = np.nan

    df = pd.DataFrame({
        'min_pericenter_dist': min_peri,
        'max_apocenter_dist':  max_apo,
        'mean_dist':           mean_r,
        'n_pericenters':       n_peri,
        'period':              period,
        'eccentricity_geo':    ecc_geo,
        'eccentricity_osculating': ecc_osc
    })

    return df


def measure_stream_LengthWidth(
    phi1: np.ndarray,
    phi2: np.ndarray,
    method: str = "kde",         # "quantile" or "kde"
    density_frac: float = 0.9,
    nbins: int = 100,                 # number of bins per stream (KDE only)
    smoothing_sigma: float = 1.0,     # Gaussian σ in bin‐units (KDE only)
    return_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray[bool]]:
    """
    Measure stream length (along phi1) and width (along phi2), for one or many streams.

    You can choose:
    - "quantile": length and width are simple [(1−f)/2, (1+f)/2] quantiles.
    - "kde": length is the span of the phi1 region enclosing `density_frac`
             using a per-stream smoothed histogram (approximate KDE),
             and width is the quantile width of phi2 within that region.

    Parameters
    ----------
    phi1 : np.ndarray, shape (N,) or (S, N)
        Along-stream coordinates (degrees or radians).
    phi2 : np.ndarray, shape (N,) or (S, N)
        Across-stream coordinates.
    method : {"quantile","kde"}
        Which estimator to use.
    density_frac : float
        Fraction of total particles to enclose (e.g. 0.95).
    nbins : int
        Number of bins for the per-stream histogram (KDE mode).
    smoothing_sigma : float
        σ of the Gaussian filter, in bin units (KDE mode).
    return_mask : bool
        Returns kde based particle masking in (S, N) shape.
    
    Returns
    -------
    lengths : np.ndarray of shape (S,) or scalar
        Enclosed span in phi1.
    widths  : np.ndarray of shape (S,) or scalar
        Span in phi2, measured as quantile width within the KDE-defined phi1 region.
    """
    single = (phi1.ndim == 1)
    phi1_arr = np.atleast_2d(phi1)
    phi2_arr = np.atleast_2d(phi2)
    S, N = phi1_arr.shape
    assert phi2_arr.shape == (S, N), "phi1/phi2 must match shapes"

    masks = None
    
    if method == "quantile":
        # -------------------------
        # 1) Compute quantile-based length & width
        # -------------------------
        lo = (1 - density_frac) / 2
        hi = 1 - lo
        q1 = np.quantile(phi1_arr, [lo, hi], axis=1)  # phi1 quantiles
        q2 = np.quantile(phi2_arr, [lo, hi], axis=1)  # phi2 quantiles
        lengths = q1[1] - q1[0]
        widths  = q2[1] - q2[0]
        
        if return_mask:
            masks = (phi1_arr >= q1[0][:, None]) & (phi1_arr <= q1[1][:, None])
        
    elif method == "kde":
        # -------------------------
        # 1) Determine per-stream min, max, and bin width for phi1
        # -------------------------
        min1 = phi1_arr.min(axis=1)
        max1 = phi1_arr.max(axis=1)
        w1   = (max1 - min1) / nbins  # per-stream bin width

        # -------------------------
        # 2) Digitize phi1 values into bin indices [0, nbins-1]
        # -------------------------
        rel = (phi1_arr - min1[:, None]) / w1[:, None]
        inds = np.floor(rel).astype(int)
        inds = np.clip(inds, 0, nbins - 1)

        # -------------------------
        # 3) Build per-stream histograms via a single bincount
        # -------------------------
        flat = inds + np.arange(S)[:, None] * nbins
        hist = np.bincount(flat.ravel(), minlength=S * nbins).reshape(S, nbins)

        # -------------------------
        # 4) Smooth histogram = approximate KDE
        # -------------------------
        smooth = gaussian_filter1d(hist.astype(float), smoothing_sigma, axis=1)

        # -------------------------
        # 5) Normalize to get a probability density over phi1
        # -------------------------
        area = smooth.sum(axis=1) * w1
        pdf  = smooth / area[:, None]

        # -------------------------
        # 6) Find threshold that encloses density_frac
        # -------------------------
        #   sort bins by density descending
        order = np.argsort(pdf, axis=1)[:, ::-1]
        #   cumulative density in sorted order
        cum   = np.cumsum(np.take_along_axis(pdf, order, axis=1) * w1[:, None], axis=1)
        #   index where cumulative ≥ density_frac
        idx   = np.argmax(cum >= density_frac, axis=1)
        #   threshold density per stream
        thr   = pdf[np.arange(S), order[np.arange(S), idx]]

        # -------------------------
        # 7) Mask bins above threshold to define the coherent phi1 region
        # -------------------------
        mask  = pdf >= thr[:, None]
        #   first and last True per stream
        first = mask.argmax(axis=1)
        last  = nbins - 1 - mask[:, ::-1].argmax(axis=1)

        # -------------------------
        # 8) Compute phi1 length: (last − first) × per-stream bin width
        # -------------------------
        lengths = (last - first) * w1

        # -------------------------
        # 9) Compute phi2 width by quantiles within the phi1 region
        # -------------------------
        lo = (1 - density_frac) / 2
        hi = 1 - lo
        widths = np.empty(S, dtype=float)

        if return_mask:
            masks = np.zeros((S, N), dtype=bool)
        
        for i in range(S):
            # select particles within the KDE-defined phi1 span
            mask_particles = (
                (phi1_arr[i] >= min1[i] + first[i] * w1[i]) &
                (phi1_arr[i] <= min1[i] + last[i]  * w1[i])
            )
            
            if return_mask:
                masks[i] = mask_particles
            
            if mask_particles.sum() < 2:
                widths[i] = 0.0
            else:
                q2 = np.quantile(phi2_arr[i, mask_particles], [lo, hi])
                widths[i] = float(q2[1] - q2[0])
    else:
        raise ValueError("method must be 'quantile' or 'kde'")

    # -------------------------
    # 10) Return scalars if input was 1D
    # -------------------------
    if single:
        if return_mask:
            return float(lengths[0]), float(widths[0]), masks[0]
        return float(lengths[0]), float(widths[0])
    
    if return_mask:
        return lengths, widths, masks
    return lengths, widths

def compute_vel_disp_stream(
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
            raise ValueError(f"phi1 must have shape (S, N) or (1, N) after normalization; got {phi1.shape}")

    # Call the (numba-accelerated) worker function
    if NUMBA_AVAILABLE and use_numba:
        sigma_t = _compute_vel_disp_numba(xv, mask, detrend, phi1, poly_degree)
    else:
        sigma_t = _compute_vel_disp_numpy(xv, mask, detrend, phi1, poly_degree)
    
    # If original input was single-stream, unwrap the first entry to preserve API
    return sigma_t[0] if was_single else sigma_t

def _compute_vel_disp_numpy(
    xv: np.ndarray,
    mask: np.ndarray[bool],
    detrend: bool,
    phi1: None | np.ndarray,
    poly_degree: int
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
                coeffs = (_lstsq(A, v, rcond=None)[0]
                          if _lstsq is np.linalg.lstsq
                          else _lstsq(A, v)[0])
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
    poly_degree
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
        Velocity dispersion per stream.
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
            # Polynomial fit via least-squares normal equations
            def polyfit_fit(xvals, yvals, deg, n):
                A = np.zeros((deg + 1, deg + 1))
                B = np.zeros(deg + 1)

                for p in range(deg + 1):
                    for q in range(deg + 1):
                        for k in range(n):
                            A[p, q] += xvals[k]**(p + q)
                    for k in range(n):
                        B[p] += yvals[k] * xvals[k]**p
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

def stream_local_binned_stats(
    phi1: np.ndarray, 
    phi2: np.ndarray, # could be any property . . . 
    scheme: str = 'bin_size',
    bin_edges: np.ndarray | None = None, 
    bin_size: float = 1/8., 
    min_particles: int = 300,
    normal_ref_size: int = 20000, 
    normal_ref_seed: int = 12345,
    # # Random control args for plots
    plot_streambins: bool = False, # whether to make the binned-stream plots. 
    axis_labels: bool = True,
    x_axis_label: str = r"\bf{\boldmath$\phi_1$ [$\ell \ ^\circ$]}",
    y_axis_label: str = r"\bf{\boldmath$\phi_2$ [$^\circ$]}",
    max_bins_to_plot: int = 24, 
    figsize_per_bin: tuple[int, int] = (2, 2),
    scatter_kwargs = None, 
    kde_bandwidth: float | None = None,
    phi1_kde_bandwidth: float | None = None,
    phi1_curve_height_frac: float = 0.50,    # max curve height as fraction of vertical span
    phi2_violin_width_frac: float = 0.35,    # fraction of bin width used for phi2 mirrored curve
    show_counts: bool = True, 
    title: str | None = None,
    phi2_ylim: tuple[float, float] | None = None,
    phi1_linewidth: float = 1.0, 
    phi2_linewidth: float = 2.0,
    phi1_color: str = 'g',
    phi2_color: str = 'r',
    axs = None,
) -> pd.DataFrame | tuple[pd.DataFrame, plt.Figure, plt.Axes,]:
    
    """
    Compute per-bin statistics and optionally plot them.
    Stats are computed using a custom function with pandas groupby.
    For Arora+2026 No stream left. Phi1 was rescaled with Length of a stream.

    Parameters
    ----------
    phi1 : np.ndarray, shape (N,) or (S, N)
            Along-stream coordinates (degrees or radians). Ideally rescaled by length. 
    phi2 : np.ndarray, shape (N,) or (S, N)
            Across-stream coordinates. Could be anything! for e.g., v_los 
    scheme: str, {'bin_size', 'equal_count'}, default='bin_size'}
            -'bin_size': Fixed bin length in phi1.
            -'equal_count': equal number of particles in each bin.
    bin_size: float, default=1/8.
                size of individual bin in length scale. Used in 'bin_size' scheme.
    min_particles: int, default=300,
                    minimum number of particles to generate a bin. Used in 'equal_counts' scheme.
    normal_ref_size: int, default=20000, 
                    Number of samples used for reference standard normal distribution.
    normal_ref_seed: int, default=12345,
                    numpy random seed to sample the reference distribution.
    plot_streambins: bool, default=False, 
                    whether to make the binned-stream plots. 
                    The rest of the flags are only active if True.
    see type hints for the plotting+etc flags....
    
    Returns
    -------
    pd.DataFrame of statistics in each bin (rows).
    
    Or, with plot_streambins=True,
     fig, ax, pd.DataFrame
     
    Plot row of bins (shared y-axis) with:
      - global phi1 KDE curve overplotted in each bin
      - mirrored KDE curves of phi2 in each bin
      - optional scatter of individual particles
    Or, with plot_streambins=False, just return the DataFrame of statistics.
    """
    
    phi1 = np.asarray(phi1)
    phi2 = np.asarray(phi2)
    assert phi1.shape == phi2.shape, "phi1, phi2 must have same length"

    # Work on copies so inputs not mutated
    phi1_local = phi1.copy()
    phi2_local = phi2.copy()

    # Detrend phi2
    if phi1_local.size >= 2:
        try:
            polynom_order = min(3, phi1_local.size-1)
            trend_coeffs = np.polyfit(phi1_local, phi2_local, int(polynom_order))
            phi2_local = phi2_local - np.polyval(trend_coeffs, phi1_local)
        except Exception:
            pass

    # Normal reference distribution
    rng = np.random.default_rng(normal_ref_seed)
    normal_ref = rng.normal(0.0, 1.0, size=normal_ref_size)
    
    # Global mean & std of the distribution:
    mean_phi2 = np.nanmean(phi2_local)
    std_phi2 = np.nanstd(phi2_local)
    global_ref = (phi2_local - mean_phi2) / (std_phi2 if std_phi2 > 0 else 1)
    
    # Bin edges
    if bin_edges is None:
        if scheme == 'bin_size':
            left, right = np.nanmin(phi1_local), np.nanmax(phi1_local)
            if right <= left:
                right = left + bin_size
            bin_edges = np.arange(left, right + bin_size, bin_size)
        elif scheme == 'equal_count':
            N = len(phi1_local)
            n_bins = max(1, N // min_particles)
            qs = np.linspace(0.0, 1.0, n_bins+1)
            bin_edges = np.quantile(phi1_local, qs)
        else:
            raise ValueError("scheme must be 'bin_size' or 'equal_count'")
    else:
        bin_edges = np.asarray(bin_edges)

    n_bins = len(bin_edges)-1
    
    if n_bins <= 0:
        raise ValueError("No bins determined (check bin_edges / bin_size).")
        
    # ---- Custom statistic function ----
    def compute_bin_stats(values: np.ndarray) -> list[int, float, float, float, float, float, float]:
        if len(values) == 0:
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            
        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if n > 1 else 0.0

        # Standardize
        z_vals = (values - mean_val) / (std_val if std_val > 0 else 1.0)

        # KS test
        try:
            ks_res = kstest(z_vals, 'norm', args=(0, 1))
            ks_stat, ks_p = ks_res.statistic, ks_res.pvalue
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        # Wasserstein normal
        try:
            wass_norm = wasserstein_distance(z_vals, normal_ref)
        except Exception:
            wass_norm = np.nan
        
        # Wasserstein versus global
        try:
            wass_glob = wasserstein_distance(z_vals, global_ref)
        except Exception:
            wass_glob = np.nan
            
        return [n, mean_val, std_val, ks_stat, ks_p, wass_norm, wass_glob]

    # ---- Compute statistics with pandas ----
    # Create a temporary DataFrame to enable grouping operations
    temp_df = pd.DataFrame({'phi1': phi1_local, 'phi2': phi2_local})
    # Assign each row to a bin based on the pre-calculated bin_edges
    temp_df['bin'] = pd.cut(temp_df['phi1'], bins=bin_edges, include_lowest=True)
    # Group by the bins and apply the custom function to the 'phi2' values in each group
    stats_per_bin = temp_df.groupby('bin', observed=False)['phi2'].apply(compute_bin_stats)
    
    column_names = [
        'count', 'mean_phi2', 'std_phi2',
        'ks_stat_vs_norm', 'ks_p_vs_norm', 
        'wass_vs_norm', 'wass_vs_global'
    ]
    
    df = pd.DataFrame(stats_per_bin.tolist(), index=stats_per_bin.index, columns=column_names)
    # Add the bin boundaries to the DataFrame for plotting and analysis
    df['bin_left'] = df.index.map(lambda x: x.left)
    df['bin_right'] = df.index.map(lambda x: x.right)
    df = df.reset_index(drop=True) # Drop the IntervalIndex for a clean default index
    
    if not plot_streambins: return df # return the results without plotting.

    # ---- Plotting code goes here now----
    # ---- Global φ1 KDE ----
    try:
        kde1_global = gaussian_kde(phi1)
        if phi1_kde_bandwidth is not None:
            kde1_global.set_bandwidth(kde1_global.factor * phi1_kde_bandwidth)
    except Exception:
        kde1_global = None

    if kde1_global is not None:
        phi1_grid = np.linspace(np.nanmin(phi1_local), np.nanmax(phi1_local), 500)
        phi1_kde_vals = kde1_global(phi1_grid)
        max_phi1_kde = phi1_kde_vals.max()
    else:
        max_phi1_kde = 1.0

    # ---- Y limits ----
    if phi2_ylim is None:
        y_min, y_max = np.nanmin(phi2_local), np.nanmax(phi2_local)
        y_pad = 0.03 * (y_max - y_min if (y_max > y_min) else 1.0)
        y_min_all, y_max_all = y_min - y_pad, y_max + y_pad
    else:
        y_min_all, y_max_all = phi2_ylim
        
    y_span = (y_max_all - y_min_all) or 1.0

    if axs is None:
        fig_w = max(6, figsize_per_bin[0]*n_bins)
        fig_h = figsize_per_bin[1]
        
        fig, axs = plt.subplots(1, n_bins, figsize=(fig_w, fig_h),
                                squeeze=False, sharey=True, dpi=300)
        plt.subplots_adjust(wspace=0.02)
        axs = axs[0]

    phi2_grid_res = 200
    
    for i, (bin_interval, points_in_bin) in enumerate(temp_df.groupby('bin', observed=False)):
        ax = axs[i]
        # Get bin edges directly from the pandas Interval object
        left, right = bin_interval.left, bin_interval.right
        ax.set_xlim(left, right)
        ax.set_ylim(y_min_all, y_max_all)

        n = len(points_in_bin) 
        if n > 0:
            if scatter_kwargs is None:
                scatter_kwargs = dict(s=5, alpha=0.18, edgecolors='none',
                                      linewidths=0, facecolor=plt.rcParams['text.color'])
            ax.scatter(points_in_bin['phi1'], points_in_bin['phi2'], **scatter_kwargs)

        if n >= 3:
            try:
                kde2 = gaussian_kde(points_in_bin['phi2'])
                if kde_bandwidth is not None:
                    kde2.set_bandwidth(kde2.factor * kde_bandwidth)

                y_grid = np.linspace(y_min_all, y_max_all, phi2_grid_res)
                dens_y = kde2(y_grid)
                dens_norm = dens_y / dens_y.max() if dens_y.max() > 0 else dens_y
                max_horiz_offset = (right-left) * phi2_violin_width_frac
                x_right = 0.5*(left+right) + dens_norm * max_horiz_offset
                x_left  = 0.5*(left+right) - dens_norm * max_horiz_offset
                ax.plot(x_right, y_grid, linewidth=phi2_linewidth, color=phi2_color, alpha=0.95)
                ax.plot(x_left,  y_grid, linewidth=phi2_linewidth, color=phi2_color, alpha=0.95)
            except Exception:
                pass

        if kde1_global is not None:
            x_grid = np.linspace(left, right, 200)
            dens_x = kde1_global(x_grid)
            y_curve = y_min_all + (dens_x / max_phi1_kde) * (phi1_curve_height_frac * y_span)
            if phi1_curve_height_frac > 0:
                ax.plot(x_grid, y_curve, linewidth=phi1_linewidth, color=phi1_color, alpha=0.95)

        if show_counts:
            ax.text(0.02, 0.94, f"$\\mathbf{{N_{{star}}}}$ = {n}",
                    transform=ax.transAxes, va='top', ha='left', fontsize=9)
    if title:
        fig.suptitle(title)
    if axis_labels:
        axs[0].set_ylabel(y_axis_label, fontsize=16)
        fig.supxlabel(x_axis_label, fontsize=16, y=-0.15)
    return df #@, fig, axs

def compute_bin_diagnostics(
    df: pd.DataFrame, 
    ddof: int = 1,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Compute simplistic metrics from the statistical tests performed in localized bins.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with expected columns: ['wass_vs_norm', 'ks_p_vs_norm', 'count', 'std_phi2'].
        Missing columns will be filled with NaN values.
    ddof : int, default 1
        Degrees of freedom to use for standard deviation of standard deviations.
        Use 1 (default) to treat bin-level std values as a sample from a population.
        
    Returns
    -------
    tuple of float
        A 6-tuple containing:
        - median_norm_wass : float
            Median normalized Wasserstein distance w.r.t 95% threshold
        - max_norm_wass : float  
            Maximum normalized Wasserstein distance w.r.t 95% threshold
        - frac_flag_wass : float
            Fraction of bins flagged as Gaussian by Wasserstein test (< 1.0 threshold)
        - frac_consecutive_wass : float
            Fraction of bins flagged as Gaussian by Wasserstein test in continuation (< 1.0 threshold)
            i.e., largest section of smooth Gaussian stream. 
        - frac_p_gt_005 : float
            Fraction of bins flagged as Gaussian by KS test (p > 0.05)
        - mean_std_bins : float
            Mean of standard deviations across all bins (average physical width)
        - std_std_bins : float
            Standard deviation of standard deviations in bins (global variation in Phi2 std)
    
    Notes
    -----
    - Normalized W.D are meaningful: values < 1 indicate bins are Gaussian at 95% confidence.
      
      * Median captures the averaged stream-wide property
      * Max captures the largest perturbation
      
    - Coefficient of variation (std/mean) is a useful relative measure.
      We leave it to the user to compute relevant quantities from mean and std of bin stds.
      
    - ddof parameter: bin-level std values are treated as a sample from a population 
      of possible bin std values, hence ddof=1 for unbiased estimation.
    """
    
    # Ensure relevant columns exist
    for col in ['wass_vs_norm', 'ks_p_vs_norm', 'count', 'std_phi2']:
        if col not in df.columns:
            df[col] = np.nan
            
    counts = df['count'].fillna(0).astype(int).values
    # W.D threshold with 95% CI fit for a reference distribution
    df['wass_thresh_95'] = wasserstein_null_threshold(power_law_fit=True)(counts)
    
    # Normalized wasserstein (observed / threshold)
    df['norm_wass_to_95'] = df['wass_vs_norm'] / df['wass_thresh_95']
    # Flag per-bin exceedance
    df['flag_wass_under_95'] = df['norm_wass_to_95'] < 1.00
    
    # Overall fraction flagged by Wasserstein
    if df['flag_wass_under_95'].notna().sum() > 0:
        frac_flag_wass = float(np.sum(df['flag_wass_under_95'].fillna(False)) / df['flag_wass_under_95'].notna().sum())
        frac_consecutive_wass = _get_max_consecutive_true_fraction(df['flag_wass_under_95'])
    else:
        frac_flag_wass = np.nan
        frac_consecutive_wass = np.nan
        
    # Null hypothesis can't be rejected at 5% significance level bins
    ks_pvals = df['ks_p_vs_norm'].values
    # Fraction of p-values > 0.05 (bins consistent with normal)
    valid_p = ~np.isnan(ks_pvals)
    if valid_p.sum() > 0:
        frac_p_gt_005 = float(np.sum(ks_pvals[valid_p] > 0.05) / valid_p.sum())
    else:
        frac_p_gt_005 = np.nan
        
    # Metrics to return for printing/flushing/saving
    median_norm_wass = float(df['norm_wass_to_95'].median())  # Use .median() - insensitive to outliers
    max_norm_wass = float(df['norm_wass_to_95'].max())
    std_std_bins = float(df['std_phi2'].std(ddof=ddof))  # Unbiased sample standard deviation estimator with ddof=1 by default
    mean_std_bins = float(df['std_phi2'].mean())  # Use .mean() for average physical width

    # Code for coef of variation of W.D metric and a spatial coherence metric.
    # # spatial coherence can be described as number of consecutive gaussian bins or fraction of stream length . . . 
    # # coef of variation σ/μ consistency of non-Gaussianity. uniform non gaussian versus (lower) very non gaussian (higher)
    # Max WD: "How bad is the worst perturbation?"
    # Median WD: "What's the typical level of non-Gaussianity?"
    # CV of WD: "How variable are the perturbation strengths?" (not spatial coherence)
    # Consecutive Gaussian bins: "How spatially coherent/fragmented is the stream?"
    
    return median_norm_wass, max_norm_wass, frac_flag_wass, frac_consecutive_wass, frac_p_gt_005, mean_std_bins, std_std_bins 

def plot_metric_stats_detailed( 
    df: pd.DataFrame, plot=True,
    mc_trials=2000, normal_ref_size=20000,
    alpha=0.05, seed=0, dpi=200
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Plot per-bin Wasserstein and KS summary (like your plot_metric_stats) AND
    compute + return:
      - median_wasserstein (median of df['wass_vs_norm'])
      - max_wasserstein  (max of df['wass_vs_norm'])
      - frac_p_gt_005    (fraction of bins with ks_p_vs_norm > 0.05)
      - median_ks_stat   (median of df['ks_stat_vs_norm'])  (lower ~ more Gaussian)
      - fisher_stat, fisher_p (combined KS p-value via Fisher)
      - bonferroni_flag, bonferroni_minp (result of bonferroni_correction on KS p-values)
      - per-bin wasserstein 95% thresholds (dict mapping count->threshold) and per-bin thresholds vector

    Parameters
    ----------
    df : pandas.DataFrame
      Expected columns: ['wass_vs_norm', 'ks_p_vs_norm', 'ks_stat_vs_norm', 'count']
    plot : bool
      If True produce the two-row summary figure (Wasserstein bars + KS p-values).
    mc_trials, normal_ref_size, alpha, seed : parameters forwarded to wasserstein_null_threshold
      These control the Monte Carlo for null Wasserstein thresholds (per distinct count).
    dpi : figure DPI.

    Returns
    -------
    fig_summary (or None), metrics (dict), df_out (copy of df with added columns)
    """
    # defensive copy
    df_out = df.copy()

    # ensure relevant columns exist
    for col in ['wass_vs_norm', 'ks_p_vs_norm', 'ks_stat_vs_norm', 'count', 'wass_vs_global']:
        if col not in df_out.columns:
            df_out[col] = np.nan

    # Basic summary numbers
    wass_vals = df_out['wass_vs_norm'].dropna().values
    median_wd = float(np.nanmedian(wass_vals)) if wass_vals.size > 0 else np.nan
    max_wd = float(np.nanmax(wass_vals)) if wass_vals.size > 0 else np.nan

    try:
        wass_vals_global = df_out['wass_vs_global'].dropna().values
        median_wd_global = float(np.nanmedian(wass_vals_global)) if wass_vals_global.size > 0 else np.nan
        max_wd_global = float(np.nanmax(wass_vals_global)) if wass_vals_global.size > 0 else np.nan
    except Exception:
        wass_vals_global, median_wd_global, max_wd_global = np.nan, np.nan, np.nan
    
    ks_pvals = df_out['ks_p_vs_norm'].values
    # fraction of p-values > 0.05 (bins consistent with normal)
    valid_p = ~np.isnan(ks_pvals)
    if valid_p.sum() > 0:
        frac_p_gt_005 = float(np.sum(ks_pvals[valid_p] > 0.05) / valid_p.sum())
    else:
        frac_p_gt_005 = np.nan

    # median KS statistic (lower ~ closer to the null)
    ks_stats = df_out['ks_stat_vs_norm'].dropna().values
    median_ks_stat = float(np.nanmedian(ks_stats)) if ks_stats.size > 0 else np.nan

    # Fisher combined p-value for KS p-values
    fisher_stat, fisher_p = combine_ks_pvalues_fisher(list(df_out['ks_p_vs_norm'].values))

    # Bonferroni test on KS p-values
    bonf_flag, bonf_minp = bonferroni_correction(list(df_out['ks_p_vs_norm'].values), alpha=alpha)

    # ----- per-count Wasserstein thresholds via Monte-Carlo -----
    counts = df_out['count'].fillna(0).astype(int).values
    # W.D threshold with 95% CI fit for a reference distribution
    df_out['wass_thresh_95'] = wasserstein_null_threshold(power_law_fit=True)(counts)
    
    # Normalized wasserstein (observed / threshold)
    df_out['norm_wass_to_95'] = df_out['wass_vs_norm'] / df_out['wass_thresh_95']
    
    # Flag per-bin exceedance
    df_out['flag_wass_under_95'] = df_out['norm_wass_to_95'] < 1.0
    
    # Overall fraction flagged by Wasserstein
    if df_out['flag_wass_under_95'].notna().sum() > 0:
        frac_flag_wass = float(np.sum(df_out['flag_wass_under_95'].fillna(False)) / df_out['flag_wass_under_95'].notna().sum())
    else:
        frac_flag_wass = np.nan

    # Compose metrics dict
    metrics = dict(
        median_wasserstein = median_wd,
        max_wasserstein = max_wd,
        median_wasserstein_global = median_wd_global,
        max_wasserstein_global = max_wd_global,
        frac_p_gt_0_05 = frac_p_gt_005,
        median_ks_stat = median_ks_stat,
        fisher_stat = fisher_stat,
        fisher_p = fisher_p,
        bonferroni_flag = bonf_flag,
        bonferroni_min_p = bonf_minp,
        frac_flag_wass_vs_95 = frac_flag_wass,
    )

    # ----- plotting -----
    fig_summary = None
    if plot:
        fig_summary, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, dpi=dpi)
        axw = axes[0]; axk = axes[1]
        x = np.arange(len(df_out))
        width = 0.35

        # Plot Wasserstein bars
        axw.bar(x - width/2, df_out['wass_vs_norm'].fillna(0.0), width=width, label='W vs N(0,1)', zorder=2)

        # Plot per-bin threshold markers (as diamonds) and optionally join with a thin line
        axw.plot(x, df_out['wass_thresh_95'], linestyle='--', marker='D', color='0.3', label=f'{int((1-alpha)*100)}% W null thresh', zorder=3)

        # Highlight bins that exceed threshold by red edge marker
        exceed_idx = np.where(~df_out['flag_wass_under_95'].fillna(True).values)[0]
        if exceed_idx.size > 0:
            axw.scatter(exceed_idx, df_out.loc[exceed_idx, 'wass_vs_norm'], marker='o', facecolor='none', edgecolor='red', s=80, linewidths=1.5, label='exceeds thresh', zorder=5)

        axw.set_ylabel('Wasserstein distance')
        axw.legend(frameon=False)
        axw.set_title('Per-bin Wasserstein distances')

        # KS p-values
        axk.plot(x, df_out['ks_p_vs_norm'], marker='o', label='KS p vs N(0,1)')
        bonf_alpha = alpha / max(1, (df_out['count']>0).sum())
        axk.axhline(alpha, linestyle='--', color='gray', label=f'alpha={alpha}')
        axk.axhline(bonf_alpha, linestyle=':', color='red', label=f'Bonferroni ({bonf_alpha:.2e})')
        axk.set_ylim(-0.02, 1.02)
        axk.set_ylabel('KS p-value')
        axk.set_xlabel('bin index')
        axk.legend(frameon=False)
        fig_summary.tight_layout()

    return metrics, df_out

def _get_max_consecutive_true_fraction(series: pd.Series | np.ndarray) -> float:
    """
    Calculate fraction of max consecutive True.
    
    Parameters
    ----------
    series : pd.Series or np.ndarray
        Boolean array-like object.
        
    Returns
    -------
    float
        Fraction of max consecutive True values (max_consecutive / total_length).
        
    Examples
    --------
    >>> arr = np.array([False, True, True, False])
    >>> get_max_consecutive_true_fraction(arr)
    0.5
    """
    arr = np.asarray(series, dtype=bool)
    if not arr.any():
        return 0.0
    
    # Reset counter whenever we hit False
    reset_points = ~arr
    run_ids = reset_points.cumsum()
    
    # Count True values in each run
    run_counts = np.bincount(run_ids[arr])
    max_consecutive = run_counts.max() if len(run_counts) > 0 else 0
    
    return float(max_consecutive / len(arr))

def wasserstein_null_threshold(
    power_law_fit: bool = True, # other params only required if the power law is false.
    num_points: int = 300,
    trials: int = 2000,
    normal_ref_size: int = 20_000,
    alpha: float = 0.05,
    normal_ref_seed: int | None = None
) -> (
    Callable[[int | np.ndarray], float | np.ndarray]  # if power_law_fit=True
    | tuple[float, np.ndarray]                       # if power_law_fit=False
):
    """
    Compute or approximate the null distribution of the Wasserstein distance between
    an empirical sample of size `n` drawn from N(0, 1) and the true N(0, 1).

    Modes
    -----
    - If `power_law_fit=True` (default):
        Returns a function ``f(ns)`` that evaluates the precomputed power-law fit
        to the (1 - alpha) quantile of the Wasserstein distance distribution.
        - ``ns`` can be a scalar or array of sample sizes.
        - The returned values are floats or NumPy arrays, respectively.

    - If `power_law_fit=False`:
        Performs a Monte Carlo simulation to estimate the null distribution.
        Returns a tuple ``(critical_value, samples)``:
        - ``critical_value`` : float
            The empirical (1 - alpha) quantile of the simulated distribution.
        - ``samples`` : np.ndarray
            Array of simulated Wasserstein distances across `trials`.

    Parameters
    ----------
    power_law_fit : bool, default=True
        Whether to use a precomputed power-law approximation instead of Monte Carlo.
    num_points : int, default=300
        Sample size for Monte Carlo draws (only used if `power_law_fit=False`).
    trials : int, default=2000
        Number of Monte Carlo trials (only used if `power_law_fit=False`).
    normal_ref_size : int, default=20000
        Size of the reference sample used to approximate N(0, 1).
    alpha : float, default=0.05
        Significance level. The function returns the (1 - alpha) quantile.
    normal_ref_seed : int | None, default=None
        Random seed for reproducibility.

    Returns
    -------
    Callable[[int | np.ndarray], float | np.ndarray]
        If `power_law_fit=True`, a function that computes the quantile threshold
        as a function of sample size(s).
    tuple[float, np.ndarray]
        If `power_law_fit=False`, the empirical threshold and the distribution
        of Wasserstein distances from the Monte Carlo simulation.
    """
    
    # precomputed power law fit for 95% CI for a reference z_std of size 20_000. 
    if power_law_fit: # this returns the compute power law fit governed by A, p, B
        A, p, B = 2.3, -0.52, 0.00590098 # these are for the 95% CI fits. 
        def _power_law(ns: float | int | np.ndarray) -> float | np.ndarray:
            """Power law: A * ns^p + B, returns np.nan for ns=0."""
            # scalar -> scalar, array-like -> numpy array
            if np.isscalar(ns):
                return np.nan if ns == 0 else A * (float(ns) ** p) + B
                
            arr = np.asarray(ns, dtype=float)
            arr[arr == 0] = np.nan  # Set zeros to nan BEFORE calculation
            return A * (arr ** p) + B            
        return _power_law

    if num_points == 0:
        return np.nan, _
    
    # Monte-Carlo branch to estimate confidence intervals
    rng = np.random.default_rng(normal_ref_seed)
    normal_ref = rng.normal(0.0, 1.0, size=normal_ref_size)
    ws = []
    for _ in range(trials):
        sample = rng.normal(0.0, 1.0, size=num_points)
        ws.append(wasserstein_distance(sample, normal_ref))
    return float(np.quantile(ws, 1.0 - alpha)) #, np.array(ws)

def combine_ks_pvalues_fisher(p_values: np.ndarray | list[float]) -> tuple[float, float]:
    """
    Fisher's method: assumes p-values are independent
    Returns combined statistic and overall p-value
    """
    # Remove NaN values
    valid_pvals = [p for p in p_values if not np.isnan(p)]
    
    if len(valid_pvals) == 0:
        return np.nan, np.nan
    
    # Fisher's method
    statistic, combined_pval = combine_pvalues(valid_pvals, method='fisher')
    return statistic, combined_pval

def bonferroni_correction(
    p_values: np.ndarray | list[float], 
    alpha: float = 0.05
) -> tuple[list[bool], float]:
    """
    Conservative approach: require significance after multiple testing correction
    """
    valid_pvals = [p for p in p_values if not np.isnan(p)]
    if len(valid_pvals) == 0:
        return False, np.nan
    
    # Bonferroni correction
    corrected_alpha = alpha / len(valid_pvals)
    min_pval = np.min(valid_pvals)
    
    is_significant = min_pval < corrected_alpha
    return is_significant, min_pval

def compute_linearized_density(
    phi1: np.ndarray,
    mask: None | np.ndarray = None,  # same shape as phi1, True = keep
    nbins: int | None = None,
    bin_width: float | None = None,  # NEW: fixed bin width in phi units
    smoothing_sigma: None | float = None,
    smoothing_frac: float | None = 0.02,
    smoothing_mode: str = "reflect",
    detrend: str = "none",            # "poly", "lowpass", or "none"
    poly_order: int = 3,              # used if detrend == "poly"
    lowpass_sigma_bins: float | None = None,  # used if detrend == "lowpass" (in bins)
    return_pdf: bool = False,
    return_mask: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute per-stream histogram -> smoothed density -> detrend -> fractional residuals.

    Parameters
    ----------
    phi1 : np.ndarray
        1D (N,) single stream or 2D (S, N) (pad ragged rows with np.nan).
    mask : None or np.ndarray
        Optional boolean mask with same shape as `phi1`. True = keep particle; False = ignore.
        If None, `np.nan` entries in `phi1` are treated as invalid.
    nbins : int or None
        Number of bins per stream. If None, must provide `bin_width`.
    bin_width : float or None
        Fixed bin width in phi units. If provided, overrides `nbins`.
        nbins will be computed as ceil((max - min) / bin_width).
    smoothing_sigma : float or None
        Gaussian sigma in bin-units. If None, computed from `smoothing_frac`.
        bandwidth_phi (KDE) = smoothing_sigma * bin_width
    smoothing_frac : float or None
        If provided and `smoothing_sigma` is None, sigma = smoothing_frac * nbins.
    smoothing_mode : str
        Mode passed to `gaussian_filter1d` (e.g. 'reflect', 'constant', 'nearest').
    detrend : {"poly","lowpass","none"}
        Detrending method applied to the smoothed density:
          - "poly": vectorized polynomial on normalized coord u in [0,1].
          - "lowpass": trend = broad Gaussian smoothing of smoothed counts.
          - "none": residuals = (density - mean) / mean.
    poly_order : int
        Polynomial order for `detrend="poly"`.
    lowpass_sigma_bins : float or None
        Sigma (in bins) for lowpass trend. If None default = max(1, 0.10*nbins).
    return_pdf : bool
        If True, return area-normalized pdf (sums to 1 over phi).
    return_mask : bool
        If True, return the boolean mask actually used (shape matches input `phi1`).

    Returns
    -------
    out : dict[str, np.ndarray]
        Dictionary containing:
          - 'hist_counts'  : ndarray (S, nbins) or (nbins,) raw counts per bin
          - 'smooth_counts': ndarray same shape, smoothed counts (approx KDE)
          - 'density'      : ndarray same shape, counts per unit-phi
          - 'pdf'          : ndarray same shape or None (area-normalized)
          - 'centers'      : ndarray (S, nbins) or (nbins,) bin centers (phi units)
          - 'w'            : ndarray (S,) or scalar, per-stream bin width (phi units)
          - 'nbins'        : ndarray (S,) or scalar, actual number of bins used
          - 'min' / 'max'  : per-stream min/max phi
          - 'trend'        : trend used for detrending (same units as density)
          - 'resid'        : fractional residuals = (density - trend)/trend
          - 'mask_used'    : optional ndarray same shape as input `phi1` (bool)
    
    Notes
    -----
    **Fixed Bin Width vs. Fixed Number of Bins**
    
    When analyzing multiple streams with different lengths, you can choose:
    
    - `nbins` (fixed): All streams have the same number of bins, but different bin widths.
      This produces different spatial resolutions per stream.
      
    - `bin_width` (fixed): All streams have the same bin width (spatial resolution), but 
      different numbers of bins. This is recommended for comparing streams consistently.
      
    **KDE Approximation and Bandwidth**
    
    The smoothing operation approximates a Gaussian kernel density estimate (KDE) by 
    applying `gaussian_filter1d` to binned histogram counts. The relationship between 
    the smoothing parameter and true KDE bandwidth is:
    
        bandwidth_phi = smoothing_sigma * w
    
    where `w` is the bin width in phi coordinates.
    
    For fixed bin width analysis with bin_width=0.25 degrees:
    - smoothing_sigma=1.0 -> bandwidth = 0.25 degrees (same as bin)
    - smoothing_sigma=0.5 -> bandwidth = 0.125 degrees (half bin)
    - smoothing_sigma=2.0 -> bandwidth = 0.5 degrees (twice bin)
    
    **Detrending Methods**
    
    Three detrending approaches are available to remove large-scale trends:
    
    - 'poly': Fits a polynomial of order `poly_order` to the smoothed density using
      least squares on normalized coordinates u ∈ [0,1]. This is vectorized across
      streams and works well for smooth, monotonic trends.
      
    - 'lowpass': Applies a broad Gaussian filter (sigma=`lowpass_sigma_bins`) to 
      the smoothed counts to define the trend. This preserves more flexible trend
      shapes than polynomial fitting.
      
    - 'none': Uses the mean density as the trend baseline. Residuals represent
      deviations from a flat background.
    
    In all cases, fractional residuals are computed as: (density - trend) / trend.
    """
    phi1 = np.asarray(phi1)

    # Validate mask if provided
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != phi1.shape:
            raise ValueError("mask must have same shape as phi1")

    # Check that either nbins or bin_width is provided
    if nbins is None and bin_width is None:
        raise ValueError("Must provide either 'nbins' or 'bin_width'")
    if nbins is not None and bin_width is not None:
        raise ValueError("Cannot provide both 'nbins' and 'bin_width'; choose one")

    # unify to 2D
    if phi1.ndim == 1:
        single = True
        phi1 = phi1[None, :]
        if mask is not None:
            mask = mask[None, :]
    else:
        single = False

    if phi1.ndim != 2:
        raise ValueError("phi1 must be 1D or 2D array")

    S, N = phi1.shape

    # Build valid_mask: True where we keep particle (masked-in and not NaN)
    if mask is None:
        valid_mask = ~np.isnan(phi1)
    else:
        # mask True means keep; also exclude NaNs
        valid_mask = mask & (~np.isnan(phi1))

    # detrend option check
    detrend = detrend.lower()
    if detrend not in ["none", "poly", "lowpass"]:
        raise ValueError("Invalid detrend choice. Must be 'none', 'poly', or 'lowpass'.")

    # ----------------- per-stream grid (min/max computed respecting valid_mask) -----------------
    phi1_masked = np.where(valid_mask, phi1, np.nan)
    min1 = np.nanmin(phi1_masked, axis=1)
    max1 = np.nanmax(phi1_masked, axis=1)
    L = max1 - min1

    # Compute nbins per stream based on mode
    if bin_width is not None:
        # Fixed bin width mode: compute nbins per stream
        nbins_arr = np.where(L > 0, np.ceil(L / bin_width).astype(int), 1)
        # Ensure minimum of 1 bin
        nbins_arr = np.maximum(nbins_arr, 1)
        # Maximum nbins for array allocation
        max_nbins = int(np.max(nbins_arr))
        # Actual bin width per stream (may differ slightly from target due to rounding)
        w = np.where(L > 0, L / nbins_arr, np.nan)
    else:
        # Fixed nbins mode: all streams have same nbins
        nbins_arr = np.full(S, nbins, dtype=int)
        max_nbins = nbins
        w = np.where(L > 0, L / nbins, np.nan)

    # ----------------- build flat bincount using the valid_mask -----------------
    flat_idx_parts: list[np.ndarray] = []
    mask_out: None | np.ndarray = (np.zeros_like(phi1, dtype=bool) if return_mask else None)

    for i in range(S):
        valid = valid_mask[i]
        if not np.any(valid) or np.isnan(w[i]) or w[i] == 0.0:
            continue
        rowvals = phi1[i, valid]  # selected particle values
        rel = (rowvals - min1[i]) / w[i]
        inds = np.floor(rel).astype(int)
        inds = np.clip(inds, 0, nbins_arr[i] - 1)
        flat_idx_parts.append(inds + i * max_nbins)
        if mask_out is not None:
            valid_idx_positions = np.nonzero(valid)[0]
            mask_out[i, valid_idx_positions] = True

    if len(flat_idx_parts) == 0:
        hist = np.zeros((S, max_nbins), dtype=float)
    else:
        flat_all = np.concatenate(flat_idx_parts)
        total_bins = S * max_nbins
        flat_hist = np.bincount(flat_all, minlength=total_bins)
        hist = flat_hist.reshape(S, max_nbins).astype(float)

    # For variable nbins: mask out unused bins (set to NaN for clarity)
    if bin_width is not None:
        for i in range(S):
            if nbins_arr[i] < max_nbins:
                hist[i, nbins_arr[i]:] = np.nan

    # ----------------- smoothing (approx KDE) -----------------
    if smoothing_sigma is None:
        # Use smoothing_frac with actual nbins per stream
        # For simplicity, use max_nbins as reference (or could use per-stream)
        sigma_bins = (smoothing_frac or 0.02) * max_nbins
    else:
        sigma_bins = float(smoothing_sigma)
    
    smooth = gaussian_filter1d(hist, sigma_bins, axis=1, mode=smoothing_mode)

    # ----------------- centers and density/pdf -----------------
    j = np.arange(max_nbins)
    centers = min1[:, None] + (j + 0.5) * w[:, None]  # shape (S, max_nbins)
    density = np.full_like(smooth, np.nan, dtype=float)
    pdf = np.full_like(smooth, np.nan, dtype=float)
    valid_streams = ~np.isnan(w) & (w > 0)
    
    if np.any(valid_streams):
        density[valid_streams] = smooth[valid_streams] / w[valid_streams][:, None]
        
        # For PDF: only integrate over valid bins per stream
        for idx in np.where(valid_streams)[0]:
            n = nbins_arr[idx]
            area = smooth[idx, :n].sum() * w[idx]
            if area > 0:
                pdf[idx, :n] = smooth[idx, :n] / area
            else:
                pdf[idx, :n] = 0.0

    # ----------------- detrending -----------------
    trend = np.full_like(density, np.nan, dtype=float)
    resid = np.full_like(density, np.nan, dtype=float)

    if detrend == "poly":
        if poly_order < 0:
            raise ValueError("poly_order must be >= 0 for poly detrend")
        
        # Detrend per stream using its actual nbins
        for i in range(S):
            if not valid_streams[i]:
                continue
            n = nbins_arr[i]
            u = (np.arange(n) + 0.5) / float(n)
            V = np.vander(u, N=poly_order + 1, increasing=True)
            pinvV = np.linalg.pinv(V)
            D = density[i, :n]
            coefs = pinvV @ D
            trend_i = V @ coefs
            trend[i, :n] = trend_i
            trend_safe = np.where(trend_i == 0, np.nan, trend_i)
            resid[i, :n] = (D - trend_safe) / trend_safe

    elif detrend == "lowpass":
        if lowpass_sigma_bins is None:
            lowpass_sigma_bins = max(1.0, 0.10 * max_nbins)
        trend_counts = gaussian_filter1d(smooth, lowpass_sigma_bins, axis=1, mode=smoothing_mode)
        trend[valid_streams] = trend_counts[valid_streams] / w[valid_streams][:, None]
        trend_safe = trend.copy()
        trend_safe[trend_safe == 0] = np.nan
        resid = (density - trend_safe) / trend_safe

    else:  # "none"
        for i in range(S):
            if not valid_streams[i]:
                continue
            n = nbins_arr[i]
            mean_d = np.nanmean(density[i, :n])
            if mean_d == 0:
                mean_d = np.nan
            trend[i, :n] = mean_d
            resid[i, :n] = (density[i, :n] - mean_d) / mean_d

    # ----------------- unwrap single -----------------
    out: dict[str, np.ndarray] = {
        "hist_counts": hist if not single else hist[0],
        "smooth_counts": smooth if not single else smooth[0],
        "density": density if not single else density[0],
        "pdf": (pdf if return_pdf else (None)) if not single else (pdf[0] if return_pdf else None),
        "centers": centers if not single else centers[0],
        "w": w if not single else w[0],
        "nbins": nbins_arr if not single else nbins_arr[0],
        "min": min1 if not single else min1[0],
        "max": max1 if not single else max1[0],
        "trend": trend if not single else trend[0],
        "resid": resid if not single else resid[0],
    }
    if return_mask:
        out["mask"] = mask_out if not single else (mask_out[0] if mask_out is not None else np.zeros(phi1.shape[1], dtype=bool))
    return out

def compute_welch_psd_for_resid(
    resid: np.ndarray,
    w: np.ndarray | float,
    fs_override: float | None = None,
    nperseg_frac: float = 0.25,
    noverlap_frac: float = 0.5,
    scaling: str = "density",
) -> dict[str, Any]:
    """
    Compute per-stream Welch PSDs and summary RMS from fractional residuals.

    This function accepts either a single-stream `resid` array of shape `(nbins,)`
    or a multi-stream `resid` array of shape `(S, nbins)` and computes a Welch
    power spectral density for each stream. Sampling frequency for each stream is
    taken as `fs = 1.0 / w[i]` (cycles per unit-phi) unless `fs_override` is set.

    Parameters
    ----------
    resid : np.ndarray
        Fractional residuals array. Shape `(nbins,)` for a single stream or
        `(S, nbins)` for multiple streams (one stream per row).
    w : np.ndarray | float
        Per-stream bin width (phi units per bin). If a scalar is provided it is
        broadcast to all streams.
    fs_override : float or None, optional
        If provided, overrides the per-stream sampling frequency `1/w` and forces
        all streams to use this sampling frequency.
    nperseg_frac : float, optional
        Fraction of `nbins` to use for `nperseg` in Welch. Default 0.125.
    noverlap_frac : float, optional
        Fraction of `nperseg` to use for overlap between segments. Default 0.5.
    scaling : str, optional
        Passed to `scipy.signal.welch` (e.g., "density" or "spectrum").

    Returns
    -------
    out : dict[str, Any]
        If `resid` was 1D (single stream) the returned dict contains keys:
          - "freqs" : np.ndarray, frequency vector for the stream
          - "psd" : np.ndarray, PSD for the stream
          - "rms_total" : float, total fractional RMS from PSD integral
          - "psd_list" : list[np.ndarray], list containing the PSD (same as "psd")
          - "freqs_list" : list[np.ndarray], list containing the freqs (same as "freqs")

        If `resid` was 2D (S streams) the returned dict contains:
          - "freqs_list" : list[np.ndarray], frequencies for each stream
          - "psd_list" : list[np.ndarray], PSD array for each stream
          - "rms_total" : np.ndarray, shape (S,), total fractional RMS per stream

    Notes
    -----
    - The function skips streams where `w` is NaN or zero and returns empty arrays
      for those entries in the lists.
    - The total RMS is computed by integrating the positive-frequency PSD values:
      `rms = sqrt( ∫_{f>0} PSD(f) df )`.
    """
    single = (resid.ndim == 1)
    if single:
        resid = resid[None, :]
    S, nbins = resid.shape

    # If w is scalar, convert to array
    if np.isscalar(w):
        w_arr = np.full(S, w, dtype=float)
    else:
        w_arr = np.asarray(w, dtype=float)

    freqs_list: List[np.ndarray] = []
    psd_list: List[np.ndarray] = []
    rms_total = np.full(S, np.nan, dtype=float)

    for i in range(S):
        if np.isnan(w_arr[i]) or w_arr[i] == 0:
            freqs_list.append(np.array([], dtype=float))
            psd_list.append(np.array([], dtype=float))
            continue

        fs = 1.0 / w_arr[i] if fs_override is None else fs_override
        # nperseg as fraction of nbins, but at least 8
        nperseg = max(8, int(max(8, nperseg_frac * nbins)))
        noverlap = int(noverlap_frac * nperseg)

        # ensure nperseg <= nbins
        if nperseg > nbins:
            nperseg = nbins
            noverlap = int(0.5 * nperseg)
        freqs, psd = welch(
            resid[i] - np.nanmean(resid[i]),
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling=scaling,
        )
        freqs_list.append(freqs)
        psd_list.append(psd)

        # total rms (integral over positive freqs, skipping DC)
        sel_pos = freqs > 0
        if sel_pos.sum() > 0:
            var = trapezoid(psd[sel_pos], freqs[sel_pos])
            rms_total[i] = np.sqrt(var)
        else:
            rms_total[i] = 0.0

    if single:
        return {
            "freqs": freqs_list[0],
            "psd": psd_list[0],
            "rms_total": rms_total[0],
            "psd_list": psd_list,
            "freqs_list": freqs_list,
        }
    else:
        return {
            "freqs_list": freqs_list,
            "psd_list": psd_list,
            "rms_total": rms_total,
        }

def compute_sampling_psd_realizations(
    hist_counts: np.ndarray,
    smoothing_sigma_bins: float,
    w: float,
    n_realizations: int = 500,
    detrend: str = "poly",            # "poly" or "constant"
    poly_order: int = 3,
    nperseg_frac: float = 0.25,
    noverlap_frac: float = 0.5,
    scaling: str = "density",
    rng_seed: int | None = None,
    sampling: str = "multinomial",    # "multinomial" or "poisson"
    p: np.ndarray | None = None,      # optional probability vector for multinomial (sum -> 1)
    null_hypothesis: bool = True,    # NEW: if True, sample from uniform distribution
) -> dict[str, Any]:
    """
    Monte-Carlo PSD realizations using either multinomial (fixed-N) or Poisson sampling.

    Parameters
    ----------
    hist_counts : np.ndarray
        1-D observed counts per bin (nbins,).
    smoothing_sigma_bins : float
        Gaussian sigma in bin units for smoothing.
    w : float
        Bin width in phi units (phi per bin). Welch fs = 1/w.
    n_realizations : int
        Number of MC realizations.
    detrend : {"poly","constant"}
        Detrending method applied per realization.
    poly_order : int
        Polynomial order when detrend == "poly".
    nperseg_frac, noverlap_frac, scaling :
        Parameters forwarded to scipy.signal.welch.
    rng_seed : int | None
        RNG seed for reproducibility.
    sampling : {"multinomial","poisson"}
        Which sampling model to use. "multinomial" preserves total N per realization.
    p : ndarray or None
        If provided and sampling == "multinomial", use this as the probability vector
        (must sum to 1). If None, `p = hist_counts / hist_counts.sum()` is used.
    null_hypothesis : bool
        If True, sample from uniform distribution to establish Poisson noise floor.
        This overrides the p parameter for multinomial sampling.

    Returns
    -------
    dict with keys:
      - 'freqs', 'psd_all', 'psd_median', 'psd_16', 'psd_84', 'psd_95', 'psd_mean', 'psd_std', ...
    """
    hist = np.asarray(hist_counts, dtype=float)
    if hist.ndim != 1:
        raise ValueError("hist_counts must be 1-D (single stream).")
    nbins = hist.size
    if nbins < 2:
        raise ValueError("hist_counts must have at least 2 bins.")
    if n_realizations <= 0:
        raise ValueError("n_realizations must be > 0")
    sampling = sampling.lower()
    if sampling not in ("multinomial", "poisson"):
        raise ValueError("sampling must be 'multinomial' or 'poisson'")

    rng = np.random.default_rng(rng_seed)

    # Welch parameters
    nperseg = max(8, int(max(8, nperseg_frac * nbins)))
    if nperseg > nbins:
        nperseg = nbins
    noverlap = int(noverlap_frac * nperseg)
    fs = 1.0 / w

    # Precompute polynomial design pseudo-inverse if using polynomial detrend
    if detrend == "poly":
        u = (np.arange(nbins) + 0.5) / float(nbins)   # normalized coordinate in [0,1]
        V = np.vander(u, N=poly_order + 1, increasing=True)   # (nbins, k)
        pinvV = np.linalg.pinv(V)

    # Build mc_counts array depending on sampling method and null_hypothesis flag
    if sampling == "multinomial":
        Ntotal = int(np.nansum(hist))
        if Ntotal <= 0:
            raise ValueError("Total count Ntotal must be > 0 for multinomial sampling.")
        
        # Determine probability vector based on null_hypothesis flag
        if null_hypothesis:
            # Uniform distribution across all bins for null hypothesis
            p_vec = np.ones(nbins) / nbins
        else:
            # Original behavior: use provided p or derive from hist
            if p is None:
                if Ntotal == 0:
                    raise ValueError("hist_counts sum is zero, cannot build p.")
                p_vec = hist / Ntotal
            else:
                p_vec = np.asarray(p, dtype=float)
                if p_vec.shape != (nbins,):
                    raise ValueError("p must have same length as hist_counts")
                # normalize safely
                sm = p_vec.sum()
                if sm <= 0:
                    raise ValueError("p must sum to a positive number")
                p_vec = p_vec / sm
        
        # Vectorized multinomial draws: shape (n_realizations, nbins)
        mc_counts = rng.multinomial(Ntotal, p_vec, size=n_realizations).astype(float)
    
    else:  # Poisson sampling
        if null_hypothesis:
            # For null hypothesis with Poisson: use uniform expected counts
            # Total expected counts divided uniformly across bins
            mean_count_per_bin = np.nansum(hist) / nbins
            uniform_lambda = np.full(nbins, mean_count_per_bin)
            mc_counts = rng.poisson(lam=uniform_lambda[None, :], size=(n_realizations, nbins)).astype(float)
        else:
            # Original behavior: use observed hist as mean
            mc_counts = rng.poisson(lam=hist[None, :], size=(n_realizations, nbins)).astype(float)

    # The remainder is: smooth, density, detrend, compute PSDs
    mc_smooth = gaussian_filter1d(mc_counts, smoothing_sigma_bins, axis=1, mode='reflect')  # (n_realizations, nbins)
    mc_density = mc_smooth / w
    # mc_density = mc_counts / w

    if detrend == "poly":
        D = mc_density.T                                # (nbins, n_realizations)
        coefs = pinvV @ D                               # (k, n_realizations)
        trend_T = (V @ coefs)                           # (nbins, n_realizations)
        trend = trend_T.T                               # (n_realizations, nbins)
        trend_safe = trend.copy()
        trend_safe[trend_safe == 0.0] = np.nan
        mc_resid = (mc_density - trend_safe) / trend_safe
    else:
        means = np.nanmean(mc_density, axis=1)          # (n_realizations,)
        means_safe = np.where(means == 0.0, np.nan, means)
        mc_resid = (mc_density - means_safe[:, None]) / means_safe[:, None]

    mc_resid = mc_resid - np.nanmean(mc_resid, axis=1)[:, None]   # (n_realizations, nbins)

    # Compute PSDs
    psd_list = []
    freqs_ref = None
    for i in range(n_realizations):
        freqs_i, psd_i = welch(
            mc_resid[i],
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling=scaling,
        )
        if freqs_ref is None:
            freqs_ref = freqs_i
        else:
            if not np.allclose(freqs_ref, freqs_i, atol=1e-12, rtol=1e-8):
                raise RuntimeError("Inconsistent frequency grids from welch across realizations.")
        psd_list.append(psd_i)
    psd_all = np.vstack(psd_list)  # shape (n_realizations, nfreq)

    # Summary statistics
    psd_median = np.median(psd_all, axis=0)
    psd_16 = np.percentile(psd_all, 16.0, axis=0)
    psd_84 = np.percentile(psd_all, 84.0, axis=0)
    psd_95 = np.percentile(psd_all, 95.0, axis=0)
    psd_mean = psd_all.mean(axis=0)
    psd_std = psd_all.std(axis=0, ddof=1)

    return {
        "freqs": freqs_ref,
        "psd_all": psd_all,
        "psd_median": psd_median,
        "psd_16": psd_16,
        "psd_84": psd_84,
        "psd_95": psd_95,
        "psd_mean": psd_mean,
        "psd_std": psd_std,
        "nperseg": nperseg,
        "noverlap": noverlap,
    }

def detect_stream_psd_metrics(
    freqs: np.ndarray,
    psd_obs: np.ndarray,
    w: float,
    *,
    psd_all: np.ndarray | None = None,
    psd_median: np.ndarray | None = None,
    psd_95: np.ndarray | None = None,
    nperseg: int | None = None,
    snr_threshold: float = 3.0,
    min_bins: int = 2,
    conservative_nyquist_frac: float = 0.9,
    method: str = "band_snr",   # "band_snr" (requires psd_all) or "ratio95" (uses psd_95)
    detailed: bool = False,
) -> tuple[float, float, float, float] | dict[str, Any]:
    """
    Detect PSD metrics and the minimum detectable wavelength (band) for a single stream.

    Parameters
    ----------
    freqs : np.ndarray
        1-D positive frequency vector (cycles per phi unit), sorted ascending.
    psd_obs : np.ndarray
        Observed PSD values at `freqs` (same length).
    w : float
        Bin width (phi units per bin). Sampling freq fs = 1/w; Nyquist = 1/(2*w).
    psd_all : np.ndarray or None
        Optional MC PSD realizations array of shape (n_realizations, nfreq_res).
        Preferred input for band-integrated SNR. If provided and its freq-grid differs
        from `freqs`, the function will interpolate each MC row onto `freqs`.
    psd_median : np.ndarray or None
        Optional median MC PSD (nfreq_res,). Used if psd_all is not provided.
    psd_95 : np.ndarray or None
        Optional 95th-percentile MC PSD (nfreq_res,). Used by method="ratio95" or as fallback.
    nperseg : int or None
        If provided, used to set a conservative low-frequency limit: df_welch = fs / nperseg.
        If None, the function uses the observed Δf to set the low-frequency limit.
    snr_threshold : float
        Threshold on band SNR to claim detection (default 3.0).
    min_bins : int
        Minimum number of frequency bins in a scanned band (helps avoid single-bin noise).
    conservative_nyquist_frac : float
        Fraction of Nyquist to trust for the upper end of the scanned band (<=1.0).
    method : {"band_snr","ratio95"}
        "band_snr": compute band-integrated SNR using psd_all (preferred).
        "ratio95": fallback per-frequency ratio test using psd_95 (conservative).
    detailed : bool
        If False (default) returns a tuple:
            (rms_obs, rms_null, excess_power, min_lambda_band)
        If True returns a dict with those plus extra arrays and band lists.

    Returns
    -------
    If detailed == False:
        (rms_obs, rms_null, excess_power, min_lambda_band)
          - rms_obs : float
              sqrt(integral_{trusted band} P_obs(f) df)
          - rms_null : float
              median of sqrt(integral_{trusted band} P_mc_i(f) df) across MC realizations,
              or sqrt(integral P_median df) if psd_all not provided.
          - excess_power : float
              integral_{trusted band} max(P_obs - mu_null, 0) df
          - min_lambda_band : float or None
              smallest wavelength (degrees) for which a contiguous band (>=min_bins)
              has SNR_band >= snr_threshold. None if no detection.

    If detailed == True:
        returns a dict with keys including:
          'rms_obs','rms_null','excess_power','min_lambda_band',
          'freqs','trusted_mask','psd_null_median','psd_null_95','psd_all_interp',
          'bands_above_threshold' (list of (i0,i1,SNR,p_emp,I_obs,lambda_min)), etc.

    Notes
    -----
    - Band-integrated observed excess:
        I_obs = ∫_{band} max( P_obs(f) - μ_null(f), 0 ) df
      For each MC realization i:
        I_mc[i] = ∫_{band} max( P_mc_i(f) - μ_null(f), 0 ) df
      Then SNR_band = I_obs / std(I_mc). Empirical p-value p_emp = mean(I_mc >= I_obs).
    - Trusted band: frequencies between f_min_trust and f_max_trust where
        f_min_trust = max( median(diff(freqs_pos)), fs / nperseg (if given) )
        f_max_trust = min( freqs.max(), conservative_nyquist_frac * f_nyq )
    - If psd_all is not provided and method == "band_snr", the function will fall back
      to "ratio95" if psd_95 is provided; otherwise it'll compute rms_null/excess from psd_median
      but cannot compute band SNR (so min_lambda_band will be None).
    """
    # -- basic validation --
    freqs = np.asarray(freqs, dtype=float)
    psd_obs = np.asarray(psd_obs, dtype=float)
    if freqs.ndim != 1 or psd_obs.ndim != 1:
        raise ValueError("freqs and psd_obs must be 1D arrays.")
    if freqs.size != psd_obs.size:
        raise ValueError("freqs and psd_obs must have same length.")
    if np.any(freqs <= 0.0):
        # allow DC in input but we expect positive-only; filter DC out
        pos_mask_all = freqs > 0.0
        freqs = freqs[pos_mask_all]
        psd_obs = psd_obs[pos_mask_all]
    if freqs.size == 0:
        raise ValueError("No positive-frequency bins remain after removing DC.")

    # -- prepare MC summaries and interpolation if needed --
    psd_all_interp = None
    mu_null = None
    null_95_interp = None
    if psd_all is not None:
        psd_all = np.asarray(psd_all, dtype=float)[:, pos_mask_all]
        if psd_all.ndim != 2:
            raise ValueError("psd_all must be 2D array (nreal, nfreq_res).")
        # If user passed psd_all on a different frequency grid, they should supply 'freqs_res' via kwarg
        # but we didn't request it; so assume psd_all already aligns with freqs or that user passed matching arrays.
        # We'll check shapes: if psd_all.shape[1] == freqs.size we accept; else require res_freqs via optional parameters.
        if psd_all.shape[1] == freqs.size:
            psd_all_interp = psd_all.copy()
        else:
            # user likely has passed psd_all computed on a different freq grid; try to find a hint
            # if caller passed psd_median/psd_95 with different length we'll handle below; here we require caller to
            # instead pass psd_all already interpolated. For safety, attempt to infer using psd_median length.
            if psd_median is not None and psd_median.size == psd_all.shape[1]:
                # assume psd_median corresponds to psd_all grid; we have no explicit res_freqs -> cannot reliably interpolate.
                raise ValueError(
                    "psd_all has a different frequency grid length than freqs, and no res grid was provided. "
                    "Please provide psd_all interpolated to the observed freqs or provide res grid."
                )
            else:
                raise ValueError(
                    "psd_all has a different number of frequency bins than freqs. Provide psd_all aligned with freqs."
                )
        # compute mu_null and psd_95 from psd_all_interp
        mu_null = np.median(psd_all_interp, axis=0)
        null_std = np.std(psd_all_interp, axis=0, ddof=1)
        null_95 = np.percentile(psd_all_interp, 95.0, axis=0)
        null_16 = np.percentile(psd_all_interp, 16.0, axis=0)
        null_84 = np.percentile(psd_all_interp, 84.0, axis=0)
        null_95_interp = null_95.copy()
    else:
        # no psd_all: try to use psd_median / psd_95 if provided (assumed aligned to freqs)
        if psd_median is not None and psd_median.size == freqs.size:
            mu_null = np.asarray(psd_median, dtype=float).copy()
        else:
            mu_null = None
        if psd_95 is not None and psd_95.size == freqs.size:
            null_95_interp = np.asarray(psd_95, dtype=float).copy()
        else:
            null_95_interp = None
        null_std = None

    # If mu_null still None and psd_all was not provided, set it to zeros to avoid crashes (but detection will be impossible)
    if mu_null is None:
        mu_null = np.zeros_like(psd_obs)

    # -- trusted frequency band (avoid DC and Nyquist issues) --
    fs = 1.0 / float(w)
    f_nyq = 0.5 * fs
    f_max_trust = min(freqs.max(), float(conservative_nyquist_frac) * f_nyq)
    df_obs = float(np.median(np.diff(freqs))) if freqs.size >= 2 else (fs / (nperseg if nperseg else 2))
    df_welch = float(fs) / float(nperseg) if (nperseg is not None and nperseg > 0) else df_obs
    f_min_trust = max(df_obs, df_welch)
    trusted_mask = (freqs >= f_min_trust) & (freqs <= f_max_trust)
    if not np.any(trusted_mask):
        # relax to any positive up to f_max_trust
        trusted_mask = (freqs > 0.0) & (freqs <= f_max_trust)
    freqs_tr = freqs[trusted_mask]
    if freqs_tr.size == 0:
        raise ValueError("Trusted frequency band is empty. Check w/nperseg/conservative_nyquist_frac.")

    # -- scalar summary metrics: rms_obs, rms_null, excess_power --
    psd_obs_tr = psd_obs[trusted_mask]
    mu_null_tr = mu_null[trusted_mask]

    rms_obs = float(np.sqrt(trapezoid(psd_obs_tr, freqs_tr)))
    if psd_all_interp is not None:
        # integrate each MC realization to get rms distribution
        nreal = psd_all_interp.shape[0]
        vars_mc = np.empty(nreal, dtype=float)
        for i in range(nreal):
            vars_mc[i] = float(trapezoid(psd_all_interp[i, trusted_mask], freqs_tr))
        rms_mc = np.sqrt(vars_mc)
        rms_null = float(np.median(rms_mc))
    else:
        # fallback to median PSD if available
        if mu_null_tr is not None:
            var_null = float(trapezoid(mu_null_tr, freqs_tr))
            rms_null = float(np.sqrt(var_null))
        else:
            rms_null = float(np.nan)

    # excess power (positive-only)
    diff_tr = psd_obs_tr - mu_null_tr
    excess_power = float(trapezoid(np.clip(diff_tr, 0.0, None), freqs_tr))

    # -- detection scan for bands --
    min_lambda_band = None
    bands_above: List[Tuple[int, int, float, float, float, float]] = []  # (i0,i1,SNR,p_emp,I_obs,lambda_min)
    best_SNR = 0.0

    if method == "band_snr":
        if psd_all_interp is None:
            # cannot compute band SNR without MC realizations; fallback to ratio95 if available
            if null_95_interp is None:
                # No MC info to test; return basic metrics and None for detection
                if not detailed:
                    return (rms_obs, rms_null, excess_power, None)
                else:
                    return {
                        "rms_obs": rms_obs,
                        "rms_null": rms_null,
                        "excess_power": excess_power,
                        "min_lambda_band": None,
                        "trusted_mask": trusted_mask,
                        "warning": "No psd_all provided; cannot compute band-integrated SNR. "
                                   "Provide psd_all or use method='ratio95' with psd_95."
                    }
        # psd_all_interp available: scan contiguous bands restricted to trusted_mask
        idxs = np.where(trusted_mask)[0]
        n_tr = idxs.size
        # precompute mu_null on trusted grid and MC arrays on trusted grid
        mu_tr = mu_null[trusted_mask]
        mc_tr = psd_all_interp[:, trusted_mask]  # (nreal, n_tr)
        obs_tr_full = psd_obs[trusted_mask]
        # scanning all contiguous bands >= min_bins
        nreal = mc_tr.shape[0]
        for a in range(0, n_tr):
            for b in range(a + min_bins - 1, n_tr):
                freqs_band = freqs_tr[a:b+1]
                # observed positive clipped vector
                diff_vec = obs_tr_full[a:b+1] - mu_tr[a:b+1]
                diff_pos = np.clip(diff_vec, 0.0, None)
                I_obs = float(trapezoid(diff_pos, freqs_band))
                # MC integrals vectorized
                mc_slice = mc_tr[:, a:b+1] - mu_tr[a:b+1][None, :]
                mc_slice_pos = np.clip(mc_slice, 0.0, None)
                I_mc = trapezoid(mc_slice_pos, freqs_band, axis=1)  # (nreal,)
                sigma_I = float(np.std(I_mc, ddof=1))
                SNR = float(I_obs / sigma_I) if sigma_I > 0 else (np.inf if I_obs > 0 else 0.0)
                p_emp = float(np.mean(I_mc >= I_obs)) if I_mc.size > 0 else float(np.nan)
                # if SNR exceeds threshold, record band
                if SNR >= snr_threshold:
                    f_high = freqs_band[-1]
                    lambda_min = float(1.0 / f_high) if f_high > 0 else None
                    bands_above.append((idxs[a], idxs[b], SNR, p_emp, I_obs, lambda_min))
                if SNR > best_SNR:
                    best_SNR = SNR
        if len(bands_above) > 0:
            # find smallest lambda across bands_above
            lambda_vals = [b[-1] for b in bands_above if b[-1] is not None]
            min_lambda_band = float(np.min(lambda_vals)) if len(lambda_vals) > 0 else None
        else:
            min_lambda_band = None

    elif method == "ratio95":
        # conservative per-frequency test: P_obs / P_null_95 >= snr_threshold
        # require psd_95 (either provided or computed from psd_all)
        if null_95_interp is None and psd_all_interp is not None:
            null_95_interp = np.percentile(psd_all_interp, 95.0, axis=0)
        if null_95_interp is None:
            # cannot run ratio95
            if not detailed:
                return (rms_obs, rms_null, excess_power, None)
            else:
                return {
                    "rms_obs": rms_obs,
                    "rms_null": rms_null,
                    "excess_power": excess_power,
                    "min_lambda_band": None,
                    "warning": "No psd_95 and no psd_all present for ratio95 method."
                }
        ratio95 = psd_obs / null_95_interp
        # only consider trusted_mask
        sel = (ratio95 >= snr_threshold) & trusted_mask
        if np.any(sel):
            idxs = np.where(sel)[0]
            # smallest lambda means pick the highest-frequency selected bin
            highest_idx = int(np.max(idxs))
            f_at = float(freqs[highest_idx])
            min_lambda_band = float(1.0 / f_at) if f_at > 0 else None
        else:
            min_lambda_band = None
    else:
        raise ValueError("method must be 'band_snr' or 'ratio95'")

    if not detailed:
        return (rms_obs, rms_null, excess_power, min_lambda_band)

    # detailed return
    out: dict[str, Any] = {
        "rms_obs": rms_obs,
        "rms_null": rms_null,
        "excess_power": excess_power,
        "min_lambda_band": min_lambda_band,
        "trusted_mask": trusted_mask,
        "f_min_trust": f_min_trust,
        "f_max_trust": f_max_trust,
        "method": method,
        "snr_threshold": snr_threshold,
        "min_bins": min_bins,
    }
    out["psd_null_median"] = mu_null
    out["psd_null_95"] = null_95_interp
    if psd_all_interp is not None:
        out["psd_all_interp"] = psd_all_interp
    out["bands_above_threshold"] = bands_above
    out["best_SNR"] = best_SNR
    return out