# -*- coding: utf-8 -*-
"""
Orbital diagnostics for an ensemble of integrated orbits.

Independent of any stream frame. This is the only module that takes orbits
rather than stream coordinates.

:func:`compute_orbit_properties` builds the ``df_orbits`` table that
:func:`stream_analysis.pipeline.build_metrics_table` consumes. Run it on the
integrated progenitor orbits, then hand its output to the pipeline, which copies
``min_pericenter_dist``, ``max_apocenter_dist``, ``mean_dist``, ``n_pericenters``
and ``period`` straight through into the metrics table.

It is upstream of the stream analysis rather than part of it, which is why
nothing else in this package calls it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import detrend as _signal_detrend

__all__ = ["compute_orbit_properties"]


def compute_orbit_properties(
    orbits: np.ndarray,
    central_mass: float | None = None,
    mu: float | None = None,
    G: float = 4.3009e-6,
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
            Expected in decreasing order, as lookback times. The routine reverses
            both times and coords on entry.
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
    times = np.vstack(orbits[:, 0])[:, ::-1]  # shape (n_orbits, n_steps)  now increasing
    coords = np.stack(orbits[:, 1])[:, ::-1, :]  # shape (n_orbits, n_steps, 6)
    r = np.linalg.norm(coords[..., :3], axis=2)  # (n_orbits, n_steps)
    dt = times[0, 1] - times[0, 0]

    # detect extrema
    # Three-point comparison. A point is a pericenter when it sits below both
    # neighbours, an apocenter when it sits above both.
    r_prev = r[:, :-2]
    r_mid = r[:, 1:-1]
    r_next = r[:, 2:]
    peri_mask = (r_mid < r_prev) & (r_mid < r_next)
    apo_mask = (r_mid > r_prev) & (r_mid > r_next)

    n_peri = peri_mask.sum(axis=1)
    min_peri = np.where(peri_mask, r_mid, np.inf).min(axis=1)
    min_peri[np.isinf(min_peri)] = np.nan
    max_apo = np.where(apo_mask, r_mid, -np.inf).max(axis=1)
    max_apo[np.isneginf(max_apo)] = np.nan
    mean_r = r.mean(axis=1)

    # FFT period estimation
    # The linear trend is removed first so that a secularly growing or shrinking
    # radius does not dominate the low-frequency end. The DC bin is then zeroed
    # explicitly before the peak search.
    r_detrended = _signal_detrend(r, axis=1, type="linear")
    Rf = np.fft.rfft(r_detrended, axis=1)
    amp = np.abs(Rf)
    amp[:, 0] = 0
    peak_idx = amp.argmax(axis=1)
    freqs = np.fft.rfftfreq(r.shape[1], d=dt)
    # protect against zero-frequency
    with np.errstate(divide="ignore", invalid="ignore"):
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
        r_vec = coords[..., :3]  # (n_orbits, n_steps, 3)
        v_vec = coords[..., 3:]  # (n_orbits, n_steps, 3)
        # h = r x v
        h = np.cross(r_vec, v_vec, axis=2)
        vxh = np.cross(v_vec, h, axis=2)
        r_norm = np.linalg.norm(r_vec, axis=2)
        # protect division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            r_hat = np.nan_to_num(r_vec / r_norm[..., None])
        e_vec = (vxh / mu_val[..., None]) - r_hat if np.ndim(mu_val) else (vxh / mu_val) - r_hat
        e_mag = np.linalg.norm(e_vec, axis=2)  # (n_orbits, n_steps)
        # take a robust central value (median)
        ecc_osc = np.median(e_mag, axis=1)
        ecc_osc[~np.isfinite(ecc_osc)] = np.nan

    df = pd.DataFrame(
        {
            "min_pericenter_dist": min_peri,
            "max_apocenter_dist": max_apo,
            "mean_dist": mean_r,
            "n_pericenters": n_peri,
            "period": period,
            "eccentricity_geo": ecc_geo,
            "eccentricity_osculating": ecc_osc,
        }
    )

    return df
