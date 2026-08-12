# -*- coding: utf-8 -*-
"""
Linearized along-stream density.

Histogram the particles along phi1, smooth to approximate a KDE, remove a trend,
and return fractional residuals. Those residuals are the input to the
power-spectrum family in :mod:`stream_analysis.spectra`, which produces the
minimum detectable scale, RMS density residual and excess power metrics of
*No Stream Left Unscathed* (Arora et al.).

This is the last stage that sees individual particles. Everything downstream
works on residual arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

__all__ = ["compute_linearized_density"]


def compute_linearized_density(
    phi1: np.ndarray,
    mask: None | np.ndarray = None,  # same shape as phi1, True = keep
    nbins: int | None = None,
    bin_width: float | None = None,  # fixed bin width in phi units
    smoothing_sigma: None | float = None,
    smoothing_frac: float | None = 0.02,
    smoothing_mode: str = "reflect",
    detrend: str = "none",  # "poly", "lowpass", or "none"
    poly_order: int = 3,  # used if detrend == "poly"
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
          - 'mask'         : optional ndarray same shape as input `phi1` (bool),
                             present only when `return_mask=True`

    Raises
    ------
    ValueError
        If `mask` does not match the shape of `phi1`, if neither or both of
        `nbins` and `bin_width` are given, if `phi1` is neither 1D nor 2D, if
        `detrend` is not recognized, or if `poly_order` is negative under
        `detrend="poly"`.

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

    When `bin_width` is given and streams have different lengths, the trailing
    bins of the shorter streams are set to NaN rather than zero, so that they do
    not enter any later integral as real empty bins.
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
    mask_out: None | np.ndarray = np.zeros_like(phi1, dtype=bool) if return_mask else None

    for i in range(S):
        valid = valid_mask[i]
        if not np.any(valid) or np.isnan(w[i]) or w[i] == 0.0:
            continue
        rowvals = phi1[i, valid]  # selected particle values
        rel = (rowvals - min1[i]) / w[i]
        inds = np.floor(rel).astype(int)
        inds = np.clip(inds, 0, nbins_arr[i] - 1)
        # Offsetting by i*max_nbins lets one bincount call do all S histograms.
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
                hist[i, nbins_arr[i] :] = np.nan

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
        out["mask"] = (
            mask_out
            if not single
            else (mask_out[0] if mask_out is not None else np.zeros(phi1.shape[1], dtype=bool))
        )
    return out
