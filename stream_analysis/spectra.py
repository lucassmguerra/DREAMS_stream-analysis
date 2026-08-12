# -*- coding: utf-8 -*-
"""
Power spectra of the along-stream density and the detection metrics built on them.

Three functions, run in order, each consuming the previous one's output.

1. :func:`compute_welch_psd` turns the fractional density residuals into a Welch
   power spectral density.
2. :func:`compute_sampling_psd_realizations` builds the noise floor by resampling
   the observed counts many times and pushing each realization through the same
   smoothing, detrending and Welch chain. Under the default null hypothesis the
   resampling draws from a uniform density, so the floor is a pure
   finite-sampling floor rather than one that already contains the signal.
3. :func:`detect_stream_psd_metrics` compares the observed spectrum to that floor
   and returns the four scalars reported in the paper, the RMS density residual,
   the null RMS, the excess power, and the minimum detectable scale.

Supports the power-spectrum family of *No Stream Left Unscathed* (Arora et al.).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

__all__ = [
    "compute_welch_psd",
    "compute_sampling_psd_realizations",
    "detect_stream_psd_metrics",
]


def compute_welch_psd(
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
        Fraction of `nbins` to use for `nperseg` in Welch. Default 0.25.
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
    - The docstring of the original stated a `nperseg_frac` default of 0.125. The
      signature has always said 0.25. The documentation above now matches the
      code. No behavior changed.
    """
    single = resid.ndim == 1
    if single:
        resid = resid[None, :]
    S, nbins = resid.shape

    # If w is scalar, convert to array
    if np.isscalar(w):
        w_arr = np.full(S, w, dtype=float)
    else:
        w_arr = np.asarray(w, dtype=float)

    freqs_list: list[np.ndarray] = []
    psd_list: list[np.ndarray] = []
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
            window="hann",
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
    detrend: str = "poly",  # "poly" or "constant"
    poly_order: int = 3,
    nperseg_frac: float = 0.25,
    noverlap_frac: float = 0.5,
    scaling: str = "density",
    rng_seed: int | None = None,
    sampling: str = "multinomial",  # "multinomial" or "poisson"
    p: np.ndarray | None = None,  # optional probability vector for multinomial (sum -> 1)
    null_hypothesis: bool = True,  # if True, sample from uniform distribution
) -> dict[str, Any]:
    """
    Monte-Carlo PSD realizations using either multinomial (fixed-N) or Poisson sampling.

    Parameters
    ----------
    hist_counts : np.ndarray
        1-D observed counts per bin (nbins,).
    smoothing_sigma_bins : float
        Gaussian sigma in bin units for smoothing. Match this to the smoothing
        applied to the observed histogram, or the floor will not be comparable.
    w : float
        Bin width in phi units (phi per bin). Welch fs = 1/w.
    n_realizations : int
        Number of MC realizations. The paper uses 1000, overriding this default.
    detrend : {"poly","constant"}
        Detrending method applied per realization.
    poly_order : int
        Polynomial order when detrend == "poly".
    nperseg_frac, noverlap_frac, scaling :
        Parameters forwarded to scipy.signal.welch.
    rng_seed : int | None
        RNG seed for reproducibility. The paper uses 12345.
    sampling : {"multinomial","poisson"}
        Which sampling model to use. "multinomial" preserves total N per realization.
    p : ndarray or None
        If provided and sampling == "multinomial", use this as the probability vector
        (must sum to 1). If None, `p = hist_counts / hist_counts.sum()` is used.
        Ignored when `null_hypothesis=True`.
    null_hypothesis : bool
        If True, sample from uniform distribution to establish Poisson noise floor.
        This overrides the p parameter for multinomial sampling.

    Returns
    -------
    dict
        Keys are 'freqs', 'psd_all', 'psd_median', 'psd_16', 'psd_84', 'psd_95',
        'psd_mean', 'psd_std', 'nperseg' and 'noverlap'. 'psd_all' has shape
        (n_realizations, nfreq).

    Raises
    ------
    ValueError
        If `hist_counts` is not 1-D or has fewer than 2 bins, if `n_realizations`
        is not positive, if `sampling` is unrecognized, if the total count is not
        positive under multinomial sampling, or if an explicit `p` has the wrong
        length or a non-positive sum.
    RuntimeError
        If the frequency grid returned by `welch` differs between realizations,
        which would mean the summary percentiles were being taken across
        mismatched bins.
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

    # Precompute polynomial design pseudo-inverse if using polynomial detrend.
    # The design matrix is the same for every realization, so the pseudo-inverse
    # is formed once and applied to all of them as one matrix product below.
    if detrend == "poly":
        u = (np.arange(nbins) + 0.5) / float(nbins)  # normalized coordinate in [0,1]
        V = np.vander(u, N=poly_order + 1, increasing=True)  # (nbins, k)
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
            mc_counts = rng.poisson(
                lam=uniform_lambda[None, :], size=(n_realizations, nbins)
            ).astype(float)
        else:
            # Original behavior: use observed hist as mean
            mc_counts = rng.poisson(lam=hist[None, :], size=(n_realizations, nbins)).astype(float)

    # The remainder is: smooth, density, detrend, compute PSDs
    mc_smooth = gaussian_filter1d(
        mc_counts, smoothing_sigma_bins, axis=1, mode="reflect"
    )  # (n_realizations, nbins)
    mc_density = mc_smooth / w
    # mc_density = mc_counts / w

    if detrend == "poly":
        D = mc_density.T  # (nbins, n_realizations)
        coefs = pinvV @ D  # (k, n_realizations)
        trend_T = V @ coefs  # (nbins, n_realizations)
        trend = trend_T.T  # (n_realizations, nbins)
        trend_safe = trend.copy()
        trend_safe[trend_safe == 0.0] = np.nan
        mc_resid = (mc_density - trend_safe) / trend_safe
    else:
        means = np.nanmean(mc_density, axis=1)  # (n_realizations,)
        means_safe = np.where(means == 0.0, np.nan, means)
        mc_resid = (mc_density - means_safe[:, None]) / means_safe[:, None]

    mc_resid = mc_resid - np.nanmean(mc_resid, axis=1)[:, None]  # (n_realizations, nbins)

    # Compute PSDs
    psd_list = []
    freqs_ref = None
    for i in range(n_realizations):
        freqs_i, psd_i = welch(
            mc_resid[i],
            fs=fs,
            window="hann",
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
    method: str = "band_snr",  # "band_snr" (requires psd_all) or "ratio95" (uses psd_95)
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

    Raises
    ------
    ValueError
        If `freqs` or `psd_obs` is not 1-D, if their lengths differ, if no
        positive frequency remains after DC is removed, if `psd_all` does not
        align with `freqs`, if the trusted band is empty, or if `method` is
        unrecognized.

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
    - The two methods answer different questions and give different numbers.
      "band_snr" reports the smallest wavelength belonging to *any* band that
      clears the SNR threshold, so it tends toward the top of the trusted band.
      "ratio95" reports the wavelength of the highest-frequency bin that
      individually exceeds the 95th percentile of the null. The published values
      of *No Stream Left Unscathed* were produced with "ratio95".
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
        # NOTE: pos_mask_all is only bound when the input carried a non-positive
        # frequency. Welch always emits DC, so every real call binds it. A caller
        # passing strictly positive freqs together with psd_all hits a
        # NameError here. Preserved as-is, reported in FINDINGS.md.
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
    bands_above: list[tuple] = []  # (i0,i1,SNR,p_emp,I_obs,lambda_min)
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
                        "Provide psd_all or use method='ratio95' with psd_95.",
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
                freqs_band = freqs_tr[a : b + 1]
                # observed positive clipped vector
                diff_vec = obs_tr_full[a : b + 1] - mu_tr[a : b + 1]
                diff_pos = np.clip(diff_vec, 0.0, None)
                I_obs = float(trapezoid(diff_pos, freqs_band))
                # MC integrals vectorized
                mc_slice = mc_tr[:, a : b + 1] - mu_tr[a : b + 1][None, :]
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
                    "warning": "No psd_95 and no psd_all present for ratio95 method.",
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
