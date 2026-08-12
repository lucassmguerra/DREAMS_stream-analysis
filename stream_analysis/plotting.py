# -*- coding: utf-8 -*-
"""
Figure helpers that are not part of any metric's definition.

:func:`summarize_bin_metrics` is the interactive superset of
:func:`stream_analysis.binned.compute_bin_diagnostics`. It computes the same
normalized-Wasserstein block, adds Fisher and Bonferroni combination of the
per-bin KS p-values, and draws a two-panel summary figure. The pipeline uses
``compute_bin_diagnostics``. This one is for looking at a single stream.

The per-bin violin figure lives with the function that computes the bins, in
:mod:`stream_analysis.binned`, because separating it would have meant rewriting
that function rather than moving it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .statistics import (
    _normalize_wasserstein,
    bonferroni_correction,
    combine_ks_pvalues_fisher,
)

__all__ = ["summarize_bin_metrics"]


def summarize_bin_metrics(
    df: pd.DataFrame,
    plot: bool = True,
    mc_trials: int = 2000,
    normal_ref_size: int = 20000,
    alpha: float = 0.05,
    seed: int = 0,
    dpi: int = 200,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Summarize a per-bin table, with an optional two-panel figure.

    Computes and returns:

      - median_wasserstein (median of df['wass_vs_norm'])
      - max_wasserstein  (max of df['wass_vs_norm'])
      - frac_p_gt_005    (fraction of bins with ks_p_vs_norm > 0.05)
      - median_ks_stat   (median of df['ks_stat_vs_norm'])  (lower ~ more Gaussian)
      - fisher_stat, fisher_p (combined KS p-value via Fisher)
      - bonferroni_flag, bonferroni_minp (result of bonferroni_correction on KS p-values)
      - per-bin wasserstein 95% thresholds and the normalized values built from them

    Parameters
    ----------
    df : pandas.DataFrame
        Expected columns: ['wass_vs_norm', 'ks_p_vs_norm', 'ks_stat_vs_norm', 'count'].
        Missing columns are filled with NaN on the returned copy. The caller's
        frame is not modified.
    plot : bool
        If True produce the two-row summary figure (Wasserstein bars + KS p-values).
    mc_trials, normal_ref_size, seed : int
        Accepted for signature compatibility with earlier versions that ran a
        Monte Carlo for the null thresholds. The thresholds now come from the
        precomputed power-law fit, so these three arguments are not used.
    alpha : float
        Significance level. Sets the KS reference lines on the figure and is
        forwarded to the Bonferroni correction.
    dpi : int
        Figure DPI.

    Returns
    -------
    metrics : dict[str, float]
        The summary numbers listed above.
    df_out : pandas.DataFrame
        A copy of `df` with `wass_thresh_95`, `norm_wass_to_95` and
        `flag_wass_under_95` added.

    Notes
    -----
    The docstring of the original described a 3-tuple return of
    ``fig_summary, metrics, df_out``. The code has always returned the 2-tuple
    ``metrics, df_out``, and the figure handle is discarded. The documentation
    above now matches the code. No behavior changed. Retrieve the figure with
    ``plt.gcf()`` immediately after the call if you need it.
    """
    # defensive copy
    df_out = df.copy()

    # ensure relevant columns exist
    for col in ["wass_vs_norm", "ks_p_vs_norm", "ks_stat_vs_norm", "count", "wass_vs_global"]:
        if col not in df_out.columns:
            df_out[col] = np.nan

    # Basic summary numbers
    wass_vals = df_out["wass_vs_norm"].dropna().values
    median_wd = float(np.nanmedian(wass_vals)) if wass_vals.size > 0 else np.nan
    max_wd = float(np.nanmax(wass_vals)) if wass_vals.size > 0 else np.nan

    try:
        wass_vals_global = df_out["wass_vs_global"].dropna().values
        median_wd_global = (
            float(np.nanmedian(wass_vals_global)) if wass_vals_global.size > 0 else np.nan
        )
        max_wd_global = float(np.nanmax(wass_vals_global)) if wass_vals_global.size > 0 else np.nan
    except Exception:
        wass_vals_global, median_wd_global, max_wd_global = np.nan, np.nan, np.nan

    ks_pvals = df_out["ks_p_vs_norm"].values
    # fraction of p-values > 0.05 (bins consistent with normal)
    valid_p = ~np.isnan(ks_pvals)
    if valid_p.sum() > 0:
        frac_p_gt_005 = float(np.sum(ks_pvals[valid_p] > 0.05) / valid_p.sum())
    else:
        frac_p_gt_005 = np.nan

    # median KS statistic (lower ~ closer to the null)
    ks_stats = df_out["ks_stat_vs_norm"].dropna().values
    median_ks_stat = float(np.nanmedian(ks_stats)) if ks_stats.size > 0 else np.nan

    # Fisher combined p-value for KS p-values
    fisher_stat, fisher_p = combine_ks_pvalues_fisher(list(df_out["ks_p_vs_norm"].values))

    # Bonferroni test on KS p-values
    bonf_flag, bonf_minp = bonferroni_correction(list(df_out["ks_p_vs_norm"].values), alpha=alpha)

    # ----- per-count Wasserstein thresholds from the precomputed null fit -----
    # Adds wass_thresh_95, norm_wass_to_95 and flag_wass_under_95 in place.
    _normalize_wasserstein(df_out)

    # Overall fraction flagged by Wasserstein
    if df_out["flag_wass_under_95"].notna().sum() > 0:
        frac_flag_wass = float(
            np.sum(df_out["flag_wass_under_95"].fillna(False))
            / df_out["flag_wass_under_95"].notna().sum()
        )
    else:
        frac_flag_wass = np.nan

    # Compose metrics dict
    metrics = dict(
        median_wasserstein=median_wd,
        max_wasserstein=max_wd,
        median_wasserstein_global=median_wd_global,
        max_wasserstein_global=max_wd_global,
        frac_p_gt_0_05=frac_p_gt_005,
        median_ks_stat=median_ks_stat,
        fisher_stat=fisher_stat,
        fisher_p=fisher_p,
        bonferroni_flag=bonf_flag,
        bonferroni_min_p=bonf_minp,
        frac_flag_wass_vs_95=frac_flag_wass,
    )

    # ----- plotting -----
    fig_summary = None
    if plot:
        fig_summary, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, dpi=dpi)
        axw = axes[0]
        axk = axes[1]
        x = np.arange(len(df_out))
        width = 0.35

        # Plot Wasserstein bars
        axw.bar(
            x - width / 2,
            df_out["wass_vs_norm"].fillna(0.0),
            width=width,
            label="W vs N(0,1)",
            zorder=2,
        )

        # Plot per-bin threshold markers (as diamonds) and optionally join with a thin line
        axw.plot(
            x,
            df_out["wass_thresh_95"],
            linestyle="--",
            marker="D",
            color="0.3",
            label=f"{int((1 - alpha) * 100)}% W null thresh",
            zorder=3,
        )

        # Highlight bins that exceed threshold by red edge marker
        exceed_idx = np.where(~df_out["flag_wass_under_95"].fillna(True).values)[0]
        if exceed_idx.size > 0:
            axw.scatter(
                exceed_idx,
                df_out.loc[exceed_idx, "wass_vs_norm"],
                marker="o",
                facecolor="none",
                edgecolor="red",
                s=80,
                linewidths=1.5,
                label="exceeds thresh",
                zorder=5,
            )

        axw.set_ylabel("Wasserstein distance")
        axw.legend(frameon=False)
        axw.set_title("Per-bin Wasserstein distances")

        # KS p-values
        axk.plot(x, df_out["ks_p_vs_norm"], marker="o", label="KS p vs N(0,1)")
        bonf_alpha = alpha / max(1, (df_out["count"] > 0).sum())
        axk.axhline(alpha, linestyle="--", color="gray", label=f"alpha={alpha}")
        axk.axhline(bonf_alpha, linestyle=":", color="red", label=f"Bonferroni ({bonf_alpha:.2e})")
        axk.set_ylim(-0.02, 1.02)
        axk.set_ylabel("KS p-value")
        axk.set_xlabel("bin index")
        axk.legend(frameon=False)
        fig_summary.tight_layout()

    return metrics, df_out
