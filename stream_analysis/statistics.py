# -*- coding: utf-8 -*-
"""
Distribution-comparison primitives.

These carry no stream-specific meaning. They are the statistical machinery the
localized binned family is built on, kept separate so that :mod:`binned` and
:mod:`plotting` can both use them without either importing the other.

Covers the null distribution of the Wasserstein distance between a finite sample
of N(0, 1) and N(0, 1) itself, the two multiple-testing corrections applied to
the per-bin KS p-values, and a run-length helper used for the spatial-coherence
metric.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, wasserstein_distance

from .config import WASSERSTEIN_NULL_FIT

__all__ = [
    "wasserstein_null_threshold",
    "combine_ks_pvalues_fisher",
    "bonferroni_correction",
]


def wasserstein_null_threshold(
    power_law_fit: bool = True,  # other params only required if the power law is false.
    num_points: int = 300,
    trials: int = 2000,
    normal_ref_size: int = 20_000,
    alpha: float = 0.05,
    normal_ref_seed: int | None = None,
) -> (
    Callable[[int | np.ndarray], float | np.ndarray]  # if power_law_fit=True
    | float  # if power_law_fit=False
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
        Performs a Monte Carlo simulation to estimate the null distribution and
        returns the empirical (1 - alpha) quantile as a float.

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
        as a function of sample size(s). It returns NaN for a sample size of 0.
    float
        If `power_law_fit=False`, the empirical (1 - alpha) quantile of the
        simulated Wasserstein distances. NaN when `num_points` is 0.

    Notes
    -----
    The docstring of the original function described the `power_law_fit=False`
    branch as returning ``(critical_value, samples)``. The code has always
    returned the critical value alone, with the sample array commented out at the
    return statement. The documentation above now matches the code. No behavior
    changed.

    The coefficients of the power law live in
    :class:`stream_analysis.config.WassersteinNullFit`. They were fitted for a
    reference sample of 20000 draws at the 95% level, so passing a different
    `normal_ref_size` or `alpha` does not change them. Those two arguments only
    affect the Monte Carlo branch.
    """

    # precomputed power law fit for 95% CI for a reference z_std of size 20_000.
    if power_law_fit:  # this returns the compute power law fit governed by A, p, B
        A, p, B = (
            WASSERSTEIN_NULL_FIT.A,
            WASSERSTEIN_NULL_FIT.p,
            WASSERSTEIN_NULL_FIT.B,
        )  # these are for the 95% CI fits.

        def _power_law(ns: float | int | np.ndarray) -> float | np.ndarray:
            """Power law: A * ns^p + B, returns np.nan for ns=0."""
            # scalar -> scalar, array-like -> numpy array
            if np.isscalar(ns):
                return np.nan if ns == 0 else A * (float(ns) ** p) + B

            arr = np.asarray(ns, dtype=float)
            arr[arr == 0] = np.nan  # Set zeros to nan BEFORE calculation
            return A * (arr**p) + B

        return _power_law

    if num_points == 0:
        # A sample of zero draws has no Wasserstein distance to anything, so the
        # threshold is undefined. The original wrote `return np.nan, _`, and `_`
        # is a local bound only by the loop below, so this branch raised
        # UnboundLocalError.
        return np.nan

    # Monte-Carlo branch to estimate confidence intervals
    rng = np.random.default_rng(normal_ref_seed)
    normal_ref = rng.normal(0.0, 1.0, size=normal_ref_size)
    ws = []
    for _ in range(trials):
        sample = rng.normal(0.0, 1.0, size=num_points)
        ws.append(wasserstein_distance(sample, normal_ref))
    return float(np.quantile(ws, 1.0 - alpha))  # , np.array(ws)


def combine_ks_pvalues_fisher(p_values: np.ndarray | list[float]) -> tuple[float, float]:
    """
    Combine per-bin KS p-values with Fisher's method.

    Fisher's method assumes the p-values are independent. Adjacent bins along a
    stream are not strictly independent, so treat the combined value as a
    summary rather than as a calibrated test.

    Parameters
    ----------
    p_values : np.ndarray or list of float
        Per-bin p-values. NaN entries are dropped.

    Returns
    -------
    statistic : float
        Fisher's combined statistic, ``-2 * sum(log p)``. NaN if no p-value is
        finite.
    combined_pval : float
        The overall p-value. NaN if no p-value is finite.
    """
    # Remove NaN values
    valid_pvals = [p for p in p_values if not np.isnan(p)]

    if len(valid_pvals) == 0:
        return np.nan, np.nan

    # Fisher's method
    statistic, combined_pval = combine_pvalues(valid_pvals, method="fisher")
    return statistic, combined_pval


def bonferroni_correction(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """
    Apply a Bonferroni correction to per-bin KS p-values.

    The conservative approach. A stream is called significantly non-Gaussian only
    if at least one bin survives correction for the number of bins tested.

    Parameters
    ----------
    p_values : np.ndarray or list of float
        Per-bin p-values. NaN entries are dropped, and the correction divides by
        the number of surviving entries.
    alpha : float, default=0.05
        Family-wise significance level before correction.

    Returns
    -------
    is_significant : bool
        True when the smallest p-value falls below ``alpha / n_valid``. False
        when no p-value is finite.
    min_pval : float
        The smallest finite p-value. NaN when none is finite.
    """
    valid_pvals = [p for p in p_values if not np.isnan(p)]
    if len(valid_pvals) == 0:
        return False, np.nan

    # Bonferroni correction
    corrected_alpha = alpha / len(valid_pvals)
    min_pval = np.min(valid_pvals)

    is_significant = min_pval < corrected_alpha
    return is_significant, min_pval


def _max_consecutive_true_fraction(series: pd.Series | np.ndarray) -> float:
    """
    Calculate fraction of max consecutive True.

    Used for the spatial-coherence metric. The longest unbroken run of
    Gaussian-looking bins, as a fraction of the stream, distinguishes a stream
    disturbed in one place from a stream disturbed everywhere.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Boolean array-like object.

    Returns
    -------
    float
        Fraction of max consecutive True values (max_consecutive / total_length).
        Returns 0.0 when no entry is True.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([False, True, True, False])
    >>> _max_consecutive_true_fraction(arr)
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


def _normalize_wasserstein(df: pd.DataFrame) -> None:
    """
    Add the normalized-Wasserstein columns to a per-bin table, in place.

    This is the block that appeared verbatim in both ``compute_bin_diagnostics``
    and ``plot_metric_stats_detailed`` in the original file. It is factored here
    so the two cannot drift apart. The statements and their order are unchanged.

    Adds three columns.

    ``wass_thresh_95``
        The 95% null threshold for each bin's particle count.
    ``norm_wass_to_95``
        Observed Wasserstein distance divided by that threshold. Values below 1
        mean the bin is Gaussian at 95% confidence.
    ``flag_wass_under_95``
        The boolean form of that comparison.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-bin table carrying at least ``count`` and ``wass_vs_norm``. Modified
        in place.
    """
    counts = df["count"].fillna(0).astype(int).values
    # W.D threshold with 95% CI fit for a reference distribution
    df["wass_thresh_95"] = wasserstein_null_threshold(power_law_fit=True)(counts)

    # Normalized wasserstein (observed / threshold)
    df["norm_wass_to_95"] = df["wass_vs_norm"] / df["wass_thresh_95"]
    df["weighted_norm_wass_to_95"] = df["wass_vs_norm"] / df["wass_thresh_95"] / np.sqrt(df["count"])
    # Flag per-bin exceedance
    df["flag_wass_under_95"] = df["norm_wass_to_95"] < 1.00
