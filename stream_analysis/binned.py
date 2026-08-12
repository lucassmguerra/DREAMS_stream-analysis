# -*- coding: utf-8 -*-
"""
Localized binned statistics along a stream.

This is the machinery behind the global disturbance, peak disturbance and width
variation metrics of *No Stream Left Unscathed* (Arora et al.).

The idea is that a smooth, unperturbed stream has a Gaussian cross-section
everywhere along its track, while a disturbed one does not. So the track is cut
into bins along phi1, and in each bin the distribution of some per-particle
quantity is compared against a standard normal, by a KS test and by a
Wasserstein distance. Collapsing those per-bin numbers gives the stream-level
metrics.

The binning is generic in the quantity. The paper applies it to phi2, the
across-track offset, and the same machinery applied to `v_los` yields the
kinematic analogues. That is why the second argument is named `quantity` rather
than `phi2`, and why `quantity_name` controls the output column suffix.

Bins
----
phi1 is expected to be rescaled by the stream length before it gets here, so the
default `bin_size` of 1/8 gives eight bins spanning the stream. That is the
``L/8`` of the paper.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, kstest, wasserstein_distance

from .statistics import _max_consecutive_true_fraction, _normalize_wasserstein

__all__ = ["compute_local_binned_stats", "compute_bin_diagnostics"]


def compute_local_binned_stats(
    phi1: np.ndarray,
    quantity: np.ndarray | None = None,  # phi2, v_los, or any other per-particle quantity
    scheme: str = "bin_size",
    bin_edges: np.ndarray | None = None,
    bin_size: float = 1 / 8.0,
    min_particles: int = 300,
    normal_ref_size: int = 20000,
    normal_ref_seed: int = 12345,
    quantity_name: str = "phi2",
    # # Random control args for plots
    plot_streambins: bool = False,  # whether to make the binned-stream plots.
    axis_labels: bool = True,
    x_axis_label: str = r"\bf{\boldmath$\phi_1$ [$\ell \ ^\circ$]}",
    y_axis_label: str = r"\bf{\boldmath$\phi_2$ [$^\circ$]}",
    max_bins_to_plot: int = 24,
    figsize_per_bin: tuple[int, int] = (2, 2),
    scatter_kwargs=None,
    kde_bandwidth: float | None = None,
    phi1_kde_bandwidth: float | None = None,
    phi1_curve_height_frac: float = 0.50,  # max curve height as fraction of vertical span
    phi2_violin_width_frac: float = 0.35,  # fraction of bin width used for phi2 mirrored curve
    show_counts: bool = True,
    title: str | None = None,
    phi2_ylim: tuple[float, float] | None = None,
    phi1_linewidth: float = 1.0,
    phi2_linewidth: float = 2.0,
    phi1_color: str = "g",
    phi2_color: str = "r",
    axs=None,
    *,
    phi2: np.ndarray | None = None,  # deprecated name for `quantity`
) -> pd.DataFrame:
    """
    Compute per-bin statistics and optionally plot them.

    Stats are computed using a custom function with pandas groupby.
    For Arora+2026 No stream left. Phi1 was rescaled with Length of a stream.

    Parameters
    ----------
    phi1 : np.ndarray, shape (N,)
        Along-stream coordinates (degrees or radians). Ideally rescaled by length.
    quantity : np.ndarray, shape (N,)
        The per-particle quantity to bin. Across-stream coordinate phi2 in the
        paper. Could be anything, for example `v_los`. Named `phi2` in earlier
        versions of this function.
    scheme : str, {'bin_size', 'equal_count'}, default='bin_size'
        - 'bin_size': Fixed bin length in phi1.
        - 'equal_count': equal number of particles in each bin.
    bin_edges : np.ndarray or None, default=None
        Explicit bin edges. Overrides `scheme` and `bin_size` when given.
    bin_size : float, default=1/8.
        Size of individual bin in length scale. Used in 'bin_size' scheme.
    min_particles : int, default=300
        Minimum number of particles to generate a bin. Used in 'equal_count' scheme.
    normal_ref_size : int, default=20000
        Number of samples used for reference standard normal distribution.
    normal_ref_seed : int, default=12345
        numpy random seed to sample the reference distribution.
    quantity_name : str, default='phi2'
        Suffix for the two per-bin columns named after the quantity, `mean_<name>`
        and `std_<name>`. Labelling and column naming only, it does not enter any
        computation. The default reproduces the historical column names exactly.
    plot_streambins : bool, default=False
        Whether to make the binned-stream plots.
        The rest of the flags are only active if True.

    See type hints for the plotting and other flags.

    Returns
    -------
    pandas.DataFrame
        One row per bin, with columns

        - ``count`` : number of particles in the bin
        - ``mean_<quantity_name>`` : bin mean of the quantity
        - ``std_<quantity_name>`` : bin sample standard deviation, ddof=1
        - ``ks_stat_vs_norm``, ``ks_p_vs_norm`` : KS test of the standardized bin
          values against N(0, 1)
        - ``wass_vs_norm`` : Wasserstein distance to a reference N(0, 1) sample
        - ``wass_vs_global`` : Wasserstein distance to the standardized
          whole-stream distribution
        - ``bin_left``, ``bin_right`` : bin boundaries

        The index is a plain RangeIndex. The interval index used internally is
        dropped.

        With ``plot_streambins=True`` the figure is drawn as a side effect and
        the same DataFrame is returned.

    Raises
    ------
    AssertionError
        If `phi1` and `quantity` do not have the same shape.
    ValueError
        If `scheme` is not recognized, or if no bins could be determined.

    Notes
    -----
    A bin that receives no particles produces a six-element row while populated
    bins produce seven, and the DataFrame construction then raises. Any stream
    with an empty bin fails. See FINDINGS.md. Preserved as-is.

    The quantity is detrended against phi1 with a cubic polynomial before
    binning, so `mean_<name>` is a residual mean rather than a raw mean.
    """

    if phi2 is not None:
        if quantity is not None:
            raise TypeError(
                "Received both 'quantity' and the deprecated 'phi2'. Pass only one."
            )
        warnings.warn(
            "The 'phi2' keyword is deprecated, use 'quantity'. The function is "
            "generic in that argument.",
            DeprecationWarning,
            stacklevel=2,
        )
        quantity = phi2
    elif quantity is None:
        raise TypeError(
            "compute_local_binned_stats() missing required argument 'quantity'"
        )

    phi1 = np.asarray(phi1)
    quantity = np.asarray(quantity)
    assert phi1.shape == quantity.shape, "phi1, phi2 must have same length"

    # Work on copies so inputs not mutated
    phi1_local = phi1.copy()
    phi2_local = quantity.copy()

    # Detrend phi2
    # A cubic in phi1 removes the large-scale curvature of the track, so that the
    # per-bin distributions measure local scatter rather than global shape. The
    # order drops for very short inputs, and a failed fit is silently skipped.
    if phi1_local.size >= 2:
        try:
            polynom_order = min(3, phi1_local.size - 1)
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
        if scheme == "bin_size":
            left, right = np.nanmin(phi1_local), np.nanmax(phi1_local)
            if right <= left:
                right = left + bin_size
            bin_edges = np.arange(left, right + bin_size, bin_size)
        elif scheme == "equal_count":
            N = len(phi1_local)
            n_bins = max(1, N // min_particles)
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            bin_edges = np.quantile(phi1_local, qs)
        else:
            raise ValueError("scheme must be 'bin_size' or 'equal_count'")
    else:
        bin_edges = np.asarray(bin_edges)

    n_bins = len(bin_edges) - 1

    if n_bins <= 0:
        raise ValueError("No bins determined (check bin_edges / bin_size).")

    # ---- Custom statistic function ----
    def compute_bin_stats(values: np.ndarray) -> list:
        if len(values) == 0:
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        n = len(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if n > 1 else 0.0

        # Standardize
        z_vals = (values - mean_val) / (std_val if std_val > 0 else 1.0)

        # KS test
        try:
            ks_res = kstest(z_vals, "norm", args=(0, 1))
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
    temp_df = pd.DataFrame({"phi1": phi1_local, "phi2": phi2_local})
    # Assign each row to a bin based on the pre-calculated bin_edges
    temp_df["bin"] = pd.cut(temp_df["phi1"], bins=bin_edges, include_lowest=True)
    # Group by the bins and apply the custom function to the 'phi2' values in each group
    stats_per_bin = temp_df.groupby("bin", observed=False)["phi2"].apply(compute_bin_stats)

    column_names = [
        "count",
        f"mean_{quantity_name}",
        f"std_{quantity_name}",
        "ks_stat_vs_norm",
        "ks_p_vs_norm",
        "wass_vs_norm",
        "wass_vs_global",
    ]

    df = pd.DataFrame(stats_per_bin.tolist(), index=stats_per_bin.index, columns=column_names)
    # Add the bin boundaries to the DataFrame for plotting and analysis
    df["bin_left"] = df.index.map(lambda x: x.left)
    df["bin_right"] = df.index.map(lambda x: x.right)
    df = df.reset_index(drop=True)  # Drop the IntervalIndex for a clean default index

    if not plot_streambins:
        return df  # return the results without plotting.

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
        fig_w = max(6, figsize_per_bin[0] * n_bins)
        fig_h = figsize_per_bin[1]

        fig, axs = plt.subplots(
            1, n_bins, figsize=(fig_w, fig_h), squeeze=False, sharey=True, dpi=300
        )
        plt.subplots_adjust(wspace=0.02)
        axs = axs[0]

    phi2_grid_res = 200

    for i, (bin_interval, points_in_bin) in enumerate(temp_df.groupby("bin", observed=False)):
        ax = axs[i]
        # Get bin edges directly from the pandas Interval object
        left, right = bin_interval.left, bin_interval.right
        ax.set_xlim(left, right)
        ax.set_ylim(y_min_all, y_max_all)

        n = len(points_in_bin)
        if n > 0:
            if scatter_kwargs is None:
                scatter_kwargs = dict(
                    s=5,
                    alpha=0.18,
                    edgecolors="none",
                    linewidths=0,
                    facecolor=plt.rcParams["text.color"],
                )
            ax.scatter(points_in_bin["phi1"], points_in_bin["phi2"], **scatter_kwargs)

        if n >= 3:
            try:
                kde2 = gaussian_kde(points_in_bin["phi2"])
                if kde_bandwidth is not None:
                    kde2.set_bandwidth(kde2.factor * kde_bandwidth)

                y_grid = np.linspace(y_min_all, y_max_all, phi2_grid_res)
                dens_y = kde2(y_grid)
                dens_norm = dens_y / dens_y.max() if dens_y.max() > 0 else dens_y
                max_horiz_offset = (right - left) * phi2_violin_width_frac
                x_right = 0.5 * (left + right) + dens_norm * max_horiz_offset
                x_left = 0.5 * (left + right) - dens_norm * max_horiz_offset
                ax.plot(x_right, y_grid, linewidth=phi2_linewidth, color=phi2_color, alpha=0.95)
                ax.plot(x_left, y_grid, linewidth=phi2_linewidth, color=phi2_color, alpha=0.95)
            except Exception:
                pass

        if kde1_global is not None:
            x_grid = np.linspace(left, right, 200)
            dens_x = kde1_global(x_grid)
            y_curve = y_min_all + (dens_x / max_phi1_kde) * (phi1_curve_height_frac * y_span)
            if phi1_curve_height_frac > 0:
                ax.plot(x_grid, y_curve, linewidth=phi1_linewidth, color=phi1_color, alpha=0.95)

        if show_counts:
            ax.text(
                0.02,
                0.94,
                f"$\\mathbf{{N_{{star}}}}$ = {n}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
            )
    if title:
        fig.suptitle(title)
    if axis_labels:
        axs[0].set_ylabel(y_axis_label, fontsize=16)
        fig.supxlabel(x_axis_label, fontsize=16, y=-0.15)
    return df  # @, fig, axs


def compute_bin_diagnostics(
    df: pd.DataFrame,
    ddof: int = 1,
    quantity_name: str = "phi2",
) -> tuple[float, float, float, float, float, float, float]:
    """
    Compute simplistic metrics from the statistical tests performed in localized bins.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with expected columns: ['wass_vs_norm', 'ks_p_vs_norm', 'count',
        'std_<quantity_name>']. Missing columns will be filled with NaN values.
        Modified in place, see Notes.
    ddof : int, default 1
        Degrees of freedom to use for standard deviation of standard deviations.
        Use 1 (default) to treat bin-level std values as a sample from a population.
    quantity_name : str, default 'phi2'
        Suffix identifying the per-bin standard-deviation column, which must
        match the `quantity_name` used to build `df`. Default reproduces the
        historical column name `std_phi2`.

    Returns
    -------
    tuple of float
        A 7-tuple containing:

        - median_norm_wass : float
            Median normalized Wasserstein distance w.r.t 95% threshold
        - max_norm_wass : float
            Maximum normalized Wasserstein distance w.r.t 95% threshold
        - frac_flag_wass : float
            Fraction of bins flagged as Gaussian by Wasserstein test (< 1.0 threshold)
        - frac_consecutive_wass : float
            Fraction of bins flagged as Gaussian by Wasserstein test in continuation
            (< 1.0 threshold), i.e., largest section of smooth Gaussian stream.
        - frac_p_gt_005 : float
            Fraction of bins flagged as Gaussian by KS test (p > 0.05)
        - mean_std_bins : float
            Mean of standard deviations across all bins (average physical width)
        - std_std_bins : float
            Standard deviation of standard deviations in bins (global variation in
            the binned quantity)

    Notes
    -----
    - Normalized W.D are meaningful: values < 1 indicate bins are Gaussian at 95%
      confidence.

      * Median captures the averaged stream-wide property
      * Max captures the largest perturbation

    - Coefficient of variation (std/mean) is a useful relative measure.
      We leave it to the user to compute relevant quantities from mean and std of
      bin stds. That ratio is the width variation metric ``C_w`` of the paper.

    - ddof parameter: bin-level std values are treated as a sample from a population
      of possible bin std values, hence ddof=1 for unbiased estimation.

    - This function writes four columns onto `df` in place, ``wass_thresh_95``,
      ``norm_wass_to_95``, ``flag_wass_under_95``, and fills any missing expected
      column with NaN. It takes no defensive copy. Pass ``df.copy()`` if the
      caller's frame must not change. Reported in FINDINGS.md, preserved as-is.

    - The docstring of the original described a 6-tuple. The code has always
      returned 7 values. The documentation above now matches the code.
    """

    std_col = f"std_{quantity_name}"

    # Ensure relevant columns exist
    for col in ["wass_vs_norm", "ks_p_vs_norm", "count", std_col]:
        if col not in df.columns:
            df[col] = np.nan

    # Adds wass_thresh_95, norm_wass_to_95 and flag_wass_under_95 in place.
    _normalize_wasserstein(df)

    # Overall fraction flagged by Wasserstein
    if df["flag_wass_under_95"].notna().sum() > 0:
        frac_flag_wass = float(
            np.sum(df["flag_wass_under_95"].fillna(False)) / df["flag_wass_under_95"].notna().sum()
        )
        frac_consecutive_wass = _max_consecutive_true_fraction(df["flag_wass_under_95"])
    else:
        frac_flag_wass = np.nan
        frac_consecutive_wass = np.nan

    # Null hypothesis can't be rejected at 5% significance level bins
    ks_pvals = df["ks_p_vs_norm"].values
    # Fraction of p-values > 0.05 (bins consistent with normal)
    valid_p = ~np.isnan(ks_pvals)
    if valid_p.sum() > 0:
        frac_p_gt_005 = float(np.sum(ks_pvals[valid_p] > 0.05) / valid_p.sum())
    else:
        frac_p_gt_005 = np.nan

    # Metrics to return for printing/flushing/saving
    median_norm_wass = float(df["norm_wass_to_95"].median())  # Use .median() - insensitive to outliers
    max_norm_wass = float(df["norm_wass_to_95"].max())
    std_std_bins = float(
        df[std_col].std(ddof=ddof)
    )  # Unbiased sample standard deviation estimator with ddof=1 by default
    mean_std_bins = float(df[std_col].mean())  # Use .mean() for average physical width

    # Code for coef of variation of W.D metric and a spatial coherence metric.
    # # spatial coherence can be described as number of consecutive gaussian bins or fraction of stream length . . .
    # # coef of variation σ/μ consistency of non-Gaussianity. uniform non gaussian versus (lower) very non gaussian (higher)
    # Max WD: "How bad is the worst perturbation?"
    # Median WD: "What's the typical level of non-Gaussianity?"
    # CV of WD: "How variable are the perturbation strengths?" (not spatial coherence)
    # Consecutive Gaussian bins: "How spatially coherent/fragmented is the stream?"

    return (
        median_norm_wass,
        max_norm_wass,
        frac_flag_wass,
        frac_consecutive_wass,
        frac_p_gt_005,
        mean_std_bins,
        std_std_bins,
    )
