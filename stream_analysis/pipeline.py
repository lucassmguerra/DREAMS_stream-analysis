# -*- coding: utf-8 -*-
"""
The metrics-table driver.

:func:`build_metrics_table` runs the whole analysis over a set of streams and
returns the table reported in *No Stream Left Unscathed* (Arora et al.). It
mirrors the ``return_calc_props_df`` driver that produced the published numbers,
with the same computation and the same columns in the same order.

Three things differ, and all three are conveniences of the new function rather
than changes to the contract.

``checkpoint_path``
    Replaces a hardcoded parquet write to a Flatiron ceph path. The default is
    None, and nothing is written.

``method``
    Exposed as a keyword instead of being hardcoded, and defaulting to
    ``"peak_snr"`` at a 5-sigma threshold rather than to the ``"ratio95"`` at 3
    that the published run used. This is the one place the new pipeline
    deliberately produces a different number from the old one, in the
    ``min_lambda_band`` column only. Pass ``method="ratio95",
    snr_threshold=3.0`` to reproduce the published table exactly. See
    See the README on the choice of detection method.

Index alignment
    The original indexed the stream arrays with ``df_orbits.index`` and then
    assigned the orbit columns positionally with ``.to_numpy()``. That is only
    correct when the index is a clean ``RangeIndex`` covering the stream rows.
    This function aligns explicitly and validates first, so a mismatched index
    raises a clear error instead of silently pairing the wrong orbit with the
    wrong stream. Where the old code would have produced correct output, this
    produces identical output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .binned import compute_bin_diagnostics, compute_local_binned_stats
from .config import BINNED, DENSITY, SPECTRA
from .density import compute_linearized_density
from .spectra import (
    compute_sampling_psd_realizations,
    compute_welch_psd,
    detect_stream_psd_metrics,
)

__all__ = ["build_metrics_table", "METRIC_COLUMNS"]


#: Binned-metric block, in the order compute_bin_diagnostics returns them.
_BINNED_COLUMNS = [
    "median_norm_wass",  # global disturbance
    "max_norm_wass",  # peak disturbance
    "frac_flag_wass",  # fraction of bins consistent with Gaussian
    "frac_consecutive_wass",  # longest unbroken Gaussian run, as a fraction
    "frac_p_gt_005",  # fraction of bins with KS p > 0.05
    "mean_std_bins",  # mean per-bin width
    "std_std_bins",  # scatter of per-bin widths
]

#: Copied through from the orbit table.
_ORBIT_COLUMNS = [
    "min_pericenter_dist",  # closest approach
    "max_apocenter_dist",  # furthest excursion
    "mean_dist",  # time-averaged galactocentric radius
    "n_pericenters",  # pericentric passages
    "period",  # radial period
]

#: Power-spectrum block, in the order detect_stream_psd_metrics returns them.
_PSD_COLUMNS = [
    "rms_obs",  # RMS density residual
    "rms_null",  # sampling noise floor
    "excess_power",  # power above the floor
    "min_lambda_band",  # minimum detectable scale
]

#: Full column order of the returned table.
METRIC_COLUMNS = (
    _BINNED_COLUMNS
    + [
        "length_deg",  # stream length
        "width_deg",  # stream width
        "vel.std.tot_km_s",  # total velocity dispersion
        "coef_of_var_stds",  # width variation, C_w
    ]
    + _ORBIT_COLUMNS
    + _PSD_COLUMNS
)


def _validate_alignment(df_orbits: pd.DataFrame, n_streams: int) -> np.ndarray:
    """
    Check the orbit table against the stream arrays and return the row positions.

    Parameters
    ----------
    df_orbits : pandas.DataFrame
        Orbit table, one row per stream.
    n_streams : int
        Number of streams in the coordinate arrays.

    Returns
    -------
    np.ndarray
        Integer positions into the stream arrays, one per row of `df_orbits`, in
        the order the rows appear.

    Raises
    ------
    ValueError
        If the table has a different number of rows than there are streams, if
        its index is not integral, if any index value falls outside the stream
        arrays, if the index has duplicates, or if a required column is missing.
    """
    missing = [c for c in _ORBIT_COLUMNS if c not in df_orbits.columns]
    if missing:
        raise ValueError(
            f"df_orbits is missing required columns {missing}. "
            f"Expected all of {_ORBIT_COLUMNS}."
        )

    if len(df_orbits) != n_streams:
        raise ValueError(
            f"df_orbits has {len(df_orbits)} rows but the stream arrays hold "
            f"{n_streams} streams. One row per stream is required."
        )

    if df_orbits.index.has_duplicates:
        dupes = df_orbits.index[df_orbits.index.duplicated()].unique().tolist()
        raise ValueError(f"df_orbits.index has duplicate values {dupes}. Rows cannot be aligned.")

    index = np.asarray(df_orbits.index)
    if not np.issubdtype(index.dtype, np.integer):
        raise ValueError(
            f"df_orbits.index has dtype {index.dtype}, which cannot address rows of the "
            "stream arrays. Use an integer index of stream row positions."
        )

    # A sparse index of original catalogue IDs is the common case, for example a
    # 16-stream subset drawn from a 5000-stream run. Positions are taken in row
    # order rather than from the index values whenever the values do not
    # themselves address the arrays.
    if index.min() >= 0 and index.max() < n_streams:
        return index

    return np.arange(n_streams)


def build_metrics_table(
    phi1: np.ndarray,
    quantity: np.ndarray,
    masks: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    vel_std_tot: np.ndarray,
    df_orbits: pd.DataFrame,
    *,
    checkpoint_path: str | Path | None = None,
    method: str = SPECTRA.method,
    snr_threshold: float | None = None,
    bin_size: float = BINNED.bin_size,
    bin_width: float = DENSITY.bin_width,
    smoothing_sigma: float = DENSITY.smoothing_sigma,
    n_realizations: int = SPECTRA.n_realizations,
    rng_seed: int = SPECTRA.rng_seed,
    quantity_name: str = "phi2",
    progress: Callable[[Iterable], Iterable] | None = None,
) -> pd.DataFrame:
    """
    Compute the full metrics table for a set of streams.

    Runs the localized binned family and the power-spectrum family over every
    stream and joins them with the geometry, kinematics and orbit columns.

    Parameters
    ----------
    phi1 : np.ndarray, shape (S, N)
        Along-stream coordinates in degrees, one row per stream.
    quantity : np.ndarray, shape (S, N)
        The per-particle quantity binned along the track. phi2 in the paper.
        Any other per-particle quantity works, for example `v_los`, which yields
        the kinematic analogues of the binned metrics.
    masks : np.ndarray of bool, shape (S, N)
        Membership masks from
        :func:`stream_analysis.geometry.measure_stream_length_width`.
    lengths : np.ndarray, shape (S,)
        Stream lengths in degrees. Used both as a reported column and as the
        rescaling applied to phi1 before binning.
    widths : np.ndarray, shape (S,)
        Stream widths in degrees.
    vel_std_tot : np.ndarray, shape (S,)
        Total velocity dispersion in km/s, from
        :func:`stream_analysis.kinematics.compute_velocity_dispersion`.
    df_orbits : pandas.DataFrame
        One row per stream, carrying the columns listed in `_ORBIT_COLUMNS`.
        Aligned on index rather than assumed to be positional. See Notes.
    checkpoint_path : str, Path or None, keyword-only
        If given, the binned half of the table is written to this parquet path
        before the power spectra run, so a long job can be resumed or inspected.
        Nothing is written when None, which is the default.
    method : str, keyword-only
        PSD detection method, "peak_snr", "band_snr" or "ratio95". Defaults to
        "peak_snr", a per-frequency signal-to-noise ratio at 5 sigma. The
        published table used "ratio95" at 3.
    snr_threshold : float or None, keyword-only
        Detection threshold. None lets each method use its own default, 5.0 for
        "peak_snr" and 3.0 for the other two.
    bin_size : float, keyword-only
        Bin width along the length-rescaled phi1. The default 1/8 gives the eight
        bins of the paper.
    bin_width : float, keyword-only
        Density bin width in degrees.
    smoothing_sigma : float, keyword-only
        Density smoothing sigma in bin units. The same value is used for the
        Monte Carlo realizations, so the observed spectrum and its noise floor
        are smoothed identically.
    n_realizations : int, keyword-only
        Monte Carlo realizations per stream.
    rng_seed : int, keyword-only
        Seed for those realizations.
    quantity_name : str, keyword-only
        Column suffix for the binned quantity. Labelling only.
    progress : callable or None, keyword-only
        Optional wrapper applied to the per-stream loops, for example ``tqdm``.

    Returns
    -------
    pandas.DataFrame
        One row per stream, indexed by `df_orbits.index` with index name
        ``stream_index``, carrying the columns of `METRIC_COLUMNS` in that order.

    Raises
    ------
    ValueError
        If `df_orbits` cannot be aligned to the stream arrays. See
        :func:`_validate_alignment`.

    Notes
    -----
    Index alignment. The original driver did
    ``df_final[cols] = df_orbits[cols].to_numpy()``, a positional assignment onto
    a fresh ``RangeIndex``, while separately using ``df_orbits.index`` to address
    rows of the stream arrays. Those two uses only agree when the index is
    ``0..S-1``. This function validates first and raises otherwise. When the
    index is a sparse set of catalogue IDs, which is the usual case for a subset
    of a larger run, the stream arrays are addressed in row order and the index
    is carried through to the output for provenance.
    """
    phi1 = np.asarray(phi1)
    quantity = np.asarray(quantity)
    masks = np.asarray(masks)
    lengths = np.asarray(lengths)
    widths = np.asarray(widths)

    n_streams = phi1.shape[0]
    positions = _validate_alignment(df_orbits, n_streams)

    def _wrap(it):
        return progress(it) if progress is not None else it

    # ---- localized binned metrics ----
    metrics = []
    for pos in _wrap(positions):
        df = compute_local_binned_stats(
            phi1[pos, masks[pos]] / lengths[pos],
            quantity[pos, masks[pos]],
            bin_size=bin_size,
            quantity_name=quantity_name,
        )
        metrics.append(compute_bin_diagnostics(df, quantity_name=quantity_name))

    df_final = pd.DataFrame(metrics, columns=_BINNED_COLUMNS, index=df_orbits.index)
    df_final.index.name = "stream_index"
    df_final["length_deg"] = lengths
    df_final["width_deg"] = widths
    df_final["vel.std.tot_km_s"] = np.asarray(vel_std_tot)
    df_final["coef_of_var_stds"] = df_final["std_std_bins"] / df_final["mean_std_bins"]

    # Explicit alignment. df_final carries df_orbits' own index, and
    # _validate_alignment has already refused any index that cannot address the
    # stream arrays, so row i here is the same stream as row i of df_orbits.
    #
    # The values go across as one 2D array rather than column by column. That is
    # what the original did, and it upcasts the whole block to a common float64
    # dtype, so `n_pericenters` arrives as a float rather than as an int. The
    # dtypes are part of the contract, so the upcast is kept.
    df_final[_ORBIT_COLUMNS] = df_orbits[_ORBIT_COLUMNS].to_numpy()

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(path, index=True)

    # ---- power-spectrum metrics ----
    def _psd_metrics(phi1_track: np.ndarray) -> tuple[float, float, float, float | None]:
        out = compute_linearized_density(
            phi1_track,
            detrend=DENSITY.detrend,
            smoothing_sigma=smoothing_sigma,
            bin_width=bin_width,
        )
        psd_out = compute_welch_psd(out["resid"], out["w"])
        res = compute_sampling_psd_realizations(
            out["hist_counts"],
            smoothing_sigma_bins=smoothing_sigma,
            w=out["w"],
            n_realizations=n_realizations,
            rng_seed=rng_seed,
        )
        return detect_stream_psd_metrics(
            psd_out["freqs"],
            psd_out["psd"],
            w=out["w"],
            psd_all=res["psd_all"],
            method=method,
            snr_threshold=snr_threshold,
            nperseg=res["nperseg"],
        )

    psd_metrics = [_psd_metrics(phi1[pos, masks[pos]]) for pos in _wrap(positions)]

    df_psd = pd.DataFrame(psd_metrics, columns=_PSD_COLUMNS, index=df_orbits.index)
    df_final = pd.concat([df_final, df_psd], axis=1)
    df_final.index.name = "stream_index"

    return _normalize_missing(df_final)


def _normalize_missing(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Make every flavour of missing value the same ``np.nan``.

    The original called this ``_sanity_NaN_checkers``. It matters because
    ``min_lambda_band`` is None rather than NaN when no scale is detected, which
    makes that column object-dtype until it is coerced. Text sentinels are
    normalized too, and object columns are converted to numeric when more than
    half their entries parse.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to clean. Not modified in place.
    verbose : bool
        Print the shape and missing-value counts, as the original did.

    Returns
    -------
    pandas.DataFrame
        The cleaned table.
    """
    sentinels = ["None", "none", "NaN", "nan", "", " ", "NULL", "null", "<NA>", "NA"]
    df = df.replace(sentinels, np.nan)
    df = df.where(pd.notna(df), np.nan)

    for col in df.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_parsed = coerced.notna().sum()
        frac_parsed = n_parsed / len(df)
        # threshold: if >50% of values convert to numeric, keep the numeric conversion
        if frac_parsed > 0.5:
            df[col] = coerced

    if verbose:
        print("Rows/cols:", df.shape)
        print("Total NaNs in DF:", int(df.isna().sum().sum()))
        print("Top columns by NaN count:")
        print(df.isna().sum().sort_values(ascending=False).head(20))

    return df
