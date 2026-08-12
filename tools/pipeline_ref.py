# -*- coding: utf-8 -*-
"""
Reference re-implementation of ``return_calc_props_df``.

The pipeline driver never lived in ``stream_analysis.py``. It lives in
``Stream_morphology_analysis.py`` in the simulation repository, and it imports
agama, zarr, seaborn and a private utility module, none of which belong in
this package or its tests.

So the end-to-end contract is defined here instead. This module reproduces that
function statement for statement against whatever ``api`` it is handed, minus
the three things that are environment rather than computation: the ``tqdm``
progress bars, the hardcoded parquet write to a cluster scratch path, and the
diagnostic prints. ``stream_analysis.pipeline.build_metrics_table`` must match
its output column for column, dtype for dtype, index and all.

The one knob is ``method``. The original hardcodes ``'ratio95'`` in its inner
``_compute_PSD_etc_metrics``, and this module keeps that as its default, so the
equality check against the original's behavior needs no argument.
``build_metrics_table`` defaults to ``'peak_snr'`` instead, which is the one
column where the two intentionally differ.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Column order of the binned-metric block, verbatim from the original.
BINNED_COLUMNS = [
    "median_norm_wass",
    "max_norm_wass",
    "frac_flag_wass",
    "frac_consecutive_wass",
    "frac_p_gt_005",
    "mean_std_bins",
    "std_std_bins",
]

#: Columns copied through from the orbit table, verbatim from the original.
ORBIT_COLUMNS = [
    "min_pericenter_dist",
    "max_apocenter_dist",
    "mean_dist",
    "n_pericenters",
    "period",
]

#: Column names of the power-spectrum block, verbatim from the original.
PSD_COLUMNS = ["rms_obs", "rms_null", "excess_power", "min_lambda_band"]


def _sanity_nan_checker(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    The original's ``_sanity_NaN_checkers``, prints made optional.

    It normalizes textual missing-value sentinels and pandas NA to ``np.nan``,
    then coerces object columns to numeric when more than half the entries
    parse. It is kept because it can change column dtypes, and dtypes are part
    of the contract.
    """
    sentinels = ["None", "none", "NaN", "nan", "", " ", "NULL", "null", "<NA>", "NA"]
    df = df.replace(sentinels, np.nan)
    df = df.where(pd.notna(df), np.nan)

    for col in df.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_parsed = coerced.notna().sum()
        frac_parsed = n_parsed / len(df)
        if frac_parsed > 0.5:
            df[col] = coerced

    if verbose:
        print("Rows/cols:", df.shape)
        print("Total NaNs in DF:", int(df.isna().sum().sum()))
        print("Top columns by NaN count:")
        print(df.isna().sum().sort_values(ascending=False).head(20))
    return df


def reference_metrics_table(
    api: Any,
    Phi1: np.ndarray,
    Phi2: np.ndarray,
    masks: np.ndarray,
    L: np.ndarray,
    W: np.ndarray,
    vel_std_tot: np.ndarray,
    df_orbits: pd.DataFrame,
    method: str = "ratio95",
) -> pd.DataFrame:
    """
    Reproduce ``return_calc_props_df`` against ``api``.

    Parameters mirror the original exactly, apart from the dropped
    ``tempdata_file`` and the added ``method``.
    """
    metrics = []
    for stream_ind in df_orbits.index:
        df = api.compute_local_binned_stats(
            Phi1[stream_ind, masks[stream_ind]] / L[stream_ind],
            Phi2[stream_ind, masks[stream_ind]],
        )
        metrics.append(api.compute_bin_diagnostics(df))

    df_final = pd.DataFrame(metrics, columns=BINNED_COLUMNS)
    df_final.index.name = "stream_index"
    df_final["length_deg"] = L
    df_final["width_deg"] = W
    df_final["vel.std.tot_km_s"] = vel_std_tot
    df_final["coef_of_var_stds"] = df_final["std_std_bins"] / df_final["mean_std_bins"]
    df_final[ORBIT_COLUMNS] = df_orbits[ORBIT_COLUMNS].to_numpy()

    def _compute_psd_etc_metrics(phi1_track):
        out = api.compute_linearized_density(
            phi1_track, detrend="poly", smoothing_sigma=1, bin_width=0.25
        )
        psd_out = api.compute_welch_psd(out["resid"], out["w"])
        res = api.compute_sampling_psd_realizations(
            out["hist_counts"],
            smoothing_sigma_bins=1,
            w=out["w"],
            n_realizations=1000,
            rng_seed=12345,
        )
        return api.detect_stream_psd_metrics(
            psd_out["freqs"],
            psd_out["psd"],
            w=out["w"],
            psd_all=res["psd_all"],
            method=method,
            nperseg=res["nperseg"],
        )

    psd_metrics = [
        _compute_psd_etc_metrics(Phi1[stream_ind, masks[stream_ind]])
        for stream_ind in df_orbits.index
    ]

    df_newcols = pd.DataFrame(psd_metrics, columns=PSD_COLUMNS)
    df_final = pd.concat([df_final, df_newcols], axis=1)

    try:
        return _sanity_nan_checker(df_final)
    except Exception as exc:  # pragma: no cover - matches the original's guard
        print(exc)
        return df_final


def pipeline_inputs(api: Any, fx: Any, state: str) -> dict[str, Any]:
    """
    Assemble the arguments the pipeline takes, from a fixture state.

    The orbit table is re-indexed to a plain positional index here, because the
    original indexes ``Phi1`` with ``df_orbits.index`` directly and so requires
    ``df_orbits.index`` to be row positions into the stream arrays. The
    index-alignment behavior of the new ``build_metrics_table`` is covered by a
    separate test that hands it the real sparse index instead.
    """
    st = fx.state(state)
    L, W, masks = fx.geometry(state)
    vel_std_tot = api.compute_velocity_dispersion(
        st.xv, masks, detrend=True, phi1=st.phi1, use_numba=False
    )
    df_orbits = fx.orbits.reset_index(drop=True)
    return {
        "Phi1": st.phi1,
        "Phi2": st.phi2,
        "masks": masks,
        "L": L,
        "W": W,
        "vel_std_tot": vel_std_tot,
        "df_orbits": df_orbits,
    }
