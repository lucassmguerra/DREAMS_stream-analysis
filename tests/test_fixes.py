# -*- coding: utf-8 -*-
"""
Tests for the issues in FINDINGS.md that were fixed.

Each of these changes behavior relative to
``reference/stream_analysis_source.py``, so none of them can have a golden.
They are pinned here instead. Where a fix is narrow, the test also checks that
everything *outside* the fix still matches the frozen source exactly, which is a
stronger statement than a stored blob would make.

The detection-scan fixes, FINDINGS entries 5 and 19, live in
``tests/test_detection.py``. Everything diverging is listed in
``tools/cases.py`` under ``DIVERGENCES``.
"""
from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import cases as case_registry  # noqa: E402


def assert_metrics_equal(got, expected, message=""):
    """
    Compare two metric tuples, treating NaN as equal to NaN.

    Plain tuple equality does not work here. ``float("nan") == float("nan")`` is
    False, and tuple comparison only papers over it when both sides happen to
    hold the same NaN object.
    """
    np.testing.assert_array_equal(np.asarray(got, dtype=float), np.asarray(expected, dtype=float),
                                  err_msg=message)


# ------------------------------------------------- entry 1, empty bins


def test_empty_bins_report_a_zero_count(api):
    """
    An empty bin now yields ``count`` 0 rather than NaN.

    ``compute_bin_stats`` returned six values for an empty bin against seven
    columns. pandas padded the row out to seven with NaN, and because all six
    were already NaN the padding was invisible in the values. The one visible
    effect was ``count`` arriving as NaN, which forced the column to float.
    """
    phi1, quantity = case_registry.synthetic_short_track()
    df = api.compute_local_binned_stats(phi1, quantity)

    empty = df["count"] == 0
    assert empty.sum() > 0, "the short-track fixture should leave some bins empty"
    assert df["count"].dtype == np.int64
    assert df.loc[empty, "mean_phi2"].isna().all()
    assert df.loc[empty, "std_phi2"].isna().all()
    assert df.loc[empty, "wass_vs_norm"].isna().all()


def test_empty_bins_change_nothing_but_the_count(api, sa_ref):
    """
    The fix is confined to the ``count`` column.

    Every other column still matches the frozen source bitwise, including on the
    empty rows.
    """
    phi1, quantity = case_registry.synthetic_short_track()
    new = api.compute_local_binned_stats(phi1, quantity)
    old = sa_ref.compute_local_binned_stats(phi1, quantity)

    assert list(new.columns) == list(old.columns)
    for col in old.columns:
        if col == "count":
            continue
        np.testing.assert_array_equal(
            new[col].to_numpy(), old[col].to_numpy(), err_msg=f"column {col}"
        )

    # And the counts agree wherever the old one was not NaN.
    populated = old["count"].notna()
    np.testing.assert_array_equal(
        new.loc[populated, "count"].to_numpy(),
        old.loc[populated, "count"].to_numpy().astype(np.int64),
    )
    assert old.loc[~populated, "count"].isna().all()


def test_empty_bins_do_not_change_any_metric(api, sa_ref):
    """
    The headline metrics are identical either way.

    ``compute_bin_diagnostics`` already did ``count.fillna(0)``, so a NaN count
    and a zero count were always equivalent downstream. This is what makes the
    fix safe for the published numbers.
    """
    phi1, quantity = case_registry.synthetic_short_track()
    new = api.compute_bin_diagnostics(api.compute_local_binned_stats(phi1, quantity))
    old = sa_ref.compute_bin_diagnostics(sa_ref.compute_local_binned_stats(phi1, quantity))
    assert_metrics_equal(new, old)


def test_real_streams_have_no_empty_bins(api, fixtures):
    """
    The paper's streams are untouched by the fix, because none of them has an
    empty bin at the default eight. Guards the claim rather than assuming it.
    """
    for state in ("unperturb", "perturb", "perturb_xtrm"):
        lengths, _widths, masks = fixtures.geometry(state)
        st = fixtures.state(state)
        for i in range(len(lengths)):
            m = masks[i]
            df = api.compute_local_binned_stats(st.phi1[i, m] / lengths[i], st.phi2[i, m])
            assert (df["count"] > 0).all(), f"{state} stream {i} has an empty bin"


# ------------------------------------ entry 2, numba worker and detrend=False


def test_numba_worker_runs_without_detrending(api):
    """
    ``compute_velocity_dispersion(xv, mask)`` used to raise ``TypingError`` with
    numba installed, because the jitted body indexes ``phi1`` inside a branch
    numba cannot prune and ``phi1`` was None.
    """
    if not api.NUMBA_AVAILABLE:
        pytest.skip("numba not installed, the fallback path was never affected")

    rng = np.random.default_rng(12345)
    xv = rng.standard_normal((4, 500, 6))
    sigma = api.compute_velocity_dispersion(xv, use_numba=True)
    assert sigma.shape == (4,)
    assert np.all(np.isfinite(sigma))


def test_undetrended_backends_agree(api):
    """
    Without detrending the two backends do the same arithmetic in a different
    order, so they agree to floating-point tolerance.

    With detrending they solve the polynomial fit differently and are not
    expected to match, which is why that case is not asserted here.
    """
    if not api.NUMBA_AVAILABLE:
        pytest.skip("numba not installed")

    rng = np.random.default_rng(12345)
    xv = rng.standard_normal((4, 500, 6))
    mask = rng.random((4, 500)) > 0.2
    np.testing.assert_allclose(
        api.compute_velocity_dispersion(xv, mask, use_numba=True),
        api.compute_velocity_dispersion(xv, mask, use_numba=False),
        rtol=1e-12,
    )


def test_numba_empty_mask_returns_zero(api):
    """
    A mask selecting nothing gives 0.0 from the numba worker and NaN from the
    NumPy one. That disagreement is in the frozen source and is preserved. Only
    the compile failure was fixed.
    """
    if not api.NUMBA_AVAILABLE:
        pytest.skip("numba not installed")

    xv = np.zeros((3, 100, 6))
    mask = np.zeros((3, 100), dtype=bool)
    np.testing.assert_array_equal(api.compute_velocity_dispersion(xv, mask, use_numba=True), np.zeros(3))
    np.testing.assert_array_equal(
        api.compute_velocity_dispersion(xv, mask, use_numba=False), np.full(3, np.nan)
    )


def test_detrended_dispersion_still_matches_the_frozen_source(api, sa_ref, fixtures):
    """
    Removing the unread ``x`` buffer from the jitted body, FINDINGS entry 18,
    did not perturb the numbers. Also covered bitwise by the contract suite.
    """
    st = fixtures.state("unperturb")
    _lengths, _widths, masks = fixtures.geometry("unperturb")
    np.testing.assert_array_equal(
        api.compute_velocity_dispersion(st.xv, masks, detrend=True, phi1=st.phi1, use_numba=True),
        sa_ref.compute_velocity_dispersion(
            st.xv, masks, detrend=True, phi1=st.phi1, use_numba=True
        ),
    )


# --------------------------------------------- entry 3, unbound pos_mask_all


def test_strictly_positive_frequencies_with_realizations(api):
    """
    ``pos_mask_all`` was bound only when the input carried a non-positive
    frequency, but was used unconditionally to slice ``psd_all``. A caller
    passing a DC-free frequency vector hit ``NameError``.
    """
    rng = np.random.default_rng(12345)
    freqs = np.linspace(0.1, 2.0, 32)
    psd = rng.random(32)
    psd_all = rng.random((50, 32))

    for method in ("peak_snr", "band_snr", "ratio95"):
        result = api.detect_stream_psd_metrics(
            freqs, psd, w=0.25, psd_all=psd_all, method=method
        )
        assert len(result) == 4
        assert np.isfinite(result[0])


def test_dc_bin_still_stripped(api):
    """The DC-carrying path is unchanged. A zero frequency is still dropped."""
    rng = np.random.default_rng(12345)
    with_dc = np.linspace(0.0, 2.0, 33)
    psd = rng.random(33)
    psd_all = rng.random((50, 33))
    detail = api.detect_stream_psd_metrics(
        with_dc, psd, w=0.25, psd_all=psd_all, method="peak_snr", detailed=True
    )
    assert detail["freqs"].size == 32
    assert (detail["freqs"] > 0).all()


# ------------------------------------ entry 4, UnboundLocalError on `_`


def test_zero_point_monte_carlo_returns_nan(api):
    """
    ``return np.nan, _`` referenced a local bound only by a later loop, so the
    branch raised ``UnboundLocalError``. A sample of zero draws has no
    Wasserstein distance, so NaN is the answer.
    """
    result = api.wasserstein_null_threshold(power_law_fit=False, num_points=0)
    assert isinstance(result, float)
    assert np.isnan(result)


def test_monte_carlo_branch_otherwise_unchanged(api, sa_ref):
    """The seeded Monte Carlo path is untouched."""
    kwargs = dict(
        power_law_fit=False, num_points=50, trials=25, normal_ref_size=500, normal_ref_seed=12345
    )
    assert api.wasserstein_null_threshold(**kwargs) == sa_ref.wasserstein_null_threshold(**kwargs)


# ---------------------------------------------- entry 7, bare @njit shim


def test_fallback_njit_accepts_both_decoration_forms():
    """
    The fallback shim was ``njit(nopython=True, cache=True, parallel=True)``, so
    a bare ``@njit`` passed the function in as ``nopython`` and returned the
    inner decorator instead of the function.

    Tested against the shim directly, so it runs on a machine that has numba.
    """
    from stream_analysis._numba import _fallback_njit

    @_fallback_njit
    def bare(x):
        return x + 1

    @_fallback_njit(parallel=True, cache=True, nopython=True)
    def called(x):
        return x + 2

    @_fallback_njit()
    def called_empty(x):
        return x + 3

    assert bare(1) == 2
    assert called(1) == 3
    assert called_empty(1) == 4


# ------------------------------------- entry 8, in-place mutation of the input


def test_diagnostics_leaves_the_caller_frame_alone(api):
    """
    ``compute_bin_diagnostics`` wrote its working columns onto the caller's
    frame. ``summarize_bin_metrics`` took a copy. The two disagreed.
    """
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 4000))
    df = api.compute_local_binned_stats(phi1, rng.standard_normal(4000))

    before_columns = list(df.columns)
    before_values = df.copy()

    api.compute_bin_diagnostics(df)

    assert list(df.columns) == before_columns, "compute_bin_diagnostics added columns in place"
    pd.testing.assert_frame_equal(df, before_values)


def test_diagnostics_is_idempotent(api):
    """
    Calling it twice on one frame now gives the same answer twice. It used to
    operate on a frame it had already modified.
    """
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 4000))
    df = api.compute_local_binned_stats(phi1, rng.standard_normal(4000))
    assert_metrics_equal(api.compute_bin_diagnostics(df), api.compute_bin_diagnostics(df))


def test_diagnostics_returns_the_same_numbers(api, sa_ref, fixtures):
    """The metrics themselves are unchanged. Only the side effect is gone."""
    for i in (0, 6, 10):
        phi1, quantity, length = fixtures.track("unperturb", i)
        new = api.compute_bin_diagnostics(
            api.compute_local_binned_stats(phi1 / length, quantity)
        )
        old = sa_ref.compute_bin_diagnostics(
            sa_ref.compute_local_binned_stats(phi1 / length, quantity)
        )
        assert_metrics_equal(new, old, f"stream {i}")


def test_diagnostics_still_tolerates_missing_columns(api, sa_ref):
    """
    Missing expected columns are treated as NaN rather than written onto the
    caller's frame, and the returned metrics are unchanged.

    Not every metric comes back NaN here. A NaN Wasserstein compares False
    against the threshold rather than propagating, so the two flagged fractions
    come out 0.0. That is the frozen source's behavior and is preserved.
    """
    df = pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]})
    result = api.compute_bin_diagnostics(df)
    assert list(df.columns) == ["unrelated"], "the caller's frame was modified"

    old_df = pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]})
    assert_metrics_equal(result, sa_ref.compute_bin_diagnostics(old_df))
    assert set(old_df.columns) > {"unrelated"}, "the frozen source used to mutate in place"


# ------------------------------------------- entry 9, max_bins_to_plot ignored


def test_max_bins_to_plot_is_honoured(api):
    """
    The parameter was accepted and never read, so a 200-bin request drew 200
    panels.
    """
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 8000))
    quantity = rng.standard_normal(8000)

    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        api.compute_local_binned_stats(
            phi1, quantity, bin_size=1 / 40.0, plot_streambins=True, max_bins_to_plot=6
        )
    assert len(plt.gcf().axes) == 6
    plt.close("all")


def test_plotting_does_not_change_the_table(api):
    """The returned DataFrame is the same whether or not the figure is drawn."""
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 4000))
    quantity = rng.standard_normal(4000)

    quiet = api.compute_local_binned_stats(phi1, quantity)
    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drawn = api.compute_local_binned_stats(phi1, quantity, plot_streambins=True)
    plt.close("all")

    pd.testing.assert_frame_equal(quiet, drawn)


def test_uncapped_plot_draws_every_bin(api):
    """Below the cap, every bin still gets a panel."""
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 4000))

    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = api.compute_local_binned_stats(
            phi1, rng.standard_normal(4000), plot_streambins=True
        )
    assert len(plt.gcf().axes) == len(df)
    plt.close("all")
