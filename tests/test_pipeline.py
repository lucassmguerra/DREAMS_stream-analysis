# -*- coding: utf-8 -*-
"""
End-to-end tests for ``build_metrics_table``.

The contract here is ``tools/pipeline_ref.py``, which reproduces Arpit's
``return_calc_props_df`` statement for statement against the frozen source. The
new driver must match it column for column, dtype for dtype, index and all, at
``method="ratio95"``, which is the setting the original hardcoded.

Also covers the index-alignment behavior that the original did not have, and a
cross-check against the metric values published in the fixture HDF5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline_ref import pipeline_inputs, reference_metrics_table


@pytest.fixture(scope="module")
def inputs(api, fixtures):
    """Pipeline arguments for the unperturbed state, with a RangeIndex orbit table."""
    return pipeline_inputs(api, fixtures, "unperturb")


@pytest.fixture(scope="module")
def reference(api, inputs):
    """The reference table, as the original driver would have produced it."""
    return reference_metrics_table(api, method="ratio95", **inputs)


@pytest.mark.slow
def test_matches_reference_driver(api, inputs, reference):
    """build_metrics_table reproduces return_calc_props_df exactly."""
    new = api.build_metrics_table(
        inputs["Phi1"],
        inputs["Phi2"],
        inputs["masks"],
        inputs["L"],
        inputs["W"],
        inputs["vel_std_tot"],
        inputs["df_orbits"],
        method="ratio95",
    )

    assert list(new.columns) == list(reference.columns), "column order differs"
    np.testing.assert_array_equal(new.index.to_numpy(), reference.index.to_numpy())

    # The one deliberate difference. The original set index.name = "stream_index"
    # and then dropped it again through the pd.concat that appends the
    # power-spectrum block, so its in-memory table came out with an unnamed
    # index. build_metrics_table keeps the name. Values, order and dtypes are
    # unaffected. See FINDINGS.md.
    assert new.index.name == "stream_index"
    assert reference.index.name is None

    for col in reference.columns:
        assert new[col].dtype == reference[col].dtype, (
            f"{col}: dtype {new[col].dtype} != reference {reference[col].dtype}"
        )
        np.testing.assert_array_equal(
            new[col].to_numpy(), reference[col].to_numpy(), err_msg=f"column {col}"
        )


@pytest.mark.slow
def test_column_order_matches_declaration(api, inputs):
    """The declared METRIC_COLUMNS is the order actually produced."""
    new = api.build_metrics_table(
        inputs["Phi1"],
        inputs["Phi2"],
        inputs["masks"],
        inputs["L"],
        inputs["W"],
        inputs["vel_std_tot"],
        inputs["df_orbits"],
        method="ratio95",
    )
    assert list(new.columns) == list(api.METRIC_COLUMNS)


@pytest.mark.slow
def test_default_method_reproduces_the_paper(api, inputs, reference):
    """
    Called with no method at all, build_metrics_table matches the paper.

    The default is ratio95, which is what return_calc_props_df passed. See
    FINDINGS.md entry 5 for why band_snr is not the default despite being the
    preference stated in the detect_stream_psd_metrics docstring.
    """
    assert api.SPECTRA.pipeline_method == "ratio95"
    new = api.build_metrics_table(
        inputs["Phi1"],
        inputs["Phi2"],
        inputs["masks"],
        inputs["L"],
        inputs["W"],
        inputs["vel_std_tot"],
        inputs["df_orbits"],
    )
    np.testing.assert_array_equal(
        new["min_lambda_band"].to_numpy(), reference["min_lambda_band"].to_numpy()
    )


@pytest.mark.slow
def test_band_snr_saturates_at_the_trusted_band_edge(api, fixtures):
    """
    Pin the defect that keeps band_snr out of the default. FINDINGS.md entry 5.

    Its min_lambda_band is 1 / f_top_trusted for every stream, so the column
    carries no per-stream information. If this test ever fails, the scan has
    been changed and the default should be revisited.
    """
    for i in range(4):
        phi1, _q, _L = fixtures.track("unperturb", i)
        out = api.compute_linearized_density(
            phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
        )
        psd = api.compute_welch_psd(out["resid"], out["w"])
        res = api.compute_sampling_psd_realizations(
            out["hist_counts"],
            smoothing_sigma_bins=1,
            w=out["w"],
            n_realizations=1000,
            rng_seed=12345,
        )
        detail = api.detect_stream_psd_metrics(
            psd["freqs"],
            psd["psd"],
            w=out["w"],
            psd_all=res["psd_all"],
            method="band_snr",
            nperseg=res["nperseg"],
            detailed=True,
        )
        positive = psd["freqs"][psd["freqs"] > 0]
        f_top_trusted = positive[detail["trusted_mask"]].max()
        assert detail["min_lambda_band"] == pytest.approx(1.0 / f_top_trusted, rel=1e-12)


@pytest.mark.slow
def test_sparse_orbit_index_gives_same_values(api, fixtures, reference):
    """
    A non-RangeIndex orbit table produces the same numbers, keeping its index.

    The fixture orbit table is indexed by the original m12i catalogue IDs,
    [2873, 1131, ...], not by row position. The original driver would have used
    those values to index the stream arrays and gone out of bounds. This one
    addresses the arrays in row order and carries the index through.
    """
    state = fixtures.state("unperturb")
    L, W, masks = fixtures.geometry("unperturb")
    vel_std_tot = api.compute_velocity_dispersion(
        state.xv, masks, detrend=True, phi1=state.phi1, use_numba=False
    )
    df_orbits = fixtures.orbits  # sparse catalogue index

    new = api.build_metrics_table(
        state.phi1, state.phi2, masks, L, W, vel_std_tot, df_orbits, method="ratio95"
    )

    np.testing.assert_array_equal(new.index.to_numpy(), df_orbits.index.to_numpy())
    assert new.index.name == "stream_index"
    for col in reference.columns:
        np.testing.assert_array_equal(
            new[col].to_numpy(), reference[col].to_numpy(), err_msg=f"column {col}"
        )


def test_rejects_row_count_mismatch(api, fixtures):
    state = fixtures.state("unperturb")
    _L, _W, masks = fixtures.geometry("unperturb")
    with pytest.raises(ValueError, match="rows but the stream arrays"):
        api.build_metrics_table(
            state.phi1,
            state.phi2,
            masks,
            np.ones(16),
            np.ones(16),
            np.ones(16),
            fixtures.orbits.iloc[:10],
        )


def test_rejects_duplicate_index(api, fixtures):
    state = fixtures.state("unperturb")
    _L, _W, masks = fixtures.geometry("unperturb")
    df = fixtures.orbits.copy()
    df.index = [0] * len(df)
    with pytest.raises(ValueError, match="duplicate values"):
        api.build_metrics_table(
            state.phi1, state.phi2, masks, np.ones(16), np.ones(16), np.ones(16), df
        )


def test_rejects_non_integer_index(api, fixtures):
    state = fixtures.state("unperturb")
    _L, _W, masks = fixtures.geometry("unperturb")
    df = fixtures.orbits.copy()
    df.index = [f"s{i}" for i in range(len(df))]
    with pytest.raises(ValueError, match="cannot address rows"):
        api.build_metrics_table(
            state.phi1, state.phi2, masks, np.ones(16), np.ones(16), np.ones(16), df
        )


def test_rejects_missing_orbit_column(api, fixtures):
    state = fixtures.state("unperturb")
    _L, _W, masks = fixtures.geometry("unperturb")
    df = fixtures.orbits.drop(columns=["period"])
    with pytest.raises(ValueError, match="missing required columns"):
        api.build_metrics_table(
            state.phi1, state.phi2, masks, np.ones(16), np.ones(16), np.ones(16), df
        )


@pytest.mark.slow
def test_checkpoint_written_when_requested(api, inputs, tmp_path):
    """checkpoint_path replaces the original's hardcoded parquet write."""
    out = tmp_path / "nested" / "checkpoint.parquet"
    api.build_metrics_table(
        inputs["Phi1"],
        inputs["Phi2"],
        inputs["masks"],
        inputs["L"],
        inputs["W"],
        inputs["vel_std_tot"],
        inputs["df_orbits"],
        checkpoint_path=out,
        method="ratio95",
    )
    assert out.exists()
    saved = pd.read_parquet(out)
    assert len(saved) == 16
    assert "median_norm_wass" in saved.columns
    # The power-spectrum block is added after the checkpoint, so it is absent.
    assert "min_lambda_band" not in saved.columns


@pytest.mark.slow
def test_no_checkpoint_by_default(api, inputs, tmp_path, monkeypatch):
    """Nothing is written when checkpoint_path is None."""
    monkeypatch.chdir(tmp_path)
    api.build_metrics_table(
        inputs["Phi1"],
        inputs["Phi2"],
        inputs["masks"],
        inputs["L"],
        inputs["W"],
        inputs["vel_std_tot"],
        inputs["df_orbits"],
        method="ratio95",
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.slow
def test_band_snr_changes_only_min_lambda(api, inputs, reference):
    """
    The two detection methods differ in exactly one column.

    rms_obs, rms_null and excess_power come from the same trusted-band integrals
    either way. Only min_lambda_band depends on the detection scan.
    """
    new = api.build_metrics_table(
        inputs["Phi1"],
        inputs["Phi2"],
        inputs["masks"],
        inputs["L"],
        inputs["W"],
        inputs["vel_std_tot"],
        inputs["df_orbits"],
        method="band_snr",
    )
    for col in reference.columns:
        if col == "min_lambda_band":
            continue
        np.testing.assert_array_equal(
            new[col].to_numpy(), reference[col].to_numpy(), err_msg=f"column {col}"
        )
    assert not np.allclose(
        new["min_lambda_band"].to_numpy(), reference["min_lambda_band"].to_numpy()
    ), "band_snr and ratio95 gave the same min_lambda_band, which is unexpected"


@pytest.mark.slow
def test_published_values_are_recovered(api, inputs, reference, request):
    """
    Cross-check against the values published in the fixture HDF5.

    Reported rather than asserted for the coordinate-dependent columns. The
    fixture coordinates come from ``nbody_streams.coords.generate_stream_coords``
    while the published values came from a different implementation of the same
    transform, and the pole-tilt optimizer lands in a slightly different place.
    So the metrics agree to roughly a part in 1e3, not bitwise.

    The orbit columns pass straight through untouched and are asserted exactly.
    """
    from conftest import FIXTURE_DIR

    published = pd.read_parquet(FIXTURE_DIR / "published_metrics_m12i_unperturb.parquet")

    for col in ["min_pericenter_dist", "max_apocenter_dist", "mean_dist", "period"]:
        np.testing.assert_allclose(
            reference[col].to_numpy(),
            published[col].to_numpy().astype(float),
            rtol=0,
            atol=0,
            err_msg=f"orbit column {col} was altered in transit",
        )

    report = []
    for col in [
        "length_deg",
        "width_deg",
        "median_norm_wass",
        "max_norm_wass",
        "coef_of_var_stds",
        "min_lambda_band",
    ]:
        ours = reference[col].to_numpy(dtype=float)
        theirs = published[col].to_numpy(dtype=float)
        rel = np.nanmax(np.abs(ours - theirs) / np.abs(theirs))
        med = np.nanmedian(np.abs(ours - theirs) / np.abs(theirs))
        report.append(f"    {col:20s} median {med:.2e}   max {rel:.2e}")
        # A loose sanity bound only. This is not a contract check, because the
        # two tables were built on different stream-frame fits. See the
        # docstring above.
        assert rel < 1.0, f"{col} differs from the published value by {rel:.2e}"

    print("\n  Published-value cross-check, unperturbed m12i, relative difference:")
    print("\n".join(report))
