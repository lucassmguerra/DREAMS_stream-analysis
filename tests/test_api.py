# -*- coding: utf-8 -*-
"""
Self-contained tests for the public API.

Everything here runs on seeded synthetic data. No simulation data, no fixtures,
no network. A fresh clone can run the whole file.

These check that each function is wired correctly, returns the shape and columns
it promises, and holds its stated invariants. They are not a numerical contract.
The bitwise contract against the original single-file implementation, its golden
outputs and the record of the twenty-two issues found and fixed, all live on the
``refactor-provenance`` branch.
"""
from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import stream_analysis as sa

SEED = 12345


# --------------------------------------------------------------- synthetic data


@pytest.fixture(scope="module")
def streams():
    """
    Three synthetic streams, (S, N, 6) phase space plus stream coordinates.

    Stream 0 is smooth. Stream 1 carries a localized bulge in phi2, the signature
    the binned family is meant to catch. Stream 2 carries a density wave, the
    signature the power-spectrum family is meant to catch.
    """
    rng = np.random.default_rng(SEED)
    S, N = 3, 4000

    phi1 = np.sort(rng.uniform(-15.0, 15.0, size=(S, N)), axis=1)
    phi2 = 0.2 * rng.standard_normal((S, N))

    # Stream 1: a localized widening between phi1 = 0 and 5.
    bulge = (phi1[1] > 0.0) & (phi1[1] < 5.0)
    phi2[1, bulge] *= 4.0

    # Stream 2: a density wave. Resample phi1 with a sinusoidal acceptance.
    wave = np.sort(rng.uniform(-15.0, 15.0, size=4 * N))
    keep = rng.random(wave.size) < 0.5 * (1.0 + 0.6 * np.sin(2.0 * np.pi * wave / 3.0))
    selected = wave[keep][:N]
    phi1[2, : selected.size] = selected
    phi1[2] = np.sort(phi1[2])

    # Phase space consistent enough for the kinematics to mean something.
    xv = np.empty((S, N, 6))
    for s in range(S):
        radius = 20.0 + 0.05 * phi1[s]
        angle = np.deg2rad(phi1[s])
        xv[s, :, 0] = radius * np.cos(angle)
        xv[s, :, 1] = radius * np.sin(angle)
        xv[s, :, 2] = phi2[s]
        xv[s, :, 3:] = 100.0 * rng.standard_normal((N, 3)) * 0.01
        xv[s, :, 3] += 0.5 * phi1[s]  # a trend for detrend=True to remove

    return {"phi1": phi1, "phi2": phi2, "xv": xv}


@pytest.fixture(scope="module")
def geometry(streams):
    return sa.measure_stream_length_width(
        streams["phi1"], streams["phi2"], method="kde", density_frac=0.9, return_mask=True
    )


@pytest.fixture(scope="module")
def orbits():
    """Four eccentric Kepler-like orbits, in the shape the function expects."""
    rng = np.random.default_rng(SEED)
    t = np.linspace(0.0, 5.0, 512)
    out = np.empty((4, 2), dtype=object)
    for k in range(4):
        a = 20.0 + 10.0 * rng.random()
        ecc = 0.1 + 0.6 * rng.random()
        theta = 2.0 * np.pi * (1.0 + rng.random()) * t + 2.0 * np.pi * rng.random()
        r = a * (1.0 - ecc**2) / (1.0 + ecc * np.cos(theta))
        coords = np.column_stack(
            [
                r * np.cos(theta),
                r * np.sin(theta),
                0.3 * a * np.sin(0.5 * theta),
                np.gradient(r * np.cos(theta), t),
                np.gradient(r * np.sin(theta), t),
                np.gradient(0.3 * a * np.sin(0.5 * theta), t),
            ]
        )
        out[k, 0] = t[::-1].copy()  # decreasing, as lookback times
        out[k, 1] = coords[::-1]
    return out


# ------------------------------------------------------------------- orbits


def test_orbit_properties_columns(orbits):
    df = sa.compute_orbit_properties(orbits, central_mass=1e12)
    assert list(df.columns) == [
        "min_pericenter_dist",
        "max_apocenter_dist",
        "mean_dist",
        "n_pericenters",
        "period",
        "eccentricity_geo",
        "eccentricity_osculating",
    ]
    assert len(df) == 4
    assert (df["max_apocenter_dist"] > df["min_pericenter_dist"]).all()
    assert (df["n_pericenters"] > 0).all()
    assert df["eccentricity_geo"].between(0.0, 1.0).all()


def test_osculating_eccentricity_needs_a_mass(orbits):
    without = sa.compute_orbit_properties(orbits)
    assert without["eccentricity_osculating"].isna().all()
    with_mu = sa.compute_orbit_properties(orbits, mu=4.3009e-6 * 1e12)
    assert with_mu["eccentricity_osculating"].notna().all()


# ----------------------------------------------------------------- geometry


def test_length_width_shapes(streams, geometry):
    lengths, widths, masks = geometry
    S, N = streams["phi1"].shape
    assert lengths.shape == (S,)
    assert widths.shape == (S,)
    assert masks.shape == (S, N)
    assert masks.dtype == bool
    assert (lengths > 0).all()
    assert (widths > 0).all()
    # The 90% contour keeps most, but not all, particles.
    kept = masks.sum(axis=1) / N
    assert ((kept > 0.5) & (kept <= 1.0)).all()


def test_bulged_stream_is_wider(geometry):
    _lengths, widths, _masks = geometry
    assert widths[1] > widths[0]


def test_single_stream_returns_scalars(streams):
    length, width, mask = sa.measure_stream_length_width(
        streams["phi1"][0], streams["phi2"][0], method="kde"
    )
    assert isinstance(length, float)
    assert isinstance(width, float)
    assert mask.shape == streams["phi1"][0].shape


def test_quantile_method(streams):
    lengths, widths = sa.measure_stream_length_width(
        streams["phi1"], streams["phi2"], method="quantile", return_mask=False
    )
    assert lengths.shape == (3,)
    assert (lengths > 0).all()


def test_unknown_method_rejected(streams):
    with pytest.raises(ValueError, match="quantile.*kde"):
        sa.measure_stream_length_width(streams["phi1"], streams["phi2"], method="nope")


# --------------------------------------------------------------- kinematics


@pytest.mark.parametrize("use_numba", [False, True])
def test_velocity_dispersion_shapes(streams, geometry, use_numba):
    _lengths, _widths, masks = geometry
    sigma = sa.compute_velocity_dispersion(
        streams["xv"], masks, detrend=True, phi1=streams["phi1"], use_numba=use_numba
    )
    assert sigma.shape == (3,)
    assert np.all(np.isfinite(sigma))
    assert (sigma > 0).all()


@pytest.mark.parametrize("use_numba", [False, True])
def test_velocity_dispersion_without_detrending(streams, use_numba):
    sigma = sa.compute_velocity_dispersion(streams["xv"], use_numba=use_numba)
    assert sigma.shape == (3,)
    assert np.all(np.isfinite(sigma))


def test_detrending_removes_the_trend(streams):
    """The synthetic streams carry a linear vx trend along phi1."""
    plain = sa.compute_velocity_dispersion(streams["xv"], use_numba=False)
    detrended = sa.compute_velocity_dispersion(
        streams["xv"], detrend=True, phi1=streams["phi1"], use_numba=False
    )
    assert (detrended < plain).all()


def test_backends_agree_without_detrending(streams):
    if not sa.NUMBA_AVAILABLE:
        pytest.skip("numba not installed")
    np.testing.assert_allclose(
        sa.compute_velocity_dispersion(streams["xv"], use_numba=True),
        sa.compute_velocity_dispersion(streams["xv"], use_numba=False),
        rtol=1e-12,
    )


def test_detrend_without_phi1_rejected(streams):
    with pytest.raises(ValueError, match="phi1 must be provided"):
        sa.compute_velocity_dispersion(streams["xv"], detrend=True)


# ------------------------------------------------------------------- binned


def test_binned_stats_columns_and_bin_count(streams, geometry):
    lengths, _widths, masks = geometry
    m = masks[0]
    df = sa.compute_local_binned_stats(streams["phi1"][0, m] / lengths[0], streams["phi2"][0, m])
    assert list(df.columns) == [
        "count",
        "mean_phi2",
        "std_phi2",
        "ks_stat_vs_norm",
        "ks_p_vs_norm",
        "wass_vs_norm",
        "wass_vs_global",
        "bin_left",
        "bin_right",
    ]
    # phi1 rescaled by length, bin_size 1/8, so eight bins span the stream.
    assert 8 <= len(df) <= 9
    assert df["count"].sum() == m.sum()
    assert isinstance(df.index, pd.RangeIndex)


def test_diagnostics_returns_seven_metrics(streams, geometry):
    lengths, _widths, masks = geometry
    m = masks[0]
    df = sa.compute_local_binned_stats(streams["phi1"][0, m] / lengths[0], streams["phi2"][0, m])
    metrics = sa.compute_bin_diagnostics(df)
    assert len(metrics) == 7
    median_norm_wass, max_norm_wass = metrics[0], metrics[1]
    assert max_norm_wass >= median_norm_wass


def test_bulged_stream_is_more_disturbed(streams, geometry):
    """The localized bulge in stream 1 should raise the peak disturbance."""
    lengths, _widths, masks = geometry
    peaks = []
    for s in (0, 1):
        m = masks[s]
        df = sa.compute_local_binned_stats(
            streams["phi1"][s, m] / lengths[s], streams["phi2"][s, m]
        )
        peaks.append(sa.compute_bin_diagnostics(df)[1])
    assert peaks[1] > peaks[0]


def test_diagnostics_does_not_mutate_its_input(streams, geometry):
    lengths, _widths, masks = geometry
    m = masks[0]
    df = sa.compute_local_binned_stats(streams["phi1"][0, m] / lengths[0], streams["phi2"][0, m])
    before = df.copy()
    sa.compute_bin_diagnostics(df)
    pd.testing.assert_frame_equal(df, before)


def test_binning_is_generic_in_its_quantity(streams, geometry):
    """quantity_name renames two columns and changes nothing else."""
    lengths, _widths, masks = geometry
    m = masks[0]
    x = streams["phi1"][0, m] / lengths[0]
    q = streams["phi2"][0, m]

    default = sa.compute_local_binned_stats(x, q)
    renamed = sa.compute_local_binned_stats(x, q, quantity_name="v_los")

    assert "mean_v_los" in renamed.columns and "std_v_los" in renamed.columns
    assert "mean_phi2" not in renamed.columns
    np.testing.assert_array_equal(
        renamed["std_v_los"].to_numpy(), default["std_phi2"].to_numpy()
    )
    for col in ("count", "ks_p_vs_norm", "wass_vs_norm"):
        np.testing.assert_array_equal(renamed[col].to_numpy(), default[col].to_numpy())

    np.testing.assert_array_equal(
        np.asarray(sa.compute_bin_diagnostics(default), dtype=float),
        np.asarray(sa.compute_bin_diagnostics(renamed, quantity_name="v_los"), dtype=float),
    )


def test_empty_bins_report_a_zero_count():
    """A clumped track leaves bins empty. They report 0, not NaN."""
    rng = np.random.default_rng(SEED)
    phi1 = np.sort(np.concatenate([rng.uniform(0, 0.12, 20), rng.uniform(0.88, 1.0, 20)]))
    df = sa.compute_local_binned_stats(phi1, 0.1 * rng.standard_normal(40))
    assert (df["count"] == 0).any()
    assert df["count"].dtype == np.int64
    assert df.loc[df["count"] == 0, "std_phi2"].isna().all()


def test_unknown_scheme_rejected():
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="bin_size.*equal_count"):
        sa.compute_local_binned_stats(
            np.sort(rng.uniform(0, 1, 500)), rng.standard_normal(500), scheme="nope"
        )


# --------------------------------------------------------------- statistics


def test_wasserstein_null_threshold_is_a_decreasing_power_law():
    f = sa.wasserstein_null_threshold(power_law_fit=True)
    values = f(np.array([10, 100, 1000, 10000], dtype=float))
    assert np.all(np.diff(values) < 0), "the threshold should fall with sample size"
    assert np.isnan(f(0))
    assert f(300) == pytest.approx(
        sa.WASSERSTEIN_NULL_FIT.A * 300.0**sa.WASSERSTEIN_NULL_FIT.p + sa.WASSERSTEIN_NULL_FIT.B
    )


def test_monte_carlo_threshold_is_seeded():
    kwargs = dict(
        power_law_fit=False, num_points=50, trials=20, normal_ref_size=500, normal_ref_seed=SEED
    )
    assert sa.wasserstein_null_threshold(**kwargs) == sa.wasserstein_null_threshold(**kwargs)
    assert np.isnan(sa.wasserstein_null_threshold(power_law_fit=False, num_points=0))


def test_fisher_and_bonferroni():
    stat, p = sa.combine_ks_pvalues_fisher([0.5, 0.01, 0.2, np.nan, 0.9])
    assert np.isfinite(stat) and 0.0 <= p <= 1.0
    assert all(np.isnan(v) for v in sa.combine_ks_pvalues_fisher([np.nan]))

    significant, min_p = sa.bonferroni_correction([0.5, 0.0001, 0.2])
    assert significant is True or significant == np.True_
    assert min_p == 0.0001
    assert sa.bonferroni_correction([0.5, 0.4, 0.2])[0] in (False, np.False_)


# ------------------------------------------------------------------ density


def test_linearized_density_keys_and_shapes(streams, geometry):
    _lengths, _widths, masks = geometry
    out = sa.compute_linearized_density(
        streams["phi1"][0, masks[0]], detrend="poly", smoothing_sigma=1, bin_width=0.25
    )
    for key in ("hist_counts", "smooth_counts", "density", "centers", "w", "nbins", "resid"):
        assert key in out
    nbins = out["hist_counts"].size
    assert out["resid"].shape == (nbins,)
    assert out["w"] == pytest.approx(0.25, rel=0.05)
    assert out["hist_counts"].sum() == masks[0].sum()
    # Fractional residuals about a fitted trend average to roughly zero.
    assert abs(np.nanmean(out["resid"])) < 0.1


@pytest.mark.parametrize("detrend", ["poly", "lowpass", "none"])
def test_all_detrend_modes(streams, geometry, detrend):
    _lengths, _widths, masks = geometry
    out = sa.compute_linearized_density(
        streams["phi1"][0, masks[0]], detrend=detrend, smoothing_sigma=1, bin_width=0.25
    )
    assert np.isfinite(out["resid"]).any()


def test_density_requires_exactly_one_binning_rule():
    with pytest.raises(ValueError, match="either 'nbins' or 'bin_width'"):
        sa.compute_linearized_density(np.linspace(0, 1, 100))
    with pytest.raises(ValueError, match="Cannot provide both"):
        sa.compute_linearized_density(np.linspace(0, 1, 100), nbins=10, bin_width=0.1)


# ------------------------------------------------------------------ spectra


@pytest.fixture(scope="module")
def spectrum(streams, geometry):
    """The full spectral chain for the density-wave stream."""
    _lengths, _widths, masks = geometry
    m = masks[2]
    dens = sa.compute_linearized_density(
        streams["phi1"][2, m], detrend="poly", smoothing_sigma=1, bin_width=0.25
    )
    psd = sa.compute_welch_psd(dens["resid"], dens["w"])
    null = sa.compute_sampling_psd_realizations(
        dens["hist_counts"],
        smoothing_sigma_bins=1,
        w=dens["w"],
        n_realizations=200,
        rng_seed=SEED,
    )
    return {"dens": dens, "psd": psd, "null": null}


def test_welch_psd_keys(spectrum):
    psd = spectrum["psd"]
    assert psd["freqs"].shape == psd["psd"].shape
    assert psd["freqs"][0] == 0.0  # DC is present
    assert psd["rms_total"] > 0


def test_realizations_shape_and_ordering(spectrum):
    null = spectrum["null"]
    assert null["psd_all"].shape == (200, null["freqs"].size)
    assert np.all(null["psd_16"] <= null["psd_median"] + 1e-12)
    assert np.all(null["psd_median"] <= null["psd_84"] + 1e-12)
    assert np.all(null["psd_84"] <= null["psd_95"] + 1e-12)


def test_realizations_are_seeded(spectrum):
    repeat = sa.compute_sampling_psd_realizations(
        spectrum["dens"]["hist_counts"],
        smoothing_sigma_bins=1,
        w=spectrum["dens"]["w"],
        n_realizations=200,
        rng_seed=SEED,
    )
    np.testing.assert_array_equal(repeat["psd_all"], spectrum["null"]["psd_all"])


@pytest.mark.parametrize("method", ["peak_snr", "band_snr", "ratio95"])
def test_detection_returns_four_metrics(spectrum, method):
    rms_obs, rms_null, excess, lam = sa.detect_stream_psd_metrics(
        spectrum["psd"]["freqs"],
        spectrum["psd"]["psd"],
        w=spectrum["dens"]["w"],
        psd_all=spectrum["null"]["psd_all"],
        nperseg=spectrum["null"]["nperseg"],
        method=method,
    )
    assert rms_obs > 0 and rms_null > 0 and excess >= 0
    assert lam is None or lam > 0


def test_scalar_metrics_do_not_depend_on_method(spectrum):
    common = dict(
        w=spectrum["dens"]["w"],
        psd_all=spectrum["null"]["psd_all"],
        nperseg=spectrum["null"]["nperseg"],
    )
    baseline = sa.detect_stream_psd_metrics(
        spectrum["psd"]["freqs"], spectrum["psd"]["psd"], method="ratio95", **common
    )[:3]
    for method in ("peak_snr", "band_snr"):
        got = sa.detect_stream_psd_metrics(
            spectrum["psd"]["freqs"], spectrum["psd"]["psd"], method=method, **common
        )[:3]
        assert got == baseline


def test_density_wave_is_detected(spectrum):
    """The 3-degree wave in stream 2 should clear the noise floor."""
    lam = sa.detect_stream_psd_metrics(
        spectrum["psd"]["freqs"],
        spectrum["psd"]["psd"],
        w=spectrum["dens"]["w"],
        psd_all=spectrum["null"]["psd_all"],
        nperseg=spectrum["null"]["nperseg"],
    )[-1]
    assert lam is not None, "a strong injected density wave should be detected"


def test_per_method_thresholds(spectrum):
    from stream_analysis.spectra import _DEFAULT_SNR_THRESHOLD

    assert _DEFAULT_SNR_THRESHOLD == {"peak_snr": 5.0, "band_snr": 3.0, "ratio95": 3.0}
    common = dict(
        w=spectrum["dens"]["w"],
        psd_all=spectrum["null"]["psd_all"],
        nperseg=spectrum["null"]["nperseg"],
    )
    for method, threshold in _DEFAULT_SNR_THRESHOLD.items():
        implicit = sa.detect_stream_psd_metrics(
            spectrum["psd"]["freqs"], spectrum["psd"]["psd"], method=method, **common
        )
        explicit = sa.detect_stream_psd_metrics(
            spectrum["psd"]["freqs"],
            spectrum["psd"]["psd"],
            method=method,
            snr_threshold=threshold,
            **common,
        )
        assert implicit == explicit


def test_unknown_detection_method_rejected():
    with pytest.raises(ValueError, match="peak_snr.*band_snr.*ratio95"):
        sa.detect_stream_psd_metrics(np.linspace(0.0, 2.0, 33), np.ones(33), w=0.25, method="nope")


# ----------------------------------------------------------------- pipeline


@pytest.fixture(scope="module")
def metrics_table(streams, geometry, orbits):
    lengths, widths, masks = geometry
    sigma_v = sa.compute_velocity_dispersion(
        streams["xv"], masks, detrend=True, phi1=streams["phi1"], use_numba=False
    )
    df_orbits = sa.compute_orbit_properties(orbits).iloc[:3]
    df_orbits.index = [101, 202, 303]  # deliberately not a RangeIndex
    return sa.build_metrics_table(
        streams["phi1"],
        streams["phi2"],
        masks,
        lengths,
        widths,
        sigma_v,
        df_orbits,
        n_realizations=200,
    )


def test_metrics_table_shape_and_columns(metrics_table):
    assert list(metrics_table.columns) == list(sa.METRIC_COLUMNS)
    assert len(metrics_table) == 3
    assert metrics_table.index.name == "stream_index"
    assert list(metrics_table.index) == [101, 202, 303]


def test_metrics_table_derived_column(metrics_table):
    np.testing.assert_allclose(
        metrics_table["coef_of_var_stds"],
        metrics_table["std_std_bins"] / metrics_table["mean_std_bins"],
    )


def test_metrics_table_rejects_a_mismatched_orbit_table(streams, geometry, orbits):
    lengths, widths, masks = geometry
    df_orbits = sa.compute_orbit_properties(orbits)  # 4 rows against 3 streams
    with pytest.raises(ValueError, match="rows but the stream arrays"):
        sa.build_metrics_table(
            streams["phi1"], streams["phi2"], masks, lengths, widths, np.ones(3), df_orbits
        )


def test_metrics_table_rejects_missing_orbit_columns(streams, geometry, orbits):
    lengths, widths, masks = geometry
    df_orbits = sa.compute_orbit_properties(orbits).iloc[:3].drop(columns=["period"])
    with pytest.raises(ValueError, match="missing required columns"):
        sa.build_metrics_table(
            streams["phi1"], streams["phi2"], masks, lengths, widths, np.ones(3), df_orbits
        )


def test_checkpoint_is_optional(streams, geometry, orbits, tmp_path):
    lengths, widths, masks = geometry
    df_orbits = sa.compute_orbit_properties(orbits).iloc[:3]
    out = tmp_path / "nested" / "checkpoint.parquet"
    sa.build_metrics_table(
        streams["phi1"],
        streams["phi2"],
        masks,
        lengths,
        widths,
        np.ones(3),
        df_orbits,
        checkpoint_path=out,
        n_realizations=50,
    )
    assert out.exists()
    saved = pd.read_parquet(out)
    assert "median_norm_wass" in saved.columns
    assert "min_lambda_band" not in saved.columns  # written before the spectra run


# ------------------------------------------------------- surface and config


OLD_NAMES = {
    "measure_stream_LengthWidth": "measure_stream_length_width",
    "compute_vel_disp_stream": "compute_velocity_dispersion",
    "stream_local_binned_stats": "compute_local_binned_stats",
    "compute_welch_psd_for_resid": "compute_welch_psd",
    "plot_metric_stats_detailed": "summarize_bin_metrics",
}


@pytest.mark.parametrize("old,new", sorted(OLD_NAMES.items()))
def test_deprecated_names_still_resolve(old, new):
    assert hasattr(sa, old)
    assert hasattr(sa, new)


def test_deprecated_name_warns_and_forwards(streams):
    with pytest.warns(DeprecationWarning, match="measure_stream_length_width"):
        old = sa.measure_stream_LengthWidth(streams["phi1"], streams["phi2"], method="kde")
    new = sa.measure_stream_length_width(streams["phi1"], streams["phi2"], method="kde")
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


def test_old_phi2_keyword_still_works():
    rng = np.random.default_rng(SEED)
    phi1 = np.sort(rng.uniform(0, 1, 2000))
    q = rng.standard_normal(2000)
    expected = sa.compute_local_binned_stats(phi1, q)
    with pytest.warns(DeprecationWarning, match="quantity"):
        got = sa.compute_local_binned_stats(phi1, phi2=q)
    pd.testing.assert_frame_equal(got, expected)


def test_all_exports_exist():
    missing = [name for name in sa.__all__ if not hasattr(sa, name)]
    assert missing == []


def test_config_matches_signature_defaults():
    import inspect

    sig = inspect.signature(sa.compute_local_binned_stats)
    assert sig.parameters["bin_size"].default == sa.BINNED.bin_size == 1 / 8.0
    assert sig.parameters["normal_ref_seed"].default == sa.BINNED.normal_ref_seed == 12345

    sig = inspect.signature(sa.measure_stream_length_width)
    assert sig.parameters["density_frac"].default == sa.GEOMETRY.density_frac == 0.9

    sig = inspect.signature(sa.compute_velocity_dispersion)
    assert sig.parameters["poly_degree"].default == sa.KINEMATICS.poly_degree == 5

    assert sa.DENSITY.bin_width == 0.25
    assert sa.SPECTRA.n_realizations == 1000
    assert sa.SPECTRA.rng_seed == 12345
    assert sa.SPECTRA.method == "peak_snr"
    assert sa.SPECTRA.published_method == "ratio95"
    assert (sa.WASSERSTEIN_NULL_FIT.A, sa.WASSERSTEIN_NULL_FIT.p) == (2.3, -0.52)


def test_package_does_not_import_heavy_optional_deps():
    """No nbody_streams, no agama. Run in a subprocess to get a clean import."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys; before = set(sys.modules); import stream_analysis; "
        "print(sorted({'nbody_streams', 'agama'} & (set(sys.modules) - before)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


# ----------------------------------------------------------------- plotting


def test_summarize_bin_metrics(streams, geometry):
    lengths, _widths, masks = geometry
    m = masks[0]
    df = sa.compute_local_binned_stats(streams["phi1"][0, m] / lengths[0], streams["phi2"][0, m])
    metrics, df_out = sa.summarize_bin_metrics(df, plot=False)
    assert "median_wasserstein" in metrics
    assert "norm_wass_to_95" in df_out.columns
    assert "norm_wass_to_95" not in df.columns  # the caller's frame is untouched


def test_bin_plot_respects_the_panel_cap(streams, geometry):
    from matplotlib import pyplot as plt

    lengths, _widths, masks = geometry
    m = masks[0]
    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sa.compute_local_binned_stats(
            streams["phi1"][0, m] / lengths[0],
            streams["phi2"][0, m],
            bin_size=1 / 40.0,
            plot_streambins=True,
            max_bins_to_plot=6,
        )
    assert len(plt.gcf().axes) == 6
    plt.close("all")
