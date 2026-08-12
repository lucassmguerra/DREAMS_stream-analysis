# -*- coding: utf-8 -*-
"""
Tests for the package surface and the compatibility shims.

The rule from the brief is that no name resolving under ``sa.`` before the
refactor may fail to resolve after it, whether through its new name or through a
shim. These tests hold that line, and check that each shim forwards without
touching the numbers.
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT

#: Every public name the original module exposed, read off the frozen source.
ORIGINAL_PUBLIC_NAMES = [
    "compute_orbit_properties",
    "measure_stream_LengthWidth",
    "compute_vel_disp_stream",
    "stream_local_binned_stats",
    "compute_bin_diagnostics",
    "plot_metric_stats_detailed",
    "wasserstein_null_threshold",
    "combine_ks_pvalues_fisher",
    "bonferroni_correction",
    "compute_linearized_density",
    "compute_welch_psd_for_resid",
    "compute_sampling_psd_realizations",
    "detect_stream_psd_metrics",
    # private, but exposed at module level and used by the source itself
    "_get_max_consecutive_true_fraction",
    "NUMBA_AVAILABLE",
]


def test_original_names_all_resolve(api):
    """Every pre-refactor name still resolves on the package."""
    missing = [n for n in ORIGINAL_PUBLIC_NAMES if not hasattr(api, n)]
    assert missing == [], f"names lost in the refactor: {missing}"


def test_frozen_source_has_no_extra_public_names(api):
    """
    Nothing public in the frozen source is unaccounted for.

    Guards the list above against drifting out of date.
    """
    source = REPO_ROOT / "reference" / "stream_analysis_source.py"
    spec = importlib.util.spec_from_file_location("sa_ref_surface", source)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    defined = {
        name
        for name, obj in vars(ref).items()
        if callable(obj) and getattr(obj, "__module__", None) == "sa_ref_surface"
    }
    # The two private velocity-dispersion workers are implementation detail and
    # are not re-exported. Everything else must be reachable.
    defined -= {"_compute_vel_disp_numpy", "_compute_vel_disp_numba"}
    unreachable = sorted(n for n in defined if not hasattr(api, n))
    assert unreachable == [], f"frozen-source names not reachable on the package: {unreachable}"


def test_all_is_importable(api):
    """Every name in __all__ actually exists."""
    missing = [n for n in api.__all__ if not hasattr(api, n)]
    assert missing == [], f"__all__ lists names that do not exist: {missing}"


@pytest.mark.parametrize(
    "old_name,new_name",
    [
        ("measure_stream_LengthWidth", "measure_stream_length_width"),
        ("compute_vel_disp_stream", "compute_velocity_dispersion"),
        ("stream_local_binned_stats", "compute_local_binned_stats"),
        ("compute_welch_psd_for_resid", "compute_welch_psd"),
        ("plot_metric_stats_detailed", "summarize_bin_metrics"),
        ("_get_max_consecutive_true_fraction", "_max_consecutive_true_fraction"),
    ],
)
def test_shim_warns(api, old_name, new_name):
    """Each shim emits a DeprecationWarning naming its replacement."""
    old = getattr(api, old_name)
    with pytest.warns(DeprecationWarning, match=new_name):
        if old_name == "_get_max_consecutive_true_fraction":
            old(np.array([True, False]))
        elif old_name == "measure_stream_LengthWidth":
            old(np.linspace(0, 1, 100), np.zeros(100))
        elif old_name == "compute_vel_disp_stream":
            old(np.zeros((10, 6)), use_numba=False)
        elif old_name == "stream_local_binned_stats":
            old(np.linspace(0, 1, 400), np.random.default_rng(0).standard_normal(400))
        elif old_name == "compute_welch_psd_for_resid":
            old(np.random.default_rng(0).standard_normal(64), 0.25)
        else:
            import pandas as pd

            old(pd.DataFrame({"count": [10, 20]}), plot=False)


def test_shim_forwards_identically(api, fixtures):
    """The old name and the new name return the same numbers."""
    state = fixtures.state("unperturb")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = api.measure_stream_LengthWidth(state.phi1, state.phi2, method="kde")
    new = api.measure_stream_length_width(state.phi1, state.phi2, method="kde")
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


def test_old_phi2_keyword_still_works(api):
    """The renamed keyword is accepted, with a warning, and changes nothing."""
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 2000))
    q = rng.standard_normal(2000)

    expected = api.compute_local_binned_stats(phi1, q)
    with pytest.warns(DeprecationWarning, match="quantity"):
        got = api.compute_local_binned_stats(phi1, phi2=q)

    assert list(got.columns) == list(expected.columns)
    for col in expected.columns:
        np.testing.assert_array_equal(got[col].to_numpy(), expected[col].to_numpy())


def test_both_quantity_names_rejected(api):
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 100))
    with pytest.raises(TypeError, match="Pass only one"):
        api.compute_local_binned_stats(phi1, rng.standard_normal(100), phi2=rng.standard_normal(100))


def test_missing_quantity_rejected(api):
    with pytest.raises(TypeError, match="missing required argument"):
        api.compute_local_binned_stats(np.linspace(0, 1, 100))


def test_quantity_name_renames_columns_only(api):
    """
    quantity_name changes the two column labels and nothing else.

    This is what makes the binned family usable on v_los without pretending the
    numbers are about phi2.
    """
    rng = np.random.default_rng(12345)
    phi1 = np.sort(rng.uniform(0, 1, 2000))
    q = rng.standard_normal(2000)

    default = api.compute_local_binned_stats(phi1, q)
    renamed = api.compute_local_binned_stats(phi1, q, quantity_name="v_los")

    assert "mean_v_los" in renamed.columns and "std_v_los" in renamed.columns
    assert "mean_phi2" not in renamed.columns

    np.testing.assert_array_equal(
        renamed["std_v_los"].to_numpy(), default["std_phi2"].to_numpy()
    )
    for col in ["count", "ks_stat_vs_norm", "ks_p_vs_norm", "wass_vs_norm", "wass_vs_global"]:
        np.testing.assert_array_equal(renamed[col].to_numpy(), default[col].to_numpy())

    # And the diagnostics read the renamed column, giving identical metrics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = api.compute_bin_diagnostics(default.copy())
        b = api.compute_bin_diagnostics(renamed.copy(), quantity_name="v_los")
    assert a == b


def test_config_values_match_signature_defaults(api):
    """The frozen config carries the same values the signatures do."""
    import inspect

    sig = inspect.signature(api.compute_local_binned_stats)
    assert sig.parameters["bin_size"].default == api.BINNED.bin_size
    assert sig.parameters["normal_ref_seed"].default == api.BINNED.normal_ref_seed
    assert sig.parameters["normal_ref_size"].default == api.BINNED.normal_ref_size

    sig = inspect.signature(api.measure_stream_length_width)
    assert sig.parameters["density_frac"].default == api.GEOMETRY.density_frac
    assert sig.parameters["nbins"].default == api.GEOMETRY.nbins

    sig = inspect.signature(api.compute_velocity_dispersion)
    assert sig.parameters["poly_degree"].default == api.KINEMATICS.poly_degree

    sig = inspect.signature(api.detect_stream_psd_metrics)
    assert sig.parameters["snr_threshold"].default == api.SPECTRA.snr_threshold
    assert sig.parameters["min_bins"].default == api.SPECTRA.min_bins
    assert (
        sig.parameters["conservative_nyquist_frac"].default
        == api.SPECTRA.conservative_nyquist_frac
    )
    assert sig.parameters["method"].default == api.SPECTRA.method

    sig = inspect.signature(api.compute_welch_psd)
    assert sig.parameters["nperseg_frac"].default == api.SPECTRA.nperseg_frac
    assert sig.parameters["noverlap_frac"].default == api.SPECTRA.noverlap_frac

    sig = inspect.signature(api.compute_linearized_density)
    assert sig.parameters["smoothing_frac"].default == api.DENSITY.smoothing_frac
    assert sig.parameters["smoothing_mode"].default == api.DENSITY.smoothing_mode
    assert sig.parameters["poly_order"].default == api.DENSITY.poly_order


def test_wasserstein_fit_coefficients_unchanged(api):
    """The published null-threshold power law is the one being evaluated."""
    fit = api.WASSERSTEIN_NULL_FIT
    assert (fit.A, fit.p, fit.B) == (2.3, -0.52, 0.00590098)
    f = api.wasserstein_null_threshold(power_law_fit=True)
    assert f(300) == fit.A * (300.0**fit.p) + fit.B


def test_package_does_not_import_heavy_optional_deps():
    """
    The package never pulls in nbody_streams, agama, zarr or seaborn.

    Run in a subprocess, because by the time the test session reaches this point
    other modules have already been imported by pytest and its plugins, and
    sys.modules no longer says anything about what the package itself needs.
    """
    import subprocess
    import sys

    code = (
        "import sys; before = set(sys.modules); import stream_analysis; "
        "leaked = {'nbody_streams', 'agama', 'zarr', 'seaborn'} & (set(sys.modules) - before); "
        "print(sorted(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]", f"stream_analysis imported {result.stdout.strip()}"
