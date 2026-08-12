# -*- coding: utf-8 -*-
"""
The golden case registry.

Every case is a single callable that takes an ``api`` namespace and a
``Fixtures`` object and returns whatever the function under test returns. The
same case list is executed twice, once against the frozen source in
``tools/build_goldens.py`` and once against the refactored package in
``tests/test_contract.py``. Nothing else defines what "the contract" means.

The ``api`` indirection exists only to absorb the Phase 2 renames. Cases are
written against the new names. ``RefAPI`` maps them back to the old names when
running against the frozen source. If a rename is ever added or changed, this
one table is the only place it has to be recorded for the tests.

Cases that are expected to raise simply raise. The runner catches the exception
and records its type and message as the golden, because reproducing a failure
is part of the contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

STATES = ("unperturb", "perturb", "perturb_xtrm")

# Streams exercised by the expensive per-stream cases. Chosen to span the length
# range in the fixture: index 0 is short, 10 is the longest at 219 degrees.
SAMPLE_STREAMS = (0, 6, 10)


# --------------------------------------------------------------------------
# name mapping
# --------------------------------------------------------------------------

#: Canonical new name -> name in reference/stream_analysis_source.py.
#: Kept in sync with the Renames section of PLAN.md and with RENAMES.md.
REF_NAMES: dict[str, str] = {
    "measure_stream_length_width": "measure_stream_LengthWidth",
    "compute_velocity_dispersion": "compute_vel_disp_stream",
    "compute_local_binned_stats": "stream_local_binned_stats",
    "compute_welch_psd": "compute_welch_psd_for_resid",
    "summarize_bin_metrics": "plot_metric_stats_detailed",
    "_max_consecutive_true_fraction": "_get_max_consecutive_true_fraction",
}


class RefAPI:
    """Adapter exposing the frozen source under the package's new names."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, REF_NAMES.get(name, name))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@dataclass
class StreamState:
    """One perturbation state of the 16 m12i fixture streams."""

    phi1: np.ndarray  # (16, 10000)
    phi2: np.ndarray  # (16, 10000)
    xv: np.ndarray  # (16, 10000, 6)
    xv_prog: np.ndarray  # (16, 6)
    stream_index: np.ndarray  # (16,)


class Fixtures:
    """
    Lazy loader for the fixture data.

    Length, width and mask are recomputed from the state on first access using
    the function under test, rather than being stored, so that the derived
    quantities feeding the later cases come from the same code path being
    tested. That is deliberate. A break in ``measure_stream_length_width``
    should cascade and fail loudly rather than be masked by a stored array.
    """

    def __init__(self, api: Any) -> None:
        self._api = api
        self._states: dict[str, StreamState] = {}
        self._geom: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._orbits: pd.DataFrame | None = None

    def state(self, name: str) -> StreamState:
        if name not in self._states:
            path = FIXTURE_DIR / f"streams_m12i_{name}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"Fixture {path} missing. Regenerate with "
                    "`python tools/build_fixtures.py`, see its docstring."
                )
            with np.load(path) as z:
                self._states[name] = StreamState(
                    phi1=z["phi1"],
                    phi2=z["phi2"],
                    xv=z["xv"],
                    xv_prog=z["xv_prog"],
                    stream_index=z["stream_index"],
                )
        return self._states[name]

    def geometry(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(lengths, widths, masks) for one state, at the paper's settings."""
        if name not in self._geom:
            st = self.state(name)
            self._geom[name] = self._api.measure_stream_length_width(
                st.phi1, st.phi2, method="kde", density_frac=0.9, return_mask=True
            )
        return self._geom[name]

    @property
    def orbits(self) -> pd.DataFrame:
        if self._orbits is None:
            self._orbits = pd.read_parquet(FIXTURE_DIR / "orbits_m12i.parquet")
        return self._orbits

    def track(self, state: str, i: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Masked, length-rescaled phi1 and masked phi2 for one stream."""
        st = self.state(state)
        L, _W, masks = self.geometry(state)
        m = masks[i]
        return st.phi1[i, m], st.phi2[i, m], float(L[i])


# --------------------------------------------------------------------------
# synthetic edge-case data, all seeded
# --------------------------------------------------------------------------


def synthetic_orbits(n_orbits: int = 4, n_steps: int = 512, seed: int = 12345) -> np.ndarray:
    """
    Build an ``orbits`` array in the shape ``compute_orbit_properties`` expects.

    Each element is ``[times, coords]`` with coords ``(n_steps, 6)``. The orbits
    are eccentric Kepler-like ellipses with a seeded random phase and axis
    ratio, so peri and apo detection, the FFT period and both eccentricity
    estimates all have something to find.

    Times run backwards, because the real caller passes lookback times and the
    function reverses them internally.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 5.0, n_steps)
    out = np.empty((n_orbits, 2), dtype=object)
    for k in range(n_orbits):
        a = 20.0 + 10.0 * rng.random()
        ecc = 0.1 + 0.6 * rng.random()
        omega = 2.0 * np.pi * (1.0 + rng.random())
        phase = 2.0 * np.pi * rng.random()
        theta = omega * t + phase
        r = a * (1.0 - ecc**2) / (1.0 + ecc * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0.3 * a * np.sin(0.5 * theta)
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        vz = np.gradient(z, t)
        coords = np.column_stack([x, y, z, vx, vy, vz])
        # Reverse so times decrease, matching the caller's convention.
        out[k, 0] = t[::-1].copy()
        out[k, 1] = coords[::-1].copy()
    return out


def synthetic_track(n: int = 4000, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """A well-behaved single stream track, phi1 already rescaled to [0, 1]."""
    rng = np.random.default_rng(seed)
    phi1 = np.sort(rng.uniform(0.0, 1.0, size=n))
    phi2 = 0.2 * rng.standard_normal(n) + 0.05 * np.sin(6.0 * np.pi * phi1)
    return phi1, phi2


def synthetic_short_track(n: int = 40, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """
    A track short enough and clumped enough that some of the 8 bins are empty.

    Particles occupy only the first and last eighth of the range, so the six
    middle bins receive nothing.
    """
    rng = np.random.default_rng(seed)
    left = rng.uniform(0.00, 0.12, size=n // 2)
    right = rng.uniform(0.88, 1.00, size=n - n // 2)
    phi1 = np.sort(np.concatenate([left, right]))
    phi2 = 0.1 * rng.standard_normal(phi1.size)
    return phi1, phi2


# --------------------------------------------------------------------------
# case registry
# --------------------------------------------------------------------------

Runner = Callable[[Any, Fixtures], Any]
CASES: dict[str, Runner] = {}


def case(name: str) -> Callable[[Runner], Runner]:
    """Register a golden case under ``name``."""

    def deco(fn: Runner) -> Runner:
        if name in CASES:
            raise KeyError(f"duplicate case name {name!r}")
        CASES[name] = fn
        return fn

    return deco


def _register_parametrized(prefix: str, values, builder) -> None:
    """Register one case per value, named ``<prefix>__<value>``."""
    for v in values:
        CASES[f"{prefix}__{v}"] = builder(v)


# ---------------------------------------------------------------- geometry

def _geom_kde(state: str) -> Runner:
    def run(api, fx):
        st = fx.state(state)
        return api.measure_stream_length_width(
            st.phi1, st.phi2, method="kde", density_frac=0.9, return_mask=True
        )

    return run


_register_parametrized("geometry_kde", STATES, _geom_kde)


def _geom_quantile(state: str) -> Runner:
    def run(api, fx):
        st = fx.state(state)
        return api.measure_stream_length_width(
            st.phi1, st.phi2, method="quantile", density_frac=0.9, return_mask=True
        )

    return run


_register_parametrized("geometry_quantile", STATES, _geom_quantile)


@case("geometry_kde_nomask")
def _(api, fx):
    st = fx.state("unperturb")
    return api.measure_stream_length_width(st.phi1, st.phi2, method="kde", return_mask=False)


@case("geometry_kde_single_stream")
def _(api, fx):
    st = fx.state("unperturb")
    return api.measure_stream_length_width(st.phi1[3], st.phi2[3], method="kde", return_mask=True)


@case("geometry_kde_narrow_frac")
def _(api, fx):
    st = fx.state("perturb")
    return api.measure_stream_length_width(
        st.phi1, st.phi2, method="kde", density_frac=0.5, nbins=40, smoothing_sigma=2.0
    )


@case("geometry_edge_all_nan")
def _(api, fx):
    phi1 = np.full(500, np.nan)
    phi2 = np.full(500, np.nan)
    return api.measure_stream_length_width(phi1, phi2, method="kde")


@case("geometry_edge_all_nan_quantile")
def _(api, fx):
    phi1 = np.full(500, np.nan)
    phi2 = np.full(500, np.nan)
    return api.measure_stream_length_width(phi1, phi2, method="quantile")


@case("geometry_edge_single_particle")
def _(api, fx):
    return api.measure_stream_length_width(np.array([1.0]), np.array([2.0]), method="kde")


@case("geometry_edge_constant_phi1")
def _(api, fx):
    # Zero span, so the per-stream bin width is zero and the digitization divides
    # by zero. Recorded as-is.
    phi1 = np.full(200, 3.0)
    phi2 = np.linspace(-1.0, 1.0, 200)
    return api.measure_stream_length_width(phi1, phi2, method="kde")


@case("geometry_edge_bad_method")
def _(api, fx):
    phi1, phi2 = synthetic_track(100)
    return api.measure_stream_length_width(phi1, phi2, method="not_a_method")


@case("geometry_edge_shape_mismatch")
def _(api, fx):
    return api.measure_stream_length_width(np.zeros(10), np.zeros(11), method="kde")


# -------------------------------------------------------------- kinematics


def _kin(state: str, detrend: bool, use_numba: bool) -> Runner:
    def run(api, fx):
        st = fx.state(state)
        _L, _W, masks = fx.geometry(state)
        return api.compute_velocity_dispersion(
            st.xv,
            masks,
            detrend=detrend,
            phi1=st.phi1 if detrend else None,
            use_numba=use_numba,
        )

    return run


for _state in STATES:
    for _backend in ("numpy", "numba"):
        CASES[f"kinematics_detrend_{_backend}__{_state}"] = _kin(
            _state, detrend=True, use_numba=(_backend == "numba")
        )
    CASES[f"kinematics_plain_numpy__{_state}"] = _kin(_state, detrend=False, use_numba=False)


@case("kinematics_single_stream_numpy")
def _(api, fx):
    st = fx.state("unperturb")
    _L, _W, masks = fx.geometry("unperturb")
    return api.compute_velocity_dispersion(
        st.xv[2], masks[2], detrend=True, phi1=st.phi1[2], use_numba=False
    )


@case("kinematics_poly_degree_2_numpy")
def _(api, fx):
    st = fx.state("unperturb")
    _L, _W, masks = fx.geometry("unperturb")
    return api.compute_velocity_dispersion(
        st.xv, masks, detrend=True, phi1=st.phi1, poly_degree=2, use_numba=False
    )


@case("kinematics_edge_empty_mask_numpy")
def _(api, fx):
    st = fx.state("unperturb")
    mask = np.zeros((16, st.xv.shape[1]), dtype=bool)
    return api.compute_velocity_dispersion(st.xv, mask, use_numba=False)


@case("kinematics_edge_single_particle_mask_numpy")
def _(api, fx):
    st = fx.state("unperturb")
    mask = np.zeros((16, st.xv.shape[1]), dtype=bool)
    mask[:, 0] = True
    return api.compute_velocity_dispersion(
        st.xv, mask, detrend=True, phi1=st.phi1, use_numba=False
    )


@case("kinematics_edge_detrend_without_phi1")
def _(api, fx):
    st = fx.state("unperturb")
    return api.compute_velocity_dispersion(st.xv, None, detrend=True, phi1=None)


@case("kinematics_edge_bad_ndim")
def _(api, fx):
    return api.compute_velocity_dispersion(np.zeros(6))


@case("kinematics_edge_bad_last_dim")
def _(api, fx):
    return api.compute_velocity_dispersion(np.zeros((10, 5)))


@case("kinematics_edge_bad_mask_length")
def _(api, fx):
    return api.compute_velocity_dispersion(np.zeros((10, 6)), np.ones(9, dtype=bool))


# ------------------------------------------------------------ binned stats


def _binned(state: str, i: int) -> Runner:
    def run(api, fx):
        phi1, phi2, L = fx.track(state, i)
        return api.compute_local_binned_stats(phi1 / L, phi2)

    return run


for _state in STATES:
    for _i in SAMPLE_STREAMS:
        CASES[f"binned_stats__{_state}_{_i}"] = _binned(_state, _i)


@case("binned_stats_vlos")
def _(api, fx):
    # Same machinery applied to a per-particle kinematic quantity instead of
    # phi2, proving the binning is generic in its second argument.
    st = fx.state("unperturb")
    L, _W, masks = fx.geometry("unperturb")
    i = 6
    m = masks[i]
    xv = st.xv[i, m]
    r = np.linalg.norm(xv[:, :3], axis=1)
    v_los = np.sum(xv[:, :3] * xv[:, 3:], axis=1) / r
    return api.compute_local_binned_stats(st.phi1[i, m] / L[i], v_los)


@case("binned_stats_equal_count")
def _(api, fx):
    phi1, phi2, L = fx.track("unperturb", 6)
    return api.compute_local_binned_stats(phi1 / L, phi2, scheme="equal_count", min_particles=300)


@case("binned_stats_custom_edges")
def _(api, fx):
    phi1, phi2, L = fx.track("unperturb", 6)
    x = phi1 / L
    edges = np.linspace(x.min(), x.max(), 13)
    return api.compute_local_binned_stats(x, phi2, bin_edges=edges)


@case("binned_stats_bin_size_quarter")
def _(api, fx):
    phi1, phi2, L = fx.track("unperturb", 6)
    return api.compute_local_binned_stats(phi1 / L, phi2, bin_size=0.25)


@case("binned_stats_synthetic")
def _(api, fx):
    phi1, phi2 = synthetic_track()
    return api.compute_local_binned_stats(phi1, phi2)


@case("binned_stats_edge_all_nan_quantity")
def _(api, fx):
    phi1, _ = synthetic_track(2000)
    return api.compute_local_binned_stats(phi1, np.full(phi1.size, np.nan))


@case("binned_stats_edge_single_particle")
def _(api, fx):
    return api.compute_local_binned_stats(np.array([0.0]), np.array([0.5]))


@case("binned_stats_edge_constant_phi1")
def _(api, fx):
    rng = np.random.default_rng(12345)
    return api.compute_local_binned_stats(np.zeros(500), rng.standard_normal(500))


@case("binned_stats_edge_bad_scheme")
def _(api, fx):
    phi1, phi2 = synthetic_track(500)
    return api.compute_local_binned_stats(phi1, phi2, scheme="not_a_scheme")


@case("binned_stats_edge_shape_mismatch")
def _(api, fx):
    return api.compute_local_binned_stats(np.zeros(10), np.zeros(11))


# ------------------------------------------------------------ diagnostics


def _diag(state: str, i: int) -> Runner:
    def run(api, fx):
        phi1, phi2, L = fx.track(state, i)
        df = api.compute_local_binned_stats(phi1 / L, phi2)
        return api.compute_bin_diagnostics(df)

    return run


for _state in STATES:
    for _i in SAMPLE_STREAMS:
        CASES[f"diagnostics__{_state}_{_i}"] = _diag(_state, _i)


@case("diagnostics_ddof0")
def _(api, fx):
    phi1, phi2, L = fx.track("unperturb", 6)
    df = api.compute_local_binned_stats(phi1 / L, phi2)
    return api.compute_bin_diagnostics(df, ddof=0)


@case("diagnostics_edge_missing_columns")
def _(api, fx):
    df = pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]})
    return api.compute_bin_diagnostics(df)


@case("diagnostics_edge_all_nan")
def _(api, fx):
    df = pd.DataFrame(
        {
            "wass_vs_norm": [np.nan] * 8,
            "ks_p_vs_norm": [np.nan] * 8,
            "count": [np.nan] * 8,
            "std_phi2": [np.nan] * 8,
        }
    )
    return api.compute_bin_diagnostics(df)


@case("diagnostics_edge_zero_counts")
def _(api, fx):
    df = pd.DataFrame(
        {
            "wass_vs_norm": [0.1, 0.2, np.nan],
            "ks_p_vs_norm": [0.5, 0.01, np.nan],
            "count": [0, 0, 0],
            "std_phi2": [0.1, 0.2, np.nan],
        }
    )
    return api.compute_bin_diagnostics(df)


@case("diagnostics_edge_empty_frame")
def _(api, fx):
    return api.compute_bin_diagnostics(pd.DataFrame())


# ------------------------------------------------------------- statistics


@case("wasserstein_power_law_scalar")
def _(api, fx):
    f = api.wasserstein_null_threshold(power_law_fit=True)
    return [f(1), f(10), f(300), f(20000), f(0)]


@case("wasserstein_power_law_array")
def _(api, fx):
    f = api.wasserstein_null_threshold(power_law_fit=True)
    return f(np.array([0, 1, 5, 50, 300, 5000], dtype=float))


@case("wasserstein_power_law_int_array")
def _(api, fx):
    f = api.wasserstein_null_threshold(power_law_fit=True)
    return f(np.array([0, 1, 5, 50, 300, 5000]))


@case("wasserstein_monte_carlo_seeded")
def _(api, fx):
    return api.wasserstein_null_threshold(
        power_law_fit=False, num_points=50, trials=25, normal_ref_size=500, normal_ref_seed=12345
    )


@case("fisher_combine")
def _(api, fx):
    return api.combine_ks_pvalues_fisher([0.5, 0.01, 0.2, np.nan, 0.9])


@case("fisher_combine_all_nan")
def _(api, fx):
    return api.combine_ks_pvalues_fisher([np.nan, np.nan])


@case("fisher_combine_empty")
def _(api, fx):
    return api.combine_ks_pvalues_fisher([])


@case("bonferroni")
def _(api, fx):
    return api.bonferroni_correction([0.5, 0.001, 0.2, np.nan, 0.9], alpha=0.05)


@case("bonferroni_not_significant")
def _(api, fx):
    return api.bonferroni_correction([0.5, 0.4, 0.2], alpha=0.05)


@case("bonferroni_all_nan")
def _(api, fx):
    return api.bonferroni_correction([np.nan, np.nan])


@case("max_consecutive_true")
def _(api, fx):
    return [
        api._max_consecutive_true_fraction(np.array([False, True, True, False])),
        api._max_consecutive_true_fraction(np.array([True, True, True, True])),
        api._max_consecutive_true_fraction(np.array([False, False])),
        api._max_consecutive_true_fraction(pd.Series([True, False, True, True, True])),
    ]


# ---------------------------------------------------------------- summary


@case("summarize_bin_metrics_noplot")
def _(api, fx):
    phi1, phi2, L = fx.track("unperturb", 6)
    df = api.compute_local_binned_stats(phi1 / L, phi2)
    return api.summarize_bin_metrics(df, plot=False)


@case("summarize_bin_metrics_plot")
def _(api, fx):
    phi1, phi2, L = fx.track("perturb_xtrm", 10)
    df = api.compute_local_binned_stats(phi1 / L, phi2)
    return api.summarize_bin_metrics(df, plot=True)


@case("summarize_bin_metrics_edge_missing_columns")
def _(api, fx):
    return api.summarize_bin_metrics(pd.DataFrame({"unrelated": [1.0, 2.0]}), plot=False)


# ----------------------------------------------------------------- density


def _density(state: str, i: int) -> Runner:
    def run(api, fx):
        phi1, _phi2, _L = fx.track(state, i)
        return api.compute_linearized_density(
            phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
        )

    return run


for _state in STATES:
    for _i in SAMPLE_STREAMS:
        CASES[f"density_poly_binwidth__{_state}_{_i}"] = _density(_state, _i)


@case("density_lowpass")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    return api.compute_linearized_density(
        phi1, detrend="lowpass", smoothing_sigma=1, bin_width=0.25
    )


@case("density_none_detrend")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    return api.compute_linearized_density(phi1, detrend="none", smoothing_sigma=1, bin_width=0.25)


@case("density_fixed_nbins")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    return api.compute_linearized_density(phi1, nbins=128, detrend="poly")


@case("density_smoothing_frac_default")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    return api.compute_linearized_density(phi1, nbins=128)


@case("density_return_pdf_and_mask")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    return api.compute_linearized_density(
        phi1, bin_width=0.25, smoothing_sigma=1, detrend="poly", return_pdf=True, return_mask=True
    )


@case("density_multistream")
def _(api, fx):
    st = fx.state("unperturb")
    _L, _W, masks = fx.geometry("unperturb")
    return api.compute_linearized_density(
        st.phi1, mask=masks, bin_width=0.25, smoothing_sigma=1, detrend="poly", return_mask=True
    )


@case("density_edge_no_binning_given")
def _(api, fx):
    return api.compute_linearized_density(np.linspace(0, 1, 100))


@case("density_edge_both_binnings_given")
def _(api, fx):
    return api.compute_linearized_density(np.linspace(0, 1, 100), nbins=10, bin_width=0.1)


@case("density_edge_bad_detrend")
def _(api, fx):
    return api.compute_linearized_density(np.linspace(0, 1, 100), nbins=10, detrend="cubic_spline")


@case("density_edge_mask_shape_mismatch")
def _(api, fx):
    return api.compute_linearized_density(
        np.linspace(0, 1, 100), mask=np.ones(99, dtype=bool), nbins=10
    )


@case("density_edge_all_nan")
def _(api, fx):
    return api.compute_linearized_density(np.full(100, np.nan), nbins=10, detrend="poly")


@case("density_edge_empty_mask")
def _(api, fx):
    return api.compute_linearized_density(
        np.linspace(0, 10, 100), mask=np.zeros(100, dtype=bool), bin_width=0.25, detrend="poly"
    )


@case("density_edge_single_particle")
def _(api, fx):
    return api.compute_linearized_density(np.array([1.0]), bin_width=0.25, detrend="poly")


@case("density_edge_constant_phi1")
def _(api, fx):
    return api.compute_linearized_density(np.full(100, 2.0), bin_width=0.25, detrend="poly")


@case("density_edge_negative_poly_order")
def _(api, fx):
    return api.compute_linearized_density(
        np.linspace(0, 10, 100), bin_width=0.25, detrend="poly", poly_order=-1
    )


@case("density_edge_3d_input")
def _(api, fx):
    return api.compute_linearized_density(np.zeros((2, 3, 4)), nbins=10)


# ----------------------------------------------------------------- spectra


def _welch(state: str, i: int) -> Runner:
    def run(api, fx):
        phi1, _p2, _L = fx.track(state, i)
        out = api.compute_linearized_density(
            phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
        )
        return api.compute_welch_psd(out["resid"], out["w"])

    return run


for _state in STATES:
    for _i in SAMPLE_STREAMS:
        CASES[f"welch_psd__{_state}_{_i}"] = _welch(_state, _i)


@case("welch_psd_multistream")
def _(api, fx):
    st = fx.state("unperturb")
    _L, _W, masks = fx.geometry("unperturb")
    out = api.compute_linearized_density(
        st.phi1, mask=masks, bin_width=0.25, smoothing_sigma=1, detrend="poly"
    )
    return api.compute_welch_psd(out["resid"], out["w"])


@case("welch_psd_spectrum_scaling")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    out = api.compute_linearized_density(phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25)
    return api.compute_welch_psd(out["resid"], out["w"], scaling="spectrum", nperseg_frac=0.5)


@case("welch_psd_fs_override")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    out = api.compute_linearized_density(phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25)
    return api.compute_welch_psd(out["resid"], out["w"], fs_override=4.0)


@case("welch_psd_edge_nan_width")
def _(api, fx):
    return api.compute_welch_psd(np.zeros((2, 64)), np.array([np.nan, 0.0]))


@case("welch_psd_edge_short_signal")
def _(api, fx):
    rng = np.random.default_rng(12345)
    return api.compute_welch_psd(rng.standard_normal(4), 0.25)


def _mc(state: str, i: int) -> Runner:
    def run(api, fx):
        phi1, _p2, _L = fx.track(state, i)
        out = api.compute_linearized_density(
            phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
        )
        return api.compute_sampling_psd_realizations(
            out["hist_counts"],
            smoothing_sigma_bins=1,
            w=out["w"],
            n_realizations=1000,
            rng_seed=12345,
        )

    return run


for _state in STATES:
    for _i in SAMPLE_STREAMS:
        CASES[f"psd_realizations__{_state}_{_i}"] = _mc(_state, _i)


@case("psd_realizations_poisson")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    out = api.compute_linearized_density(phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25)
    return api.compute_sampling_psd_realizations(
        out["hist_counts"],
        smoothing_sigma_bins=1,
        w=out["w"],
        n_realizations=200,
        rng_seed=12345,
        sampling="poisson",
    )


@case("psd_realizations_poisson_observed")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    out = api.compute_linearized_density(phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25)
    return api.compute_sampling_psd_realizations(
        out["hist_counts"],
        smoothing_sigma_bins=1,
        w=out["w"],
        n_realizations=200,
        rng_seed=12345,
        sampling="poisson",
        null_hypothesis=False,
    )


@case("psd_realizations_multinomial_observed")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    out = api.compute_linearized_density(phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25)
    return api.compute_sampling_psd_realizations(
        out["hist_counts"],
        smoothing_sigma_bins=1,
        w=out["w"],
        n_realizations=200,
        rng_seed=12345,
        null_hypothesis=False,
    )


@case("psd_realizations_explicit_p")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    out = api.compute_linearized_density(phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25)
    n = out["hist_counts"].size
    p = np.linspace(1.0, 2.0, n)
    return api.compute_sampling_psd_realizations(
        out["hist_counts"],
        smoothing_sigma_bins=1,
        w=out["w"],
        n_realizations=100,
        rng_seed=12345,
        null_hypothesis=False,
        p=p,
    )


@case("psd_realizations_constant_detrend")
def _(api, fx):
    phi1, _p2, _L = fx.track("unperturb", 6)
    out = api.compute_linearized_density(phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25)
    return api.compute_sampling_psd_realizations(
        out["hist_counts"],
        smoothing_sigma_bins=1,
        w=out["w"],
        n_realizations=200,
        rng_seed=12345,
        detrend="constant",
    )


@case("psd_realizations_edge_2d_input")
def _(api, fx):
    return api.compute_sampling_psd_realizations(np.zeros((2, 10)), 1.0, 0.25)


@case("psd_realizations_edge_one_bin")
def _(api, fx):
    return api.compute_sampling_psd_realizations(np.array([5.0]), 1.0, 0.25)


@case("psd_realizations_edge_zero_realizations")
def _(api, fx):
    return api.compute_sampling_psd_realizations(np.ones(32), 1.0, 0.25, n_realizations=0)


@case("psd_realizations_edge_bad_sampling")
def _(api, fx):
    return api.compute_sampling_psd_realizations(np.ones(32), 1.0, 0.25, sampling="binomial")


@case("psd_realizations_edge_zero_counts")
def _(api, fx):
    return api.compute_sampling_psd_realizations(np.zeros(32), 1.0, 0.25, n_realizations=10)


def _detect(state: str, i: int, method: str) -> Runner:
    def run(api, fx):
        phi1, _p2, _L = fx.track(state, i)
        out = api.compute_linearized_density(
            phi1, detrend="poly", smoothing_sigma=1, bin_width=0.25
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

    return run


for _state in STATES:
    for _i in SAMPLE_STREAMS:
        CASES[f"detect_ratio95__{_state}_{_i}"] = _detect(_state, _i, "ratio95")

# Deliberately absent from the frozen contract, because their behavior changed
# on purpose after the refactor. See DIVERGENCES below and tests/test_detection.py.
#
#   detect_band_snr__*          band_snr's min_lambda_band selection was fixed
#   detect_band_snr_detailed    same, plus new keys in the detailed dict
#   detect_ratio95_detailed     new keys in the detailed dict
#   detect_edge_bad_method      method is now validated earlier, new message
#   pipeline_band_snr__*        follows from the band_snr fix


@case("detect_edge_no_mc_input")
def _(api, fx):
    freqs = np.linspace(0.0, 2.0, 33)
    rng = np.random.default_rng(12345)
    psd = rng.random(33)
    return api.detect_stream_psd_metrics(freqs, psd, w=0.25, method="band_snr")


@case("detect_edge_no_mc_input_detailed")
def _(api, fx):
    freqs = np.linspace(0.0, 2.0, 33)
    rng = np.random.default_rng(12345)
    psd = rng.random(33)
    return api.detect_stream_psd_metrics(freqs, psd, w=0.25, method="band_snr", detailed=True)


@case("detect_edge_ratio95_no_null")
def _(api, fx):
    freqs = np.linspace(0.0, 2.0, 33)
    rng = np.random.default_rng(12345)
    psd = rng.random(33)
    return api.detect_stream_psd_metrics(freqs, psd, w=0.25, method="ratio95")


@case("detect_edge_length_mismatch")
def _(api, fx):
    return api.detect_stream_psd_metrics(np.linspace(0.1, 2.0, 10), np.zeros(9), w=0.25)


@case("detect_edge_2d_input")
def _(api, fx):
    return api.detect_stream_psd_metrics(np.zeros((2, 5)), np.zeros((2, 5)), w=0.25)


@case("detect_edge_psd_median_only")
def _(api, fx):
    freqs = np.linspace(0.0, 2.0, 33)
    rng = np.random.default_rng(12345)
    psd = rng.random(33)
    # psd_median must align with the post-DC-removal freqs, so 32 entries.
    return api.detect_stream_psd_metrics(
        freqs, psd, w=0.25, psd_median=np.full(32, 0.4), psd_95=np.full(32, 0.8), method="ratio95"
    )


# ------------------------------------------------------------------ orbits


@case("orbit_properties_plain")
def _(api, fx):
    return api.compute_orbit_properties(synthetic_orbits())


@case("orbit_properties_with_central_mass")
def _(api, fx):
    return api.compute_orbit_properties(synthetic_orbits(), central_mass=1e12)


@case("orbit_properties_with_mu")
def _(api, fx):
    return api.compute_orbit_properties(synthetic_orbits(), mu=4.3009e-6 * 1e12)


@case("orbit_properties_short_series")
def _(api, fx):
    return api.compute_orbit_properties(synthetic_orbits(n_orbits=2, n_steps=8))


# ---------------------------------------------------------------- pipeline

# The end-to-end table. Defined by tools/pipeline_ref.py, which reproduces
# return_calc_props_df statement for statement. build_metrics_table has to match
# these column for column and dtype for dtype.


def _pipeline(state: str, method: str) -> Runner:
    def run(api, fx):
        from pipeline_ref import pipeline_inputs, reference_metrics_table

        kwargs = pipeline_inputs(api, fx, state)
        return reference_metrics_table(api, method=method, **kwargs)

    return run


for _state in STATES:
    CASES[f"pipeline_ratio95__{_state}"] = _pipeline(_state, "ratio95")


# --------------------------------------------------------------- divergences

#: Behavior that changed on purpose after the refactor, so it is not part of the
#: frozen contract and has no golden. Each entry names where it is tested
#: instead. Nothing else in this file diverges from the frozen source.
DIVERGENCES: dict[str, str] = {
    "detect_stream_psd_metrics(method='band_snr')": (
        "min_lambda_band now comes from bands of exactly min_bins frequency bins "
        "rather than from the upper edge of any qualifying band, which used to "
        "pin it to the top of the trusted range for every stream. "
        "tests/test_detection.py"
    ),
    "detect_stream_psd_metrics(method='peak_snr')": (
        "New. Per-frequency SNR against the Monte Carlo floor. Now the default "
        "method, at a 5-sigma threshold. tests/test_detection.py"
    ),
    "detect_stream_psd_metrics(detailed=True)": (
        "The detailed dict gained freqs, psd_null_std and snr_per_freq. "
        "tests/test_detection.py"
    ),
    "detect_stream_psd_metrics(method=<unknown>)": (
        "Validated at the top of the function rather than after the scalar "
        "metrics are computed, so the message names all three methods. "
        "tests/test_detection.py"
    ),
    "build_metrics_table(method=...)": (
        "Defaults to peak_snr at 5 sigma rather than to the published ratio95 at "
        "3. Only min_lambda_band differs. tests/test_pipeline.py"
    ),
    "compute_local_binned_stats, empty bins": (
        "An empty bin reports count 0 rather than NaN, so the column stays "
        "integral. Every other column is unchanged, and no metric moves, because "
        "compute_bin_diagnostics already did count.fillna(0). FINDINGS entry 1. "
        "tests/test_fixes.py"
    ),
    "compute_local_binned_stats(max_bins_to_plot=...)": (
        "Now honoured. It was accepted and never read. Figures only, no numbers. "
        "FINDINGS entry 9. tests/test_fixes.py"
    ),
    "compute_velocity_dispersion(detrend=False, use_numba=True)": (
        "Compiles and runs. It used to raise TypingError because the jitted body "
        "indexes a None phi1 in a branch numba cannot prune. FINDINGS entry 2. "
        "tests/test_fixes.py"
    ),
    "compute_bin_diagnostics": (
        "Works on a copy. It used to write its working columns onto the caller's "
        "frame. The returned metrics are unchanged. FINDINGS entry 8. "
        "tests/test_fixes.py"
    ),
    "detect_stream_psd_metrics(freqs strictly positive)": (
        "Works. pos_mask_all was bound only when the input carried a "
        "non-positive frequency but used unconditionally to slice psd_all, so "
        "this raised NameError. FINDINGS entry 3. tests/test_fixes.py"
    ),
    "wasserstein_null_threshold(power_law_fit=False, num_points=0)": (
        "Returns NaN. It used to raise UnboundLocalError on a stray `_`. "
        "FINDINGS entry 4. tests/test_fixes.py"
    ),
    "_numba fallback njit": (
        "Accepts a bare @njit as well as @njit(...). Only reachable when numba "
        "is absent. FINDINGS entry 7. tests/test_fixes.py"
    ),
}
