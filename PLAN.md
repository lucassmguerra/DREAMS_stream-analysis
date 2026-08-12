# PLAN

Proposed module split for the `stream_analysis` package, derived from the call
graph and external call sites in `INVENTORY.md`.

---

## 1. Shape of the problem

The call graph is nearly flat. Fourteen of sixteen functions call nothing
internal. There is exactly one shared helper, `wasserstein_null_threshold`, and
one private worker pair behind `compute_vel_disp_stream`. So the split cannot be
driven by dependency clustering, because there are almost no dependencies. It has
to be driven by the analysis stage each function serves, which is the same thing
as the paper's metric families.

Three stages exist in the pipeline, and they chain through plain arrays rather
than through function calls.

```
stage A   phase space -> phi1, phi2, mask, L, W, sigma_v
stage B   phi1, phi2  -> per-bin table -> seven binned metrics
stage C   phi1        -> linearized density -> Welch PSD -> four spectral metrics
```

The split below follows that, with the statistical primitives that stage B and
the plotting helper share pulled out to their own module so neither has to import
the other.

---

## 2. Modules

### `_numba.py`
The numba import guard, `NUMBA_AVAILABLE`, `njit`, `prange`, and the pure-NumPy
fallback shim.

Isolates the one import-time side effect and the one optional dependency, so
every other module imports numpy and nothing conditional.

Contains: the lines 38 to 48 block, moved verbatim. The shim keeps its current
`njit(nopython=True, cache=True, parallel=True)` signature so that behavior does
not change. The bare-decoration hazard is reported in FINDINGS, not fixed.

### `config.py`
Frozen dataclasses holding the analysis hyperparameters at their present values.

Gives the paper's numbers one place to be read from and one place to be audited,
without taking them out of the function signatures where they are overridable.

Contains: `BinnedConfig` (`bin_size=1/8.`, `min_particles=300`,
`normal_ref_size=20000`, `normal_ref_seed=12345`, `phi2_detrend_order=3`,
`ddof=1`, `wass_flag_threshold=1.0`, `ks_alpha=0.05`),
`GeometryConfig` (`density_frac=0.9`, `nbins=100`, `smoothing_sigma=1.0`),
`DensityConfig` (`bin_width=0.25`, `smoothing_sigma=1.0`, `poly_order=3`,
`smoothing_mode="reflect"`, `smoothing_frac=0.02`),
`SpectraConfig` (`n_realizations=1000`, `rng_seed=12345`, `nperseg_frac=0.25`,
`noverlap_frac=0.5`, `scaling="density"`, `snr_threshold=3.0`, `min_bins=2`,
`conservative_nyquist_frac=0.9`),
`WassersteinNullFit` (`A=2.3`, `p=-0.52`, `B=0.00590098`),
`KinematicsConfig` (`poly_degree=5`).

Note on scope. Every dataclass field is the value already hardcoded. Function
signatures keep their present literal defaults so that a call written today
against `sa.` keeps working with no config object in sight. The dataclasses are
the documented source of those literals and are what `pipeline.py` reads.

### `orbits.py`
Orbital diagnostics computed from integrated orbits, independent of any stream
coordinate frame.

Contains: `compute_orbit_properties`.

Justification for its own module rather than a merge: it is the only function
that takes orbits rather than stream coordinates, and it is the only consumer of
`scipy.signal.detrend`. It sits upstream of the stream analysis, building the
`df_orbits` table the pipeline consumes, so a separate file matches where it
falls in the workflow.

(First pass called this a dead-code candidate on the strength of a grep that
found no call site. Wrong. Corrected in INVENTORY.md and LEGACY.md.)

### `geometry.py`
Stream extent in the stream frame, length along phi1, width along phi2, and the
KDE membership mask that every later stage filters on.

Contains: `measure_stream_LengthWidth` renamed `measure_stream_length_width`.

### `kinematics.py`
Velocity dispersion, with the optional polynomial detrend against phi1.

Contains: `compute_vel_disp_stream` renamed `compute_velocity_dispersion`, plus
its two private workers `_compute_vel_disp_numpy` and `_compute_vel_disp_numba`.
Imports `_numba`.

The three move together because the two workers exist only to serve the wrapper
and are numerically distinguishable from each other, so they must stay adjacent
for the backend contract to be reviewable.

### `statistics.py`
Distribution-comparison primitives that carry no stream-specific meaning.

Contains: `wasserstein_null_threshold`, `combine_ks_pvalues_fisher`,
`bonferroni_correction`, `_get_max_consecutive_true_fraction`, and a new private
`_normalize_wasserstein(df)` holding the six-statement block currently duplicated
between `compute_bin_diagnostics` and `plot_metric_stats_detailed`.

This module exists to break the only fan-in in the graph. `binned.py` and
`plotting.py` both need `wasserstein_null_threshold`. Putting it in either would
force one to import the other.

On `_normalize_wasserstein`. This is the one place the refactor factors rather
than moves. The six statements are byte-identical between the two call sites
except that one operates on `df` and the other on `df_out`. Extraction is
mechanical and the golden tests cover both callers. If the extraction produces
any difference at all, it is reverted and the duplication is kept.

### `binned.py`
The localized binned-statistics family, Section-level support for the global
disturbance, peak disturbance and width variation metrics.

Contains: `stream_local_binned_stats` renamed `compute_local_binned_stats`, and
`compute_bin_diagnostics`. Imports `statistics.py` and `config.py`.

The per-bin figure branch of `compute_local_binned_stats` stays inside that
function, in this module, rather than moving to `plotting.py`. Splitting it would
mean rewriting the function rather than moving it, and the prime directive
forbids that. The cost is that `binned.py` imports matplotlib. Accepted.

### `density.py`
Linearized along-stream density, smoothing as an approximate KDE, and detrending
to fractional residuals. The input to the power-spectrum family.

Contains: `compute_linearized_density`.

Separate from `spectra.py` because it is the only function that touches particle
coordinates in that chain. Everything downstream of it operates on residual
arrays and never sees a particle again. That is a real seam, and the fixture and
golden boundaries fall naturally on it.

### `spectra.py`
Welch power spectra, the Monte Carlo sampling noise floor, and the detection
metrics. Support for the minimum detectable scale, RMS density residual and
excess power.

Contains: `compute_welch_psd_for_resid` renamed `compute_welch_psd`,
`compute_sampling_psd_realizations`, `detect_stream_psd_metrics`.

These three are one pipeline. The output dict of each feeds the next by key. They
are never called separately in any call site found.

### `plotting.py`
Figure helpers that are not part of any metric's definition.

Contains: `plot_metric_stats_detailed` renamed `summarize_bin_metrics`. Imports
`statistics.py`.

The rename is because the function's return value is a metrics dict and a
DataFrame, the figure is built and dropped, and it is called for its numbers as
often as for its picture. `plot=True` stays the default so behavior is unchanged.

### `pipeline.py`
The metrics-table driver.

Contains: `build_metrics_table`, new, mirroring `return_calc_props_df` from
`Stream_morphology_analysis.py` line 24. Imports
`binned.py`, `density.py`, `spectra.py`, `config.py`.

### `legacy.py`
Quarantine. Empty at the end of Phase 0. Nothing is proposed for it yet. The two
candidates that might have landed here, `compute_orbit_properties` and
`plot_metric_stats_detailed`, both stay public, the first because it is a
complete documented API even though nothing calls it, the second because it has
external call sites.

### `_compat.py`
Thin `DeprecationWarning` wrappers under every old name.

### `__init__.py`
Flat re-export of every public name, new and old, with an explicit `__all__`.

---

## 3. Import graph

No cycles. Depth 3.

```
_numba      <- kinematics
config      <- binned, density, spectra, geometry, kinematics, pipeline
statistics  <- binned, plotting
binned      <- pipeline
density     <- pipeline
spectra     <- pipeline
orbits, geometry, plotting   (leaves)
_compat     <- __init__   (imports everything, imported by nothing but __init__)
```

---

## 4. Renames

Old name on the left, new name on the right. Positional argument order is
unchanged everywhere.

| Old | New | Why |
|---|---|---|
| `measure_stream_LengthWidth` | `measure_stream_length_width` | Mixed case with an embedded acronym-style `LengthWidth`. Snake case matches every other name in the file. |
| `compute_vel_disp_stream` | `compute_velocity_dispersion` | `vel_disp` is an abbreviation of an abbreviation, and the trailing `_stream` is redundant inside a package called `stream_analysis`. |
| `stream_local_binned_stats` | `compute_local_binned_stats` | Same redundant `stream_` prefix. `compute_` prefix matches its siblings. |
| `compute_welch_psd_for_resid` | `compute_welch_psd` | `for_resid` describes the argument, not the operation, and the argument is already named `resid`. |
| `plot_metric_stats_detailed` | `summarize_bin_metrics` | It returns metrics and a table. The figure is a side effect. `_detailed` distinguished it from nothing, since no plainer variant exists. |
| `_get_max_consecutive_true_fraction` | `_max_consecutive_true_fraction` | Drops the `get_` on a pure function. Stays private. |

Unchanged: `compute_orbit_properties`, `compute_bin_diagnostics`,
`wasserstein_null_threshold`, `combine_ks_pvalues_fisher`,
`bonferroni_correction`, `compute_linearized_density`,
`compute_sampling_psd_realizations`, `detect_stream_psd_metrics`.

### Keyword renames

Only one, and it is the one the brief calls out.

`compute_local_binned_stats` currently names its second positional argument
`phi2`, and names two output columns `mean_phi2` and `std_phi2`, although the
function is generic in that argument and its own inline comment at line 704 says
so. Renaming:

- second positional argument `phi2` becomes `quantity`. Position is unchanged, so
  every positional call site is unaffected. Callers passing `phi2=` by keyword
  are carried by the deprecation shim.
- new keyword `quantity_name: str = "phi2"`, used only for the output column
  suffix and the default axis label. At its default the output columns are
  `mean_phi2` and `std_phi2`, byte-identical to today.
- `compute_bin_diagnostics` gains a matching `quantity_name: str = "phi2"` so it
  knows which column to read. At its default it reads `std_phi2` exactly as now.
- the plotting kwargs `y_axis_label`, `phi2_ylim`, `phi2_violin_width_frac`,
  `phi2_linewidth`, `phi2_color` keep their names. Renaming them would churn a
  large kwarg surface for no gain, and they are genuinely about the vertical axis
  of the figure, whatever is plotted on it.

Binning is untouched. Eight bins, `bin_size=1/8.` of the rescaled phi1, same
edges, same `pd.cut`, same `observed=False` grouping.

---

## 5. Phase 1 test plan

Goldens come from `reference/stream_analysis_source.py` only, loaded as `sa_ref`
by path. Fixtures come from the m12i forecasting HDF5,
which holds 16 streams in three states, `unperturb`, `perturb` and
`perturb_xtrm`, each `(16, 10000, 6)`, plus `progenitor_present` `(16, 6)`. All
three states become fixtures, so the disturbance metrics are exercised across
their range.

The file also carries a `stream_metadata` group with Arpit's own published
values for `median_norm_wass`, `max_norm_wass`, `coef_of_var_stds`,
`min_lambda_band` and `length_deg`. Those are an independent end-to-end check on
`build_metrics_table` beyond the golden comparison, and they will be reported,
not asserted, since the metadata was produced by a different code path.

Backend pinning. `compute_velocity_dispersion` goldens are generated twice, once
with `use_numba=True` and once with `use_numba=False`, and compared per backend.
The two backends are not bitwise equal to each other in the source either, so
cross-backend equality is not part of the contract.

---

## 6. Checkpoint questions

Six questions. The first three block Phase 1 or Phase 2. The last three block
only the README and Phase 5.

**Q1. `detect_stream_psd_metrics` method for `build_metrics_table`.**
Your `return_calc_props_df` calls it with `method='ratio95'`. The function's own
default and its docstring's stated preference are `band_snr`, and the brief
directs the new pipeline to use `band_snr`. These give different numbers for
`min_lambda_band`. The published `stream_metadata/min_lambda_band` in the HDF5
came from the `ratio95` path.

There is also a cost difference. `band_snr` scans every contiguous frequency band
with an O(n_bands squared) double loop, integrating 1000 Monte Carlo realizations
inside each. For a stream with 40 trusted frequency bins that is roughly 800 band
integrals over a (1000, k) array, per stream. `ratio95` is a single vectorized
comparison.

Which should `build_metrics_table` default to? Options: `ratio95`, to reproduce
the paper exactly. `band_snr`, per the brief, accepting that the pipeline number
changes. Or `band_snr` as the default with the README stating plainly that the
paper used `ratio95`.

**Q2. Empty bins raise today.**
`compute_bin_stats` at line 823 returns six values for an empty bin and seven for
a populated one. `pd.DataFrame(stats_per_bin.tolist(), columns=[7 names])` then
raises. Any stream with an empty phi1 bin fails, which is exactly the short-stream
edge case the brief asks me to cover.

The brief says to record the exception as the golden, and separately says never to
change a number. Those agree here, so my default is to record the raised
exception as the contract and report the defect in FINDINGS without fixing it.
Confirming, because the alternative reading is that you want the one-line fix
(pad to seven NaNs) and a re-run. Do you want the exception preserved?

**Q3. Orbit-table columns absent from the fixture.**
`build_metrics_table` copies `min_pericenter_dist`, `max_apocenter_dist`,
`mean_dist`, `n_pericenters` and `period` from `df_orbits`. The HDF5's
`stream_metadata` group has only the first two, plus `eccentricity`. It has no
`mean_dist`, `n_pericenters` or `period`.

Is there a parquet of orbit properties for these 16 streams I should point
`tools/build_fixtures.py` at? If not, I will build the fixture orbit table with
the two real columns and the three missing ones as NaN, which still exercises the
alignment and copy logic, and I will say so in the README.

**Q4. Metric table TBD rows.**
Once the table is drafted I will need paper names or a "diagnostic only" ruling
for `frac_flag_wass`, `frac_consecutive_wass`, `frac_p_gt_005`, `rms_null`,
`mean_std_bins` and `std_std_bins` as standalone entries, `length_deg`,
`width_deg` and `vel.std.tot_km_s`. Not blocking until Phase 4.

**Q5. Legacy quarantine.**
Nothing is proposed for `legacy.py`. Answered: `compute_orbit_properties` looked
dead to a grep but produces `df_orbits` upstream, so it stays public.

**Q6. Repository name.**
Needed before Phase 5.

---

## 7. What happens on approval

Phase 1 generates fixtures and goldens from the frozen source. Phase 2 lands
`_numba.py`, `config.py` and `statistics.py` serially, then runs `geometry`,
`kinematics`, `orbits`, `binned`, `density`, `spectra` and `plotting` as parallel
agents on disjoint files. Phase 3 assembles the package and writes
`build_metrics_table`. Phases 4 and 5 follow without stopping.
