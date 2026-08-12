# INVENTORY

Inventory of `reference/stream_analysis_source.py`, a frozen copy of
`/mnt/d/Research/GC_streams/src/stream_analysis.py` (2239 lines, 16 top-level
functions, 0 classes).

---

## 1. Top-level functions

| Line | Signature | Purpose |
|---|---|---|
| 50 | `compute_orbit_properties(orbits, central_mass=None, mu=None, G=4.3009e-6) -> pd.DataFrame` | Per-orbit peri/apo, mean radius, pericenter count, FFT period, geometric and osculating eccentricity. |
| 226 | `measure_stream_LengthWidth(phi1, phi2, method="kde", density_frac=0.9, nbins=100, smoothing_sigma=1.0, return_mask=True)` | Stream length along phi1 and width along phi2, by quantile or by smoothed-histogram KDE, plus the membership mask. |
| 387 | `compute_vel_disp_stream(xv, mask=None, detrend=False, phi1=None, poly_degree=5, use_numba=True) -> float \| np.ndarray` | Public wrapper for 3D velocity dispersion. Normalizes shapes, dispatches to the numba or NumPy worker. |
| 504 | `_compute_vel_disp_numpy(xv, mask, detrend, phi1, poly_degree) -> np.ndarray` | NumPy/SciPy worker for the above. Uses `scipy.linalg.lstsq` when importable. |
| 583 | `_compute_vel_disp_numba(xv, mask, detrend, phi1, poly_degree)` | numba `@njit(parallel=True)` worker. Normal-equations polyfit plus Horner evaluation, hand-rolled. |
| 702 | `stream_local_binned_stats(phi1, phi2, scheme='bin_size', bin_edges=None, bin_size=1/8., min_particles=300, normal_ref_size=20000, normal_ref_seed=12345, ...30 plotting kwargs...) -> pd.DataFrame` | Detrends phi2 against phi1, bins along phi1, and per bin returns count, mean, std, KS statistic and p versus N(0,1), Wasserstein versus N(0,1), and Wasserstein versus the global standardized distribution. Optionally draws the per-bin violin figure. |
| 962 | `compute_bin_diagnostics(df, ddof=1) -> tuple[float x 7]` | Collapses the per-bin table into the seven headline binned metrics. |
| 1059 | `plot_metric_stats_detailed(df, plot=True, mc_trials=2000, normal_ref_size=20000, alpha=0.05, seed=0, dpi=200) -> tuple[dict, pd.DataFrame]` | Richer diagnostic summary of the same per-bin table, with Fisher and Bonferroni combination of KS p-values and a two-panel figure. |
| 1195 | `_get_max_consecutive_true_fraction(series) -> float` | Longest run of `True` divided by array length. |
| 1229 | `wasserstein_null_threshold(power_law_fit=True, num_points=300, trials=2000, normal_ref_size=20000, alpha=0.05, normal_ref_seed=None)` | Returns the 95% null Wasserstein threshold. Default branch returns a closure evaluating a precomputed power law in sample size; the alternative branch runs Monte Carlo. |
| 1311 | `combine_ks_pvalues_fisher(p_values) -> tuple[float, float]` | Fisher combination of per-bin KS p-values, NaNs dropped. |
| 1326 | `bonferroni_correction(p_values, alpha=0.05) -> tuple[bool, float]` | Bonferroni significance flag and minimum p-value. |
| 1344 | `compute_linearized_density(phi1, mask=None, nbins=None, bin_width=None, smoothing_sigma=None, smoothing_frac=0.02, smoothing_mode="reflect", detrend="none", poly_order=3, lowpass_sigma_bins=None, return_pdf=False, return_mask=False) -> dict` | Histogram, then Gaussian smoothing as an approximate KDE, then detrending, then fractional density residuals. Supports fixed `nbins` or fixed `bin_width`. |
| 1637 | `compute_welch_psd_for_resid(resid, w, fs_override=None, nperseg_frac=0.25, noverlap_frac=0.5, scaling="density") -> dict` | Per-stream Welch PSD of the fractional residuals plus total RMS from the PSD integral. |
| 1758 | `compute_sampling_psd_realizations(hist_counts, smoothing_sigma_bins, w, n_realizations=500, detrend="poly", poly_order=3, nperseg_frac=0.25, noverlap_frac=0.5, scaling="density", rng_seed=None, sampling="multinomial", p=None, null_hypothesis=True) -> dict` | Monte Carlo noise floor. Resamples the counts, repeats the smooth, detrend and Welch chain, returns all realizations and percentile summaries. |
| 1937 | `detect_stream_psd_metrics(freqs, psd_obs, w, *, psd_all=None, psd_median=None, psd_95=None, nperseg=None, snr_threshold=3.0, min_bins=2, conservative_nyquist_frac=0.9, method="band_snr", detailed=False)` | Observed RMS, null RMS, excess power, and the minimum detectable wavelength, by band-integrated SNR or by a per-frequency 95th-percentile ratio test. |

No classes are defined.

---

## 2. Call graph

Internal edges only. Everything else is numpy, scipy, pandas, matplotlib.

```
compute_orbit_properties            -> (none)
measure_stream_LengthWidth          -> (none)
compute_vel_disp_stream             -> _compute_vel_disp_numba | _compute_vel_disp_numpy
_compute_vel_disp_numpy             -> (none)
_compute_vel_disp_numba             -> (none)
stream_local_binned_stats           -> (none)
compute_bin_diagnostics             -> wasserstein_null_threshold
                                    -> _get_max_consecutive_true_fraction
plot_metric_stats_detailed          -> combine_ks_pvalues_fisher
                                    -> bonferroni_correction
                                    -> wasserstein_null_threshold
_get_max_consecutive_true_fraction  -> (none)
wasserstein_null_threshold          -> (none)
combine_ks_pvalues_fisher           -> (none)
bonferroni_correction               -> (none)
compute_linearized_density          -> (none)
compute_welch_psd_for_resid         -> (none)
compute_sampling_psd_realizations   -> (none)
detect_stream_psd_metrics           -> (none)
```

The graph is almost flat. The only fan-in node is `wasserstein_null_threshold`,
reached from both `compute_bin_diagnostics` and `plot_metric_stats_detailed`.

### External call sites

Counted across `/mnt/d/Research/GC_streams/**/*.py` and `**/*.ipynb` on the `sa.`
prefix, so these are the names Arpit actually calls.

`measure_stream_LengthWidth`, `stream_local_binned_stats`, `compute_vel_disp_stream`,
`plot_metric_stats_detailed`, `compute_linearized_density`, `compute_welch_psd_for_resid`,
`compute_sampling_psd_realizations`, `compute_bin_diagnostics`, `detect_stream_psd_metrics`,
`wasserstein_null_threshold`.

The pipeline driver `return_calc_props_df` lives outside this file, in
`/mnt/d/Research/GC_streams/src/Stream_morphology_analysis.py` line 24.

---

## 3. Dead-code candidates

None.

`compute_orbit_properties` was listed here in the first pass, because no `sa.`
call site for it appears in any `.py` or `.ipynb` under
`/mnt/d/Research/GC_streams/`. Corrected after Arpit pointed it out. It builds
the `df_orbits` table that the pipeline consumes, in the orbit-integration step
upstream of that directory, so the parquet the pipeline reads is its output. It
is live and on the critical path.

`combine_ks_pvalues_fisher`, `bonferroni_correction` and
`_get_max_consecutive_true_fraction` have no external call sites but are reached
internally, so they are live too.

The lesson for anyone auditing this file later is that a grep for `sa.<name>`
only finds the consumers inside one directory. It does not find a function whose
output has already been serialized to disk by an earlier stage.

---

## 4. Superseded candidates

No `_v2` suffixes and no commented-out call sites exist. One genuine overlap:

- `plot_metric_stats_detailed` (line 1059) recomputes, line for line, the
  `wass_thresh_95`, `norm_wass_to_95`, `flag_wass_under_95` and `frac_flag_wass`
  block that `compute_bin_diagnostics` computes at lines 1017 to 1032. The two
  are consistent today. They are two copies of the same six statements and will
  drift if either is edited. `compute_bin_diagnostics` is the one the pipeline
  uses. `plot_metric_stats_detailed` is the interactive superset and is still
  called externally, so it is not dead, only duplicated.

Recommendation: keep both public, factor the shared block into one private helper
that both call, and verify bitwise identity through the golden tests. Not a
quarantine candidate.

---

## 5. numba dependence

| Function | numba use | Works under the pure-NumPy shim? |
|---|---|---|
| `_compute_vel_disp_numba` | Decorated `@njit(parallel=True)` at line 582. Uses `prange`, closures defined inside the jitted body, `np.linalg.solve`. | Yes as written, because `@njit(parallel=True)` supplies the keyword the shim's signature requires and `prange` falls back to `range`. The nested `polyfit_fit` and `eval_poly` closures are valid pure Python. Note the loop variable `i` is reused inside `eval_poly` at line 660, shadowing the outer `prange(S)` index; under numba each closure call gets its own frame so this is harmless, and it is equally harmless in pure Python because `eval_poly` binds `i` locally. |
| `compute_vel_disp_stream` | Reads `NUMBA_AVAILABLE` at line 496 to choose the worker. | Yes. Falls through to `_compute_vel_disp_numpy`. |

No other function touches numba.

The shim itself is at lines 44 to 47:

```python
def njit(nopython=True, cache=True, parallel=True):  # Dummy decorator
    def decorator(func):
        return func
    return decorator
```

This only works when `njit` is called with parentheses. A bare `@njit`
decoration would pass the function as `nopython` and return the inner
`decorator` instead of the function. The single decoration site in this file
does use parentheses, so the file works today, but the shim is a trap for the
next `@njit` added. Reported in FINDINGS.

Note also that the numba and NumPy workers are not numerically equivalent by
construction. The numba path solves the polynomial fit by normal equations with
`np.linalg.solve` on a Vandermonde-Gram matrix, the NumPy path uses
`scipy.linalg.lstsq` on the Vandermonde matrix directly. Same model, different
conditioning, so results agree to fit tolerance and not bitwise. Golden outputs
must therefore be generated and compared per backend, with `use_numba` pinned.

---

## 6. Hyperparameters and hardcoded constants

Defaults that act as analysis choices, plus literals buried in bodies.

### Defaults in signatures

| Line | Name | Value |
|---|---|---|
| 54 | `G` | `4.3009e-6` |
| 229 | `method` | `"kde"` |
| 230 | `density_frac` | `0.9` |
| 231 | `nbins` | `100` |
| 232 | `smoothing_sigma` | `1.0` |
| 392 | `poly_degree` | `5` |
| 705 | `scheme` | `'bin_size'` |
| 707 | `bin_size` | `1/8.` |
| 708 | `min_particles` | `300` |
| 709 | `normal_ref_size` | `20000` |
| 710 | `normal_ref_seed` | `12345` |
| 964 | `ddof` | `1` |
| 1061 | `mc_trials`, `normal_ref_size` | `2000`, `20000` |
| 1062 | `alpha`, `seed`, `dpi` | `0.05`, `0`, `200` |
| 1230 | `num_points` | `300` |
| 1231 | `trials` | `2000` |
| 1232 | `normal_ref_size` | `20_000` |
| 1233 | `alpha` | `0.05` |
| 1234 | `normal_ref_seed` | `None` |
| 1328 | `alpha` | `0.05` |
| 1350 | `smoothing_frac` | `0.02` |
| 1351 | `smoothing_mode` | `"reflect"` |
| 1352 | `detrend` | `"none"` |
| 1353 | `poly_order` | `3` |
| 1642 | `nperseg_frac` | `0.25` |
| 1643 | `noverlap_frac` | `0.5` |
| 1644 | `scaling` | `"density"` |
| 1762 | `n_realizations` | `500` |
| 1763 | `detrend` | `"poly"` |
| 1764 | `poly_order` | `3` |
| 1946 | `snr_threshold` | `3.0` |
| 1947 | `min_bins` | `2` |
| 1948 | `conservative_nyquist_frac` | `0.9` |
| 1949 | `method` | `"band_snr"` |

### Literals inside bodies

| Line | Literal | Meaning |
|---|---|---|
| 785 | `min(3, phi1_local.size-1)` | Cubic detrend of phi2 against phi1 before binning. |
| 829 | `ddof=1` | Per-bin sample std. |
| 1024 | `< 1.00` | Bin is Gaussian if the normalized Wasserstein is below the 95% threshold. |
| 1039, 1113 | `> 0.05` | KS p-value cut. |
| 1287 | `A, p, B = 2.3, -0.52, 0.00590098` | The precomputed 95% null-Wasserstein power law. Fitted for a reference sample of 20000. |
| 1551 | `smoothing_frac or 0.02` | Second hardcoded copy of the 0.02 default. |
| 1601 | `max(1.0, 0.10 * max_nbins)` | Default lowpass trend sigma. |
| 1716, 1823 | `max(8, ...)` | Welch `nperseg` floor. |
| 1722 | `int(0.5 * nperseg)` | Overlap when `nperseg` is clipped to `nbins`. |
| 1844 | `np.ones(nbins) / nbins` | Uniform null probability vector. |
| 1876 | `mode='reflect'` | Hardcoded, unlike the configurable mode in `compute_linearized_density`. |
| 1918 to 1920 | `16.0`, `84.0`, `95.0` | Percentile summaries. |
| 2068 to 2070 | `95.0`, `16.0`, `84.0` | Same, on the interpolated realizations. |
| 2191 | `95.0` | ratio95 fallback percentile. |

### Pipeline-level values, set at the call site in `return_calc_props_df`

These are not in this file but are part of the paper's contract.

| Value | Where |
|---|---|
| `bin_width=0.25` | `compute_linearized_density` call |
| `smoothing_sigma=1` | same call |
| `detrend="poly"` | same call |
| `smoothing_sigma_bins=1` | `compute_sampling_psd_realizations` call |
| `n_realizations=1000` | same call, overriding the default 500 |
| `rng_seed=12345` | same call |
| `method='ratio95'` | `detect_stream_psd_metrics` call, overriding the default `band_snr` |
| `L/8` | implicit, phi1 is divided by `L` before `stream_local_binned_stats` runs with `bin_size=1/8.` |

The `method` disagreement is a genuine contract ambiguity and is raised as a
checkpoint question.

---

## 7. Module-level side effects, prints, writes, globals

| Line | Item |
|---|---|
| 25 | `__docformat__ = "numpy"` global. |
| 38 to 48 | `try: from numba import ...` with an `except ImportError` fallback. Sets the module globals `NUMBA_AVAILABLE`, `prange`, `njit`. |
| 48 | `print("Numba not found. Running with pure NumPy/SciPy ...")` at import time. The only module-level print. |

No file writes, no directory creation, no matplotlib style mutation, no random
seeding at import. Nothing else executes at import.

Two prints occur inside functions in the external pipeline driver, not here.

One in-function mutation worth recording: `compute_bin_diagnostics` writes four
new columns onto the caller's DataFrame in place, at lines 1015, 1019, 1022 and
1024. It takes no copy. `plot_metric_stats_detailed` does take a defensive copy
at line 1090.

---

## 8. Docstring and code name mismatches

| Line | Mismatch |
|---|---|
| 1212 | The `Examples` block of `_get_max_consecutive_true_fraction` calls `get_max_consecutive_true_fraction`, without the leading underscore. No such name exists. |
| 1063 to 1087 | `plot_metric_stats_detailed` documents its return as `fig_summary (or None), metrics (dict), df_out`, a 3-tuple. The code returns the 2-tuple `metrics, df_out` at line 1193, and the type hint at line 1063 also says 2-tuple. The figure is created and discarded. |
| 1279 to 1282 | `wasserstein_null_threshold` documents the `power_law_fit=False` branch as returning `tuple[float, np.ndarray]`. Line 1309 returns a bare float, with the array return commented out. |
| 1665 | `compute_welch_psd_for_resid` documents `nperseg_frac` as "Default 0.125". The signature says `0.25`. |
| 965, 980 | `compute_bin_diagnostics` is annotated and documented as returning a 6-tuple. It returns 7 values. |
| 823 to 825 | `compute_bin_stats` is annotated `list[int, float x 6]`, 7 entries, and returns 7 on the populated path but only 6 on the empty-bin path. |
| 1806 | `compute_sampling_psd_realizations` documents a `'psd_mean'` and `'psd_std'` key list ending in an ellipsis. The actual dict also carries `nperseg` and `noverlap`. Accurate, just abbreviated.

---

## 9. Imports

Used: `numpy`, `pandas`, `matplotlib.pyplot`, `scipy.stats.gaussian_kde`,
`kstest`, `wasserstein_distance`, `combine_pvalues`,
`scipy.ndimage.gaussian_filter1d`, `scipy.integrate.trapezoid`,
`scipy.signal.welch`, `scipy.signal.detrend`, `typing.Any`, `typing.Callable`.

Unused: `matplotlib.colors.LogNorm` (line 30).

Referenced but never imported: `List` at line 1704 and 1705, `Tuple` and `List`
at line 2130. These appear only in local variable annotations, which
`from __future__ import annotations` leaves unevaluated, so they do not raise.
They would raise if the annotations were ever evaluated, for example by
`typing.get_type_hints`.

Shadowing: the module imports `scipy.signal.detrend` at line 36 and then uses
`detrend` as a parameter name in `compute_vel_disp_stream`,
`_compute_vel_disp_numpy`, `_compute_vel_disp_numba`, `compute_linearized_density`
and `compute_sampling_psd_realizations`. The only body that calls the scipy
function is `compute_orbit_properties` at line 172, which has no such parameter,
so the shadowing is currently harmless.
