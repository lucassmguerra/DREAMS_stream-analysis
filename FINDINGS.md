# FINDINGS

Logical issues found while refactoring `reference/stream_analysis_source.py`.
Reported, not fixed, and pinned by a golden test. The two exceptions are marked
FIXED and listed under Divergences at the end. Numbering is stable, so entries
keep their original severity heading even where the assessment changed.

## bug

**1. `binned.compute_local_binned_stats`, source line 823.**
`compute_bin_stats` returns six values for an empty bin and seven for a populated
one, so the `pd.DataFrame(..., columns=[7 names])` at line 869 raises.
Bites on any stream with an empty phi1 bin, which is every stream short or clumpy
enough that one of the eight bins receives no particles.
```python
if len(values) == 0:
    return [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
```
Severity: `bug`. Golden `binned_stats_edge_empty_bins`.

**2. `kinematics._compute_vel_disp_numba`, source line 636.**
numba types the whole body regardless of the runtime value of `detrend`, so
`phi1[i, j]` inside the `if detrend:` branch fails to compile when `phi1` is None.
Bites on `compute_velocity_dispersion(xv, mask)` with numba installed and
`detrend=False`, which raises `TypingError` instead of returning a dispersion.
```python
# in the public wrapper, before dispatch
if phi1 is None:
    phi1 = np.zeros((S, N), dtype=np.float64)
```
Severity: `bug`. Golden `kinematics_plain_numba__unperturb`.

**3. `spectra.detect_stream_psd_metrics`, source line 2043.**
`pos_mask_all` is bound only inside `if np.any(freqs <= 0.0)`, but is used
unconditionally when `psd_all` is given.
Bites when a caller passes a strictly positive frequency vector together with
`psd_all`, which raises `NameError` rather than working.
```python
pos_mask_all = freqs > 0.0          # bind unconditionally, before the filter
freqs, psd_obs = freqs[pos_mask_all], psd_obs[pos_mask_all]
```
Severity: `bug`.

**4. `statistics.wasserstein_null_threshold`, source line 1300.**
`return np.nan, _` references `_`, which is a local bound only by the loop 6
lines below, so the branch raises `UnboundLocalError`.
Bites on `wasserstein_null_threshold(power_law_fit=False, num_points=0)`.
```python
return np.nan
```
Severity: `bug`. Golden `wasserstein_monte_carlo_zero_points`.

## fragile

**5. `spectra.detect_stream_psd_metrics`, source lines 2159 to 2183. FIXED.**
Under `method="band_snr"`, `min_lambda_band` was the smallest wavelength in *any*
band clearing the SNR threshold, and once a wide low-frequency band cleared it
the answer was always the top of the trusted band.
Measured on all 16 fixture streams it returned exactly `1 / f_top_trusted` every
time, so the column was a constant of the binning and carried no stream-specific
information, while `ratio95` on the same data spanned 0.99 to 8.06 degrees.
```python
if (b - a + 1) == min_bins and lambda_min is not None:
    narrow_lambdas.append(lambda_min)      # only the narrowest bands set lambda
```
Severity: `bug`. Fixed on Arpit's instruction, the only behavior change in the
package. `tests/test_detection.py::test_band_snr_no_longer_saturates`.

**6. `pipeline`, `Stream_morphology_analysis.return_calc_props_df` line 109.**
`df_final[cols] = df_orbits[cols].to_numpy()` assigns positionally onto a fresh
`RangeIndex`, while line 85 uses `df_orbits.index` to index the stream arrays.
Bites whenever `df_orbits.index` is not `0..S-1`, silently pairing each stream
with the wrong orbit, or raising `IndexError` if the index runs past the arrays.
```python
positions = validate_alignment(df_orbits, n_streams)   # raise on mismatch
df_final = pd.DataFrame(metrics, columns=cols, index=df_orbits.index)
```
Severity: `fragile`. Fixed in `build_metrics_table`, which validates and raises.

**7. `_numba`, source line 44.**
The fallback shim is `def njit(nopython=True, cache=True, parallel=True)`, so it
only works when called as `@njit(...)`.
Bites the next time anyone writes a bare `@njit` decoration, which passes the
function in as `nopython` and returns the inner `decorator` instead of the
function.
```python
def njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]                      # bare @njit
    return lambda func: func                # @njit(...)
```
Severity: `fragile`.

**8. `binned.compute_bin_diagnostics`, source lines 1013 to 1024.**
Writes `wass_thresh_95`, `norm_wass_to_95` and `flag_wass_under_95` onto the
caller's DataFrame in place, and fills missing expected columns with NaN, taking
no copy.
Bites anyone who calls it twice on one frame, or who inspects their frame
afterwards and finds columns they did not create. `plot_metric_stats_detailed`
does take a defensive copy, so the two disagree.
```python
def compute_bin_diagnostics(df, ddof=1):
    df = df.copy()
```
Severity: `fragile`.

**9. `binned.compute_local_binned_stats`, source line 716.**
`max_bins_to_plot: int = 24` is accepted and never read anywhere in the body.
Bites a user who sets it to limit a 200-bin figure and gets 200 panels.
```python
# either honour it
n_bins_plotted = min(n_bins, max_bins_to_plot)
```
Severity: `fragile`.

**10. `spectra.detect_stream_psd_metrics`, source line 2227.**
The `detailed` dict returns `trusted_mask` sized to the DC-stripped frequency
vector, but does not return that vector, and the docstring promises a `'freqs'`
key that does not exist.
Bites anyone who tries `freqs[out["trusted_mask"]]` with their own `freqs`, which
raises `IndexError` on the length mismatch.
```python
out["freqs"] = freqs        # the DC-stripped vector the mask indexes
```
Severity: `fragile`.

**11. `binned`, `kinematics`, `density`, `spectra`, source line 36.**
`scipy.signal.detrend` is imported at module scope and then shadowed by a
parameter named `detrend` in five functions.
Bites the next person who calls the scipy function inside one of those bodies and
gets a bool or a string instead.
```python
from scipy.signal import detrend as signal_detrend
```
Severity: `fragile`. Done in `orbits`, which is the only module that calls it.

**12. `statistics`, source lines 1017 to 1032 and 1128 to 1142.**
The six statements computing `wass_thresh_95`, `norm_wass_to_95`,
`flag_wass_under_95` and `frac_flag_wass` are duplicated verbatim between
`compute_bin_diagnostics` and `plot_metric_stats_detailed`.
Bites the next time either is edited, because the two will disagree silently and
the pipeline uses only one of them.
```python
_normalize_wasserstein(df)      # one implementation, both callers
```
Severity: `fragile`. Factored into `statistics._normalize_wasserstein`.

## cosmetic

**13 to 17. Docstrings that disagree with their code.**
Five signatures whose documentation states something the code does not do.
Each bites anyone who codes against the documentation instead of reading the body.
All five corrected in the moved docstrings. No behavior changed.

| # | Location, source line | Documented | Actual |
|---|---|---|---|
| 13 | `plot_metric_stats_detailed`, 1087 | returns `fig, metrics, df_out` | returns `metrics, df_out`, figure discarded |
| 14 | `wasserstein_null_threshold`, 1280 | MC branch returns `(float, ndarray)` | returns a bare float |
| 15 | `compute_welch_psd_for_resid`, 1665 | `nperseg_frac` default 0.125 | 0.25 |
| 16 | `compute_bin_diagnostics`, 965 | returns a 6-tuple | returns 7 values |
| 17 | `_get_max_consecutive_true_fraction`, 1212 | example calls `get_max_...` | no such name, it is `_get_max_...` |

Severity: `cosmetic`.

**18. `_compute_vel_disp_numba`, source lines 622 and 631.**
The `x` buffer is allocated and filled with `xv[i, j, 0]` every iteration and
never read.
Costs one array allocation and N writes per stream, for nothing.
```python
# delete both lines
```
Severity: `cosmetic`.

**19. `detect_stream_psd_metrics`, source lines 2067, 2069 and 2070. FIXED.**
`null_std`, `null_16` and `null_84` were computed from the realizations and never
used. `null_std` is the denominator of a per-frequency SNR, so this reads as an
abandoned implementation rather than as dead code.
It bit by leaving the function with no true signal-to-noise test at all, which is
what `ratio95` at a threshold of 3 was standing in for.
```python
snr_per_freq = (psd_obs - mu_null) / null_std      # method="peak_snr"
```
Severity: `bug`. `null_std` now drives the new `peak_snr` method, which is the
default. `null_16` and `null_84` were removed. See the note below.

**20. `return_calc_props_df`, `Stream_morphology_analysis.py` lines 103 and 142.**
`df_final.index.name = "stream_index"` is set, then dropped again by the
`pd.concat` that appends the power-spectrum block.
Bites anyone reading the in-memory table, whose index comes back unnamed while
the parquet on disk carries a name.
```python
df_final = pd.concat([df_final, df_newcols], axis=1)
df_final.index.name = "stream_index"
```
Severity: `cosmetic`. `build_metrics_table` keeps the name.

**21. Module header, source line 30.**
`from matplotlib.colors import LogNorm` is never used.
Costs nothing but a reader's time.
```python
# delete the import
```
Severity: `cosmetic`. Dropped.

**22. Module header, source lines 1704, 1705 and 2130.**
`List` and `Tuple` are used in local variable annotations but never imported.
Harmless today because `from __future__ import annotations` leaves local
annotations unevaluated, but `typing.get_type_hints` on those frames would raise.
```python
freqs_list: list[np.ndarray] = []
```
Severity: `cosmetic`. Changed to the builtin generics.

---

## Deliberate divergences from the frozen source

Entries 5 and 19 were fixed on Arpit's instruction, because the defect was an
error of understanding rather than a quirk worth preserving. These are the only
places where the package does not reproduce the frozen source. Everything else
above is preserved bitwise and has a golden.

**`method="peak_snr"`, new, and now the default everywhere.**
`SNR(f) = ( P_obs(f) - mu_null(f) ) / sigma_null(f)`, reporting `1/f` at the
highest trusted frequency clearing the threshold. Default 5.0, the familiar
5-sigma. The only one of the three methods that is an actual signal-to-noise
ratio. `sigma_null` is the `null_std` of entry 19.

**`method="band_snr"`, corrected.** `min_lambda_band` now comes from bands of
exactly `min_bins` frequency bins, which keeps the test local. Its statistic
`I_obs / std(I_mc)` and its 3.0 threshold are unchanged.

**`method="ratio95"`, untouched.** `P_obs / P_null_95 >= 3`. A power ratio, not
an SNR. It produced the published values and still reproduces them, via
`SPECTRA.published_method` and `SPECTRA.published_snr_threshold`.

**`build_metrics_table`** defaults to `peak_snr` at 5 sigma. Only the
`min_lambda_band` column differs from the published table. The `detailed=True`
dict gained `freqs`, `psd_null_std` and `snr_per_freq`, and an unknown `method`
is now rejected before any work rather than after.

`min_lambda_band` in degrees, unperturbed m12i:

| stream | `peak_snr` 5 | `band_snr` 3 | `ratio95` 3 | old `band_snr` |
|---|---|---|---|---|
| 0 | 0.59 | 0.95 | 1.18 | 0.59 |
| 2 | 1.83 | 0.61 | 1.83 | 0.61 |
| 3 | 3.08 | 0.58 | 3.08 | 0.58 |
| 5 | 2.47 | 1.65 | 2.47 | 0.55 |
| 6 | 2.48 | 0.62 | 8.06 | 0.56 |
| 8 | 4.32 | 0.76 | 4.32 | 0.56 |
| 10 | 0.60 | 0.59 | 6.81 | 0.56 |

The old column was `1 / f_top_trusted` in every row.
