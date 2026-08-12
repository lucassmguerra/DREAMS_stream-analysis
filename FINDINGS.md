# FINDINGS

Twenty-two logical issues found while refactoring
`reference/stream_analysis_source.py`. All fixed. Each entry gives the location
in the frozen source, what was wrong, when it bit, the patch, and a severity.

Entries 1 to 12 changed behavior and so have no golden. They are pinned by
`tests/test_fixes.py` and `tests/test_detection.py`, which check the fix and,
where the fix is narrow, that everything outside it still matches the frozen
source bitwise. `tools/cases.py` lists them under `DIVERGENCES`. Entries 13 to 22
touched only documentation, dead code or the new pipeline, so every golden still
holds.

## bug

**1. `spectra.detect_stream_psd_metrics`, source lines 2159 to 2183.**
Under `method="band_snr"`, `min_lambda_band` was the smallest wavelength in *any*
band clearing the threshold, and a band spanning the trusted range always clears,
so the answer was always that range's upper edge.
It returned exactly `1 / f_top_trusted` on all 16 fixture streams, a constant of
the binning, while `ratio95` on the same data spanned 0.99 to 8.06 degrees.
```python
if (b - a + 1) == min_bins and lambda_min is not None:
    narrow_lambdas.append(lambda_min)      # only the narrowest bands set lambda
```
Severity: `bug`.

**2. `spectra.detect_stream_psd_metrics`, source lines 2067 to 2070.**
`null_std`, `null_16` and `null_84` were computed from the realizations and never
used, and `null_std` is the denominator of a per-frequency SNR, so this reads as
an abandoned implementation rather than as dead code.
It bit by leaving the function with no true signal-to-noise test at all, which is
what `ratio95` at a threshold of 3 was standing in for.
```python
snr_per_freq = (psd_obs - mu_null) / null_std      # new method="peak_snr"
```
Severity: `bug`. `peak_snr` is now the default, at 5 sigma.

**3. `kinematics._compute_vel_disp_numba`, source line 636.**
numba types the whole body regardless of the runtime value of `detrend`, so
`phi1[i, j]` inside the `if detrend:` branch could not compile when `phi1` was
None.
Bit on `compute_velocity_dispersion(xv, mask)` with numba installed and
`detrend=False`, which raised `TypingError` instead of returning a dispersion.
```python
if phi1 is None:                                   # in the wrapper, before dispatch
    phi1 = np.zeros((S, N), dtype=np.float64)
```
Severity: `bug`.

**4. `spectra.detect_stream_psd_metrics`, source line 2043.**
`pos_mask_all` was bound only inside `if np.any(freqs <= 0.0)` but used
unconditionally to slice `psd_all`.
Bit when a caller passed a strictly positive frequency vector together with
`psd_all`, raising `NameError`. Welch always emits DC, so real calls dodged it.
```python
pos_mask_all = freqs > 0.0          # bind unconditionally, before the filter
if not np.all(pos_mask_all):
    freqs, psd_obs = freqs[pos_mask_all], psd_obs[pos_mask_all]
```
Severity: `bug`.

**5. `statistics.wasserstein_null_threshold`, source line 1300.**
`return np.nan, _` referenced `_`, a local bound only by the loop six lines
below, so the branch raised `UnboundLocalError`.
Bit on `wasserstein_null_threshold(power_law_fit=False, num_points=0)`.
```python
return np.nan
```
Severity: `bug`.

## fragile

**6. `binned.compute_local_binned_stats`, source line 823.**
`compute_bin_stats` returned six values for an empty bin against seven columns.
pandas padded the row with NaN rather than raising, and because all six were
already NaN the padding was invisible.
It bit only through `count` arriving as NaN instead of 0, forcing that column to
float. It would have bitten much harder the moment anyone put a non-NaN value in
that return, because the padding shifts silently.
```python
if len(values) == 0:
    return [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
```
Severity: `fragile`. No metric moves, `compute_bin_diagnostics` already did
`count.fillna(0)`, and none of the paper's streams has an empty bin.

**7. `pipeline`, `Stream_morphology_analysis.return_calc_props_df` line 109.**
`df_final[cols] = df_orbits[cols].to_numpy()` assigned positionally onto a fresh
`RangeIndex`, while line 85 used `df_orbits.index` to index the stream arrays.
Bit whenever `df_orbits.index` was not `0..S-1`, silently pairing each stream with
the wrong orbit, or raising `IndexError` if the index ran past the arrays.
```python
positions = _validate_alignment(df_orbits, n_streams)   # raise on mismatch
df_final = pd.DataFrame(metrics, columns=cols, index=df_orbits.index)
```
Severity: `fragile`. `build_metrics_table` validates and raises.

**8. `binned.compute_bin_diagnostics`, source lines 1013 to 1024.**
Wrote `wass_thresh_95`, `norm_wass_to_95` and `flag_wass_under_95` onto the
caller's DataFrame in place and filled missing expected columns with NaN, taking
no copy.
Bit anyone calling it twice on one frame, and disagreed with
`plot_metric_stats_detailed`, which did take a copy.
```python
df = df.copy()
```
Severity: `fragile`. The returned metrics are unchanged.

**9. `_numba`, source line 44.**
The fallback shim was `def njit(nopython=True, cache=True, parallel=True)`, so it
only worked when called.
Would have bitten the next bare `@njit` decoration, which passes the function in
as `nopython` and returns the inner `decorator` instead of the function.
```python
if len(args) == 1 and callable(args[0]) and not kwargs:
    return args[0]                      # bare @njit
```
Severity: `fragile`. Reachable only when numba is absent.

**10. `binned.compute_local_binned_stats`, source line 716.**
`max_bins_to_plot: int = 24` was accepted and never read anywhere in the body.
Bit a user who set it to limit a 200-bin figure and got 200 panels.
```python
n_bins_plotted = min(n_bins, max_bins_to_plot)
```
Severity: `fragile`. Figures only, no numbers.

**11. `spectra.detect_stream_psd_metrics`, source line 2227.**
The `detailed` dict returned `trusted_mask` sized to the DC-stripped frequency
vector, did not return that vector, and promised a `'freqs'` key that did not
exist.
Bit anyone trying `freqs[out["trusted_mask"]]` with their own `freqs`, raising
`IndexError` on the length mismatch.
```python
out["freqs"] = freqs        # the DC-stripped vector the mask indexes
```
Severity: `fragile`. `psd_null_std` and `snr_per_freq` were added alongside it.

**12. `statistics`, source lines 1017 to 1032 and 1128 to 1142.**
The six statements computing `wass_thresh_95`, `norm_wass_to_95`,
`flag_wass_under_95` and `frac_flag_wass` were duplicated verbatim between
`compute_bin_diagnostics` and `plot_metric_stats_detailed`.
Would have bitten the next time either was edited, because the two would disagree
silently and the pipeline uses only one of them.
```python
_normalize_wasserstein(df)      # one implementation, both callers
```
Severity: `fragile`. Factored into `statistics._normalize_wasserstein`.

## cosmetic

**13 to 17. Docstrings that disagreed with their code.**
Five signatures whose documentation stated something the code did not do.
Each bit anyone coding against the documentation instead of reading the body.
All five corrected in the moved docstrings. No behavior changed.

| # | Location, source line | Documented | Actual |
|---|---|---|---|
| 13 | `plot_metric_stats_detailed`, 1087 | returns `fig, metrics, df_out` | returns `metrics, df_out`, figure discarded |
| 14 | `wasserstein_null_threshold`, 1280 | MC branch returns `(float, ndarray)` | returns a bare float |
| 15 | `compute_welch_psd_for_resid`, 1665 | `nperseg_frac` default 0.125 | 0.25 |
| 16 | `compute_bin_diagnostics`, 965 | returns a 6-tuple | returns 7 values |
| 17 | `_get_max_consecutive_true_fraction`, 1212 | example calls `get_max_...` | no such name |

Severity: `cosmetic`.

**18. `_compute_vel_disp_numba`, source lines 622 and 631.**
The `x` buffer was allocated and filled with `xv[i, j, 0]` on every iteration and
never read.
Cost one array allocation and N writes per stream, for nothing.
```python
# both lines deleted
```
Severity: `cosmetic`. The detrended dispersion is still bitwise identical.

**19. `binned`, `kinematics`, `density`, `spectra`, source line 36.**
`scipy.signal.detrend` was imported at module scope and then shadowed by a
parameter named `detrend` in five functions.
Would have bitten the next person calling the scipy function inside one of those
bodies and getting a bool or a string instead.
```python
from scipy.signal import detrend as _signal_detrend
```
Severity: `cosmetic`. Only `orbits` calls it, and only `orbits` imports it now.

**20. `return_calc_props_df`, `Stream_morphology_analysis.py` lines 103 and 142.**
`df_final.index.name = "stream_index"` was set, then dropped again by the
`pd.concat` appending the power-spectrum block.
Bit anyone reading the in-memory table, whose index came back unnamed while the
parquet on disk carried a name.
```python
df_final = pd.concat([df_final, df_psd], axis=1)
df_final.index.name = "stream_index"
```
Severity: `cosmetic`. `build_metrics_table` keeps the name.

**21. Module header, source line 30.**
`from matplotlib.colors import LogNorm` was never used.
Cost nothing but a reader's time.
```python
# import dropped
```
Severity: `cosmetic`.

**22. Module header, source lines 1704, 1705 and 2130.**
`List` and `Tuple` were used in local variable annotations but never imported.
Harmless, because `from __future__ import annotations` leaves local annotations
unevaluated, but `typing.get_type_hints` on those frames would have raised.
```python
freqs_list: list[np.ndarray] = []
```
Severity: `cosmetic`.

---

## What this means for the published numbers

Nothing in the metrics table moves except `min_lambda_band`, and only because the
default detection method changed on purpose.

- Entries 1 and 2 change `min_lambda_band`. Pass `method=SPECTRA.published_method`
  and `snr_threshold=SPECTRA.published_snr_threshold` to reproduce the paper.
- Entry 6 changes the `count` column's dtype on tables containing an empty bin.
  No metric depends on it, and none of the paper's streams has one.
- Entries 3, 4, 5, 9 and 11 only affect inputs that used to raise.
- Entries 7, 8, 10, 12 and 13 to 22 change no number at any input.

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
