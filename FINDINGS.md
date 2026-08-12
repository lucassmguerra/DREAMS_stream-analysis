# FINDINGS

Logical issues found while refactoring `reference/stream_analysis_source.py`.
Reported, not fixed. Every one of them is preserved bitwise in the package and
pinned by a golden test.

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

**5. `spectra.detect_stream_psd_metrics`, source lines 2159 to 2183.**
Under `method="band_snr"`, `min_lambda_band` is the smallest wavelength in *any*
band clearing the SNR threshold, and once a wide low-frequency band clears it the
answer is always the top of the trusted band.
Measured on all 16 fixture streams, `band_snr` returns exactly
`1 / f_top_trusted` every time, so the column is a constant of the binning and
carries no stream-specific information, while `ratio95` on the same data spans
0.99 to 8.06 degrees.
```python
# report the highest-frequency band that clears the threshold on its own,
# not the top edge of the widest band that does
lambda_vals = [1.0 / freqs_tr[b] for (a, b) in bands if (b - a + 1) == min_bins]
```
Severity: `fragile`.

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

**13. `plot_metric_stats_detailed`, source lines 1087 and 1193.**
Docstring promises `fig_summary (or None), metrics, df_out`, the code returns
`metrics, df_out` and drops the figure handle.
Bites anyone unpacking three values, and anyone who wanted the figure.
```python
return fig_summary, metrics, df_out
```
Severity: `cosmetic`. Docstring corrected in `plotting.summarize_bin_metrics`.

**14. `wasserstein_null_threshold`, source lines 1280 and 1309.**
Docstring promises `tuple[float, np.ndarray]` from the Monte Carlo branch, the
code returns a bare float with the array commented out at the return.
Bites anyone unpacking two values.
```python
return float(np.quantile(ws, 1.0 - alpha)), np.array(ws)
```
Severity: `cosmetic`. Docstring corrected.

**15. `compute_welch_psd_for_resid`, source line 1665.**
Docstring says `nperseg_frac` defaults to 0.125, the signature says 0.25.
Bites anyone sizing their Welch segments from the documentation.
```python
nperseg_frac : float, optional
    Fraction of `nbins` to use for `nperseg` in Welch. Default 0.25.
```
Severity: `cosmetic`. Docstring corrected.

**16. `compute_bin_diagnostics`, source lines 965 and 980.**
Annotated and documented as returning a 6-tuple, returns 7 values.
Bites anyone unpacking by the documentation.
```python
) -> tuple[float, float, float, float, float, float, float]:
```
Severity: `cosmetic`. Docstring corrected.

**17. `_get_max_consecutive_true_fraction`, source line 1212.**
The `Examples` block calls `get_max_consecutive_true_fraction`, without the
leading underscore. No such name exists.
Bites anyone who runs the doctest.
```python
>>> _max_consecutive_true_fraction(arr)
```
Severity: `cosmetic`. Corrected.

**18. `_compute_vel_disp_numba`, source lines 622 and 631.**
The `x` buffer is allocated and filled with `xv[i, j, 0]` every iteration and
never read.
Costs one array allocation and N writes per stream, for nothing.
```python
# delete both lines
```
Severity: `cosmetic`.

**19. `detect_stream_psd_metrics`, source lines 2067, 2069 and 2070.**
`null_std`, `null_16` and `null_84` are computed from the realizations and never
used, and `null_std` is then set to None on the other branch.
Costs three full passes over a (1000, nfreq) array per stream.
```python
# delete all three, or return them in the detailed dict
```
Severity: `cosmetic`.

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
