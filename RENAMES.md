# RENAMES

Every old name still resolves. The shims in `stream_analysis/_compat.py` forward
to the new function unchanged and emit a `DeprecationWarning`. Nothing has to be
renamed for old code to keep working, so treat the sed block at the bottom as
optional cleanup rather than a migration you owe anyone.

## Functions

| Old name | New name |
|---|---|
| `measure_stream_LengthWidth` | `measure_stream_length_width` |
| `compute_vel_disp_stream` | `compute_velocity_dispersion` |
| `stream_local_binned_stats` | `compute_local_binned_stats` |
| `compute_welch_psd_for_resid` | `compute_welch_psd` |
| `plot_metric_stats_detailed` | `summarize_bin_metrics` |
| `_get_max_consecutive_true_fraction` | `_max_consecutive_true_fraction` |

Unchanged: `compute_orbit_properties`, `compute_bin_diagnostics`,
`wasserstein_null_threshold`, `combine_ks_pvalues_fisher`,
`bonferroni_correction`, `compute_linearized_density`,
`compute_sampling_psd_realizations`, `detect_stream_psd_metrics`.

## Keyword arguments

| Function | Old keyword | New keyword |
|---|---|---|
| `compute_local_binned_stats` | `phi2` | `quantity` |

Positional argument order is unchanged everywhere, so any call that passed the
quantity positionally needs no edit at all. Only `phi2=` by keyword is affected,
and that still works with a warning.

The rename is because the function was never about phi2. It bins any
per-particle quantity along the track. Applied to `v_los` it gives the kinematic
analogues of the same metrics. The new `quantity_name` keyword sets the suffix on
the two columns named after the quantity, `mean_<name>` and `std_<name>`. Its
default is `"phi2"`, so the output columns are unchanged unless you ask for
something else.

`compute_bin_diagnostics` takes a matching `quantity_name`, defaulting to
`"phi2"`, so it knows which standard-deviation column to read.

## New keyword arguments

| Function | Keyword | Default | Purpose |
|---|---|---|---|
| `compute_local_binned_stats` | `quantity_name` | `"phi2"` | Column suffix and axis label. Enters no computation. |
| `compute_bin_diagnostics` | `quantity_name` | `"phi2"` | Which `std_<name>` column to read. |

## sed

Run against a copy first. These are word-boundary matches, so they will not
touch a longer name that happens to contain one of these.

```bash
# Notebooks and scripts, in place.
find . -name '*.py' -o -name '*.ipynb' | xargs sed -i \
  -e 's/\bmeasure_stream_LengthWidth\b/measure_stream_length_width/g' \
  -e 's/\bcompute_vel_disp_stream\b/compute_velocity_dispersion/g' \
  -e 's/\bstream_local_binned_stats\b/compute_local_binned_stats/g' \
  -e 's/\bcompute_welch_psd_for_resid\b/compute_welch_psd/g' \
  -e 's/\bplot_metric_stats_detailed\b/summarize_bin_metrics/g' \
  -e 's/\b_get_max_consecutive_true_fraction\b/_max_consecutive_true_fraction/g'
```

The keyword rename, only if you passed it by name:

```bash
find . -name '*.py' -o -name '*.ipynb' | xargs sed -i \
  -e 's/\(compute_local_binned_stats([^)]*\)\bphi2=/\1quantity=/g'
```

That last one only catches single-line calls. Check the result before committing.

## Checking your notebooks

Turn the warnings into errors and run a cell. Anything still on an old name will
raise and name its replacement.

```python
import warnings
warnings.simplefilter("error", DeprecationWarning)
```
