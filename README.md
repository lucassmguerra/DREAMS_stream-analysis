# stream-analysis

Disturbance metrics for N-body globular-cluster streams. This is the library
behind the measurements in *No Stream Left Unscathed* (Arora et al.).

A stream that has been left alone stays smooth. One that has passed through
something does not. The package measures that difference two ways. It cuts the
stream into bins along its track and asks, in each bin, whether the
cross-sectional distribution still looks Gaussian. And it takes the power
spectrum of the along-stream density and asks what scale of structure rises
above the noise you would get from finite sampling alone.

It is a library of helper functions, not a script. Nothing here imports
`nbody_streams` or `agama`, and nothing writes to disk unless you ask it to.

## What changed, and what did not

This is a reorganization of a single 2239-line file into a package. Not one
number moved. Every function returns bitwise-identical output for identical
input, and the test suite proves it against golden outputs generated from a
frozen copy of the original, which lives at
`reference/stream_analysis_source.py` and is never imported by the package.

Several functions were renamed. Every old name still works through a
deprecation shim. See [RENAMES.md](RENAMES.md).

Logical issues found along the way are reported in [FINDINGS.md](FINDINGS.md)
and were deliberately not fixed. Two of them change what you should pass. Read
entries 1, 2 and 5 before running anything unfamiliar.

## Install and import

The package is meant to be imported off `PYTHONPATH`, the way it always was.

```bash
export PYTHONPATH=~/py_scripts:$PYTHONPATH
```

```python
import stream_analysis as sa
```

An editable install works too, if you would rather not manage the path.

```bash
cd ~/py_scripts/stream_analysis
pip install -e .
pip install -e '.[numba]'     # optional, accelerates the velocity dispersion
```

Requires Python 3.10 or later, numpy, scipy, pandas and matplotlib. numba is
optional. There is no console entry point.

## Quickstart

End to end, from a phase-space array to the metrics table.

```python
import numpy as np
import pandas as pd
import stream_analysis as sa

# 1. Stream coordinates. This step lives outside the package, because the
#    transform belongs to nbody_streams. xv is (S, N, 6) galactocentric phase
#    space, xv_prog is (S, 6) present-day progenitor vectors.
from nbody_streams import coords

phi1, phi2 = coords.generate_stream_coords(
    xv=xv, xv_prog=xv_prog, optimizer_fit=True,
)

# 2. Length, width, and the membership mask everything downstream filters on.
lengths, widths, masks = sa.measure_stream_length_width(
    phi1, phi2, method="kde", density_frac=0.9, return_mask=True,
)

# 3. Total velocity dispersion, with the along-track trend removed.
sigma_v = sa.compute_velocity_dispersion(
    xv, masks, detrend=True, phi1=phi1,
)

# 4. The whole table. df_orbits carries one row per stream with the columns
#    min_pericenter_dist, max_apocenter_dist, mean_dist, n_pericenters, period.
table = sa.build_metrics_table(
    phi1, phi2, masks, lengths, widths, sigma_v, df_orbits,
)
```

`table` has one row per stream and the twenty columns of the reference table
below, in that order, indexed by `df_orbits.index` under the name
`stream_index`.

Pass `checkpoint_path="somewhere.parquet"` to have the binned half written out
before the power spectra run, which is worth doing on a few thousand streams.
Pass `progress=tqdm` to watch it.

### Working one stream at a time

The pieces are usable on their own, and this is what `build_metrics_table` does
per stream.

```python
i, m = 0, masks[0]

# Localized binned family. phi1 is divided by the stream length first, so the
# default bin_size of 1/8 gives the eight bins of the paper.
bins = sa.compute_local_binned_stats(phi1[i, m] / lengths[i], phi2[i, m])
(median_norm_wass, max_norm_wass, frac_flag_wass, frac_consecutive_wass,
 frac_p_gt_005, mean_std_bins, std_std_bins) = sa.compute_bin_diagnostics(bins)

# Power-spectrum family.
dens = sa.compute_linearized_density(
    phi1[i, m], detrend="poly", smoothing_sigma=1, bin_width=0.25,
)
psd = sa.compute_welch_psd(dens["resid"], dens["w"])
null = sa.compute_sampling_psd_realizations(
    dens["hist_counts"], smoothing_sigma_bins=1, w=dens["w"],
    n_realizations=1000, rng_seed=12345,
)
rms_obs, rms_null, excess_power, min_lambda_band = sa.detect_stream_psd_metrics(
    psd["freqs"], psd["psd"], w=dens["w"],
    psd_all=null["psd_all"], nperseg=null["nperseg"], method="ratio95",
)
```

`compute_bin_diagnostics` writes three columns onto `bins` in place and takes no
copy. Pass `bins.copy()` if that matters to you. See FINDINGS entry 8.

### The binned family is generic in its quantity

Nothing in `compute_local_binned_stats` is about phi2. It bins any per-particle
quantity along the track and tests its local distribution against a normal. The
paper applies it to phi2. Applied to the line-of-sight velocity it gives the
kinematic analogues of the same three metrics.

```python
v_los = np.sum(xv[i, m, :3] * xv[i, m, 3:], axis=1) / np.linalg.norm(xv[i, m, :3], axis=1)

bins_v = sa.compute_local_binned_stats(
    phi1[i, m] / lengths[i], v_los, quantity_name="v_los",
)
metrics_v = sa.compute_bin_diagnostics(bins_v, quantity_name="v_los")
```

`quantity_name` sets the suffix on the two columns named after the quantity,
`mean_v_los` and `std_v_los`. It enters no computation. The bins, the binning
and every statistic are identical either way.

## Metric reference

One row per column of `build_metrics_table`. Sign convention says which
direction means more disturbed.

### Localized binned family

| Code name | Paper name | Symbol | What it is | Units | Sign |
|---|---|---|---|---|---|
| `median_norm_wass` | Global disturbance | TBD | Median across bins of the Wasserstein distance from the local distribution to a standard normal, divided by the 95% null threshold for that bin's particle count. A value of 1 is the edge of consistency with Gaussian. | dimensionless | higher is more disturbed |
| `max_norm_wass` | Peak disturbance | TBD | The same quantity at its worst bin. Catches a stream disturbed in one place that is otherwise clean. | dimensionless | higher is more disturbed |
| `coef_of_var_stds` | Width variation | `C_w` | `std_std_bins / mean_std_bins`. How much the stream's width varies along its own length, relative to its mean width. | dimensionless | higher is more disturbed |
| `mean_std_bins` | diagnostic only | | Mean across bins of the per-bin standard deviation of the binned quantity. The average local width. | degrees | higher is wider, not more disturbed |
| `std_std_bins` | diagnostic only | | Scatter across bins of that same per-bin standard deviation. The numerator of `C_w`. | degrees | higher is more disturbed |
| `frac_flag_wass` | diagnostic only | | Fraction of bins whose normalized Wasserstein distance falls below 1, so still consistent with Gaussian at 95%. | fraction, 0 to 1 | lower is more disturbed |
| `frac_consecutive_wass` | diagnostic only | | Longest unbroken run of such bins, as a fraction of the bin count. Separates a stream disturbed in one place from one disturbed throughout. | fraction, 0 to 1 | lower is more disturbed |
| `frac_p_gt_005` | diagnostic only | | Fraction of bins whose KS test against a standard normal returns p > 0.05. The KS counterpart of `frac_flag_wass`. | fraction, 0 to 1 | lower is more disturbed |

### Power-spectrum family

| Code name | Paper name | Symbol | What it is | Units | Sign |
|---|---|---|---|---|---|
| `min_lambda_band` | Minimum detectable scale | `lambda_min` | Shortest along-stream wavelength at which the observed density power rises above the sampling noise floor. The finest structure the data can actually resolve. | degrees | lower means finer structure is detectable |
| `rms_obs` | RMS density residual | `RMS_delta` | Square root of the observed power integrated across the trusted frequency band. The total amplitude of fractional density fluctuation. | dimensionless | higher is more structured |
| `excess_power` | Excess power | `P_excess` | Integral across the trusted band of the observed power above the null median, clipped at zero. The part of the fluctuation that finite sampling does not explain. | dimensionless, power times frequency | higher is more disturbed |
| `rms_null` | diagnostic only | | The same integral as `rms_obs`, taken on the Monte Carlo realizations and reduced by their median. The noise floor `rms_obs` should be read against. | dimensionless | reference level, not a measurement of the stream |

### Stream properties

| Code name | Paper name | Symbol | What it is | Units | Sign |
|---|---|---|---|---|---|
| `length_deg` | Length | `ℓ` | Angular span along phi1 of the region enclosing 90% of the smoothed particle density. | degrees | not a disturbance metric |
| `width_deg` | Width | `W` | Quantile width in phi2 of the particles inside that region, at the same 90%. | degrees | not a disturbance metric |
| `vel.std.tot_km_s` | TBD | `v_LOS` | Total 3D velocity dispersion of the masked particles, `sqrt(var_vx + var_vy + var_vz)`, after a degree-5 polynomial in phi1 is removed from each component. | km/s | not a disturbance metric |

### Orbit properties

Copied through from `df_orbits` untouched. `min_pericenter_dist`,
`max_apocenter_dist` and `mean_dist` in kpc, `n_pericenters` a count, `period`
in the time units of the orbit integration.

### Two notes on this table

`vel.std.tot_km_s` is symbol `v_LOS` above because that is the paper's symbol,
but the code computes the **total 3D** dispersion, not the line-of-sight
component. The description column says what the code does. If the paper reports
a line-of-sight dispersion, that is a different quantity from this column and
the mapping needs another look.

The three `TBD` symbols are unfilled because inventing a symbol is worse than
leaving a gap. Same for the `vel.std.tot_km_s` paper name.

### `min_lambda_band` and the choice of method

`build_metrics_table` defaults to `method="ratio95"`, which is what produced the
published values.

The other method, `"band_snr"`, is described in the `detect_stream_psd_metrics`
docstring as preferred. It does not behave as one. Its scan records, for every
contiguous frequency band clearing an SNR of 3, the wavelength at the band's
**upper** edge, then reports the minimum. Any stream with real structure has some
wide band reaching the top of the trusted range that clears the threshold
easily, so the minimum is pinned there. Measured across all 16 fixture streams,
`band_snr` returns exactly `1 / f_top_trusted` every time, a constant of the
binning, while `ratio95` on the same streams spans 0.99 to 8.06 degrees. See
[FINDINGS.md](FINDINGS.md) entry 5. `method="band_snr"` remains available.

## Module map

| Module | Covers |
|---|---|
| `geometry` | Stream length, width, and the KDE membership mask. |
| `kinematics` | Velocity dispersion, with an optional polynomial detrend along the track. Dispatches to a numba or a NumPy worker. |
| `binned` | Localized binned statistics and the diagnostics collapsed from them. |
| `density` | Linearized along-stream density and its fractional residuals. |
| `spectra` | Welch power spectra, the Monte Carlo sampling floor, and the detection metrics. |
| `statistics` | Wasserstein null thresholds, Fisher and Bonferroni combination, run-length helper. |
| `orbits` | Peri, apo, mean radius, period and eccentricity from integrated orbits. |
| `plotting` | Figure helpers that are not part of any metric's definition. |
| `pipeline` | `build_metrics_table`. |
| `config` | Every analysis hyperparameter, at the paper's values. |
| `legacy` | Quarantine. Empty, see [LEGACY.md](LEGACY.md). |
| `_compat` | Deprecation shims for the old names. |
| `_numba` | The optional numba import and its pure-NumPy fallback. |

`config` collects the hyperparameters that were scattered across signatures and
function bodies. It is documentation with a type, not a control surface. Every
function keeps its own literal defaults, so nothing has to change to call them.

```python
sa.BINNED.bin_size            # 0.125, the L/8 of the paper
sa.SPECTRA.rng_seed           # 12345
sa.WASSERSTEIN_NULL_FIT.A     # 2.3, from the precomputed 95% null power law
```

## Testing

```bash
mamba activate aarora_py
python -m pytest tests/ -q                 # 206 tests, about 90 seconds
python -m pytest tests/ -q -m "not slow"   # skip the end-to-end pipeline cases
```

The suite is a contract, not a set of unit tests. `tests/test_contract.py`
parametrizes over 174 cases defined in `tools/cases.py`, runs each against the
package, and compares it bitwise against a golden generated from the frozen
source before any code moved. Float scalars are compared through their exact
hexadecimal representation, arrays with `numpy.testing.assert_array_equal`, and
DataFrames column for column, dtype for dtype, index and all.

Twenty-seven of those cases record a raised exception rather than a value. That
is deliberate. Reproducing a failure is part of the contract, so a function that
raises on an all-NaN column has to keep raising the same exception with the same
message.

`tests/test_pipeline.py` checks `build_metrics_table` against a re-implementation
of the original `return_calc_props_df` in `tools/pipeline_ref.py`.
`tests/test_compat.py` checks that no name that resolved before the refactor
fails to resolve after it.

### Regenerating the fixtures

The stream-coordinate fixtures total 19 MB and are gitignored. Rebuild them from
the raw HDF5, which is also gitignored and never leaves `/mnt/d`.

```bash
mkdir -p data
cp /mnt/d/Research/GC_streams/example_m12i_streams_forecasting.hdf5 data/
PYTHONPATH=~/py_scripts python tools/build_fixtures.py
```

That is the only script allowed to import `nbody_streams`. It runs the
coordinate transform once, at `optimizer_fit=True`, and freezes the result.

The goldens are committed and must not be regenerated after a refactor, which
would make the tests vacuous. `tools/build_goldens.py` refuses to overwrite
without `--force`.

### On the fixture coordinates

The fixtures are built with `nbody_streams.coords.generate_stream_coords`, while
the published m12i table was built with a different implementation of the same
transform. The pole-tilt optimizer lands in a slightly different place, so the
two disagree a little. Measured on the 16 unperturbed fixture streams, as a
relative difference:

| Column | median | max |
|---|---|---|
| `length_deg` | 5e-05 | 5e-04 |
| `min_lambda_band` | 1e-04 | 1e-01 |
| `max_norm_wass` | 2e-04 | 4e-02 |
| `coef_of_var_stds` | 2e-04 | 3e-03 |
| `median_norm_wass` | 1e-03 | 1e-01 |
| `width_deg` | 5e-03 | 2e-01 |

This is a property of the coordinate frame, not of this package. The contract
tests compare against goldens computed on these same fixtures, so they are
unaffected. `tests/test_pipeline.py` reports the table above rather than
asserting on it.

## Citation

If you use these metrics, cite the paper.

> Arora et al., *No Stream Left Unscathed*.

Full reference to be added on publication.

## License

MIT. See [LICENSE](LICENSE).
