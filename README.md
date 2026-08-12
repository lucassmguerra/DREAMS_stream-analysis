# stream-analysis

Disturbance metrics for N-body globular-cluster streams. This is the library
behind the measurements in *No Stream Left Unscathed*
([arXiv:2605.16200](https://arxiv.org/abs/2605.16200)).

A stream that has been left alone stays smooth. One that has passed through
something does not. The package measures that difference two ways. It cuts the
stream into bins along its track and asks, in each bin, whether the
cross-sectional distribution still looks Gaussian. And it takes the power
spectrum of the along-stream density and asks what scale of structure rises
above the noise you would get from finite sampling alone.

It is a library of helper functions, not a script. Nothing here imports
`nbody_streams` or `agama`, and nothing writes to disk unless you ask it to.

## Install and import

```bash
pip install -e .
pip install -e '.[numba]'     # optional, accelerates the velocity dispersion
```

Or put the parent directory on `PYTHONPATH` and import it directly. Either way:

```python
import stream_analysis as sa
```

Python 3.10 or later, with numpy, scipy, pandas and matplotlib. numba is
optional. There is no console entry point.

## Quickstart

```python
import stream_analysis as sa

# Stream coordinates come from nbody_streams, outside this package.
from nbody_streams import coords
phi1, phi2 = coords.generate_stream_coords(xv=xv, xv_prog=xv_prog, optimizer_fit=True)

# Orbit properties, one row per stream.
df_orbits = sa.compute_orbit_properties(orbits)

# Length, width, and the membership mask everything downstream filters on.
lengths, widths, masks = sa.measure_stream_length_width(
    phi1, phi2, method="kde", density_frac=0.9, return_mask=True,
)

# Total 3D velocity dispersion, with the along-track trend removed.
sigma_v = sa.compute_velocity_dispersion(xv, masks, detrend=True, phi1=phi1)

# The whole metrics table.
table = sa.build_metrics_table(phi1, phi2, masks, lengths, widths, sigma_v, df_orbits)
```

`table` has one row per stream and the twenty columns below, in that order,
indexed by `df_orbits.index` under the name `stream_index`.

**[EXAMPLE.md](EXAMPLE.md) walks through all five stages one at a time**, with
real output at every step and the per-stream calls behind `build_metrics_table`.

## Metric reference

Sign convention says which direction means more disturbed.

### Localized binned family

| Column | Paper name | Symbol | What it is | Units | Sign |
|---|---|---|---|---|---|
| `median_norm_wass` | Global disturbance | $D_{\rm global}$ | Median across bins of the Wasserstein distance from the local distribution to a standard normal, divided by the 95% null threshold for that bin's particle count. A value of 1 is the edge of consistency with Gaussian. | dimensionless | higher is more disturbed |
| `max_norm_wass` | Peak disturbance | $D_{\rm peak}$ | The same quantity at its worst bin. Catches a stream disturbed in one place that is otherwise clean. | dimensionless | higher is more disturbed |
| `coef_of_var_stds` | Width variation | $C_w$ | `std_std_bins / mean_std_bins`. How much the stream's width varies along its own length, relative to its mean width. | dimensionless | higher is more disturbed |
| `mean_std_bins` | diagnostic only | | Mean across bins of the per-bin standard deviation. The average local width, and the denominator of $C_w$. | degrees | wider, not more disturbed |
| `std_std_bins` | diagnostic only | | Scatter of those per-bin standard deviations. The numerator of $C_w$. | degrees | higher is more disturbed |
| `frac_flag_wass` | diagnostic only | | Share of bins still consistent with Gaussian at 95%. | fraction | lower is more disturbed |
| `frac_consecutive_wass` | diagnostic only | | Longest unbroken run of those bins, as a fraction. Separates a locally disturbed stream from a globally disturbed one. | fraction | lower is more disturbed |
| `frac_p_gt_005` | diagnostic only | | The KS counterpart of `frac_flag_wass`, bins with p > 0.05. | fraction | lower is more disturbed |

### Power-spectrum family

| Column | Paper name | Symbol | What it is | Units | Sign |
|---|---|---|---|---|---|
| `min_lambda_band` | Minimum detectable scale | $\lambda_{\min}$ | Shortest along-stream wavelength at which the observed density power rises 5σ above the sampling noise floor. The finest structure the data can resolve. NaN when nothing clears. | degrees | lower means finer structure is detectable |
| `rms_obs` | RMS density fluctuation | $\mathrm{RMS}_\delta$ | Square root of the observed power integrated across the trusted frequency band. The total amplitude of fractional density fluctuation. | dimensionless | higher is more structured |
| `excess_power` | Excess power | $P_{\rm excess}$ | Integral across the trusted band of the observed power above the null median, clipped at zero. The part finite sampling does not explain. | dimensionless | higher is more disturbed |
| `rms_null` | diagnostic only | | The same integral taken on the Monte Carlo realizations, reduced by their median. The floor `rms_obs` should be read against. | dimensionless | reference level |

### Stream and orbit properties

| Column | Paper name | Symbol | Units |
|---|---|---|---|
| `length_deg` | Length | $\ell$ | degrees |
| `width_deg` | Width | $W$ | degrees |
| `vel.std.tot_km_s` | TBD | TBD | km/s |
| `min_pericenter_dist`, `max_apocenter_dist`, `mean_dist` | | | kpc |
| `n_pericenters` | | | count |
| `period` | | | time units of the orbit integration |

`vel.std.tot_km_s` is the total **3D** dispersion,
`sqrt(var_vx + var_vy + var_vz)`, not a line-of-sight dispersion. If you want
$v_{\rm LOS}$, project the velocities first and pass the result to
`compute_local_binned_stats`, which is generic in its binned quantity. The orbit
columns are copied through from `df_orbits` untouched.

### The localized binned family is generic

Nothing in `compute_local_binned_stats` is about `phi2`. It bins any
per-particle quantity along the track and tests its local distribution against a
normal. The paper applies it to `phi2`. The same machinery applied to
$v_{\rm LOS}$ yields the kinematic analogues of the same three metrics.

```python
bins_v = sa.compute_local_binned_stats(phi1[i, m] / lengths[i], v_los, quantity_name="v_los")
metrics_v = sa.compute_bin_diagnostics(bins_v, quantity_name="v_los")
```

`quantity_name` renames the two columns named after the quantity and nothing
else. Bins, binning and every statistic are computed identically.

### `min_lambda_band` and the choice of method

Three methods, three different statistics. `snr_threshold` means something
different in each, which is why leaving it as `None` gives each its own default.

| `method` | Statistic | Default | Notes |
|---|---|---|---|
| `peak_snr` | $(P_{\rm obs} - \mu_{\rm null}) / \sigma_{\rm null}$, per frequency | 5.0 | The default. A true signal-to-noise ratio in units of the null scatter, so 5 is the familiar 5σ. Reports $1/f$ at the highest trusted frequency clearing it. |
| `band_snr` | $I_{\rm obs} / \mathrm{std}(I_{\rm mc})$, over a band of `min_bins` bins | 3.0 | More conservative, since it needs coherent excess across adjacent bins, and noisier, since each integral covers only two points. |
| `ratio95` | $P_{\rm obs} / P_{95}$, per frequency | 3.0 | Not an SNR, despite the parameter name. A power ratio against the 95th percentile of the null. Faster and cheaper than the other two, since it needs no per-frequency scatter and no band scan, and it reproduces the published values to good accuracy for most streams. |

`build_metrics_table` defaults to `peak_snr` at 5σ. To reproduce the published
table:

```python
table = sa.build_metrics_table(
    phi1, phi2, masks, lengths, widths, sigma_v, df_orbits,
    method=sa.SPECTRA.published_method,                  # "ratio95"
    snr_threshold=sa.SPECTRA.published_snr_threshold,    # 3.0
)
```

`ratio95` at 3 was used as a proxy for a 5σ detection, and it is a good one.
`peak_snr` at 5 reproduces it exactly on many streams and differs on others,
because the null PSD distribution is chi-squared-like rather than Gaussian, so
$P_{95}/\mu_{\rm null}$ is not a fixed multiple of
$\sigma_{\rm null}/\mu_{\rm null}$.

## Module map

| Module | Covers |
|---|---|
| `geometry` | Stream length, width, and the KDE membership mask. |
| `kinematics` | Velocity dispersion, optionally detrended along the track. |
| `binned` | Localized binned statistics and the diagnostics collapsed from them. |
| `density` | Linearized along-stream density and its fractional residuals. |
| `spectra` | Welch power spectra, the sampling noise floor, the detection metrics. |
| `statistics` | Wasserstein null thresholds, Fisher and Bonferroni combination. |
| `orbits` | Peri, apo, mean radius, period and eccentricity from integrated orbits. |
| `plotting` | Figure helpers that are not part of any metric's definition. |
| `pipeline` | `build_metrics_table`. |
| `config` | Every analysis hyperparameter, at the paper's values. |

Every public name is re-exported flat, so `sa.anything(...)` works.

`config` collects the hyperparameters that were scattered across signatures and
function bodies. It is documentation with a type, not a control surface. Every
function keeps its own literal defaults.

```python
sa.BINNED.bin_size            # 0.125, the L/8 of the paper
sa.SPECTRA.rng_seed           # 12345
sa.WASSERSTEIN_NULL_FIT.A     # 2.3, from the precomputed 95% null power law
```

## Testing

```bash
python -m pytest tests/ -q     # 56 tests, about 15 seconds
```

The suite is self-contained. It runs on seeded synthetic streams, needs no
simulation data and no network, so a fresh clone can run all of it.

## Provenance

This package is a reorganization of a single 2239-line module, `stream_analysis.py`,
which produced the results in the paper. The reorganization moved code and
changed nothing. Every function returned bitwise-identical output for identical
input, checked against golden outputs generated from a frozen copy of the
original before any code moved.

Twenty-two logical issues were then found and fixed. Ten touched only
documentation or dead code. Ten change behavior on inputs that used to raise, or
remove a side effect, without moving any number. Two change a number, both in the
PSD detection scan, and together they touch exactly one column of the metrics
table, `min_lambda_band`. To reproduce the published values, pass
`method=sa.SPECTRA.published_method` and
`snr_threshold=sa.SPECTRA.published_snr_threshold` as shown above.

The full record is on the [`refactor-provenance`](https://github.com/appy2806/stream-analysis/tree/refactor-provenance)
branch: the frozen source, the inventory and module plan, all twenty-two issues
with their locations and patches, the golden outputs, and the bitwise contract
suite that checks them.

Several functions were renamed. Every old name still works through a deprecation
shim. See [RENAMES.md](RENAMES.md).

## Citation

> Arora et al., *No Stream Left Unscathed*, [arXiv:2605.16200](https://arxiv.org/abs/2605.16200).

## License

MIT. See [LICENSE](LICENSE).
