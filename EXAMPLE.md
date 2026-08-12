# Worked example

Five calls, in the order the analysis runs, on the 16 m12i streams in the test
fixture. Every number below was produced by running this file top to bottom, so
if your output differs, something in your environment does.

```python
import numpy as np
import pandas as pd
import stream_analysis as sa
```

The five calls, in one line each:

| Step | Call | Produces |
|---|---|---|
| 1 | `compute_orbit_properties` | `df_orbits`, the peri, apo, period table |
| 2 | `measure_stream_length_width` | `lengths`, `widths`, `masks` |
| 3 | `compute_velocity_dispersion` | `sigma_v` |
| 4 | `compute_local_binned_stats` then `compute_bin_diagnostics` | the localized binned metrics |
| 5 | `compute_linearized_density` then `compute_welch_psd`, `compute_sampling_psd_realizations`, `detect_stream_psd_metrics` | the power-spectrum metrics |

Step 6 at the bottom does all five for every stream in one call.

---

## Step 1. Orbit properties

Takes integrated progenitor orbits and produces the `df_orbits` table the
pipeline consumes. This is upstream of everything else and does not touch stream
coordinates at all.

`orbits` is an object array of shape `(n_orbits, 2)`, where each row is
`[times, coords]`, `times` has shape `(n_steps,)` and `coords` has shape
`(n_steps, 6)` holding `x, y, z, vx, vy, vz`. Times are expected in **decreasing**
order, as lookback times, because the function reverses them on entry.

```python
z = np.load("orbit_results.npz", allow_pickle=True)      # from your orbit integration
times, xyz, vxyz = z["t_win"], z["orbit_xyz"], z["orbit_vxyz"]   # (216,), (16,216,3), (16,216,3)

orbits = np.empty((xyz.shape[0], 2), dtype=object)
for k in range(xyz.shape[0]):
    orbits[k, 0] = times[::-1].copy()                      # decreasing
    orbits[k, 1] = np.hstack([xyz[k], vxyz[k]])[::-1]      # (n_steps, 6)

df_orbits = sa.compute_orbit_properties(orbits, central_mass=1e12)
df_orbits.index = z["indices"]           # the original catalogue IDs
df_orbits.index.name = "stream_index"
```

```
              min_pericenter_dist  max_apocenter_dist  mean_dist  n_pericenters    period  eccentricity_geo  eccentricity_osculating
stream_index
2873                    22.571949           46.673698  33.821916              6  0.932087          0.348062                 0.754635
1131                    17.740186           37.843437  27.148922              8  0.699065          0.361676                 0.810020
1525                    19.123039           62.037556  40.091333              5  1.118505          0.528761                 0.788596
2871                    15.821710           43.358882  29.832604              7  0.798932          0.465307                 0.804613
```

Distances in kpc, period in the time units of `times`. `central_mass` is optional
and only feeds `eccentricity_osculating`, which is a point-mass approximation and
has local meaning only. Pass `mu` directly if your units are not kpc, km/s, Msun.

The index does not have to be a `RangeIndex`. These are the original m12i
catalogue IDs of a 16-stream subset drawn from a 5000-stream run.
`build_metrics_table` validates the index and carries it through.

---

## Step 2. Length, width and the membership mask

Takes stream-frame coordinates. Producing those is outside this package, because
the transform belongs to `nbody_streams`.

```python
from nbody_streams import coords
phi1, phi2 = coords.generate_stream_coords(xv=xv, xv_prog=xv_prog, optimizer_fit=True)
```

```python
lengths, widths, masks = sa.measure_stream_length_width(
    phi1, phi2, method="kde", density_frac=0.9, return_mask=True,
)
```

```
lengths[:4]  [19.426 21.899 22.235 37.425]      degrees
widths[:4]   [ 0.423  0.476  0.550  0.379]      degrees
mask keeps   [8940 8934 8975 8993] of 10000     particles
```

The mask matters as much as the two scalars. Everything downstream is computed on
the masked track, not on the raw particle list, so the 90% density contour is
what defines "the stream" for every later metric.

`method="quantile"` is the simpler alternative, where length and width are plain
`[(1-f)/2, (1+f)/2]` quantiles. `method="kde"` is what the paper uses.

---

## Step 3. Velocity dispersion

```python
sigma_v = sa.compute_velocity_dispersion(xv, masks, detrend=True, phi1=phi1)
```

```
sigma_v[:4]  [0.7405 0.9239 0.9474 1.0822]      km/s
```

This is the **total 3D** dispersion, `sqrt(var_vx + var_vy + var_vz)`, not a
line-of-sight dispersion. `detrend=True` removes a degree-5 polynomial in phi1
from each velocity component first, so the result measures local scatter rather
than the bulk velocity gradient along the track.

Two backends compute this. They are not bitwise equal to each other, because the
numba one solves the polynomial fit by normal equations and the NumPy one uses
`scipy.linalg.lstsq`. Pin `use_numba` when you need reproducibility across
machines.

---

## Step 4. Localized statistics in phi1 bins

The idea: an undisturbed stream has a Gaussian cross-section everywhere along its
track. Cut the track into bins, test each bin's distribution against a standard
normal, then collapse.

phi1 is divided by the stream length first, so the default `bin_size` of `1/8`
gives the eight bins of the paper. That is the `L/8` of the method section.

```python
i, m = 6, masks[6]
bins = sa.compute_local_binned_stats(phi1[i, m] / lengths[i], phi2[i, m])
```

```
   count  mean_phi2  std_phi2  ks_stat_vs_norm  ks_p_vs_norm  wass_vs_norm  wass_vs_global bin_left bin_right
0    506     0.1363    1.9163           0.0508        0.1418        0.0784          0.1724  -0.6100   -0.4840
1    992     0.0531    1.6860           0.0414        0.0649        0.0888          0.1493  -0.4840   -0.3590
2    987    -0.4175    0.9392           0.0344        0.1892        0.0713          0.1132  -0.3590   -0.2340
3   1102     0.0625    0.5317           0.0191        0.8102        0.0286          0.1508  -0.2340   -0.1090
4   1253     0.4041    0.4902           0.0197        0.7081        0.0336          0.1567  -0.1090    0.0164
5   1103    -0.0283    0.4881           0.0156        0.9476        0.0284          0.1374   0.0164    0.1410
6   2184    -0.1784    0.7409           0.0689        0.0000        0.1268          0.0768   0.1410    0.2660
7    803     0.1694    1.3976           0.0814        0.0000        0.1674          0.1781   0.2660    0.3910
```

`mean_phi2` is a residual mean, because a cubic in phi1 is removed from the
quantity before binning. `wass_vs_norm` is the Wasserstein distance from the
standardized bin values to a seeded 20000-draw N(0, 1) reference.
`wass_vs_global` compares against the whole stream instead.

Then collapse to the seven headline numbers:

```python
(median_norm_wass, max_norm_wass, frac_flag_wass, frac_consecutive_wass,
 frac_p_gt_005, mean_std_bins, std_std_bins) = sa.compute_bin_diagnostics(bins)

coef_of_var_stds = std_std_bins / mean_std_bins
```

```
  median_norm_wass         0.919033
  max_norm_wass            2.636233
  frac_flag_wass           0.500000
  frac_consecutive_wass    0.375000
  frac_p_gt_005            0.750000
  mean_std_bins            1.023762
  std_std_bins             0.570200
  coef_of_var_stds         0.556966
```

The `norm_` prefix means the raw Wasserstein distance has been divided by the 95%
null threshold for that bin's particle count, so **1.0 is the edge of consistency
with Gaussian**. A median of 0.92 says the typical bin is marginally Gaussian, a
max of 2.64 says the worst bin is not.

`compute_bin_diagnostics` works on a copy and leaves `bins` untouched.

### The same machinery on any quantity

Nothing here is about phi2. Pass a different per-particle quantity and get the
kinematic analogues.

```python
r = np.linalg.norm(xv[i, m, :3], axis=1)
v_los = np.sum(xv[i, m, :3] * xv[i, m, 3:], axis=1) / r

bins_v = sa.compute_local_binned_stats(phi1[i, m] / lengths[i], v_los, quantity_name="v_los")
diag_v = sa.compute_bin_diagnostics(bins_v, quantity_name="v_los")
```

```
median_norm_wass 0.66555   max_norm_wass 2.81424   frac_flag_wass 0.75
columns: ['count', 'mean_v_los', 'std_v_los', 'ks_stat_vs_norm', 'ks_p_vs_norm',
          'wass_vs_norm', 'wass_vs_global', 'bin_left', 'bin_right']
```

`quantity_name` renames the two columns named after the quantity and nothing
else. The bins, the binning and every statistic are computed identically.

---

## Step 5. Power-spectrum metrics

Four calls in a chain, each consuming the previous one's output.

```python
# 5a. Histogram the track, smooth to approximate a KDE, detrend, take residuals.
dens = sa.compute_linearized_density(
    phi1[i, m], detrend="poly", smoothing_sigma=1, bin_width=0.25,
)

# 5b. Welch PSD of those residuals.
psd = sa.compute_welch_psd(dens["resid"], dens["w"])

# 5c. The sampling noise floor. Resample the counts from a uniform density many
#     times and push each realization through the identical chain.
null = sa.compute_sampling_psd_realizations(
    dens["hist_counts"], smoothing_sigma_bins=1, w=dens["w"],
    n_realizations=1000, rng_seed=12345,
)

# 5d. Compare the observed spectrum to that floor.
rms_obs, rms_null, excess_power, min_lambda_band = sa.detect_stream_psd_metrics(
    psd["freqs"], psd["psd"], w=dens["w"],
    psd_all=null["psd_all"], nperseg=null["nperseg"],
)
```

```
nbins 516   w 0.249920   nfreq 65   nperseg 129

method     rms_obs    rms_null   excess_power   min_lambda_band
peak_snr   0.233551   0.120074   0.041375       2.4800
band_snr   0.233551   0.120074   0.041375       0.6200
ratio95    0.233551   0.120074   0.041375       8.0599
```

`bin_width=0.25` degrees rather than a fixed bin count, so streams of different
lengths are compared at the same spatial resolution. `smoothing_sigma=1` in bin
units makes the effective KDE bandwidth 0.25 degrees, and the same value goes to
the realizations so the observed spectrum and its floor are smoothed identically.

The first three scalars do not depend on the detection method. Only
`min_lambda_band` does, and the three methods are three different statistics.
`peak_snr` at 5 sigma is the default. `ratio95` at 3 is faster and cheaper, and
is what produced the published values. Pass
`method=sa.SPECTRA.published_method, snr_threshold=sa.SPECTRA.published_snr_threshold`
to reproduce them. See the README for the comparison.

---

## Step 6. All of it, for every stream

```python
table = sa.build_metrics_table(phi1, phi2, masks, lengths, widths, sigma_v, df_orbits)
```

```
shape (16, 20)   index name 'stream_index'

              median_norm_wass  max_norm_wass  frac_flag_wass  frac_consecutive_wass  frac_p_gt_005  mean_std_bins  std_std_bins  length_deg
stream_index
2873                    0.4505         0.9748           1.000                   1.00          1.000         0.1252        0.0359     19.4263
1131                    0.5805         0.8960           1.000                   1.00          1.000         0.1339        0.0237     21.8986
1525                    0.6656         1.2603           0.875                   0.75          0.875         0.1616        0.0548     22.2345

              width_deg  vel.std.tot_km_s  coef_of_var_stds  min_pericenter_dist  max_apocenter_dist  mean_dist  n_pericenters  period  rms_obs  rms_null  excess_power  min_lambda_band
stream_index
2873             0.4226            0.7405            0.2870              22.5719             46.6737    33.8219            6.0  0.9321   0.1230    0.0385        0.0137           0.5914
1131             0.4762            0.9239            0.1773              17.7402             37.8434    27.1489            8.0  0.6991   0.0615    0.0344        0.0026           1.0943
1525             0.5497            0.9474            0.3393              19.1230             62.0376    40.0913            5.0  1.1185   0.1237    0.0430        0.0135           1.8319
```

Useful keywords:

```python
table = sa.build_metrics_table(
    phi1, phi2, masks, lengths, widths, sigma_v, df_orbits,
    checkpoint_path="partial.parquet",   # write the binned half before the spectra run
    progress=tqdm,                       # wrap both per-stream loops
    method="ratio95", snr_threshold=3.0, # the published detection settings
    quantity_name="v_los",               # bin something other than phi2
)
```

---

## Bookkeeping

Which call produces each column, and how it maps to the paper.

### Disturbance metrics

| Column | Paper name | Symbol | Produced by | Units | Higher means |
|---|---|---|---|---|---|
| `median_norm_wass` | Global disturbance | $D_{\rm global}$ | Step 4 | dimensionless | more disturbed |
| `max_norm_wass` | Peak disturbance | $D_{\rm peak}$ | Step 4 | dimensionless | more disturbed |
| `coef_of_var_stds` | Width variation | $C_w$ | Step 4, `std_std_bins / mean_std_bins` | dimensionless | more disturbed |
| `min_lambda_band` | Minimum detectable scale | $\lambda_{\min}$ | Step 5d | degrees | coarser structure only |
| `rms_obs` | RMS density fluctuation | $\mathrm{RMS}_\delta$ | Step 5d | dimensionless | more structured |
| `excess_power` | Excess power | $P_{\rm excess}$ | Step 5d | dimensionless | more disturbed |

### Stream properties

| Column | Paper name | Symbol | Produced by | Units |
|---|---|---|---|---|
| `length_deg` | Length | $\ell$ | Step 2 | degrees |
| `width_deg` | Width | $W$ | Step 2 | degrees |
| `vel.std.tot_km_s` | TBD | TBD | Step 3 | km/s |

`vel.std.tot_km_s` is the total 3D dispersion, not $v_{\rm LOS}$. If you want
the line-of-sight quantity, project first and feed it through Step 4, which is
generic in its quantity.

### Diagnostics, not paper metrics

| Column | Produced by | Units | What it tells you |
|---|---|---|---|
| `frac_flag_wass` | Step 4 | fraction | Share of bins still consistent with Gaussian at 95%. Lower is more disturbed. |
| `frac_consecutive_wass` | Step 4 | fraction | Longest unbroken run of those bins. Separates locally from globally disturbed. |
| `frac_p_gt_005` | Step 4 | fraction | The KS counterpart of `frac_flag_wass`. |
| `mean_std_bins` | Step 4 | degrees | Mean local width. The denominator of `C_w`. |
| `std_std_bins` | Step 4 | degrees | Scatter of local widths. The numerator of `C_w`. |
| `rms_null` | Step 5d | dimensionless | The noise floor `rms_obs` should be read against. |

### Orbit properties

Copied through from `df_orbits` untouched, produced by Step 1.
`min_pericenter_dist`, `max_apocenter_dist`, `mean_dist` in kpc,
`n_pericenters` a count, `period` in the time units of the orbit integration.

`compute_orbit_properties` also returns `eccentricity_geo` and
`eccentricity_osculating`, which the metrics table does not carry.

### Hyperparameters

Every analysis choice above is readable from `sa.config`, at the paper's values.

```python
sa.BINNED.bin_size                  # 0.125, the L/8
sa.BINNED.normal_ref_seed           # 12345
sa.GEOMETRY.density_frac            # 0.9
sa.KINEMATICS.poly_degree           # 5
sa.DENSITY.bin_width                # 0.25
sa.SPECTRA.n_realizations           # 1000
sa.SPECTRA.rng_seed                 # 12345
sa.SPECTRA.method                   # "peak_snr"
sa.SPECTRA.published_method         # "ratio95"
sa.WASSERSTEIN_NULL_FIT.A           # 2.3, from the 95% null power law
```

---

## Reference

> Arora et al., *No Stream Left Unscathed*, [arXiv:2605.16200](https://arxiv.org/abs/2605.16200).
