# LEGACY

`stream_analysis/legacy.py` is empty. Nothing from the original 2239-line file
was retired.

Two functions were considered and kept public.

- `compute_orbit_properties`, no caller anywhere in `/mnt/d/Research/GC_streams/`, but a complete documented API for quantities the metrics table carries, so it stays in `orbits.py`.
- `plot_metric_stats_detailed`, now `summarize_bin_metrics`, duplicated part of `compute_bin_diagnostics` but has eight external call sites, so it stays in `plotting.py` with the duplicated block factored into `statistics._normalize_wasserstein`.

Nothing else was even close. Every other function is reached either from the
pipeline or from a notebook.
