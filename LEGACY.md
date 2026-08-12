# LEGACY

`stream_analysis/legacy.py` is empty. Nothing from the original 2239-line file
was retired.

One function was considered and kept public.

- `plot_metric_stats_detailed`, now `summarize_bin_metrics`, duplicated part of `compute_bin_diagnostics` but has eight external call sites, so it stays in `plotting.py` with the duplicated block factored into `statistics._normalize_wasserstein`.

Nothing else was even close. Every function is reached from the pipeline, from a
notebook, or from the upstream orbit-integration step.

`compute_orbit_properties` was briefly listed here as a dead-code candidate,
because no call site appears in the analysis repository. That was wrong.
It builds the `df_orbits` table that `build_metrics_table` consumes, from the
integrated orbits, upstream of anything in that repository. It is live and on
the critical path.
