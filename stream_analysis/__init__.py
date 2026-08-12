# -*- coding: utf-8 -*-
"""
Stream analysis
===============

Helper functions for measuring how disturbed an N-body globular-cluster stream
is. This is the library behind the metrics of *No Stream Left Unscathed*
(Arora et al.).

Every public name is re-exported flat, so

    import stream_analysis as sa
    sa.measure_stream_length_width(phi1, phi2)

works, and so does every pre-refactor name through a deprecation shim.

Layout
------
``geometry``
    Stream length, width and the KDE membership mask.
``kinematics``
    Velocity dispersion, optionally detrended along the track.
``binned``
    Localized binned statistics and the diagnostics collapsed from them.
    Global disturbance, peak disturbance, width variation.
``density``
    Linearized along-stream density and its fractional residuals.
``spectra``
    Welch power spectra, the sampling noise floor, and the detection metrics.
    Minimum detectable scale, RMS density residual, excess power.
``statistics``
    Distribution-comparison primitives the binned family is built on.
``orbits``
    Orbital diagnostics, peri, apo, period and eccentricity.
``plotting``
    Figure helpers that are not part of any metric's definition.
``pipeline``
    The metrics-table driver.
``config``
    Every analysis hyperparameter, at the paper's values.

Docstring style
---------------
All docstrings follow the NumPy/SciPy convention
(see https://numpydoc.readthedocs.io/).

Notes
-----
- This is a library of helper functions, **not** a standalone script.
- Type hints follow PEP 484/PEP 604, with postponed evaluation enabled via
  ``from __future__ import annotations``.
- Nothing here imports ``nbody_streams`` or ``agama``.

Author: Arpit Arora
Created: Sept 28, 2025
"""
from __future__ import annotations

__docformat__ = "numpy"

from . import (
    binned,
    config,
    density,
    geometry,
    kinematics,
    legacy,
    orbits,
    pipeline,
    plotting,
    spectra,
    statistics,
)
from ._compat import (
    _get_max_consecutive_true_fraction,
    compute_vel_disp_stream,
    compute_welch_psd_for_resid,
    measure_stream_LengthWidth,
    plot_metric_stats_detailed,
    stream_local_binned_stats,
)
from ._numba import NUMBA_AVAILABLE
from .binned import compute_bin_diagnostics, compute_local_binned_stats
from .config import (
    BINNED,
    DENSITY,
    GEOMETRY,
    KINEMATICS,
    SPECTRA,
    WASSERSTEIN_NULL_FIT,
    BinnedConfig,
    DensityConfig,
    GeometryConfig,
    KinematicsConfig,
    SpectraConfig,
    WassersteinNullFit,
)
from .density import compute_linearized_density
from .geometry import measure_stream_length_width
from .kinematics import compute_velocity_dispersion
from .orbits import compute_orbit_properties
from .pipeline import METRIC_COLUMNS, build_metrics_table
from .plotting import summarize_bin_metrics
from .spectra import (
    compute_sampling_psd_realizations,
    compute_welch_psd,
    detect_stream_psd_metrics,
)
from .statistics import (
    _max_consecutive_true_fraction,
    bonferroni_correction,
    combine_ks_pvalues_fisher,
    wasserstein_null_threshold,
)

__version__ = "1.0.0"

__all__ = [
    # geometry
    "measure_stream_length_width",
    # kinematics
    "compute_velocity_dispersion",
    # binned
    "compute_local_binned_stats",
    "compute_bin_diagnostics",
    # density
    "compute_linearized_density",
    # spectra
    "compute_welch_psd",
    "compute_sampling_psd_realizations",
    "detect_stream_psd_metrics",
    # statistics
    "wasserstein_null_threshold",
    "combine_ks_pvalues_fisher",
    "bonferroni_correction",
    # orbits
    "compute_orbit_properties",
    # plotting
    "summarize_bin_metrics",
    # pipeline
    "build_metrics_table",
    "METRIC_COLUMNS",
    # config
    "GeometryConfig",
    "KinematicsConfig",
    "BinnedConfig",
    "DensityConfig",
    "SpectraConfig",
    "WassersteinNullFit",
    "GEOMETRY",
    "KINEMATICS",
    "BINNED",
    "DENSITY",
    "SPECTRA",
    "WASSERSTEIN_NULL_FIT",
    # environment
    "NUMBA_AVAILABLE",
    "__version__",
    # submodules
    "binned",
    "config",
    "density",
    "geometry",
    "kinematics",
    "legacy",
    "orbits",
    "pipeline",
    "plotting",
    "spectra",
    "statistics",
    # deprecated aliases, see RENAMES.md
    "measure_stream_LengthWidth",
    "compute_vel_disp_stream",
    "stream_local_binned_stats",
    "compute_welch_psd_for_resid",
    "plot_metric_stats_detailed",
    # private, exported because the original exposed it at module level
    "_max_consecutive_true_fraction",
    "_get_max_consecutive_true_fraction",
]
