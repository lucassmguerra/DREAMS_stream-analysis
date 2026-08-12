# -*- coding: utf-8 -*-
"""
Compatibility shims for the pre-refactor names.

Every renamed public function keeps a thin wrapper here under its old name. The
wrapper emits a ``DeprecationWarning`` and forwards, positionally and by keyword,
to the new one. Nothing else happens, so the numbers are identical.

One keyword was renamed as well. ``compute_local_binned_stats`` used to call its
second argument ``phi2``. It is generic in that argument, so it is now
``quantity``. Passing ``phi2=`` by keyword still works and warns.

Old names are exported from the package top level, so notebooks written against
``import stream_analysis as sa`` keep running unchanged. See RENAMES.md for the
table and a block of sed commands.
"""
from __future__ import annotations

import functools
import warnings
from typing import Any, Callable

from .binned import compute_local_binned_stats
from .geometry import measure_stream_length_width
from .kinematics import compute_velocity_dispersion
from .plotting import summarize_bin_metrics
from .spectra import compute_welch_psd
from .statistics import _max_consecutive_true_fraction

__all__ = [
    "measure_stream_LengthWidth",
    "compute_vel_disp_stream",
    "stream_local_binned_stats",
    "compute_welch_psd_for_resid",
    "plot_metric_stats_detailed",
    "_get_max_consecutive_true_fraction",
]

#: Old name -> new name. Kept in sync with RENAMES.md and tools/cases.py.
RENAMES: dict[str, str] = {
    "measure_stream_LengthWidth": "measure_stream_length_width",
    "compute_vel_disp_stream": "compute_velocity_dispersion",
    "stream_local_binned_stats": "compute_local_binned_stats",
    "compute_welch_psd_for_resid": "compute_welch_psd",
    "plot_metric_stats_detailed": "summarize_bin_metrics",
    "_get_max_consecutive_true_fraction": "_max_consecutive_true_fraction",
}


def _deprecated(old_name: str, new_func: Callable) -> Callable:
    """Build a forwarding wrapper that warns once per call site."""

    @functools.wraps(new_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{old_name} is deprecated, use {new_func.__name__} instead. "
            "The old name forwards unchanged and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    wrapper.__doc__ = (
        f"Deprecated alias for :func:`{new_func.__name__}`.\n\n"
        f"{new_func.__doc__ or ''}"
    )
    return wrapper


measure_stream_LengthWidth = _deprecated(
    "measure_stream_LengthWidth", measure_stream_length_width
)
compute_vel_disp_stream = _deprecated("compute_vel_disp_stream", compute_velocity_dispersion)
compute_welch_psd_for_resid = _deprecated("compute_welch_psd_for_resid", compute_welch_psd)
plot_metric_stats_detailed = _deprecated("plot_metric_stats_detailed", summarize_bin_metrics)
_get_max_consecutive_true_fraction = _deprecated(
    "_get_max_consecutive_true_fraction", _max_consecutive_true_fraction
)


@functools.wraps(compute_local_binned_stats)
def stream_local_binned_stats(*args: Any, **kwargs: Any):
    """
    Deprecated alias for :func:`compute_local_binned_stats`.

    Also accepts the old ``phi2`` keyword for what is now called ``quantity``.
    """
    warnings.warn(
        "stream_local_binned_stats is deprecated, use compute_local_binned_stats instead. "
        "The old name forwards unchanged and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    # The `phi2` keyword is handled by compute_local_binned_stats itself, which
    # emits its own DeprecationWarning for it.
    return compute_local_binned_stats(*args, **kwargs)


stream_local_binned_stats.__name__ = "stream_local_binned_stats"
stream_local_binned_stats.__qualname__ = "stream_local_binned_stats"
