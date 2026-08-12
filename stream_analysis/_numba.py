# -*- coding: utf-8 -*-
"""
Optional numba acceleration, with a pure-NumPy fallback.

This module isolates the package's one optional dependency and its one
import-time side effect. Every other module imports numpy unconditionally and
never touches numba directly.

When numba is installed, ``njit`` and ``prange`` are the real ones and
``NUMBA_AVAILABLE`` is True. When it is not, ``prange`` becomes the builtin
``range`` and ``njit`` becomes a decorator factory that returns the function
unchanged, so a jitted function still runs, just interpreted.

Caveat, preserved from the original
-----------------------------------
The fallback ``njit`` must be *called*, as ``@njit(...)``. A bare ``@njit``
decoration passes the decorated function in as ``nopython`` and returns the
inner ``decorator`` rather than the function, which then fails at the call site
with a confusing signature error. The only decoration in this package uses
parentheses, so nothing breaks today. The shape of the shim is kept exactly as
it was in the source, because changing it is a behavior change. See FINDINGS.md.
"""
from __future__ import annotations

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

    def njit(nopython=True, cache=True, parallel=True):  # Dummy decorator
        def decorator(func):
            return func

        return decorator

    print("Numba not found. Running with pure NumPy/SciPy (might be slower for some operations).")

__all__ = ["NUMBA_AVAILABLE", "njit", "prange"]
