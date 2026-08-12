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

Both decoration forms work
--------------------------
The fallback ``njit`` accepts ``@njit`` and ``@njit(...)`` alike. The original
shim was ``def njit(nopython=True, cache=True, parallel=True)``, which only
worked when called. A bare ``@njit`` passed the decorated function in as
``nopython`` and returned the inner ``decorator`` instead of the function, which
then failed at the call site with a confusing signature error.
"""
from __future__ import annotations


def _fallback_njit(*args, **kwargs):
    """
    No-op stand-in for ``numba.njit``, used when numba is not installed.

    Handles both ``@njit`` and ``@njit(nopython=True, cache=True, ...)``. Any
    keyword numba would accept is swallowed and ignored.

    Defined unconditionally rather than inside the ``except ImportError`` branch,
    so that it can be tested on a machine that does have numba.
    """
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]  # bare @njit

    def decorator(func):
        return func

    return decorator


try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range
    njit = _fallback_njit

    print("Numba not found. Running with pure NumPy/SciPy (might be slower for some operations).")

__all__ = ["NUMBA_AVAILABLE", "njit", "prange"]
