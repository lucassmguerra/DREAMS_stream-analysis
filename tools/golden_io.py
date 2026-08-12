# -*- coding: utf-8 -*-
"""
Serialization for golden outputs.

The contract tests need to compare arbitrary return values, arrays, scalars,
tuples, dicts, DataFrames and raised exceptions, bitwise. Rather than pickling,
which ties the contract to a pandas and numpy version, every result is flattened
into two parts.

- A JSON *spec* describing the structure, with scalars inlined.
- An npz of every array leaf, keyed by its path in the structure.

Comparison then works by flattening the new result the same way, checking the
specs are equal as JSON, and checking each array leaf with
``numpy.testing.assert_array_equal``. Nothing is ever reconstructed, so no
version-dependent unpickling is involved.

Float scalars are stored in the spec as their exact hexadecimal representation,
so that "bitwise identical" means bitwise identical rather than identical to 17
significant digits.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _float_repr(x: float) -> str:
    """Exact, round-trippable text form of a Python float."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x).hex()


def flatten(obj: Any, prefix: str = "r") -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """
    Split a return value into a JSON-safe spec and a dict of array leaves.

    Parameters
    ----------
    obj : Any
        The value to flatten. Supported leaves are None, bool, int, float, str,
        numpy scalars, numpy arrays, pandas Series and DataFrame, and the
        containers tuple, list and dict.
    prefix : str
        Path prefix for array keys. Callers pass the case name.

    Returns
    -------
    spec : dict
        JSON-serializable description of the structure.
    arrays : dict[str, np.ndarray]
        Array leaves keyed by their structural path.
    """
    arrays: dict[str, np.ndarray] = {}

    def walk(o: Any, path: str) -> dict[str, Any]:
        if o is None:
            return {"t": "none"}

        if isinstance(o, (bool, np.bool_)):
            return {"t": "bool", "v": bool(o)}

        if isinstance(o, (int, np.integer)):
            return {"t": "int", "v": int(o)}

        if isinstance(o, (float, np.floating)):
            return {"t": "float", "v": _float_repr(float(o))}

        if isinstance(o, str):
            return {"t": "str", "v": o}

        if isinstance(o, np.ndarray):
            # Object arrays cannot go into an npz reliably; none are expected.
            if o.dtype == object:
                raise TypeError(f"object-dtype array at {path}")
            arrays[path] = o
            return {"t": "array", "k": path, "dtype": str(o.dtype), "shape": list(o.shape)}

        if isinstance(o, pd.Series):
            return {
                "t": "series",
                "name": None if o.name is None else str(o.name),
                "dtype": str(o.dtype),
                "values": walk(o.to_numpy(), path + ".values"),
                "index": walk(o.index.to_numpy(), path + ".index"),
                "index_name": None if o.index.name is None else str(o.index.name),
            }

        if isinstance(o, pd.DataFrame):
            return {
                "t": "frame",
                "columns": [str(c) for c in o.columns],
                "dtypes": [str(d) for d in o.dtypes],
                "index_name": None if o.index.name is None else str(o.index.name),
                "index": walk(o.index.to_numpy(), path + ".index"),
                "cols": {
                    str(c): walk(o[c].to_numpy(), f"{path}.col.{c}") for c in o.columns
                },
            }

        if isinstance(o, tuple):
            return {"t": "tuple", "items": [walk(v, f"{path}.{i}") for i, v in enumerate(o)]}

        if isinstance(o, list):
            return {"t": "list", "items": [walk(v, f"{path}.{i}") for i, v in enumerate(o)]}

        if isinstance(o, dict):
            return {
                "t": "dict",
                "items": {str(k): walk(v, f"{path}.{k}") for k, v in o.items()},
            }

        raise TypeError(f"unsupported type {type(o)!r} at {path}")

    return walk(obj, prefix), arrays


#: Colour codes, and the file paths and line numbers a compiler error embeds.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PY_PATH = re.compile(r"[\w./\\-]*[\w\\-]+\.py")
_LINE_REF = re.compile(r"\bline \d+\b|\(\d+\)$", re.MULTILINE)


def normalize_message(message: str) -> str:
    """
    Strip the parts of an exception message that depend on where the code lives.

    A numba ``TypingError`` quotes the source file and line of the offending
    expression. Moving that expression from ``reference/stream_analysis_source.py``
    to ``stream_analysis/kinematics.py`` changes the text without changing the
    failure. Terminal colour codes vary the same way.

    The exception type and the substance of the message are still compared
    exactly. Only paths, line numbers and colour codes are neutralized. The
    verbatim message is kept in the spec under ``_message_verbatim`` for reading,
    and is not compared.
    """
    text = _ANSI.sub("", message)
    text = _PY_PATH.sub("<file>.py", text)
    text = _LINE_REF.sub("<line>", text)
    return text


def exception_spec(exc: BaseException) -> dict[str, Any]:
    """
    Record a raised exception as a golden.

    Reproducing a failure is part of the contract, so the exception type and its
    message are stored the same way a value would be. The message is normalized
    by :func:`normalize_message` first.
    """
    raw = str(exc)
    return {
        "t": "exception",
        "exc_type": type(exc).__name__,
        "message": normalize_message(raw),
        "_message_verbatim": raw,
    }


def save(out_dir: Path, name: str, spec: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """Write one golden case to ``<out_dir>/<name>.json`` plus ``<name>.npz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    # Always write the npz, even when empty, so the pair is always present.
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)


def load(out_dir: Path, name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read one golden case back."""
    spec = json.loads((out_dir / f"{name}.json").read_text())
    with np.load(out_dir / f"{name}.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return spec, arrays


def compare(
    spec_golden: dict[str, Any],
    arrays_golden: dict[str, np.ndarray],
    spec_new: dict[str, Any],
    arrays_new: dict[str, np.ndarray],
) -> None:
    """
    Assert that a new result matches a golden bitwise.

    Raises
    ------
    AssertionError
        On any structural or numerical difference. The message names the
        structural path of the first difference found.
    """
    spec_golden = _drop_uncompared(spec_golden)
    spec_new = _drop_uncompared(spec_new)
    if spec_golden != spec_new:
        _diff_specs(spec_golden, spec_new, "")

    assert set(arrays_golden) == set(arrays_new), (
        f"array keys differ\n  only in golden: {sorted(set(arrays_golden) - set(arrays_new))}"
        f"\n  only in new:    {sorted(set(arrays_new) - set(arrays_golden))}"
    )

    for key in sorted(arrays_golden):
        g, n = arrays_golden[key], arrays_new[key]
        assert g.dtype == n.dtype, f"{key}: dtype {n.dtype} != golden {g.dtype}"
        assert g.shape == n.shape, f"{key}: shape {n.shape} != golden {g.shape}"
        np.testing.assert_array_equal(n, g, err_msg=f"array mismatch at {key}")


def _drop_uncompared(spec: Any) -> Any:
    """Recursively remove informational keys, those whose name starts with '_'."""
    if isinstance(spec, dict):
        return {k: _drop_uncompared(v) for k, v in spec.items() if not k.startswith("_")}
    if isinstance(spec, list):
        return [_drop_uncompared(v) for v in spec]
    return spec


def _diff_specs(g: Any, n: Any, path: str) -> None:
    """Raise an AssertionError naming the first structural difference."""
    if type(g) is not type(n):
        raise AssertionError(f"{path or '<root>'}: type {type(n).__name__} != golden {type(g).__name__}")

    if isinstance(g, dict):
        gk, nk = set(g), set(n)
        if gk != nk:
            raise AssertionError(
                f"{path or '<root>'}: keys differ, only in golden {sorted(gk - nk)}, "
                f"only in new {sorted(nk - gk)}"
            )
        for k in sorted(gk):
            if g[k] != n[k]:
                _diff_specs(g[k], n[k], f"{path}.{k}")
        return

    if isinstance(g, list):
        if len(g) != len(n):
            raise AssertionError(f"{path}: length {len(n)} != golden {len(g)}")
        for i, (gv, nv) in enumerate(zip(g, n)):
            if gv != nv:
                _diff_specs(gv, nv, f"{path}[{i}]")
        return

    raise AssertionError(f"{path or '<root>'}: {n!r} != golden {g!r}")
