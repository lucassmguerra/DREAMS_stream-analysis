# -*- coding: utf-8 -*-
"""
The contract tests.

Every case in ``tools/cases.py`` is run against the refactored package and
compared, bitwise, against the golden generated from
``reference/stream_analysis_source.py`` before any code moved. Array leaves are
compared with ``numpy.testing.assert_array_equal``. Scalars are compared through
their exact hexadecimal float representation. Structure, dtypes, column order,
index and index name are all part of the comparison.

Bitwise identity is the default and the target. There is no tolerance knob here
on purpose. If a case cannot be made bitwise identical, the fix is to restructure
the new code to match the old, not to relax this file.
"""
from __future__ import annotations

import pytest

import cases as case_registry
import golden_io
from conftest import GOLDEN_DIR

CASE_NAMES = sorted(case_registry.CASES)

#: Cases that run the full metrics table over 16 streams. Slow enough to be
#: worth a marker, fast enough to stay in the default run.
SLOW_PREFIXES = ("pipeline_",)


def _mark(name: str):
    if name.startswith(SLOW_PREFIXES):
        return pytest.param(name, marks=pytest.mark.slow)
    return name


@pytest.mark.parametrize("case_name", [_mark(n) for n in CASE_NAMES])
def test_matches_golden(case_name, api, fixtures):
    """The refactored package reproduces the frozen source, bitwise."""
    golden_json = GOLDEN_DIR / f"{case_name}.json"
    if not golden_json.exists():
        pytest.fail(
            f"No golden for case {case_name!r}. Generate it with "
            f"`python tools/build_goldens.py --only {case_name}`."
        )

    spec_golden, arrays_golden = golden_io.load(GOLDEN_DIR, case_name)

    try:
        result = case_registry.CASES[case_name](api, fixtures)
    except Exception as exc:
        spec_new, arrays_new = golden_io.exception_spec(exc), {}
        if spec_golden.get("t") != "exception":
            raise AssertionError(
                f"{case_name} raised {type(exc).__name__}: {exc}\n"
                f"The golden holds a value, so this is a regression."
            ) from exc
    else:
        if spec_golden.get("t") == "exception":
            raise AssertionError(
                f"{case_name} returned a value, but the golden records "
                f"{spec_golden['exc_type']}: {spec_golden['message']}\n"
                "Reproducing the exception is part of the contract."
            )
        spec_new, arrays_new = golden_io.flatten(result, prefix=case_name)

    golden_io.compare(spec_golden, arrays_golden, spec_new, arrays_new)


def test_every_golden_has_a_case():
    """No orphaned goldens. A deleted case must have its golden deleted too."""
    stored = {p.stem for p in GOLDEN_DIR.glob("*.json")}
    known = set(CASE_NAMES)
    assert stored - known == set(), f"goldens with no case: {sorted(stored - known)}"


def test_every_case_has_a_golden():
    """No case runs without a contract behind it."""
    stored = {p.stem for p in GOLDEN_DIR.glob("*.json")}
    known = set(CASE_NAMES)
    assert known - stored == set(), f"cases with no golden: {sorted(known - stored)}"
