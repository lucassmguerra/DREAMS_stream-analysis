#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate the golden outputs from the frozen source.

Every case in ``tools/cases.py`` is run against
``reference/stream_analysis_source.py``, loaded by path as a module named
``sa_ref``. The result, or the exception it raised, is serialized into
``tests/golden/``.

The frozen source is loaded by path and never imported as a package, so nothing
here can accidentally pick up the refactored code instead.

This script is run once, before any refactoring, and re-run only when a case is
added. Re-running it after a refactor would defeat its purpose, so it refuses to
overwrite an existing golden unless ``--force`` is given.

Run from the repository root:

    python tools/build_goldens.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")  # Figures are built and discarded. Never open a window.

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import cases as case_registry  # noqa: E402
import golden_io  # noqa: E402

SOURCE_PATH = REPO_ROOT / "reference" / "stream_analysis_source.py"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"


def load_reference() -> ModuleType:
    """Load the frozen source by path, as ``sa_ref``."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Frozen source missing at {SOURCE_PATH}")
    spec = importlib.util.spec_from_file_location("sa_ref", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build a module spec for {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(name: str, runner, api, fx):
    """
    Execute one case and flatten its outcome.

    A raised exception is a valid outcome. Its type and message become the
    golden, because reproducing a failure is part of the contract.
    """
    try:
        result = runner(api, fx)
    except Exception as exc:
        return golden_io.exception_spec(exc), {}
    return golden_io.flatten(result, prefix=name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite goldens that already exist."
    )
    parser.add_argument("--only", default=None, help="Substring filter on case names.")
    args = parser.parse_args()

    sa_ref = load_reference()
    api = case_registry.RefAPI(sa_ref)
    fx = case_registry.Fixtures(api)

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    names = sorted(case_registry.CASES)
    if args.only:
        names = [n for n in names if args.only in n]

    existing = [n for n in names if (GOLDEN_DIR / f"{n}.json").exists()]
    if existing and not args.force:
        print(
            f"{len(existing)} goldens already exist and would be overwritten.\n"
            "Goldens are the contract. Re-generating them after a refactor would "
            "make the tests vacuous.\n"
            "Pass --force only if you are regenerating from the frozen source on "
            "purpose, or --only <name> to add new cases.",
            file=sys.stderr,
        )
        return 1

    n_ok = n_exc = 0
    t_start = time.time()
    for name in names:
        t0 = time.time()
        spec, arrays = run_case(name, case_registry.CASES[name], api, fx)
        golden_io.save(GOLDEN_DIR, name, spec, arrays)
        dt = time.time() - t0
        if spec.get("t") == "exception":
            n_exc += 1
            flag = f"raised {spec['exc_type']}"
        else:
            n_ok += 1
            flag = f"{len(arrays)} arrays"
        print(f"  {name:52s} {dt:6.2f}s  {flag}")

    print(
        f"\n{len(names)} goldens written to {GOLDEN_DIR} in {time.time() - t_start:.1f}s\n"
        f"  {n_ok} returned a value, {n_exc} raised."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # pragma: no cover
        traceback.print_exc()
        raise SystemExit(2)
