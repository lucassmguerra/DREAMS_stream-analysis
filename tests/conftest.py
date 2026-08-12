# -*- coding: utf-8 -*-
"""
Shared pytest setup for the contract suite.

Puts ``tools/`` on the path so the case registry and the serializer are
importable, forces a headless matplotlib backend so the plotting cases never try
to open a window, and exposes the package under test plus the fixture loader.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end pipeline cases")


@pytest.fixture(scope="session")
def api():
    """The refactored package, under its new names."""
    import stream_analysis

    return stream_analysis


@pytest.fixture(scope="session")
def sa_ref():
    """
    The frozen source, loaded by path under its new names.

    Used only by the tests that check the compatibility shims and the
    reference pipeline. The contract tests compare against stored goldens, not
    against a live reference, so that they still mean something if
    ``/mnt/d/Research`` is not mounted.
    """
    import importlib.util

    import cases as case_registry

    source = REPO_ROOT / "reference" / "stream_analysis_source.py"
    spec = importlib.util.spec_from_file_location("sa_ref", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return case_registry.RefAPI(module)


@pytest.fixture(scope="session")
def fixtures(api):
    """Fixture loader bound to the package under test."""
    import cases as case_registry

    if not (FIXTURE_DIR / "streams_m12i_unperturb.npz").exists():
        pytest.skip(
            "Stream fixtures missing. Regenerate with "
            "`PYTHONPATH=~/py_scripts python tools/build_fixtures.py`."
        )
    return case_registry.Fixtures(api)
