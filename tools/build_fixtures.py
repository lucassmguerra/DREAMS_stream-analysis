#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the test fixtures for the stream_analysis contract tests.

This script is the only place in the repository allowed to import
``nbody_streams``. It converts the raw m12i forecasting HDF5 into stream
coordinates once, writes the result to ``tests/fixtures/``, and after that the
package and its tests never need the simulation data or the N-body code again.

Inputs
------
``data/example_m12i_streams_forecasting.hdf5``
    16 selected m12i streams in three perturbation states, each of shape
    (16, 10000, 6) in galactocentric phase space, plus the present-day
    progenitor phase-space vectors and a ``stream_metadata`` group carrying
    Arpit's published metric values for those same 16 streams.

``/mnt/d/Research/GC_streams/analysis/streams_m12i_unperturb_stats_phi2_phi1.parquet``
    The published metrics table for all 5000 m12i streams. The 16 fixture
    streams are selected out of it by the HDF5 ``index`` dataset. This supplies
    the orbit-property columns, which the HDF5 alone does not carry in full, and
    it supplies an end-to-end reference for the pipeline.

Outputs
-------
``tests/fixtures/streams_m12i_<state>.npz``
    One file per state in {unperturb, perturb, perturb_xtrm}, holding phi1,
    phi2, the raw phase space, the progenitor vectors and the stream indices.

``tests/fixtures/orbits_m12i.parquet``
    The df_orbits-style table for the 16 streams, indexed by the original m12i
    stream index. The index is deliberately not a RangeIndex, because the
    pipeline must align on it rather than assume position.

``tests/fixtures/published_metrics_m12i_unperturb.parquet``
    The published 20-column metrics table rows for the same 16 streams, used as
    an independent end-to-end reference. Reported, never asserted, because it
    was produced by a different code path.

Notes
-----
The coordinate transform is run with ``optimizer_fit=True``, matching
``Stream_morphology_analysis.main``. It is the expensive step, roughly a minute
per state, which is exactly why its output is frozen into a fixture.

Run from the repository root:

    PYTHONPATH=~/py_scripts python tools/build_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
HDF5_PATH = REPO_ROOT / "data" / "example_m12i_streams_forecasting.hdf5"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

PUBLISHED_TABLE = Path(
    "/mnt/d/Research/GC_streams/analysis/streams_m12i_unperturb_stats_phi2_phi1.parquet"
)

STATES = ("unperturb", "perturb", "perturb_xtrm")

# The df_orbits columns build_metrics_table copies through to its output.
ORBIT_COLUMNS = [
    "min_pericenter_dist",
    "max_apocenter_dist",
    "mean_dist",
    "n_pericenters",
    "period",
]


def build_stream_fixture(state: str, hdf5_path: Path, out_dir: Path) -> Path:
    """
    Convert one perturbation state to stream coordinates and save it.

    Parameters
    ----------
    state : str
        One of "unperturb", "perturb", "perturb_xtrm".
    hdf5_path : Path
        Source HDF5 file.
    out_dir : Path
        Directory to write the npz into.

    Returns
    -------
    Path
        The written npz path.
    """
    from nbody_streams import coords

    with h5py.File(hdf5_path, "r") as f:
        xv = f[state][:]
        xv_prog = f["progenitor_present"][:]
        stream_index = f["index"][:]

    print(f"  {state}: xv {xv.shape}, running generate_stream_coords ...", flush=True)
    phi1, phi2 = coords.generate_stream_coords(
        xv=xv,
        xv_prog=xv_prog,
        optimizer_fit=True,
    )

    out_path = out_dir / f"streams_m12i_{state}.npz"
    np.savez_compressed(
        out_path,
        phi1=phi1,
        phi2=phi2,
        xv=xv,
        xv_prog=xv_prog,
        stream_index=stream_index,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  {state}: wrote {out_path.name} ({size_mb:.1f} MB)", flush=True)
    return out_path


def build_orbit_fixture(hdf5_path: Path, out_dir: Path) -> pd.DataFrame:
    """
    Build the df_orbits-style table for the 16 fixture streams.

    The table is indexed by the original m12i stream index, which is a sparse
    integer index rather than a RangeIndex. This is intentional. The pipeline
    must align on the index rather than assume row position, and this fixture is
    what proves it.

    Parameters
    ----------
    hdf5_path : Path
        Source HDF5 file, used for the stream index.
    out_dir : Path
        Directory to write the parquet into.

    Returns
    -------
    pandas.DataFrame
        The orbit table.
    """
    with h5py.File(hdf5_path, "r") as f:
        stream_index = f["index"][:]

    if not PUBLISHED_TABLE.exists():
        raise FileNotFoundError(
            f"Published metrics table not found at {PUBLISHED_TABLE}. "
            "It supplies mean_dist, n_pericenters and period, which the HDF5 "
            "stream_metadata group does not carry."
        )

    published = pd.read_parquet(PUBLISHED_TABLE)
    missing = set(stream_index) - set(published.index)
    if missing:
        raise KeyError(f"Stream indices absent from the published table: {sorted(missing)}")

    selected = published.loc[stream_index]

    df_orbits = selected[ORBIT_COLUMNS].copy()
    df_orbits.index.name = "stream_index"

    out_path = out_dir / "orbits_m12i.parquet"
    df_orbits.to_parquet(out_path, index=True)
    print(f"  orbits: wrote {out_path.name}, index {list(df_orbits.index[:4])}...", flush=True)

    published_out = out_dir / "published_metrics_m12i_unperturb.parquet"
    selected.to_parquet(published_out, index=True)
    print(f"  reference: wrote {published_out.name} ({selected.shape[1]} columns)", flush=True)

    return df_orbits


def main() -> int:
    if not HDF5_PATH.exists():
        print(
            f"Source data not found at {HDF5_PATH}.\n"
            "Copy it from /mnt/d/Research/GC_streams/example_m12i_streams_forecasting.hdf5 "
            "into data/ first. The data directory is gitignored.",
            file=sys.stderr,
        )
        return 1

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Building orbit fixture ...", flush=True)
    build_orbit_fixture(HDF5_PATH, FIXTURE_DIR)

    print("Building stream-coordinate fixtures ...", flush=True)
    for state in STATES:
        build_stream_fixture(state, HDF5_PATH, FIXTURE_DIR)

    total_mb = sum(p.stat().st_size for p in FIXTURE_DIR.glob("*")) / 1e6
    print(f"\nFixtures total {total_mb:.1f} MB in {FIXTURE_DIR}")
    if total_mb > 10:
        print(
            "Over the 10 MB commit threshold. Keep tests/fixtures/ gitignored and "
            "regenerate with this script."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
