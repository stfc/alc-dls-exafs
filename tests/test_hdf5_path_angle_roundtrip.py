"""End-to-end round-trip test for the 3-body ``angle`` field through HDF5.

This exercises the full path: ``write_site_results_batch`` (raw per-frame
storage) -> HDF5 -> ``iter_path_contributions`` (read-back generator that
feeds ``PathAggregator``). A prior version of this pipeline stored ``angle``
correctly but never read it back, silently discarding it for any pipeline
run that persists paths to disk (as opposed to keeping them in memory).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from larch_cli_wrapper.hdf5_store import ExafsHDF5Store


def _k_chi():
    k = np.linspace(0.0, 10.0, 20)
    return k, np.sin(k)


def test_angle_survives_hdf5_round_trip():
    """A 3-leg path's angle is preserved through write -> HDF5 -> read."""
    k, chi = _k_chi()
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "results.h5"
        store = ExafsHDF5Store(hdf5_path, mode="w", store_paths=True)
        store.write_site_results_batch(
            [
                {
                    "frame_index": 0,
                    "site_index": 0,
                    "k": k,
                    "chi": chi,
                    "absorber_element": "Fe",
                    "success": True,
                    "path_contributions": [
                        {
                            "k": k,
                            "chi": chi,
                            "r_eff": 3.5,
                            "path_index": 1,
                            "nlegs": 3,
                            "degeneracy": 2.0,
                            "scatterer": "C-N",
                            "cw_ratio": 5.0,
                            "angle": 159.5,
                        }
                    ],
                }
            ]
        )
        store.close()

        store2 = ExafsHDF5Store(hdf5_path, mode="r")
        results = list(store2.iter_path_contributions())
        store2.close()

    assert len(results) == 1
    _path_key, info, _frame_idx, _site_idx = results[0]
    assert info["angle"] == 159.5


def test_2body_angle_is_none_after_round_trip():
    """A 2-leg path has no angle, and this survives as None (not NaN)."""
    k, chi = _k_chi()
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "results.h5"
        store = ExafsHDF5Store(hdf5_path, mode="w", store_paths=True)
        store.write_site_results_batch(
            [
                {
                    "frame_index": 0,
                    "site_index": 0,
                    "k": k,
                    "chi": chi,
                    "absorber_element": "Fe",
                    "success": True,
                    "path_contributions": [
                        {
                            "k": k,
                            "chi": chi,
                            "r_eff": 2.85,
                            "path_index": 1,
                            "nlegs": 2,
                            "degeneracy": 6.0,
                            "scatterer": "Cl",
                            "cw_ratio": 50.0,
                            "angle": None,
                        }
                    ],
                }
            ]
        )
        store.close()

        store2 = ExafsHDF5Store(hdf5_path, mode="r")
        results = list(store2.iter_path_contributions())
        store2.close()

    assert len(results) == 1
    _path_key, info, _frame_idx, _site_idx = results[0]
    assert info["angle"] is None


def test_mixed_2body_and_3body_paths_round_trip_correctly():
    """Multiple paths with a mix of angle/no-angle round-trip independently."""
    k, chi = _k_chi()
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "results.h5"
        store = ExafsHDF5Store(hdf5_path, mode="w", store_paths=True)
        store.write_site_results_batch(
            [
                {
                    "frame_index": 0,
                    "site_index": 0,
                    "k": k,
                    "chi": chi,
                    "absorber_element": "Fe",
                    "success": True,
                    "path_contributions": [
                        {
                            "k": k,
                            "chi": chi,
                            "r_eff": 2.85,
                            "path_index": 1,
                            "nlegs": 2,
                            "degeneracy": 6.0,
                            "scatterer": "Cl",
                            "cw_ratio": 50.0,
                            "angle": None,
                        },
                        {
                            "k": k,
                            "chi": chi,
                            "r_eff": 3.5,
                            "path_index": 1,
                            "nlegs": 3,
                            "degeneracy": 2.0,
                            "scatterer": "N-C",
                            "cw_ratio": 5.0,
                            "angle": 90.0,
                        },
                    ],
                }
            ]
        )
        store.close()

        store2 = ExafsHDF5Store(hdf5_path, mode="r")
        results = {
            info["nlegs"]: info["angle"]
            for _pk, info, _fi, _si in store2.iter_path_contributions()
        }
        store2.close()

    assert results[2] is None
    assert results[3] == 90.0
