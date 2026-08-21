"""Tests for ``aggregate_store_paths`` (the averaged-paths engine).

Builds a synthetic ``path_results`` table via ``write_site_results_batch`` and
checks that the streaming aggregator clusters and averages path contributions
correctly across frames and sites.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from larch_cli_wrapper.exafs_data import aggregate_store_paths
from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.hdf5_store import ExafsHDF5Store

_K = np.linspace(0.0, 14.0, 80)


def _cl_path(scale=1.0, r=2.85):
    return {
        "k": _K,
        "chi": scale * np.sin(2 * _K * r) * np.exp(-_K / 10),
        "r_eff": r,
        "path_index": 1,
        "nlegs": 2,
        "degeneracy": 6.0,
        "scatterer": "Cl",
        "cw_ratio": 100.0,
        "angle": None,
    }


def _nc_path(scale=1.0, r=3.50, angle=160.0, scatterer="N-C"):
    return {
        "k": _K,
        "chi": scale * np.sin(2 * _K * r) * np.exp(-_K / 12),
        "r_eff": r,
        "path_index": 2,
        "nlegs": 3,
        "degeneracy": 2.0,
        "scatterer": scatterer,
        "cw_ratio": 8.0,
        "angle": angle,
    }


def _write_store(tmp_path: Path, records) -> Path:
    h5 = tmp_path / "results.h5"
    store = ExafsHDF5Store(h5, mode="w", store_paths=True, store_path_params=False)
    store.write_site_results_batch(records)
    store.close()
    return h5


def _record(frame, site, paths):
    return {
        "frame_index": frame,
        "site_index": site,
        "k": _K,
        "chi": np.sin(_K),
        "absorber_element": "Fe",
        "success": True,
        "path_contributions": paths,
    }


def test_empty_store_returns_empty(tmp_path):
    h5 = tmp_path / "empty.h5"
    store = ExafsHDF5Store(h5, mode="w", store_paths=True)
    store.write_site_results_batch(
        [_record(0, 0, [])]  # site result, no paths
    )
    store.close()
    overall, per_site = aggregate_store_paths(
        h5, FeffConfig().fourier_params, max_workers=1
    )
    assert overall == {}
    assert per_site == {}


def test_two_distinct_paths_form_two_clusters(tmp_path):
    records = [_record(f, 0, [_cl_path(), _nc_path()]) for f in range(4)]
    h5 = _write_store(tmp_path, records)
    overall, per_site = aggregate_store_paths(
        h5, FeffConfig().fourier_params, max_workers=1
    )
    assert len(overall) == 2
    nlegs = sorted(pc.nlegs for pc in overall.values())
    assert nlegs == [2, 3]
    # each path appears once per frame -> 4 samples per cluster
    assert all(pc.n_samples == 4 for pc in overall.values())
    assert set(per_site) == {0}
    assert len(per_site[0]) == 2


def test_scatterer_label_order_is_canonicalised(tmp_path):
    """``N-C`` and ``C-N`` describe the same physical path and must pool."""
    records = [
        _record(0, 0, [_nc_path(scatterer="N-C")]),
        _record(1, 0, [_nc_path(scatterer="C-N")]),
    ]
    h5 = _write_store(tmp_path, records)
    overall, _ = aggregate_store_paths(h5, FeffConfig().fourier_params, max_workers=1)
    assert len(overall) == 1
    (pc,) = overall.values()
    assert pc.n_samples == 2


def test_averaged_chi_is_mean_of_samples(tmp_path):
    records = [
        _record(0, 0, [_cl_path(scale=1.0)]),
        _record(1, 0, [_cl_path(scale=3.0)]),
    ]
    h5 = _write_store(tmp_path, records)
    overall, _ = aggregate_store_paths(h5, FeffConfig().fourier_params, max_workers=1)
    (pc,) = overall.values()
    expected = _cl_path(scale=2.0)["chi"]  # mean of scale 1 and 3
    np.testing.assert_allclose(pc.chi, expected, rtol=1e-5, atol=1e-6)


def test_per_site_separation(tmp_path):
    records = [
        _record(0, 0, [_cl_path()]),
        _record(0, 1, [_cl_path()]),
        _record(1, 0, [_cl_path()]),
    ]
    h5 = _write_store(tmp_path, records)
    overall, per_site = aggregate_store_paths(
        h5, FeffConfig().fourier_params, max_workers=1
    )
    assert set(per_site) == {0, 1}
    # site 0 pooled two frames, site 1 one frame
    (pc0,) = per_site[0].values()
    (pc1,) = per_site[1].values()
    assert pc0.n_samples == 2
    assert pc1.n_samples == 1
    # overall pools everything
    (pc_all,) = overall.values()
    assert pc_all.n_samples == 3


def test_distant_paths_not_merged(tmp_path):
    """Same scatterer/nlegs but r_eff beyond r_bin stay in separate clusters."""
    records = [
        _record(0, 0, [_cl_path(r=2.85), _cl_path(r=4.50)]),
    ]
    h5 = _write_store(tmp_path, records)
    overall, _ = aggregate_store_paths(
        h5, FeffConfig().fourier_params, r_bin=0.15, max_workers=1
    )
    assert len(overall) == 2
