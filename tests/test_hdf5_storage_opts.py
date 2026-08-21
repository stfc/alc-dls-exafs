"""Storage-efficiency guarantees for the results HDF5.

Locks in the disk optimisations:
- large arrays stored as float32 with the gzip ``shuffle`` filter,
- ``store_path_params`` gates persistence of amp/pha/lam/rep,
- dropping path params yields a strictly smaller file.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from larch_cli_wrapper.hdf5_store import ExafsHDF5Store

_NFINE = 60
_PCOARSE = 40


def _k_chi():
    k = np.linspace(0.0, 12.0, _NFINE)
    return k, np.sin(k) * np.exp(-k / 8.0)


def _path_dict(i: int, *, with_params: bool):
    k, chi = _k_chi()
    d = {
        "k": k,
        "chi": chi * (1.0 + 0.01 * i),
        "r_eff": 2.5 + 0.1 * i,
        "path_index": i,
        "nlegs": 2,
        "degeneracy": 1.0,
        "scatterer": "Cl",
        "cw_ratio": 100.0 / (i + 1),
        "angle": None,
    }
    if with_params:
        kp = np.linspace(0.0, 18.0, _PCOARSE)
        d.update(
            amp=np.linspace(1.0, 0.1, _PCOARSE),
            pha=np.linspace(0.0, 3.0, _PCOARSE),
            lam=np.full(_PCOARSE, 7.0),
            rep=np.linspace(0.0, 9.0, _PCOARSE),
            k_param=kp,
        )
    return d


def _write(tmp_path: Path, *, store_path_params: bool, n_paths: int = 20) -> Path:
    k, chi = _k_chi()
    p = tmp_path / f"results_{store_path_params}.h5"
    store = ExafsHDF5Store(
        p, mode="w", store_paths=True, store_path_params=store_path_params
    )
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
                    _path_dict(i, with_params=store_path_params) for i in range(n_paths)
                ],
            }
        ]
    )
    store.close()
    return p


def test_site_and_path_chi_are_float32_with_shuffle(tmp_path):
    p = _write(tmp_path, store_path_params=True)
    with h5py.File(p, "r") as f:
        site_chi = f["site_results"]["chi"]
        path_chi = f["path_results"]["chi"]
        assert site_chi.dtype == np.float32
        assert path_chi.dtype == np.float32
        # shuffle byte filter enabled for better float compression
        assert site_chi.shuffle is True
        assert path_chi.shuffle is True
        assert site_chi.compression == "gzip"


def test_path_params_stored_as_float32_when_enabled(tmp_path):
    p = _write(tmp_path, store_path_params=True)
    with h5py.File(p, "r") as f:
        pr = f["path_results"]
        for name in ("amp", "pha", "lam", "rep"):
            assert name in pr, f"{name} should be stored when store_path_params=True"
            assert pr[name].dtype == np.float32
        assert "k_grid_params" in pr


def test_path_params_omitted_when_disabled(tmp_path):
    p = _write(tmp_path, store_path_params=False)
    with h5py.File(p, "r") as f:
        pr = f["path_results"]
        for name in ("amp", "pha", "lam", "rep", "k_grid_params"):
            assert name not in pr, f"{name} must be absent when store_path_params=False"
        # chi and metadata are still present
        assert "chi" in pr
        assert "cw_ratio" in pr


def test_dropping_params_shrinks_file(tmp_path):
    big = _write(tmp_path, store_path_params=True, n_paths=200)
    small = _write(tmp_path, store_path_params=False, n_paths=200)
    assert small.stat().st_size < big.stat().st_size


def test_stored_chi_roundtrips_within_float32_precision(tmp_path):
    k, chi = _k_chi()
    p = _write(tmp_path, store_path_params=False, n_paths=1)
    with h5py.File(p, "r") as f:
        stored = np.array(f["site_results"]["chi"][0], dtype=np.float64)
    np.testing.assert_allclose(stored, chi, rtol=1e-6, atol=1e-7)
