"""Tests for output-tree migration and the pickle result cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from ase import Atoms

from larch_cli_wrapper.cache_utils import (
    get_structure_hash,
    load_from_cache,
    save_to_cache,
)
from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.hdf5_store import ExafsHDF5Store
from larch_cli_wrapper.pipeline import FeffBatch, FeffExecutor, FeffTask

from .conftest import write_fake_feff_outputs


# --------------------------------------------------------------------------- #
# from_existing_output_dir migration
# --------------------------------------------------------------------------- #
def _build_output_tree(root: Path, n_frames: int, n_sites: int) -> None:
    for frame in range(n_frames):
        for site in range(n_sites):
            write_fake_feff_outputs(root / f"frame_{frame:04d}" / f"site_{site:04d}")


def test_migration_imports_all_sites(tmp_path):
    out = tmp_path / "tree"
    _build_output_tree(out, n_frames=3, n_sites=2)
    h5 = tmp_path / "migrated.h5"

    store = ExafsHDF5Store.from_existing_output_dir(out, hdf5_path=h5, store_paths=True)
    store.close()

    with h5py.File(h5, "r") as f:
        assert f["site_results"]["chi"].shape[0] == 6  # 3 frames x 2 sites
        assert "path_results" in f
        assert f["path_results"]["chi"].shape[0] == 6  # one path per site


def test_migration_without_paths(tmp_path):
    out = tmp_path / "tree"
    _build_output_tree(out, n_frames=2, n_sites=1)
    h5 = tmp_path / "migrated_nopaths.h5"

    store = ExafsHDF5Store.from_existing_output_dir(
        out, hdf5_path=h5, store_paths=False
    )
    store.close()

    with h5py.File(h5, "r") as f:
        assert f["site_results"]["chi"].shape[0] == 2
        assert "path_results" not in f


def test_migration_skips_sites_without_chi(tmp_path):
    out = tmp_path / "tree"
    _build_output_tree(out, n_frames=2, n_sites=1)
    # A site dir with no chi.dat must be skipped, not fatal.
    (out / "frame_0000" / "site_0009").mkdir(parents=True)
    h5 = tmp_path / "migrated.h5"

    store = ExafsHDF5Store.from_existing_output_dir(
        out, hdf5_path=h5, store_paths=False
    )
    store.close()
    with h5py.File(h5, "r") as f:
        assert f["site_results"]["chi"].shape[0] == 2


def test_migration_default_hdf5_path(tmp_path):
    out = tmp_path / "tree"
    _build_output_tree(out, n_frames=1, n_sites=1)
    store = ExafsHDF5Store.from_existing_output_dir(out, store_paths=False)
    store.close()
    assert (out / "results.h5").exists()


# --------------------------------------------------------------------------- #
# cache_utils
# --------------------------------------------------------------------------- #
def _atoms():
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


def test_structure_hash_is_deterministic_and_sensitive():
    a = _atoms()
    b = _atoms()
    assert get_structure_hash(a) == get_structure_hash(b)
    moved = _atoms()
    moved.positions[0][0] += 0.5
    assert get_structure_hash(moved) != get_structure_hash(a)


def test_cache_roundtrip(tmp_path):
    k = np.linspace(0, 10, 50)
    chi = np.sin(k) + 1j * np.cos(k)
    save_to_cache("key1", chi, k, tmp_path)
    assert (tmp_path / "key1.pkl").exists()
    loaded_k, loaded_chi = load_from_cache("key1", tmp_path)
    np.testing.assert_array_equal(loaded_k, k)
    np.testing.assert_array_equal(loaded_chi, chi)


def test_cache_miss_returns_none(tmp_path):
    assert load_from_cache("absent", tmp_path) is None


def test_cache_disabled_when_no_dir():
    assert load_from_cache("k", None) is None
    # save with no dir is a no-op (must not raise)
    save_to_cache("k", np.zeros(3), np.zeros(3), None)


def test_force_recalculate_ignores_existing_cache(tmp_path):
    k = np.arange(5.0)
    save_to_cache("k2", k, k, tmp_path)
    assert load_from_cache("k2", tmp_path, force_recalculate=True) is None


def test_corrupt_cache_file_is_removed(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not a pickle")
    assert load_from_cache("bad", tmp_path) is None
    # corrupt file is cleaned up so the next run recomputes cleanly
    assert not bad.exists()


# --------------------------------------------------------------------------- #
# FeffExecutor <-> pkl cache wiring
# --------------------------------------------------------------------------- #
def _cache_batch(tmp_path: Path) -> FeffBatch:
    """A two-task batch with inputs already on disk, ready to execute."""
    out = tmp_path / "run"
    tasks = []
    for frame in range(2):
        feff_dir = out / f"frame_{frame:04d}" / "site_0000"
        feff_dir.mkdir(parents=True, exist_ok=True)
        inp = feff_dir / "feff.inp"
        inp.write_text(f"TITLE frame {frame}\nCONTROL 1 1 1 1 1 1\nEND\n")
        tasks.append(FeffTask(input_file=inp, site_index=0, frame_index=frame))
    return FeffBatch(
        tasks=tasks, output_dir=out, config=FeffConfig(cleanup_feff_files=False)
    )


def test_executor_populates_pkl_cache(tmp_path, fake_feff):
    """Results are cached under the feff.inp hash, and reused on a second run."""
    cache_dir = tmp_path / "cache"
    batch = _cache_batch(tmp_path)

    with patch(
        "larch_cli_wrapper.pipeline.run_multi_site_feff_calculations", fake_feff
    ):
        results = FeffExecutor(cache_dir=cache_dir).execute_batch(batch, parallel=False)

    assert all(results.values())
    cached = sorted(p.name for p in cache_dir.glob("*.pkl"))
    assert len(cached) == 2, cached
    assert "None.pkl" not in cached

    # A second run is served entirely from the cache: FEFF is never invoked.
    def _explode(*_a, **_kw):
        raise AssertionError("FEFF should not run on a full cache hit")

    with patch("larch_cli_wrapper.pipeline.run_multi_site_feff_calculations", _explode):
        again = FeffExecutor(cache_dir=cache_dir).execute_batch(batch, parallel=False)
    assert all(again.values())


def test_force_recalculate_refreshes_pkl_cache(tmp_path, fake_feff):
    """--force-recalculate must rewrite the cache, not bypass and abandon it.

    The cache key is the hash of feff.inp, so it has to be computed even when
    the existing entry is being ignored; otherwise the refreshed result is
    written under a null key and the stale entry survives forever.
    """
    cache_dir = tmp_path / "cache"
    batch = _cache_batch(tmp_path)
    executor = FeffExecutor(cache_dir=cache_dir, force_recalculate=True)

    with patch(
        "larch_cli_wrapper.pipeline.run_multi_site_feff_calculations", fake_feff
    ):
        results = executor.execute_batch(batch, parallel=False)

    assert all(results.values())
    names = sorted(p.name for p in cache_dir.glob("*.pkl"))
    assert len(names) == 2, names
    assert "None.pkl" not in names
