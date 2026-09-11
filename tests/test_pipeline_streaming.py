"""Tests for disk-bounded streaming execution in ``FeffExecutor.execute_batch``.

Uses the ``fake_feff`` stub (no FEFF binary) to drive the real
parse -> recompute -> HDF5 store -> cleanup path, verifying:
- results are independent of ``stream_chunk_size`` (chunk invariance),
- every site is stored exactly once,
- per-path ``feffNNNN.dat`` files are deleted after each chunk,
- ``chunk_progress_callback`` reports a sensible sequence,
- ``store_min_cw_ratio`` prunes weak paths at write time.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

import larch_cli_wrapper.pipeline as pl
from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.hdf5_store import ExafsHDF5Store
from larch_cli_wrapper.pipeline import FeffBatch, FeffExecutor, FeffTask


def _make_batch(tmp_path: Path, n_frames: int, cfg: FeffConfig) -> FeffBatch:
    out = tmp_path / "out"
    tasks = []
    for frame in range(n_frames):
        d = out / f"frame_{frame:04d}" / "site_0000"
        d.mkdir(parents=True, exist_ok=True)
        inp = d / "feff.inp"
        inp.write_text("CONTROL 0 0 0 1 1 1\nPRINT 0 0 0 0 0 3\n")
        tasks.append(
            FeffTask(
                input_file=inp.resolve(),
                site_index=0,
                frame_index=frame,
                absorber_element="Fe",
            )
        )
    return FeffBatch(tasks=tasks, output_dir=out, config=cfg)


def _run(tmp_path, n_frames, chunk_size, fake_feff, *, min_cw=None, cleanup=True):
    cfg = FeffConfig(
        keep_path_files=True,
        cleanup_feff_files=cleanup,
        stream_chunk_size=chunk_size,
        store_min_cw_ratio=min_cw,
    )
    batch = _make_batch(tmp_path, n_frames, cfg)
    h5 = tmp_path / "results.h5"
    store = ExafsHDF5Store(h5, config=cfg, store_paths=True, store_path_params=False)
    ex = FeffExecutor(max_workers=2, hdf5_store=store)
    chunks: list[tuple[int, int]] = []
    with patch.object(pl, "run_multi_site_feff_calculations", fake_feff):
        results = ex.execute_batch(
            batch,
            parallel=True,
            chunk_progress_callback=lambda i, n: chunks.append((i, n)),
        )
    store.close()
    return h5, batch, results, chunks


def _read_store(h5: Path):
    with h5py.File(h5, "r") as f:
        sr = f["site_results"]
        frames = np.array(sr["frame_index"])
        sites = np.array(sr["site_index"])
        chi = np.array(sr["chi"])
        order = np.lexsort((sites, frames))
        n_paths = (
            int(f["path_results"]["frame_index"].shape[0])
            if "path_results" in f and "frame_index" in f["path_results"]
            else 0
        )
        return frames[order], sites[order], chi[order], n_paths


def test_all_sites_stored_exactly_once(tmp_path, fake_feff):
    h5, batch, results, _ = _run(
        tmp_path, n_frames=5, chunk_size=2, fake_feff=fake_feff
    )
    frames, sites, _chi, n_paths = _read_store(h5)
    assert len(frames) == 5
    # unique (frame, site) pairs
    assert len({(int(f), int(s)) for f, s in zip(frames, sites, strict=True)}) == 5
    assert all(results.values())
    assert n_paths == 5  # one path (feff0001.dat) per calculation


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 100, 0])
def test_results_independent_of_chunk_size(tmp_path, fake_feff, chunk_size):
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    h5_ref, *_ = _run(ref_dir, n_frames=6, chunk_size=100, fake_feff=fake_feff)
    f_ref, s_ref, chi_ref, np_ref = _read_store(h5_ref)

    run_dir = tmp_path / f"chunk_{chunk_size}"
    run_dir.mkdir()
    h5, *_ = _run(run_dir, n_frames=6, chunk_size=chunk_size, fake_feff=fake_feff)
    f, s, chi, npths = _read_store(h5)

    np.testing.assert_array_equal(f, f_ref)
    np.testing.assert_array_equal(s, s_ref)
    np.testing.assert_allclose(chi, chi_ref, rtol=0, atol=0)
    assert npths == np_ref


def test_feff_path_files_deleted_after_processing(tmp_path, fake_feff):
    """With cleanup on, feffNNNN.dat are removed (bounding on-disk usage)."""
    h5, batch, _r, _c = _run(tmp_path, n_frames=4, chunk_size=2, fake_feff=fake_feff)
    leftover = [p for task in batch.tasks for p in task.feff_dir.glob("feff[0-9]*.dat")]
    assert leftover == [], f"path files not cleaned up: {leftover}"
    # chi.dat is intentionally kept for Stage C result loading
    assert all((t.feff_dir / "chi.dat").exists() for t in batch.tasks)


def test_path_files_retained_when_cleanup_disabled(tmp_path, fake_feff):
    h5, batch, _r, _c = _run(
        tmp_path, n_frames=3, chunk_size=2, fake_feff=fake_feff, cleanup=False
    )
    assert all((t.feff_dir / "feff0001.dat").exists() for t in batch.tasks)


def test_chunk_progress_callback_sequence(tmp_path, fake_feff):
    _h5, _b, _r, chunks = _run(tmp_path, n_frames=5, chunk_size=2, fake_feff=fake_feff)
    # 5 frames / chunk 2 -> 3 chunks
    assert chunks == [(1, 3), (2, 3), (3, 3)]


def test_single_chunk_emits_no_chunk_callback(tmp_path, fake_feff):
    _h5, _b, _r, chunks = _run(
        tmp_path, n_frames=4, chunk_size=100, fake_feff=fake_feff
    )
    assert chunks == []  # only fires when work is split into >1 chunk


def test_store_min_cw_ratio_prunes_weak_paths(tmp_path, fake_feff):
    # fake feff paths have cw_ratio == 100; threshold above that prunes them all.
    h5, _b, _r, _c = _run(
        tmp_path, n_frames=3, chunk_size=2, fake_feff=fake_feff, min_cw=150.0
    )
    _f, _s, _chi, n_paths = _read_store(h5)
    assert n_paths == 0  # all pruned
    # site spectra are still stored (total spectrum unaffected by pruning)
    assert len(_f) == 3


def test_store_min_cw_ratio_below_threshold_keeps_paths(tmp_path, fake_feff):
    h5, _b, _r, _c = _run(
        tmp_path, n_frames=3, chunk_size=2, fake_feff=fake_feff, min_cw=50.0
    )
    _f, _s, _chi, n_paths = _read_store(h5)
    assert n_paths == 3
