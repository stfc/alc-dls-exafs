"""Tests for md_exafs.execution (ADR 0002)."""

from pathlib import Path

import numpy as np

from md_exafs.execution import (
    DEFAULT_K_GRID,
    FeffTask,
    merge_shards,
    parse_task_outputs,
    write_batch_shard,
)
from md_exafs.hdf5 import ArchiveReader, BatchShardWriter
from md_exafs.paths import PathResult


def test_feff_task():
    t = FeffTask(
        frame_idx=1, site_idx=2, input_dir=Path("/tmp/feff_test"), absorber_element="Cu"
    )
    assert t.task_id == "frame_0001_site_0002"
    assert t.feff_dir == Path("/tmp/feff_test").resolve()


def test_merge_shards(tmp_path: Path):
    s1_path = tmp_path / "shard_1.h5"
    s2_path = tmp_path / "shard_2.h5"
    ensemble_path = tmp_path / "ensemble_results.h5"

    k_grid = np.linspace(2.0, 15.0, 100)
    chi1 = np.sin(k_grid)
    chi2 = np.sin(k_grid) * 1.1

    p1 = PathResult(
        frame_idx=0,
        site_idx=0,
        r_eff=2.5,
        nlegs=2,
        degeneracy=12.0,
        scatterer="Cu",
        cw_ratio=100.0,
        k=np.linspace(0.0, 20.0, 20),
        feff_data=np.ones((20, 6)),
    )

    with BatchShardWriter(s1_path, k_grid=k_grid) as w1:
        w1.add_task_result(
            frame_idx=0, site_idx=0, absorber_element="Cu", chi=chi1, paths=[p1]
        )

    with BatchShardWriter(s2_path, k_grid=k_grid) as w2:
        w2.add_task_result(
            frame_idx=1, site_idx=0, absorber_element="Cu", chi=chi2, paths=[p1]
        )

    merged = merge_shards([s1_path, s2_path], ensemble_path=ensemble_path)
    assert merged.exists()

    reader = ArchiveReader(ensemble_path)
    assert reader.is_ensemble
    assert np.allclose(reader.chi, 1.05 * chi1, atol=1e-5)
    assert reader.chir_mag is not None
    assert len(reader.chir_mag) > 0

    # Verify site and frame averages
    site_0 = reader.get_site_average(0)
    assert np.allclose(site_0["chi"], 1.05 * chi1, atol=1e-5)

    frame_0 = reader.get_frame_average(0)
    assert np.allclose(frame_0["chi"], chi1, atol=1e-5)

    frame_1 = reader.get_frame_average(1)
    assert np.allclose(frame_1["chi"], chi2, atol=1e-5)


def _write_chi_dat(directory: Path, k: np.ndarray, chi: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savetxt(directory / "chi.dat", np.column_stack([k, chi]), header="k chi")


def test_write_batch_shard_collects_completed_runs(tmp_path: Path):
    """A driver that ran FEFF itself can serialise the results via the public API."""
    k_native = np.linspace(0.05, 19.95, 200)
    chi_a = np.sin(k_native)
    chi_b = np.cos(k_native)

    _write_chi_dat(tmp_path / "snap_a", k_native, chi_a)
    _write_chi_dat(tmp_path / "snap_b", k_native, chi_b)

    tasks = [
        FeffTask(
            frame_idx=0,
            site_idx=0,
            input_dir=tmp_path / "snap_a",
            absorber_element="Cu",
        ),
        FeffTask(
            frame_idx=1,
            site_idx=0,
            input_dir=tmp_path / "snap_b",
            absorber_element="Cu",
        ),
    ]

    out, n_written = write_batch_shard(
        tasks, tmp_path / "batch_shard.h5", store_paths=False
    )

    assert n_written == 2
    reader = ArchiveReader(out)
    assert reader.is_shard
    assert np.allclose(reader.k, DEFAULT_K_GRID)
    expected = np.mean(
        [
            np.interp(DEFAULT_K_GRID, k_native, chi_a, left=0.0, right=0.0),
            np.interp(DEFAULT_K_GRID, k_native, chi_b, left=0.0, right=0.0),
        ],
        axis=0,
    )
    assert np.allclose(reader.chi, expected, atol=1e-5)


def test_write_batch_shard_omits_unparseable_tasks(tmp_path: Path):
    """A task with no usable chi.dat must be omitted, never zero-filled.

    A zero spectrum is indistinguishable from a real result once averaged, so
    writing one would silently bias the ensemble mean toward zero.
    """
    k_native = np.linspace(0.05, 19.95, 200)
    chi = np.sin(k_native)
    _write_chi_dat(tmp_path / "snap_ok", k_native, chi)
    (tmp_path / "snap_broken").mkdir()  # ran, but produced no chi.dat

    tasks = [
        FeffTask(
            frame_idx=0,
            site_idx=0,
            input_dir=tmp_path / "snap_ok",
            absorber_element="Cu",
        ),
        FeffTask(
            frame_idx=1,
            site_idx=0,
            input_dir=tmp_path / "snap_broken",
            absorber_element="Cu",
        ),
    ]

    out, n_written = write_batch_shard(
        tasks, tmp_path / "batch_shard.h5", store_paths=False
    )

    assert n_written == 1
    reader = ArchiveReader(out)
    # The shard mean must equal the one good spectrum, not half of it.
    expected = np.interp(DEFAULT_K_GRID, k_native, chi, left=0.0, right=0.0)
    assert np.allclose(reader.chi, expected, atol=1e-5)


def test_parse_task_outputs_reports_missing_chi(tmp_path: Path):
    (tmp_path / "snap").mkdir()
    task = FeffTask(frame_idx=0, site_idx=0, input_dir=tmp_path / "snap")
    k, chi, paths = parse_task_outputs(task, store_paths=False)
    assert k is None
    assert chi is None
    assert paths == []
