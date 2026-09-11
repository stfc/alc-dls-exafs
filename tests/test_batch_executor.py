"""Tests for md_exafs.execution (ADR 0002)."""

from pathlib import Path

import numpy as np

from md_exafs.execution import FeffTask, merge_shards
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
