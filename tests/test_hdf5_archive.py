"""Tests for md_exafs.hdf5 (ADR 0002, ADR 0003, ADR 0004)."""

from pathlib import Path

import numpy as np

from md_exafs.hdf5 import ArchiveReader, BatchShardWriter, EnsembleWriter
from md_exafs.paths import PathResult


def test_batch_shard_writer_and_reader(tmp_path: Path):
    shard_file = tmp_path / "batch_shard.h5"
    k_grid = np.linspace(2.0, 15.0, 100)

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

    with BatchShardWriter(shard_file, k_grid=k_grid, threshold=5.0) as writer:
        writer.add_task_result(
            frame_idx=0,
            site_idx=0,
            absorber_element="Cu",
            chi=np.sin(k_grid),
            paths=[p1],
        )

    reader = ArchiveReader(shard_file)
    assert reader.is_shard
    assert not reader.is_ensemble
    assert np.allclose(reader.k, k_grid)
    assert np.allclose(reader.chi, np.sin(k_grid), atol=1e-6)

    paths = list(reader.iter_paths())
    assert len(paths) == 1
    assert paths[0].scatterer == "Cu"
    assert paths[0].r_eff == 2.5


def test_ensemble_writer_and_reader(tmp_path: Path):
    ensemble_file = tmp_path / "ensemble_results.h5"
    k_grid = np.linspace(2.0, 15.0, 100)
    chi = np.sin(k_grid)

    with EnsembleWriter(ensemble_file, k_grid=k_grid) as writer:
        writer.set_overall_average(
            k_grid,
            chi,
            chi_std=0.01 * np.ones_like(chi),
            ft_results={
                "r": np.linspace(0, 6, 50),
                "chir_mag": np.ones(50),
                "chir_re": np.ones(50),
                "chir_im": np.zeros(50),
            },
        )
        writer.add_site_average(
            site_idx=0,
            k=k_grid,
            chi=chi,
            ft_results={
                "r": np.linspace(0, 6, 50),
                "chir_mag": np.ones(50),
            },
        )

    reader = ArchiveReader(ensemble_file)
    assert reader.is_ensemble
    assert not reader.is_shard
    assert np.allclose(reader.k, k_grid)
    assert np.allclose(reader.chi, chi)
    assert reader.chir_mag is not None
    assert len(reader.chir_mag) == 50

    site_avg = reader.get_site_average(0)
    assert np.allclose(site_avg["k"], k_grid)
