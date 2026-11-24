"""Tests for md_exafs.execution (ADR 0002)."""

import shutil
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from md_exafs.execution import (
    DEFAULT_K_GRID,
    FeffTask,
    merge_shards,
    parse_task_outputs,
    write_batch_shard,
)
from md_exafs.feff_input import FeffConfig
from md_exafs.hdf5 import ArchiveReader, BatchShardWriter
from md_exafs.paths import PathResult, path_chi, read_feff_path_dat


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
            np.interp(DEFAULT_K_GRID, k_native, chi_a),
            np.interp(DEFAULT_K_GRID, k_native, chi_b),
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
    expected = np.interp(DEFAULT_K_GRID, k_native, chi)
    assert np.allclose(reader.chi, expected, atol=1e-5)


def test_parse_task_outputs_reports_missing_chi(tmp_path: Path):
    (tmp_path / "snap").mkdir()
    task = FeffTask(frame_idx=0, site_idx=0, input_dir=tmp_path / "snap")
    k, chi, paths = parse_task_outputs(task, store_paths=False)
    assert k is None
    assert chi is None
    assert paths == []


def _synthetic_feff_data(k: np.ndarray, amp_scale: float) -> np.ndarray:
    """Smooth, physically shaped columns ordered as ``FEFF_DATA_COLS``."""
    return np.column_stack(
        [
            13.0 - 0.05 * k,  # real_phc
            amp_scale * k * np.exp(-0.5 * k / 4.0) / (1.0 + 0.1 * k),  # mag_feff
            -10.0 - 0.6 * k,  # pha_feff
            1.0 - 0.01 * k,  # red_fact
            8.0 + 0.5 * k,  # lam
            np.sqrt(k**2 + 1.5),  # rep
        ]
    )


def _write_task_shard(
    path: Path,
    k_grid: np.ndarray,
    tasks: list[list[PathResult]],
    first_frame: int,
    config: FeffConfig | None = None,
) -> None:
    with BatchShardWriter(path, k_grid=k_grid, config=config) as w:
        for i, paths in enumerate(tasks):
            chi = np.zeros_like(k_grid)
            for p in paths:
                chi += path_chi(
                    p.k, p.feff_data, p.r_eff, p.degeneracy, k_grid, sigma2=p.sig2
                )
            w.add_task_result(
                frame_idx=first_frame + i,
                site_idx=0,
                absorber_element="Cu",
                chi=chi,
                paths=paths,
            )


def test_merge_shards_v4_top_paths_sum_and_metadata(tmp_path: Path):
    """Complete path sets: top_paths must sum to the overall average.

    Each task's chi is the exact sum of its stored paths, so any per-instance
    disorder damping lost in grouping (e.g. evaluating at the group-mean r_eff)
    shows up as a residual against the float64 overall average.
    """
    k_native = np.linspace(0.0, 20.0, 60)
    k_grid = np.linspace(0.05, 15.0, 150)
    cu_data = _synthetic_feff_data(k_native, 0.5)
    o_data = _synthetic_feff_data(k_native, 0.3)

    def cu(r_eff: float, frame: int) -> PathResult:
        return PathResult(
            frame_idx=frame,
            site_idx=0,
            r_eff=r_eff,
            nlegs=2,
            degeneracy=12.0,
            scatterer="Cu",
            cw_ratio=100.0,
            k=k_native,
            feff_data=cu_data,
            sig2=0.004,
        )

    def oxy(r_eff: float, frame: int) -> PathResult:
        return PathResult(
            frame_idx=frame,
            site_idx=0,
            r_eff=r_eff,
            nlegs=2,
            degeneracy=4.0,
            scatterer="O",
            cw_ratio=40.0,
            k=k_native,
            feff_data=o_data,
            sig2=0.006,
        )

    cfg = FeffConfig(radius=5.5)
    shard1, shard2 = tmp_path / "s1.h5", tmp_path / "s2.h5"
    _write_task_shard(
        shard1,
        k_grid,
        [[cu(2.50, 0), oxy(3.55, 0)], [cu(2.55, 1), oxy(3.60, 1)]],
        0,
        cfg,
    )
    _write_task_shard(shard2, k_grid, [[cu(2.60, 2)]], 2, cfg)

    ensemble = tmp_path / "ensemble.h5"
    merge_shards([shard1, shard2], ensemble_path=ensemble)

    reader = ArchiveReader(ensemble)
    assert reader.is_ensemble
    assert reader.format_version == 4
    assert reader.radius == 5.5
    assert reader.n_tasks == 3

    top_paths = reader.get_top_paths()
    assert [(p["scatterer"], p["count"]) for p in top_paths] == [("Cu", 3), ("O", 2)]

    chi_sum = sum(p["chi"].astype(np.float64) for p in top_paths)
    assert np.allclose(reader.chi, chi_sum, atol=1e-4)

    # Non-tautology guard: the old v3 behaviour (chi at the group-mean r_eff)
    # discards the per-instance dephasing and must differ substantially.
    mean_chi = path_chi(k_native, cu_data, 2.55, 12.0, k_grid, sigma2=0.004)
    assert np.max(np.abs(mean_chi - top_paths[0]["chi"])) > 1e-3

    # v3 archives (no radius, no n_tasks) must stay readable.
    v3 = tmp_path / "ensemble_v3.h5"
    shutil.copy(ensemble, v3)
    with h5py.File(v3, "r+") as f:
        f["meta"].attrs["format_version"] = 3
        del f["meta"].attrs["radius"]
        del f["top_paths"].attrs["n_tasks"]
    legacy = ArchiveReader(v3)
    assert legacy.is_ensemble is True
    assert legacy.radius is None
    assert legacy.n_tasks is None


def _write_triangle_dat(
    path: Path, scatterers: list[tuple[str, int, int, tuple[float, float, float]]]
):
    """Write a synthetic 3-leg feff0001.dat (absorber Fe at the origin)."""
    lines = [
        " comment: synthetic non-isosceles triangle         Feff8L (EXAFS)       0.1",
        " Source:",
        " Structure Summary:  Test",
        " space group: (P1), space number:  (1)",
        " abc:  10.0 10.0 10.0",
        " angles: 90.0 90.0 90.0",
        " sites: 3",
        " POT  SCF  30  4.0000   0, core-hole, AFOLP (folp(0)= 1.150)",
        " Abs   Z=26 Rmt= 1.256 Rnm= 1.439 Fe  shell",
        " Pot 1 Z= 8 Rmt= 0.701 Rnm= 1.029",
        " Pot 2 Z= 6 Rmt= 0.749 Rnm= 1.019",
        " Gam_ch=6.796E-01 H-L exch",
        " Mu=-1.236E+01 kf=2.747E+00 Vint=-2.078E+01 Rs_int= 1.320",
        " PATH  Rmax= 4.000,  Keep_limit= 0.00, Heap_limit 0.00  Pwcrit= 0.00%",
        " Path    1      icalc       2",
        " -----------------------------------------------------------------------",
        "   3   2.000   3.4380    1.9836  -12.36283 nleg, deg, reff, rnrmav(bohr), edge",  # noqa: E501
        "        x         y         z   pot at#",
        "     0.0000    0.0000    0.0000  0  26 Fe       absorbing atom",
    ]
    for symbol, ipot, z, (x, y, zc) in scatterers:
        lines.append(f"  {x:8.4f}  {y:8.4f}  {zc:8.4f}  {ipot}  {z:2d} {symbol}")
    lines.append(
        "    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda     real[p]@#"
    )
    for k in np.arange(0.0, 20.01, 0.5):
        lines.append(f"  {k:.3f}  1.0E+01  1.0E-01 -1.0E+01  1.0E+00  3.0E+01  1.5E+00")
    path.write_text("\n".join(lines) + "\n")


def test_merge_shards_time_reversal_non_isosceles_triangle(tmp_path: Path):
    """A 3-leg path and its time reverse are ONE FEFF path (degeneracy covers both).

    The triangle Fe(0,0,0), O(2,0,0), C(2.8,1.5,0) has very different internal
    angles at O and C, so any grouping keyed on a leg-order-dependent angle would
    split the two orientations.
    """
    o_leg = ("O", 1, 8, (2.0, 0.0, 0.0))
    c_leg = ("C", 2, 6, (2.8, 1.5, 0.0))
    for name, legs in (("fwd", [o_leg, c_leg]), ("rev", [c_leg, o_leg])):
        (tmp_path / name).mkdir()
        _write_triangle_dat(tmp_path / name / "feff0001.dat", legs)

    fwd = read_feff_path_dat(tmp_path / "fwd" / "feff0001.dat")
    rev = read_feff_path_dat(tmp_path / "rev" / "feff0001.dat")
    assert fwd is not None and rev is not None
    assert fwd.angle is not None and rev.angle is not None
    assert abs(fwd.angle - rev.angle) < 1e-6
    assert fwd.scatterer == rev.scatterer == "C-O"
    assert fwd.degeneracy == rev.degeneracy == 2.0

    k_grid = np.linspace(0.05, 15.0, 100)
    specs = [
        (fwd, "O-C", 0.00, 0.0),
        (rev, "C-O", 0.01, 1.0),
        (fwd, "C-O", -0.01, -1.0),
        (rev, "O-C", 0.02, 0.5),
    ]
    tasks = [
        [
            replace(
                base,
                frame_idx=i,
                scatterer=scatterer,
                r_eff=base.r_eff + dr,
                angle=base.angle + dang,
            )
        ]
        for i, (base, scatterer, dr, dang) in enumerate(specs)
    ]
    shard1, shard2 = tmp_path / "s1.h5", tmp_path / "s2.h5"
    _write_task_shard(shard1, k_grid, tasks[:2], 0)
    _write_task_shard(shard2, k_grid, tasks[2:], 2)

    ensemble = tmp_path / "ensemble.h5"
    merge_shards([shard1, shard2], ensemble_path=ensemble)

    top_paths = ArchiveReader(ensemble).get_top_paths()
    assert len(top_paths) == 1
    assert top_paths[0]["count"] == 4
    assert top_paths[0]["degeneracy"] == 2.0
    assert top_paths[0]["scatterer"] == "C-O"
