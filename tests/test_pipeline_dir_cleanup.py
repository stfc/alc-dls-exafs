"""Tests for on-the-fly scratch directory eviction and lazy input generation.

The pipeline's disk footprint used to scale with the trajectory: every
frame/site directory was created up front and kept until the run finished.
These tests pin down the streaming replacement -- plan the tasks, materialize
one chunk of directories at a time, and evict each one as soon as its chi(k) is
safely in the HDF5 archive -- and, just as importantly, the cases where nothing
may be deleted:

- a lazily planned batch must produce byte-identical ``feff.inp`` files to an
  eager one, including under ``--reuse-potentials``;
- a spectrum that cannot be trusted is reported as a failure, kept out of the
  archive, and left on disk;
- a failed HDF5 commit aborts eviction, because the archive is the only other
  copy of the result.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase.build import bulk

from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.hdf5_store import ExafsHDF5Store
from larch_cli_wrapper.pipeline import (
    FeffBatch,
    FeffExecutor,
    FeffTask,
    InputGenerator,
    PipelineProcessor,
    ResultProcessor,
)

from .conftest import write_fake_feff_outputs

MOCK_FEFF = "larch_cli_wrapper.pipeline.run_multi_site_feff_calculations"


def _make_atoms(n_frames: int = 2):
    cu = bulk("Cu", "fcc", a=3.61)
    return [cu.copy() for _ in range(n_frames)]


def _control_line(feff_inp: Path) -> str:
    return next(
        line.strip()
        for line in feff_inp.read_text().splitlines()
        if line.strip().startswith("CONTROL")
    )


# ---------------------------------------------------------------------------
# Stage A: lazy planning
# ---------------------------------------------------------------------------


def test_lazy_input_generation_creates_no_directories(tmp_path: Path):
    """Lazy generation plans tasks without touching the filesystem."""
    out_dir = tmp_path / "lazy_out"
    structures = _make_atoms(4)
    cfg = FeffConfig()
    gen = InputGenerator(cfg)

    batch = gen.generate_trajectory_inputs(
        structures=structures,
        absorber="Cu",
        output_dir=out_dir,
        lazy=True,
    )

    assert batch.lazy is True
    assert len(batch.tasks) == 4
    assert not out_dir.exists()

    for i, task in enumerate(batch.tasks):
        assert task.frame_index == i
        assert task.site_index == 0
        assert task.absorber_element == "Cu"
        assert task.structure is not None
        assert not task.input_file.exists()

    # Materializing one task creates only that task's directory.
    task0 = batch.tasks[0]
    task0.materialize(batch.config)
    assert task0.input_file.exists()
    assert (task0.feff_dir / "feff.inp").exists()
    assert all(not t.feff_dir.exists() for t in batch.tasks[1:])


def test_eager_batch_is_not_marked_lazy(tmp_path: Path):
    """A batch written up front reports ``lazy=False`` and exists on disk."""
    out_dir = tmp_path / "eager_out"
    gen = InputGenerator(FeffConfig())

    batch = gen.generate_trajectory_inputs(
        structures=_make_atoms(2),
        absorber="Cu",
        output_dir=out_dir,
        lazy=False,
    )

    assert batch.lazy is False
    assert all(t.input_file.exists() for t in batch.tasks)


def test_materialize_without_structure_raises(tmp_path: Path):
    """An absent input file with nothing to regenerate it from is an error."""
    task = FeffTask(input_file=tmp_path / "gone" / "feff.inp", site_index=0)
    with pytest.raises(FileNotFoundError, match="no structure provided"):
        task.materialize(FeffConfig())


@pytest.mark.parametrize("precompute", [False, True])
def test_lazy_and_eager_inputs_are_identical(tmp_path: Path, precompute: bool):
    """Deferring input generation must not change the generated input.

    Regression test: the precompute path swaps in a path-only config
    (``CONTROL 0 0 0 1 1 1``) while writing the main tasks, then restores the
    caller's config.  A lazy batch writes its inputs *after* that restore, so
    handing back the restored config made every frame recompute the potentials
    it was supposed to be reusing -- silently, and only with --reuse-potentials.
    """
    structures = _make_atoms(2)
    generated: dict[bool, list[str]] = {}

    for lazy in (False, True):
        out_dir = tmp_path / f"lazy_{lazy}"
        cfg = FeffConfig(radius=4.0)
        batch = InputGenerator(cfg).generate_trajectory_inputs(
            structures=structures,
            absorber="Cu",
            output_dir=out_dir,
            precompute_potentials=precompute,
            precompute_potentials_structure=structures[0] if precompute else None,
            lazy=lazy,
        )
        for task in batch.tasks:
            task.materialize(batch.config)
        generated[lazy] = [t.input_file.read_text() for t in batch.tasks]

        # The caller's own config must come back untouched either way.
        assert cfg.control is None

    assert generated[True] == generated[False]

    # And, for the precompute case, that shared text is the path-only variant.
    expected = "CONTROL 0 0 0 1 1 1" if precompute else "CONTROL 1 1 1 1 1 1"
    assert all(expected in text for text in generated[True])


def test_lazy_generation_rejects_bad_absorber_up_front(tmp_path: Path):
    """A bad absorber fails during planning, not hours into the run."""
    gen = InputGenerator(FeffConfig())
    with pytest.raises((ValueError, IndexError)):
        gen.generate_trajectory_inputs(
            structures=_make_atoms(2),
            absorber="Pt",  # not present in a Cu cell
            output_dir=tmp_path / "bad_absorber",
            lazy=True,
        )


# ---------------------------------------------------------------------------
# Spectrum validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "chi", "expected"),
    [
        (np.linspace(0, 15, 100), np.sin(np.linspace(0, 15, 100)), None),
        (None, np.zeros(3), "no k/chi arrays"),
        (np.zeros(3), None, "no k/chi arrays"),
        (np.array([]), np.array([]), "empty k grid"),
        (np.linspace(0, 15, 100), np.zeros(50), "length mismatch"),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), "NaN/Inf"),
        (np.array([1.0, np.inf]), np.array([1.0, 2.0]), "NaN/Inf"),
        (np.array([1.0, 2.0]), np.array([1.0, np.nan]), "NaN/Inf"),
        (np.linspace(0, 15, 100), np.zeros(100), "identically zero"),
    ],
)
def test_spectrum_rejection_reason(k, chi, expected):
    """Unusable spectra are described; usable ones return ``None``."""
    reason = FeffExecutor._spectrum_rejection_reason(k, chi)
    if expected is None:
        assert reason is None
    else:
        assert reason is not None and expected in reason


# ---------------------------------------------------------------------------
# Stage B: streaming eviction
# ---------------------------------------------------------------------------


def test_clean_scratch_evicts_directories_on_the_fly(tmp_path: Path, fake_feff):
    """Every scratch directory is gone by the end of a clean run."""
    out_dir = tmp_path / "pipeline_run"
    h5_path = out_dir / "results.h5"

    cfg = FeffConfig(
        cleanup_feff_files=True,
        clean_scratch=True,
        stream_chunk_size=2,  # 4 tasks in 2 chunks of 2
    )
    proc = PipelineProcessor(cfg, max_workers=2, hdf5_path=h5_path)

    with patch(MOCK_FEFF, fake_feff):
        overall, frame_avgs, _site_avgs, individual = proc.process_trajectory(
            structures=_make_atoms(4),
            absorber="Cu",
            output_dir=out_dir,
            parallel=True,
        )

    assert overall is not None
    assert len(frame_avgs) == 4
    assert len(individual) == 4
    assert h5_path.exists()

    with ExafsHDF5Store(h5_path, mode="r") as store:
        assert len(list(store.iter_site_results())) == 4

    # Both the site directories and their now-empty frame parents are gone.
    assert list(out_dir.glob("frame_*")) == []


def test_failed_task_directory_is_preserved_for_debugging(tmp_path: Path):
    """A FEFF failure keeps its directory; its successful peers still go."""
    out_dir = tmp_path / "fail_run"
    h5_path = out_dir / "results.h5"

    cfg = FeffConfig(cleanup_feff_files=True, clean_scratch=True)
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=h5_path)

    def mock_feff(input_files, **_kw):
        results = []
        for i, inp in enumerate(input_files):
            d = Path(inp).parent
            if i == 0:  # first task fails, the rest succeed
                results.append((d, False))
            else:
                write_fake_feff_outputs(d)
                results.append((d, True))
        return results

    with patch(MOCK_FEFF, mock_feff):
        proc.process_trajectory(
            structures=_make_atoms(2),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    failed_dir = out_dir / "frame_0000" / "site_0000"
    assert failed_dir.exists(), "failed calculation directory should be preserved"
    assert (failed_dir / "feff.inp").exists()
    assert not (out_dir / "frame_0001" / "site_0000").exists()


def test_invalid_spectrum_is_rejected_not_archived(tmp_path: Path, caplog):
    """A zero spectrum is failed, kept off the archive, and left on disk.

    Validation is not merely a delete guard: a spectrum that cannot be trusted
    must not reach the ensemble average either, or the run is silently wrong.
    """
    out_dir = tmp_path / "invalid_run"
    h5_path = out_dir / "results.h5"

    cfg = FeffConfig(cleanup_feff_files=True, clean_scratch=True)
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=h5_path)

    def mock_feff(input_files, **_kw):
        results = []
        for i, inp in enumerate(input_files):
            d = Path(inp).parent
            write_fake_feff_outputs(d)
            if i == 0:
                # Physically impossible output that FEFF still exits 0 on.
                from .conftest import write_valid_chi_dat

                k = np.linspace(0.0, 15.0, 120)
                write_valid_chi_dat(d, k=k, chi=np.zeros_like(k))
            results.append((d, True))
        return results

    with caplog.at_level("WARNING"), patch(MOCK_FEFF, mock_feff):
        _overall, _frames, _sites, individual = proc.process_trajectory(
            structures=_make_atoms(3),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    # Rejected: excluded from the averages...
    assert len(individual) == 2
    assert "frame_0000_site_0000" not in individual
    # ...and from the archive...
    with ExafsHDF5Store(h5_path, mode="r") as store:
        stored = {(r.frame_index, r.site_index) for r in store.iter_site_results()}
    assert stored == {(1, 0), (2, 0)}
    # ...but kept on disk, with a reason the user can act on.
    assert (out_dir / "frame_0000" / "site_0000" / "chi.dat").exists()
    assert "identically zero" in caplog.text


def test_rejected_directory_drops_bulky_path_files(tmp_path: Path):
    """A retained directory keeps its diagnostics, not its per-path files.

    Keeping rejected runs for inspection must not reintroduce the unbounded
    disk growth this whole feature exists to prevent.
    """
    out_dir = tmp_path / "reject_paths"
    h5_path = out_dir / "results.h5"

    # keep_path_files drives ExafsHDF5Store(store_paths=...), so this run
    # parses and stores per-path contributions.
    cfg = FeffConfig(cleanup_feff_files=True, clean_scratch=True, keep_path_files=True)
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=h5_path)

    def mock_feff(input_files, **_kw):
        from .conftest import write_valid_chi_dat

        results = []
        for inp in input_files:
            d = Path(inp).parent
            write_fake_feff_outputs(d)
            k = np.linspace(0.0, 15.0, 120)
            write_valid_chi_dat(d, k=k, chi=np.zeros_like(k))
            results.append((d, True))
        return results

    with patch(MOCK_FEFF, mock_feff):
        proc.process_trajectory(
            structures=_make_atoms(1),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    site_dir = out_dir / "frame_0000" / "site_0000"
    assert site_dir.exists()
    assert (site_dir / "feff.inp").exists(), "inputs needed to reproduce the failure"
    assert (site_dir / "chi.dat").exists(), "the offending spectrum itself"
    assert list(site_dir.glob("feff[0-9]*.dat")) == [], "bulky path files must go"


def test_unparseable_paths_retain_directory_but_keep_the_spectrum(tmp_path: Path):
    """A path-parsing problem costs you the paths, not the calculation.

    chi(k) and the per-path decomposition fail independently.  A spectrum that
    reads back perfectly well is still a valid ensemble member even if larch
    cannot parse the feffNNNN.dat files next to it, so only the directory is
    held back for inspection.
    """
    out_dir = tmp_path / "bad_paths"
    h5_path = out_dir / "results.h5"

    cfg = FeffConfig(cleanup_feff_files=True, clean_scratch=True, keep_path_files=True)
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=h5_path)

    def mock_feff(input_files, **_kw):
        results = []
        for inp in input_files:
            d = Path(inp).parent
            write_fake_feff_outputs(d)
            # Valid chi.dat, but a path file larch cannot make sense of.
            (d / "feff0001.dat").write_text("not a FEFF path file\n")
            results.append((d, True))
        return results

    with patch(MOCK_FEFF, mock_feff):
        _overall, _frames, _sites, individual = proc.process_trajectory(
            structures=_make_atoms(2),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    # The spectra survive, in memory and in the archive...
    assert len(individual) == 2
    with ExafsHDF5Store(h5_path, mode="r") as store:
        assert len(list(store.iter_site_results())) == 2
    # ...while the directories are held back for inspection.
    for i in range(2):
        site_dir = out_dir / f"frame_{i:04d}" / "site_0000"
        assert site_dir.exists()
        assert (site_dir / "chi.dat").exists()


def test_hdf5_write_failure_aborts_directory_cleanup(tmp_path: Path, fake_feff):
    """A failed commit leaves the directories alone -- they are the only copy."""
    out_dir = tmp_path / "write_fail_run"
    h5_path = out_dir / "results.h5"

    cfg = FeffConfig(cleanup_feff_files=True, clean_scratch=True)
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=h5_path)

    def mock_write_fail(*_a, **_kw):
        raise OSError("Simulated disk error or permission failure")

    with (
        patch(MOCK_FEFF, fake_feff),
        patch.object(proc._hdf5_store, "write_site_results_batch", mock_write_fail),
    ):
        proc.process_trajectory(
            structures=_make_atoms(2),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    assert len(list(out_dir.glob("frame_*"))) == 2
    assert all(
        (out_dir / f"frame_{i:04d}" / "site_0000" / "chi.dat").exists()
        for i in range(2)
    )


def test_no_hdf5_fallback_retains_all_directories(tmp_path: Path, fake_feff):
    """Without an archive to evict into, nothing is deleted."""
    out_dir = tmp_path / "no_hdf5_run"

    cfg = FeffConfig(cleanup_feff_files=True, clean_scratch=True)
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=None)

    with patch(MOCK_FEFF, fake_feff):
        overall, _frames, _sites, individual = proc.process_trajectory(
            structures=_make_atoms(2),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    assert overall is not None
    assert len(individual) == 2
    for i in range(2):
        site_dir = out_dir / f"frame_{i:04d}" / "site_0000"
        assert site_dir.exists()
        assert (site_dir / "chi.dat").exists()


def test_clean_scratch_disabled_retains_all_directories(tmp_path: Path, fake_feff):
    """``--no-clean-scratch`` keeps every directory even with HDF5 enabled."""
    out_dir = tmp_path / "keep_run"
    h5_path = out_dir / "results.h5"

    cfg = FeffConfig(cleanup_feff_files=True, clean_scratch=False)
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=h5_path)

    with patch(MOCK_FEFF, fake_feff):
        proc.process_trajectory(
            structures=_make_atoms(2),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    for i in range(2):
        assert (out_dir / f"frame_{i:04d}" / "site_0000" / "chi.dat").exists()


def test_results_match_with_and_without_eviction(tmp_path: Path, fake_feff):
    """Deleting scratch directories must not change a single number."""
    spectra = {}
    for clean in (False, True):
        out_dir = tmp_path / f"clean_{clean}"
        cfg = FeffConfig(
            cleanup_feff_files=True, clean_scratch=clean, stream_chunk_size=2
        )
        proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=out_dir / "results.h5")
        with patch(MOCK_FEFF, fake_feff):
            overall, _f, _s, individual = proc.process_trajectory(
                structures=_make_atoms(4),
                absorber="Cu",
                output_dir=out_dir,
                parallel=False,
            )
        spectra[clean] = (overall, individual)

    overall_keep, ind_keep = spectra[False]
    overall_evict, ind_evict = spectra[True]
    assert len(ind_keep) == len(ind_evict) == 4
    assert {g.task_id for g in ind_keep} == {g.task_id for g in ind_evict}
    np.testing.assert_allclose(overall_keep.chi, overall_evict.chi)
    np.testing.assert_allclose(overall_keep.chir_mag, overall_evict.chir_mag)


def test_lazy_run_never_exceeds_one_chunk_of_directories(tmp_path: Path):
    """The live directory count stays bounded by ``stream_chunk_size``."""
    out_dir = tmp_path / "bounded"
    chunk_size = 2
    n_frames = 8
    peak = 0

    def counting_feff(input_files, **_kw):
        nonlocal peak
        results = []
        for inp in input_files:
            d = Path(inp).parent
            write_fake_feff_outputs(d)
            results.append((d, True))
        peak = max(peak, len(list(out_dir.glob("frame_*/site_*"))))
        return results

    cfg = FeffConfig(
        cleanup_feff_files=True, clean_scratch=True, stream_chunk_size=chunk_size
    )
    proc = PipelineProcessor(cfg, max_workers=1, hdf5_path=out_dir / "results.h5")
    with patch(MOCK_FEFF, counting_feff):
        proc.process_trajectory(
            structures=_make_atoms(n_frames),
            absorber="Cu",
            output_dir=out_dir,
            parallel=False,
        )

    assert peak <= chunk_size, f"{peak} live directories exceeds chunk size"
    assert list(out_dir.glob("frame_*")) == []


# ---------------------------------------------------------------------------
# Stage C: sourcing spectra without the scratch directories
# ---------------------------------------------------------------------------


def test_result_processor_prefers_in_memory_groups():
    """Stage C reuses Stage B's groups rather than re-reading chi.dat."""
    from larch import Group

    cfg = FeffConfig()
    task = FeffTask(
        input_file=Path("/nonexistent/feff.inp"), site_index=0, frame_index=0
    )

    group = Group()
    group.k = np.linspace(0, 10, 50)
    group.chi = np.sin(group.k)
    group.task_id = task.task_id
    group.site_idx = 0
    group.frame_idx = 0

    batch = FeffBatch(tasks=[task], output_dir=Path("/nonexistent"), config=cfg)
    loaded = ResultProcessor(cfg).load_successful_results(
        batch, {task.task_id: True}, loaded_groups={task.task_id: group}
    )

    assert loaded[task.task_id] is group


def test_result_processor_falls_back_to_hdf5(tmp_path: Path):
    """With the directory gone and no in-memory group, the archive is used."""
    cfg = FeffConfig()
    h5_path = tmp_path / "results.h5"
    k = np.linspace(0.0, 15.0, 120)
    chi = np.sin(k) * np.exp(-k / 10.0)

    with ExafsHDF5Store(h5_path, config=cfg) as store:
        store.write_site_results_batch(
            [
                {
                    "frame_index": 0,
                    "site_index": 0,
                    "k": k,
                    "chi": chi,
                    "absorber_element": "Cu",
                    "success": True,
                    "path_contributions": None,
                }
            ]
        )

    task = FeffTask(
        input_file=tmp_path / "gone" / "feff.inp",
        site_index=0,
        frame_index=0,
        absorber_element="Cu",
    )
    batch = FeffBatch(tasks=[task], output_dir=tmp_path, config=cfg)

    with ExafsHDF5Store(h5_path, mode="r") as store:
        loaded = ResultProcessor(cfg).load_successful_results(
            batch, {task.task_id: True}, hdf5_store=store
        )

    group = loaded[task.task_id]
    np.testing.assert_allclose(group.chi, chi)
    assert group.frame_idx == 0
    assert group.site_idx == 0
    # The r-space transform is reapplied with the configured FT parameters.
    assert hasattr(group, "chir_mag")
