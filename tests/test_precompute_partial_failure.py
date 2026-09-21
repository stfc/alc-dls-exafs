"""Tests for partial failures of the ``--reuse-potentials`` precompute stage.

Potentials are precomputed once per absorber site and then distributed into
every frame's task directory.  A FEFF crash on a single site (e.g. an atom
sitting exactly on the FMS boundary) must not stop the *other* sites from
running: previously ``_distribute_potentials`` was all-or-nothing, so one bad
site left every task directory without ``phase.pad``/``pot.pad`` and all
path-only runs aborted with ``#ERROR STOP cannot find phase.pad in rdxsph``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import larch_cli_wrapper.pipeline as pl
from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.pipeline import FeffBatch, FeffExecutor, FeffTask

from .conftest import write_fake_feff_outputs

_POTENTIALS = ("phase.pad", "pot.pad")


def _make_batch(
    tmp_path: Path, n_frames: int = 3, sites: tuple[int, ...] = (0, 1, 2)
) -> FeffBatch:
    """Batch of ``n_frames x sites`` path-only tasks plus one precompute task/site."""
    out = tmp_path / "out"

    precompute_tasks = []
    for site in sites:
        d = out / "precomputed_potentials" / f"site_{site:04d}"
        d.mkdir(parents=True, exist_ok=True)
        inp = d / "feff.inp"
        inp.write_text("CONTROL 1 1 1 0 0 0\n")
        precompute_tasks.append(
            FeffTask(
                input_file=inp.resolve(),
                site_index=site,
                frame_index=-1,
                absorber_element="Fe",
            )
        )

    tasks = []
    for frame in range(n_frames):
        for site in sites:
            d = out / f"frame_{frame:04d}" / f"site_{site:04d}"
            d.mkdir(parents=True, exist_ok=True)
            inp = d / "feff.inp"
            inp.write_text("CONTROL 0 0 0 1 1 1\n")
            tasks.append(
                FeffTask(
                    input_file=inp.resolve(),
                    site_index=site,
                    frame_index=frame,
                    absorber_element="Fe",
                )
            )

    return FeffBatch(
        tasks=tasks,
        output_dir=out,
        config=FeffConfig(potential_link_mode="copy"),
        precompute_tasks=precompute_tasks,
    )


def _fake_feff_factory(failing_sites: set[int], *, silent_failure: bool = False):
    """Fake ``run_multi_site_feff_calculations``.

    Precompute runs (``require_chi=False``) write ``phase.pad``/``pot.pad`` for
    every site except those in *failing_sites*.  With ``silent_failure`` the
    failing sites still report success but produce no files, mimicking output
    that vanished after a reported-OK run.
    """

    def _site_index(feff_dir: Path) -> int:
        return int(feff_dir.name.split("_")[-1])

    def _run(input_files, *_a, require_chi: bool = True, progress_callback=None, **_k):
        results = []
        total = len(input_files)
        for i, inp in enumerate(input_files, start=1):
            feff_dir = Path(inp).parent
            if not require_chi:  # precompute
                failed = _site_index(feff_dir) in failing_sites
                if not failed:
                    for filename in _POTENTIALS:
                        (feff_dir / filename).write_bytes(b"binary-potentials")
                reported = silent_failure if failed else True
                results.append((feff_dir, reported))
            else:  # path-only main run
                write_fake_feff_outputs(feff_dir)
                results.append((feff_dir, True))
            if progress_callback:
                progress_callback(i, total)
        return results

    return _run


def _execute(batch: FeffBatch, fake) -> dict[str, bool]:
    ex = FeffExecutor()
    with patch.object(pl, "run_multi_site_feff_calculations", fake):
        return ex.execute_batch(batch, parallel=False)


def test_one_failed_site_does_not_block_the_others(tmp_path):
    batch = _make_batch(tmp_path)
    original_tasks = list(batch.tasks)

    results = _execute(batch, _fake_feff_factory({1}))

    good = [t for t in original_tasks if t.site_index != 1]
    bad = [t for t in original_tasks if t.site_index == 1]

    # Healthy sites received their potentials and ran.
    for task in good:
        for filename in _POTENTIALS:
            assert (task.feff_dir / filename).exists(), (
                f"{filename} missing for {task.task_id}"
            )
        assert results[task.task_id] is True

    # The failed site was skipped, not silently run without potentials.
    for task in bad:
        assert not (task.feff_dir / "phase.pad").exists()
        assert results[task.task_id] is False
    assert batch.tasks == good


def test_reported_success_without_files_is_treated_as_failure(tmp_path):
    """A site whose potentials are absent on disk is unusable, exit code or not."""
    batch = _make_batch(tmp_path)
    original_tasks = list(batch.tasks)

    results = _execute(batch, _fake_feff_factory({2}, silent_failure=True))

    assert {t.site_index for t in batch.tasks} == {0, 1}
    skipped = [t for t in original_tasks if t.site_index == 2]
    assert skipped and all(results[t.task_id] is False for t in skipped)
    assert all(results[t.task_id] is True for t in batch.tasks)


def test_all_sites_failing_aborts_instead_of_running_doomed_tasks(tmp_path):
    batch = _make_batch(tmp_path)
    fake = _fake_feff_factory({0, 1, 2})

    with pytest.raises(RuntimeError, match="every absorber site"):
        _execute(batch, fake)

    # Nothing was run: no chi.dat anywhere.
    assert not list(batch.output_dir.rglob("chi.dat"))


def test_no_failures_behaves_as_before(tmp_path):
    batch = _make_batch(tmp_path)
    original_tasks = list(batch.tasks)

    results = _execute(batch, _fake_feff_factory(set()))

    assert batch.tasks == original_tasks
    assert all(results.values())
    for task in original_tasks:
        for filename in _POTENTIALS:
            assert (task.feff_dir / filename).exists()


def test_progress_reaches_total_after_skipping(tmp_path):
    """The progress bar must complete against the *remaining* task count."""
    batch = _make_batch(tmp_path)
    seen: list[tuple[int, int]] = []
    ex = FeffExecutor()
    with patch.object(pl, "run_multi_site_feff_calculations", _fake_feff_factory({1})):
        ex.execute_batch(
            batch,
            parallel=False,
            progress_callback=lambda c, t: seen.append((c, t)),
        )
    assert seen[-1] == (len(batch.tasks), len(batch.tasks))
    assert all(c <= t for c, t in seen)


def test_distribute_potentials_honours_valid_sites(tmp_path):
    batch = _make_batch(tmp_path, n_frames=2)
    for site in (0, 1, 2):
        d = batch.output_dir / "precomputed_potentials" / f"site_{site:04d}"
        for filename in _POTENTIALS:
            (d / filename).write_bytes(b"x")

    ex = FeffExecutor()
    n = ex._distribute_potentials(batch, parallel=False, valid_sites={0, 2})

    assert n == 2 * 2 * len(_POTENTIALS)  # 2 frames x 2 sites x 2 files
    for task in batch.tasks:
        placed = (task.feff_dir / "phase.pad").exists()
        assert placed is (task.site_index in {0, 2})
