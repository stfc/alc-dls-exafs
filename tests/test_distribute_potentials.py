"""Tests for precomputed-potential distribution (copy/hardlink/symlink).

Covers ``FeffExecutor._distribute_potentials`` and ``_place_potential_file``:
placement correctness, link modes, parallel vs serial equivalence, progress
reporting, and graceful handling of missing precompute directories.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from larch_cli_wrapper.feff_utils import FeffConfig
from larch_cli_wrapper.pipeline import FeffBatch, FeffExecutor, FeffTask

# A representative subset of the files the precompute step produces.
_FILES = ("phase.pad", "pot.pad", "genfmt.json")


def _make_batch(tmp_path: Path, mode: str, n_tasks: int = 3, n_sites: int = 1):
    """Create a batch with a populated precompute dir and empty task dirs."""
    out = tmp_path / "out"
    for site in range(n_sites):
        pc = out / "precomputed_potentials" / f"site_{site:04d}"
        pc.mkdir(parents=True)
        for fn in _FILES:
            (pc / fn).write_text(f"data-{site}-{fn}")

    tasks = []
    for frame in range(n_tasks):
        for site in range(n_sites):
            inp = out / f"frame_{frame:04d}" / f"site_{site:04d}" / "feff.inp"
            tasks.append(
                FeffTask(
                    input_file=inp.resolve(),
                    site_index=site,
                    frame_index=frame,
                    absorber_element="Fe",
                )
            )
    cfg = FeffConfig(potential_link_mode=mode)
    return FeffBatch(tasks=tasks, output_dir=out, config=cfg)


def _verify_placed(batch: FeffBatch, mode: str) -> None:
    for task in batch.tasks:
        for fn in _FILES:
            dst = task.feff_dir / fn
            src = (
                batch.output_dir
                / "precomputed_potentials"
                / f"site_{task.site_index:04d}"
                / fn
            )
            assert dst.exists(), f"{dst} missing"
            assert dst.read_text() == src.read_text()
            if mode == "symlink":
                assert dst.is_symlink()
            if mode == "hardlink":
                assert dst.stat().st_ino == src.stat().st_ino


@pytest.mark.parametrize("mode", ["copy", "hardlink", "symlink"])
def test_distribute_all_modes_place_correct_content(tmp_path, mode):
    batch = _make_batch(tmp_path, mode)
    ex = FeffExecutor()
    n = ex._distribute_potentials(batch, parallel=True)
    assert n == len(batch.tasks) * len(_FILES)
    _verify_placed(batch, mode)


def test_progress_callback_reaches_total(tmp_path):
    batch = _make_batch(tmp_path, "copy", n_tasks=4)
    ex = FeffExecutor()
    seen: list[tuple[int, int]] = []
    ex._distribute_potentials(
        batch, parallel=True, progress_callback=lambda c, t: seen.append((c, t))
    )
    total = len(batch.tasks) * len(_FILES)
    assert seen[0] == (0, total)
    assert seen[-1] == (total, total)
    # Monotonic non-decreasing completion counts.
    assert [c for c, _ in seen] == sorted(c for c, _ in seen)


def test_parallel_and_serial_equivalent(tmp_path):
    b1 = _make_batch(tmp_path / "a", "copy")
    b2 = _make_batch(tmp_path / "b", "copy")
    ex = FeffExecutor()
    n_par = ex._distribute_potentials(b1, parallel=True)
    n_ser = ex._distribute_potentials(b2, parallel=False)
    assert n_par == n_ser
    _verify_placed(b1, "copy")
    _verify_placed(b2, "copy")


def test_missing_precompute_dir_is_skipped(tmp_path, caplog):
    batch = _make_batch(tmp_path, "copy", n_tasks=2)
    # Add a task for a site with no precompute dir; it should be skipped, not fatal.
    orphan = FeffTask(
        input_file=(
            batch.output_dir / "frame_0099" / "site_0007" / "feff.inp"
        ).resolve(),
        site_index=7,
        frame_index=99,
        absorber_element="Fe",
    )
    batch.tasks.append(orphan)
    ex = FeffExecutor()
    n = ex._distribute_potentials(batch, parallel=False)
    # Only the 2 valid tasks are placed.
    assert n == 2 * len(_FILES)
    assert not orphan.feff_dir.exists() or not any(orphan.feff_dir.iterdir())


def test_place_potential_file_hardlink_fallback_to_copy(tmp_path, monkeypatch):
    """If hardlinking fails (e.g. cross-device), it falls back to a copy."""
    import larch_cli_wrapper.pipeline as pl

    src = tmp_path / "src.pad"
    src.write_text("payload")
    dst = tmp_path / "dst.pad"

    def _boom(*_a, **_k):
        raise OSError("Invalid cross-device link")

    monkeypatch.setattr(pl.os, "link", _boom)
    ex = FeffExecutor()
    ok = ex._place_potential_file(src, dst, "hardlink")
    assert ok
    assert dst.exists() and dst.read_text() == "payload"
    assert not dst.is_symlink()
    assert dst.stat().st_ino != src.stat().st_ino  # a real copy, not a link


def test_symlink_mode_creates_no_extra_bytes(tmp_path):
    """Symlinks point at the source rather than duplicating data."""
    batch = _make_batch(tmp_path, "symlink", n_tasks=3)
    ex = FeffExecutor()
    ex._distribute_potentials(batch, parallel=False)
    for task in batch.tasks:
        for fn in _FILES:
            link = task.feff_dir / fn
            assert link.is_symlink()
            assert Path(link.resolve()).exists()
