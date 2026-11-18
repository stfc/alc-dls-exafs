"""Stop requests: the SIGTERM handler, and chunk loops that exit cleanly."""

from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
from typer.testing import CliRunner

import md_exafs.execution as execution
import md_exafs.pipeline as pl
from md_exafs.cli import app
from md_exafs.execution import BatchExecutor, FeffTask
from md_exafs.feff_utils import FeffConfig
from md_exafs.hdf5 import BatchShardWriter
from md_exafs.hdf5_store import ExafsHDF5Store
from md_exafs.pipeline import FeffBatch, FeffExecutor
from md_exafs.stop import STOP, RunStoppedError, install_sigterm_handler

_CHI_HEADER = "       k          chi          mag           phase @#"


@pytest.fixture(autouse=True)
def _reset_stop():
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    STOP.clear()
    yield
    STOP.clear()
    for sig, handler in handlers.items():
        signal.signal(sig, handler)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGUSR1])
def test_signal_sets_stop_event(sig: signal.Signals) -> None:
    assert install_sigterm_handler() is True
    os.kill(os.getpid(), sig)
    assert STOP.is_set()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_install_is_idempotent() -> None:
    assert install_sigterm_handler() is True
    handler = signal.getsignal(signal.SIGTERM)
    assert install_sigterm_handler() is True
    assert signal.getsignal(signal.SIGTERM) is handler

    os.kill(os.getpid(), signal.SIGTERM)
    assert STOP.is_set()


def test_install_off_main_thread_is_noop() -> None:
    before = signal.getsignal(signal.SIGTERM)
    result: dict[str, bool] = {}

    thread = threading.Thread(
        target=lambda: result.update(ok=install_sigterm_handler())
    )
    thread.start()
    thread.join()

    assert result == {"ok": False}
    assert signal.getsignal(signal.SIGTERM) is before
    assert not STOP.is_set()


def _write_chi(input_dir: Path) -> None:
    k = np.linspace(0.0, 15.0, 120)
    chi = np.sin(k) * np.exp(-k / 10.0)
    data = np.column_stack([k, chi, np.abs(chi), np.zeros_like(k)])
    input_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(input_dir / "chi.dat", data, header=_CHI_HEADER, fmt="%.8e")


def _fake_feff(input_dir, feff_executable="feff", timeout=300.0) -> bool:
    _write_chi(Path(input_dir))
    return True


def _scratch_tasks(tmp_path: Path, n: int) -> list[FeffTask]:
    tasks = []
    for i in range(n):
        input_dir = tmp_path / "scratch" / f"t{i:02d}"
        input_dir.mkdir(parents=True)
        tasks.append(
            FeffTask(
                frame_idx=i,
                site_idx=0,
                input_dir=input_dir,
                absorber_element="Fe",
            )
        )
    return tasks


def _shard_task_count(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return len(f["tasks"])


def test_batch_executor_stops_before_first_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution, "_execute_feff_subprocess", _fake_feff)
    tasks = _scratch_tasks(tmp_path, 4)
    out = tmp_path / "shard.h5"
    STOP.set()

    executor = BatchExecutor(tasks, out, chunk_size=2, n_workers=1, clean_scratch=True)
    with pytest.raises(RunStoppedError):
        executor.run()

    assert _shard_task_count(out) == 0
    assert all(t.input_dir.exists() for t in tasks)


def test_batch_executor_finishes_in_flight_chunk_then_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def feff_requesting_stop(input_dir, feff_executable="feff", timeout=300.0) -> bool:
        calls.append(Path(input_dir))
        if len(calls) == 2:
            STOP.set()
        return _fake_feff(input_dir)

    monkeypatch.setattr(execution, "_execute_feff_subprocess", feff_requesting_stop)
    tasks = _scratch_tasks(tmp_path, 4)
    out = tmp_path / "shard.h5"

    executor = BatchExecutor(tasks, out, chunk_size=2, n_workers=1)
    with pytest.raises(RunStoppedError):
        executor.run()

    assert len(calls) == 2
    assert _shard_task_count(out) == 2


def test_batch_executor_removes_scratch_only_after_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(execution, "_execute_feff_subprocess", _fake_feff)
    tasks = _scratch_tasks(tmp_path, 2)
    scratch_present_at_flush: list[list[bool]] = []
    original_flush = BatchShardWriter.flush

    def checking_flush(self: BatchShardWriter) -> None:
        scratch_present_at_flush.append([t.input_dir.exists() for t in tasks])
        original_flush(self)

    monkeypatch.setattr(BatchShardWriter, "flush", checking_flush)
    executor = BatchExecutor(
        tasks,
        tmp_path / "shard.h5",
        chunk_size=2,
        n_workers=1,
        clean_scratch=True,
    )
    executor.run()

    assert scratch_present_at_flush == [[True, True]]
    assert not any(t.input_dir.exists() for t in tasks)


def test_failed_task_in_stopped_chunk_keeps_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def feff_killed_on_second_task(input_dir, feff_executable="feff", timeout=300.0):
        STOP.set()
        if Path(input_dir).name == "t01":
            return False
        return _fake_feff(input_dir)

    monkeypatch.setattr(
        execution, "_execute_feff_subprocess", feff_killed_on_second_task
    )
    tasks = _scratch_tasks(tmp_path, 3)
    out = tmp_path / "shard.h5"

    executor = BatchExecutor(tasks, out, chunk_size=2, n_workers=1, clean_scratch=True)
    with pytest.raises(RunStoppedError):
        executor.run()

    assert _shard_task_count(out) == 1
    assert not tasks[0].input_dir.exists()
    assert tasks[1].input_dir.exists()
    assert tasks[2].input_dir.exists()


def _pipeline_batch(tmp_path: Path, n_frames: int, chunk_size: int) -> FeffBatch:
    out = tmp_path / "out"
    cfg = FeffConfig(
        keep_path_files=False,
        cleanup_feff_files=True,
        clean_scratch=True,
        stream_chunk_size=chunk_size,
    )
    tasks = []
    for frame in range(n_frames):
        d = out / f"frame_{frame:04d}" / "site_0000"
        d.mkdir(parents=True, exist_ok=True)
        inp = d / "feff.inp"
        inp.write_text("CONTROL 0 0 0 1 1 1\nPRINT 0 0 0 0 0 3\n")
        tasks.append(
            pl.FeffTask(
                input_file=inp.resolve(),
                site_index=0,
                frame_index=frame,
                absorber_element="Fe",
            )
        )
    return FeffBatch(tasks=tasks, output_dir=out, config=cfg)


def test_pipeline_stops_before_any_chunk(tmp_path: Path, fake_feff) -> None:
    batch = _pipeline_batch(tmp_path, 4, chunk_size=2)
    h5 = tmp_path / "results.h5"
    store = ExafsHDF5Store(h5, config=batch.config, store_paths=False)
    executor = FeffExecutor(max_workers=2, hdf5_store=store)
    STOP.set()

    with patch.object(pl, "run_multi_site_feff_calculations", fake_feff):
        results = executor.execute_batch(batch, parallel=True)
    store.close()

    assert executor.stopped
    assert not any(results.values())
    with h5py.File(h5, "r") as f:
        assert "site_results" not in f


def test_pipeline_stops_between_chunks(tmp_path: Path, fake_feff) -> None:
    batch = _pipeline_batch(tmp_path, 4, chunk_size=2)
    h5 = tmp_path / "results.h5"
    store = ExafsHDF5Store(h5, config=batch.config, store_paths=False)
    executor = FeffExecutor(max_workers=2, hdf5_store=store)
    chunk_sizes: list[int] = []

    def feff_requesting_stop(input_files, **kwargs):
        chunk_sizes.append(len(input_files))
        results = fake_feff(input_files, **kwargs)
        STOP.set()
        return results

    with patch.object(pl, "run_multi_site_feff_calculations", feff_requesting_stop):
        results = executor.execute_batch(batch, parallel=True)
    store.close()

    assert executor.stopped
    assert chunk_sizes == [2]
    done = sorted(task_id for task_id, ok in results.items() if ok)
    assert done == [t.task_id for t in batch.tasks[:2]]
    with h5py.File(h5, "r") as f:
        assert f["site_results/chi"].shape[0] == 2


def test_pipeline_failed_task_in_stopped_chunk_is_not_complete(
    tmp_path: Path, fake_feff
) -> None:
    batch = _pipeline_batch(tmp_path, 4, chunk_size=2)
    h5 = tmp_path / "results.h5"
    store = ExafsHDF5Store(h5, config=batch.config, store_paths=False)
    executor = FeffExecutor(max_workers=2, hdf5_store=store)

    def feff_killed_on_second_task(input_files, **kwargs):
        results = fake_feff(input_files, **kwargs)
        STOP.set()
        feff_dir, _ = results[1]
        return [results[0], (feff_dir, False)]

    with patch.object(
        pl, "run_multi_site_feff_calculations", feff_killed_on_second_task
    ):
        results = executor.execute_batch(batch, parallel=True)
    store.close()

    ok_id, failed_id = batch.tasks[0].task_id, batch.tasks[1].task_id
    assert results[ok_id] is True
    assert results[failed_id] is False
    assert not batch.tasks[0].feff_dir.exists()
    assert batch.tasks[1].feff_dir.exists()


def test_cli_pipeline_reports_stop_as_failure(tmp_structure_file: Path) -> None:
    def stopped_processor(*_args, **_kwargs):
        proc = Mock()
        proc.process_trajectory.side_effect = RunStoppedError(
            "Run stopped before all calculations finished."
        )
        return proc

    with (
        patch("md_exafs.cli.ase_read") as mock_read,
        patch("md_exafs.cli.PipelineProcessor", side_effect=stopped_processor),
        patch("md_exafs.cli.parse_absorber_specification") as mock_parse,
    ):
        atoms = Mock()
        atoms.__len__ = Mock(return_value=3)
        mock_read.return_value = atoms
        mock_parse.return_value = {"absorber": 0, "description": "Fe (site 0)"}
        result = CliRunner().invoke(app, ["pipeline", str(tmp_structure_file), "Fe"])

    assert result.exit_code == 1
    assert "Stopped" in result.output
    assert "Pipeline completed successfully" not in result.output
