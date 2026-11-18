"""Results files must stay readable when a run is killed (Slurm time limit)."""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from md_exafs.hdf5_store import (
    AveragedPathsStore,
    ExafsHDF5Store,
    UnreadableHDF5Error,
)

N_COMMITTED = 5

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

_STORE_CHILD = textwrap.dedent(
    """
    import sys
    import numpy as np
    from md_exafs.hdf5_store import ExafsHDF5Store

    store = ExafsHDF5Store(sys.argv[1], store_paths=False)
    k = np.linspace(0.0, 10.0, 20)
    for i in range(10_000):
        store.write_site_results_batch([
            {
                "frame_index": i,
                "site_index": 0,
                "absorber_element": "Fe",
                "success": True,
                "k": k,
                "chi": np.sin(k + i),
            }
        ])
        print(i + 1, flush=True)
        sys.stdin.readline()
    """
)

_SHARD_CHILD = textwrap.dedent(
    """
    import sys
    import numpy as np
    from md_exafs.hdf5 import BatchShardWriter

    k = np.linspace(0.0, 10.0, 20)
    writer = BatchShardWriter(sys.argv[1], k_grid=k)
    for i in range(10_000):
        writer.add_task_result(
            frame_idx=i, site_idx=0, absorber_element="Fe", chi=np.sin(k + i)
        )
        writer.flush()
        print(i + 1, flush=True)
        sys.stdin.readline()
    """
)


def _kill_after_commits(script: str, path: Path, sig: signal.Signals) -> None:
    """Send *sig* once the child has committed N_COMMITTED batches.

    The child blocks on stdin after reporting each commit, so the signal lands
    while it is idle, never in the middle of an HDF5 write.  That makes the
    committed state exactly N_COMMITTED batches, every run.
    """
    proc = subprocess.Popen(  # noqa: S603 - fixed child script, our interpreter
        [sys.executable, "-c", script, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdin is not None and proc.stdout is not None
        for n in range(1, N_COMMITTED + 1):
            assert int(proc.stdout.readline()) == n
            if n < N_COMMITTED:
                proc.stdin.write("go\n")
                proc.stdin.flush()
        proc.send_signal(sig)
    finally:
        proc.wait(timeout=60)
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_store_batches_survive_kill(tmp_path: Path, sig: signal.Signals) -> None:
    path = tmp_path / "results.h5"
    _kill_after_commits(_STORE_CHILD, path, sig)

    with h5py.File(path, "r") as f:
        n_rows = f["site_results/chi"].shape[0]
    assert n_rows == N_COMMITTED


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_shard_writer_flush_survives_kill(tmp_path: Path, sig: signal.Signals) -> None:
    path = tmp_path / "shard.h5"
    _kill_after_commits(_SHARD_CHILD, path, sig)

    with h5py.File(path, "r") as f:
        n_tasks = len(f["tasks"])
    assert n_tasks == N_COMMITTED


def _write_one_batch(path: Path) -> None:
    store = ExafsHDF5Store(path, store_paths=False)
    k = np.linspace(0.0, 10.0, 20)
    store.write_site_results_batch(
        [
            {
                "frame_index": 0,
                "site_index": 0,
                "absorber_element": "Fe",
                "success": True,
                "k": k,
                "chi": np.sin(k),
            }
        ]
    )
    store.close()


def test_garbage_file_raises_unreadable_error(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"
    garbage = b"this is not hdf5" * 50
    path.write_bytes(garbage)

    with pytest.raises(UnreadableHDF5Error, match="unreadable") as exc_info:
        ExafsHDF5Store(path)

    assert isinstance(exc_info.value, OSError)
    assert str(path) in str(exc_info.value)
    assert path.read_bytes() == garbage


def test_truncated_file_raises_unreadable_error(tmp_path: Path) -> None:
    path = tmp_path / "results.h5"
    _write_one_batch(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(UnreadableHDF5Error, match="interrupted run"):
        ExafsHDF5Store(path)


def test_averaged_paths_store_raises_unreadable_error(tmp_path: Path) -> None:
    path = tmp_path / "averaged_paths.h5"
    path.write_bytes(b"garbage" * 100)

    with pytest.raises(UnreadableHDF5Error, match="unreadable"):
        AveragedPathsStore(path)


def test_missing_file_in_read_mode_is_not_reported_as_corrupt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ExafsHDF5Store(tmp_path / "missing.h5", mode="r")


def test_lock_error_is_not_reported_as_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "results.h5"
    _write_one_batch(path)

    def locked(*_args, **_kwargs):
        raise OSError(
            "Unable to synchronously open file (unable to lock file, errno = 11, "
            "error message = 'Resource temporarily unavailable')"
        )

    monkeypatch.setattr(h5py, "File", locked)

    with pytest.raises(OSError, match="unable to lock file") as exc_info:
        ExafsHDF5Store(path)
    assert not isinstance(exc_info.value, UnreadableHDF5Error)


def test_close_logs_flush_failure_instead_of_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ExafsHDF5Store(tmp_path / "results.h5")
    real_h5 = store._h5
    store._h5 = Mock(id=Mock(valid=True), flush=Mock(side_effect=OSError("disk full")))

    with caplog.at_level(logging.ERROR, logger="md_exafs.hdf5_store"):
        store.close()

    real_h5.close()
    assert "disk full" in caplog.text


def test_close_twice_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = ExafsHDF5Store(tmp_path / "results.h5")
    store.close()

    with caplog.at_level(logging.ERROR, logger="md_exafs.hdf5_store"):
        store.close()

    assert caplog.text == ""
