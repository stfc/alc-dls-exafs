"""Unified shard-and-stream batch execution engine (ADR 0002).

Provides:
- FeffTask: atomic representation of a single FEFF calculation task.
- BatchExecutor: bounded-memory streaming executor running FEFF and serialising to HDF5.
- write_batch_shard: serialises FEFF runs that some other driver already executed.
- merge_shards: consolidates K batch shards into an ensemble archive.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .hdf5 import ArchiveReader, BatchShardWriter, EnsembleWriter
from .paths import PathResult, path_chi, read_paths_from_dir
from .potentials import PotentialsManager
from .spectra import average_chi_arrays, xftf_arrays
from .viz import group_path_results

logger = logging.getLogger("md_exafs.execution")

#: Default photoelectron wavenumber grid (Å⁻¹) that every batch shard is written on.
#: Shards must share a grid for :func:`merge_shards` to average them, so callers that
#: write shards outside :class:`BatchExecutor` should use this unless they pass an
#: explicit grid everywhere.
DEFAULT_K_GRID: np.ndarray = np.arange(0.05, 20.0, 0.05)


@dataclass
class FeffTask:
    """Represents a single FEFF calculation task."""

    frame_idx: int
    site_idx: int
    input_dir: Path
    absorber_element: str = ""
    task_id: str = ""

    def __post_init__(self) -> None:
        """Initialize and resolve input paths and task identifiers."""
        self.input_dir = Path(self.input_dir).resolve()
        if not self.task_id:
            self.task_id = f"frame_{self.frame_idx:04d}_site_{self.site_idx:04d}"

    @property
    def feff_dir(self) -> Path:
        """Alias for directory containing input and calculation files."""
        return self.input_dir


def _execute_feff_subprocess(
    feff_dir: Path,
    feff_executable: str = "feff",
    timeout: float = 300.0,
) -> bool:
    """Execute FEFF in a task directory."""
    feff_inp = feff_dir / "feff.inp"
    if not feff_inp.exists():
        logger.warning(f"feff.inp not found in {feff_dir}")
        return False

    log_path = feff_dir / "feff.log"
    try:
        with open(log_path, "w") as log_file:
            res = subprocess.run(
                [feff_executable],
                cwd=feff_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        return res.returncode == 0
    except Exception as exc:
        logger.warning(f"FEFF execution failed in {feff_dir}: {exc}")
        return False


def parse_task_outputs(
    task: FeffTask,
    threshold: float = 5.0,
    store_paths: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, list[PathResult]]:
    """Parse ``chi.dat`` and the path files of a single completed FEFF task.

    Public API: downstream drivers (e.g. the ``aiida-feff`` batch runner) reuse
    this so that a spectrum parsed outside :class:`BatchExecutor` is byte-for-byte
    the same as one parsed inside it.

    Args:
        task: The completed task; ``task.input_dir`` must be the directory FEFF
            ran in.
        threshold: Minimum curved-wave amplitude ratio (percent) for a scattering
            path to be retained. Ignored when ``store_paths`` is False.
        store_paths: Whether to read the individual ``feffNNNN.dat`` path files.

    Returns:
        ``(k, chi, paths)``. ``k`` and ``chi`` are ``None`` when ``chi.dat`` is
        missing or unparseable — callers must treat that as a failed task rather
        than substituting zeros, because a zero spectrum is indistinguishable
        from a real result once averaged.
    """
    chi_dat = task.input_dir / "chi.dat"
    k_arr: np.ndarray | None = None
    chi_arr: np.ndarray | None = None

    if chi_dat.exists():
        try:
            from larch.io import read_ascii

            grp = read_ascii(str(chi_dat))
            k_arr = np.asarray(grp.k, dtype=float)
            chi_arr = np.asarray(grp.chi, dtype=float)
        except Exception:
            # Fallback simple 2-column reader
            try:
                data = np.loadtxt(chi_dat, comments=["#", "*"])
                if data.ndim == 2 and data.shape[1] >= 2:
                    k_arr = data[:, 0]
                    chi_arr = data[:, 1]
            except Exception as exc:
                logger.debug(f"Could not parse chi.dat in {task.input_dir}: {exc}")

    paths: list[PathResult] = []
    if store_paths:
        try:
            paths = read_paths_from_dir(
                task.input_dir,
                threshold=threshold,
                frame_idx=task.frame_idx,
                site_idx=task.site_idx,
            )
        except Exception as exc:
            logger.debug(f"Could not read paths from {task.input_dir}: {exc}")

    return k_arr, chi_arr, paths


def collect_task_into_shard(
    writer: BatchShardWriter,
    task: FeffTask,
    k_grid: np.ndarray,
    threshold: float = 5.0,
    store_paths: bool = True,
) -> bool:
    """Parse one completed FEFF task and append it to an open batch shard.

    Public API: this is the single place where a FEFF run on disk becomes a row
    in a shard, so that shards written by :class:`BatchExecutor` and shards
    written by external drivers (e.g. ``aiida-feff``) are constructed
    identically.

    A task whose ``chi.dat`` could not be parsed is **not** written. Zero-filling
    it would put an indistinguishable all-zero spectrum into the shard, which
    then silently biases every downstream ensemble average toward zero.

    Args:
        writer: An open :class:`~md_exafs.hdf5.BatchShardWriter`.
        task: The completed task to collect.
        k_grid: Wavenumber grid to interpolate χ(k) onto. Must match the grid the
            ``writer`` was opened with.
        threshold: Minimum curved-wave amplitude ratio (percent) for path retention.
        store_paths: Whether to read and store individual scattering paths.

    Returns:
        True if the task was written, False if it had no usable spectrum.
    """
    k_res, chi_res, paths = parse_task_outputs(
        task, threshold=threshold, store_paths=store_paths
    )
    if k_res is None or chi_res is None:
        return False

    chi_interp = np.interp(k_grid, k_res, chi_res, left=0.0, right=0.0)
    writer.add_task_result(
        frame_idx=task.frame_idx,
        site_idx=task.site_idx,
        absorber_element=task.absorber_element,
        chi=chi_interp,
        paths=paths if store_paths else None,
    )
    return True


def write_batch_shard(
    tasks: list[FeffTask],
    output_h5: Path | str,
    k_grid: np.ndarray | None = None,
    threshold: float = 5.0,
    store_paths: bool = True,
) -> tuple[Path, int]:
    """Collect already-completed FEFF runs on disk into a batch shard.

    Public API for drivers that run FEFF themselves (a scheduler, an AiiDA
    ``CalcJob``) and only need md-exafs to serialise the results. Use
    :class:`BatchExecutor` instead if you also want md-exafs to run FEFF.

    Args:
        tasks: Completed tasks; each ``task.input_dir`` must contain FEFF's output.
        output_h5: Destination shard path.
        k_grid: Wavenumber grid; defaults to :data:`DEFAULT_K_GRID`.
        threshold: Minimum curved-wave amplitude ratio (percent) for path retention.
        store_paths: Whether to read and store individual scattering paths.

    Returns:
        ``(path, n_written)`` — the shard path and how many of ``tasks`` produced a
        usable spectrum. ``n_written < len(tasks)`` means some runs failed to parse.
    """
    out = Path(output_h5)
    grid = DEFAULT_K_GRID if k_grid is None else k_grid
    n_written = 0
    with BatchShardWriter(out, k_grid=grid, threshold=threshold) as writer:
        for task in tasks:
            if collect_task_into_shard(
                writer, task, k_grid=grid, threshold=threshold, store_paths=store_paths
            ):
                n_written += 1
            else:
                logger.warning(
                    f"No parseable chi.dat for {task.input_dir}; task omitted from shard."
                )
    return out, n_written


class BatchExecutor:
    """Streaming, chunked executor for FEFF calculation batches (ADR 0002)."""

    def __init__(
        self,
        tasks: list[FeffTask],
        output_h5: Path | str,
        chunk_size: int = 256,
        n_workers: int | None = None,
        feff_executable: str = "feff",
        threshold: float = 5.0,
        potentials_dir: Path | str | None = None,
        k_grid: np.ndarray | None = None,
        store_paths: bool = True,
        clean_scratch: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize BatchExecutor."""
        self.tasks = tasks
        self.output_h5 = Path(output_h5).resolve()
        self.chunk_size = max(1, chunk_size) if chunk_size > 0 else len(tasks)
        self.n_workers = n_workers or min(os.cpu_count() or 1, 16)
        self.feff_executable = feff_executable
        self.threshold = threshold
        self.potentials_dir = Path(potentials_dir).resolve() if potentials_dir else None
        self.k_grid = k_grid
        self.store_paths = store_paths
        self.clean_scratch = clean_scratch
        self.progress_callback = progress_callback

    def run(self) -> Path:
        """Execute all tasks in streaming chunks, writing to output_h5."""
        if not self.tasks:
            logger.warning("BatchExecutor received 0 tasks.")
            return self.output_h5

        # 1. Distribute potentials if precomputed
        if self.potentials_dir and self.potentials_dir.exists():
            logger.info(
                f"Distributing potentials from {self.potentials_dir} to {len(self.tasks)} tasks"
            )
            target_dirs = [t.input_dir for t in self.tasks]
            PotentialsManager.distribute(
                self.potentials_dir,
                target_dirs,
                mode="symlink",
                n_workers=self.n_workers,
            )

        # 2. Determine k_grid if not provided
        k_reference = self.k_grid if self.k_grid is not None else DEFAULT_K_GRID

        total_tasks = len(self.tasks)
        completed = 0

        with BatchShardWriter(
            self.output_h5, k_grid=k_reference, threshold=self.threshold
        ) as writer:
            # Process in chunks of chunk_size
            for chunk_start in range(0, total_tasks, self.chunk_size):
                chunk = self.tasks[chunk_start : chunk_start + self.chunk_size]

                # Run FEFF in parallel for this chunk
                if self.n_workers > 1 and len(chunk) > 1:
                    with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                        futures = {
                            pool.submit(
                                _execute_feff_subprocess,
                                t.input_dir,
                                self.feff_executable,
                            ): t
                            for t in chunk
                        }
                        for _fut in as_completed(futures):
                            pass
                else:
                    for t in chunk:
                        _execute_feff_subprocess(t.input_dir, self.feff_executable)

                # Parse and write results sequentially to HDF5
                for t in chunk:
                    written = collect_task_into_shard(
                        writer,
                        t,
                        k_grid=k_reference,
                        threshold=self.threshold,
                        store_paths=self.store_paths,
                    )
                    if not written:
                        logger.warning(
                            f"No parseable chi.dat for {t.input_dir}; task omitted from shard."
                        )

                    # Bounded disk cleanup: remove scratch directory contents immediately
                    if self.clean_scratch:
                        try:
                            shutil.rmtree(t.input_dir)
                        except Exception as exc:
                            logger.debug(
                                f"Failed to remove scratch dir {t.input_dir}: {exc}"
                            )

                    completed += 1
                    if self.progress_callback:
                        self.progress_callback(completed, total_tasks)

        logger.info(
            f"Batch execution complete: {completed}/{total_tasks} written to {self.output_h5}"
        )
        return self.output_h5


def merge_shards(
    shard_paths: Sequence[Path | str],
    ensemble_path: Path | str,
    fourier_params: dict[str, Any] | None = None,
    top_n_paths: int = 25,
) -> Path:
    """Merge K batch shard archives into a single consolidated ensemble archive (ADR 0002).

    Calculates:
    - Grand ensemble average <chi(k)> with sample standard deviation.
    - Per-frame and per-site average spectra.
    - Fourier transforms of all averages via Larch xftf.
    - Pre-evaluated top-N composite paths on standard fine k and R grids (ADR 0003).

    Args:
        shard_paths: List of paths to batch_shard.h5 files.
        ensemble_path: Target path for ensemble_results.h5.
        fourier_params: Optional Fourier transform settings.
        top_n_paths: Number of dominant composite paths to evaluate and store.

    Returns:
        Path to the generated ensemble_results.h5.
    """
    valid_shards = [Path(p).resolve() for p in shard_paths if Path(p).exists()]
    if not valid_shards:
        raise FileNotFoundError("No valid batch shard files provided for merging.")

    readers = [ArchiveReader(p) for p in valid_shards]
    k_ref = readers[0].k

    # Collect all frame/site data across all shards
    frame_chis: dict[int, list[np.ndarray]] = {}
    site_chis: dict[int, list[np.ndarray]] = {}
    all_chis: list[np.ndarray] = []
    all_paths: list[PathResult] = []

    for r in readers:
        # Collect paths
        for path in r.iter_paths():
            all_paths.append(path)

        # Collect chi by task from shard
        with r._open() as f:
            if "tasks" in f:
                for _tname, tgrp in f["tasks"].items():
                    f_idx = int(tgrp.attrs.get("frame_idx", 0))
                    s_idx = int(tgrp.attrs.get("site_idx", 0))
                    if "chi" in tgrp:
                        chi = np.array(tgrp["chi"])
                        all_chis.append(chi)
                        frame_chis.setdefault(f_idx, []).append(chi)
                        site_chis.setdefault(s_idx, []).append(chi)

    if not all_chis:
        raise ValueError("No calculated chi spectra found across the provided shards.")

    # Grand ensemble average
    k_ref, mean_chi, std_chi = average_chi_arrays([k_ref] * len(all_chis), all_chis)
    ft_overall = xftf_arrays(k_ref, mean_chi, fourier_params)

    out_path = Path(ensemble_path).resolve()
    with EnsembleWriter(
        out_path, k_grid=k_ref, fourier_params=fourier_params
    ) as writer:
        writer.set_overall_average(
            k_ref, mean_chi, chi_std=std_chi, ft_results=ft_overall
        )

        # Frame averages
        for f_idx, c_list in sorted(frame_chis.items()):
            _, f_mean, _ = average_chi_arrays([k_ref] * len(c_list), c_list)
            writer.add_frame_average(f_idx, k_ref, f_mean)

        # Site averages
        for s_idx, c_list in sorted(site_chis.items()):
            _, s_mean, _ = average_chi_arrays([k_ref] * len(c_list), c_list)
            writer.add_site_average(s_idx, k_ref, s_mean)

        # Top-N composite paths (ADR 0003)
        if all_paths:
            grouped = group_path_results(all_paths, r_bin_width=0.15)
            # Rank by cw_ratio * degeneracy or cw_ratio
            ranked = sorted(grouped, key=lambda x: x["cw_ratio"], reverse=True)[
                :top_n_paths
            ]

            top_evaluated = []
            for g in ranked:
                if g["feff_data"] is not None and g["k"] is not None:
                    p_chi = path_chi(
                        k_native=g["k"],
                        feff_data=g["feff_data"],
                        r_eff=g["r_eff"],
                        degeneracy=g["degeneracy"],
                        k_out=k_ref,
                        sigma2=g.get("sig2", 0.0),
                    )
                    ft_p = xftf_arrays(k_ref, p_chi, fourier_params)
                    g_eval = dict(g)
                    g_eval["chi"] = p_chi
                    g_eval["chir_mag"] = ft_p["chir_mag"]
                    top_evaluated.append(g_eval)
                else:
                    top_evaluated.append(g)

            writer.add_top_paths(top_evaluated)

    logger.info(f"Consolidated ensemble archive written to {out_path}")
    return out_path


__all__ = [
    "DEFAULT_K_GRID",
    "FeffTask",
    "BatchExecutor",
    "merge_shards",
    "parse_task_outputs",
    "collect_task_into_shard",
    "write_batch_shard",
]
