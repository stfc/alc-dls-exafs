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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from md_exafs.stop import STOP, RunStoppedError

from .debye_waller import canonical_scatterer_key
from .feff_input import FeffConfig
from .hdf5 import ArchiveReader, BatchShardWriter, EnsembleWriter
from .paths import (
    PathResult,
    cluster_path_category,
    make_path_key,
    path_chi,
    read_paths_from_dir,
)
from .potentials import PotentialsManager
from .spectra import average_chi_arrays, resample_chi, xftf_arrays

logger = logging.getLogger("md_exafs.execution")

#: Default photoelectron wavenumber grid (Å⁻¹) that every batch shard is written on.
#: Shards must share a grid for :func:`merge_shards` to average them, so callers that
#: write shards outside :class:`BatchExecutor` should use this unless they pass an
#: explicit grid everywhere.
#:
#: Built by exact division rather than ``np.arange(0.05, 20.0, 0.05)``, whose
#: accumulated float error puts the last point at 19.950000000000003 — beyond a
#: ``chi.dat`` that ends at exactly 19.95, which would then resample to NaN.
DEFAULT_K_GRID: np.ndarray = np.arange(1, 400) / 20.0


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

    chi_interp = resample_chi(k_res, chi_res, k_grid)
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
    config: FeffConfig | None = None,
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
        config: Optional FEFF configuration recorded in the shard metadata (its
            ``radius`` is carried into merged ensemble archives).

    Returns:
        ``(path, n_written)`` — the shard path and how many of ``tasks`` produced a
        usable spectrum. ``n_written < len(tasks)`` means some runs failed to parse.
    """
    out = Path(output_h5)
    grid = DEFAULT_K_GRID if k_grid is None else k_grid
    n_written = 0
    with BatchShardWriter(
        out, k_grid=grid, threshold=threshold, config=config
    ) as writer:
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
        config: FeffConfig | None = None,
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
        self.config = config

    def run(self) -> Path:
        """Execute all tasks in streaming chunks, writing to output_h5.

        Raises RunStoppedError if :data:`md_exafs.stop.STOP` is set; the shard
        is then left incomplete.
        """
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

        stopped = False
        with BatchShardWriter(
            self.output_h5,
            k_grid=k_reference,
            threshold=self.threshold,
            config=self.config,
        ) as writer:
            # Process in chunks of chunk_size
            for chunk_start in range(0, total_tasks, self.chunk_size):
                if STOP.is_set():
                    logger.warning(
                        f"Stop requested before task {chunk_start}/{total_tasks}; "
                        f"{completed} tasks processed, written so far to "
                        f"{self.output_h5}. The shard is incomplete."
                    )
                    stopped = True
                    break
                chunk = self.tasks[chunk_start : chunk_start + self.chunk_size]

                # Run FEFF in parallel for this chunk
                if self.n_workers > 1 and len(chunk) > 1:
                    with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
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
                scratch_to_remove: list[Path] = []
                chunk_complete = True
                for t in chunk:
                    written = collect_task_into_shard(
                        writer,
                        t,
                        k_grid=k_reference,
                        threshold=self.threshold,
                        store_paths=self.store_paths,
                    )
                    if not written:
                        chunk_complete = False
                        logger.warning(
                            f"No parseable chi.dat for {t.input_dir}; task omitted from shard."
                        )

                    # A failed run in a stopped chunk may have been killed mid-way;
                    # keep its scratch for inspection.
                    if self.clean_scratch and (written or not STOP.is_set()):
                        scratch_to_remove.append(t.input_dir)

                    completed += 1
                    if self.progress_callback:
                        self.progress_callback(completed, total_tasks)

                # Scratch goes only after this chunk's results are on disk.
                writer.flush()
                for scratch_dir in scratch_to_remove:
                    try:
                        shutil.rmtree(scratch_dir)
                    except Exception as exc:
                        logger.debug(
                            f"Failed to remove scratch dir {scratch_dir}: {exc}"
                        )

                if STOP.is_set() and not chunk_complete:
                    stopped = True
                    break

        if stopped:
            raise RunStoppedError(
                f"BatchExecutor stopped with {completed}/{total_tasks} tasks "
                f"processed; {self.output_h5} is incomplete."
            )
        logger.info(
            f"Batch execution complete: {completed}/{total_tasks} written to {self.output_h5}"
        )
        return self.output_h5


def merge_shards(
    shard_paths: Sequence[Path | str],
    ensemble_path: Path | str,
    fourier_params: dict[str, Any] | None = None,
    top_n_paths: int = 25,
    radius: float | None = None,
    cluster_mode: str = "running_mean",
) -> Path:
    """Merge K batch shard archives into a single consolidated ensemble archive (ADR 0002).

    Calculates:
    - Grand ensemble average <chi(k)> with sample standard deviation.
    - Per-frame and per-site average spectra.
    - Fourier transforms of all averages via Larch xftf.
    - Top-N path groups (clustered by ``r_eff`` and, for 3-leg paths, angle), each
      stored as its exact ensemble-weighted contribution to the overall average.

    Args:
        shard_paths: List of paths to batch_shard.h5 files.
        ensemble_path: Target path for ensemble_results.h5.
        fourier_params: Optional Fourier transform settings.
        top_n_paths: Number of dominant path groups to evaluate and store.
        radius: FEFF cluster radius in Å to record in the archive. Defaults to
            the radius stored in the shards' FEFF config, or ``None`` if absent.
        cluster_mode: How FEFF path samples are grouped into ``/top_paths``:
            ``"running_mean"`` (default) or ``"valley"``.

    Returns:
        Path to the generated ensemble_results.h5.
    """
    from .debye_waller import CLUSTER_MODES

    if cluster_mode not in CLUSTER_MODES:
        raise ValueError(
            f"cluster_mode must be one of {CLUSTER_MODES}, got {cluster_mode!r}"
        )

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
    overall = average_chi_arrays([k_ref] * len(all_chis), all_chis)
    k_ref = overall.k
    ft_overall = xftf_arrays(k_ref, overall.mean, fourier_params)

    resolved_radius: float | None = None
    if radius is not None:
        resolved_radius = float(radius)
    else:
        shard_radii = [r.radius for r in readers if r.radius is not None]
        if shard_radii:
            resolved_radius = float(shard_radii[0])
            if len({float(x) for x in shard_radii}) > 1:
                logger.warning(
                    f"Shards have differing radii {sorted(set(shard_radii))}; "
                    f"using {resolved_radius}."
                )

    out_path = Path(ensemble_path).resolve()
    with EnsembleWriter(
        out_path,
        k_grid=k_ref,
        fourier_params=fourier_params,
        radius=resolved_radius,
    ) as writer:
        writer.set_overall_average(
            k_ref,
            overall.mean,
            chi_std=overall.std,
            ft_results=ft_overall,
            n_contributors=overall.n_contributors,
        )

        # Frame averages
        for f_idx, c_list in sorted(frame_chis.items()):
            avg = average_chi_arrays([k_ref] * len(c_list), c_list)
            writer.add_frame_average(
                f_idx, k_ref, avg.mean, n_contributors=avg.n_contributors
            )

        # Site averages
        for s_idx, c_list in sorted(site_chis.items()):
            avg = average_chi_arrays([k_ref] * len(c_list), c_list)
            writer.add_site_average(
                s_idx, k_ref, avg.mean, n_contributors=avg.n_contributors
            )

        # Top-N path groups: exact ensemble-weighted contribution to the average
        n_tasks = len(all_chis)
        top_evaluated: list[dict[str, Any]] = []
        if all_paths:
            scored = _rank_path_clusters(
                all_paths, n_tasks=n_tasks, cluster_mode=cluster_mode
            )

            denom = overall.n_contributors
            safe_denom = np.where(denom > 0, denom, 1)
            used_keys: set[str] = set()
            for _score, canon_scatterer, nlegs, cluster in scored[:top_n_paths]:
                chi_sum = np.zeros_like(k_ref, dtype=np.float64)
                for p in cluster:
                    if len(p.k) == 0:
                        continue
                    # Avoid UnivariateSpline extrapolation outside the FEFF k range
                    in_range = (k_ref >= float(p.k[0]) - 1e-12) & (
                        k_ref <= float(p.k[-1]) + 1e-12
                    )
                    if np.any(in_range):
                        chi_sum[in_range] += path_chi(
                            k_native=p.k,
                            feff_data=p.feff_data,
                            r_eff=p.r_eff,
                            degeneracy=p.degeneracy,
                            k_out=k_ref[in_range],
                            sigma2=p.sig2,
                        )
                chi_g = np.where(denom > 0, chi_sum / safe_denom, 0.0)
                top_evaluated.append(
                    _build_top_path_record(
                        canon_scatterer=canon_scatterer,
                        nlegs=nlegs,
                        cluster=cluster,
                        chi_g=chi_g,
                        k_ref=k_ref,
                        fourier_params=fourier_params,
                        used_keys=used_keys,
                    )
                )

        writer.add_top_paths(top_evaluated, n_tasks=n_tasks)

    logger.info(f"Consolidated ensemble archive written to {out_path}")
    return out_path


def _attr_or_key(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _rank_path_clusters(
    items: Sequence[Any],
    *,
    n_tasks: int,
    r_bin: float = 0.15,
    angle_tol: float = 15.0,
    cluster_mode: str = "running_mean",
) -> list[tuple[float, tuple[str, ...], int, list[Any]]]:
    """Cluster path items by (scatterer, nlegs, angle, r_eff) and rank by importance."""
    categories: dict[tuple[tuple[str, ...], int], list[Any]] = {}
    for p in items:
        cat = (
            canonical_scatterer_key(str(_attr_or_key(p, "scatterer", ""))),
            int(_attr_or_key(p, "nlegs", 0)),
        )
        categories.setdefault(cat, []).append(p)

    scored: list[tuple[float, tuple[str, ...], int, list[Any]]] = []
    for (canon_scatterer, nlegs), cat_paths in categories.items():
        for cluster in cluster_path_category(
            nlegs,
            cat_paths,
            r_bin=r_bin,
            angle_tol=angle_tol,
            cluster_mode=cluster_mode,
        ):
            if not cluster:
                continue
            mean_cw = float(
                np.mean([float(_attr_or_key(p, "cw_ratio", 0.0)) for p in cluster])
            )
            score = (len(cluster) / max(n_tasks, 1)) * mean_cw
            scored.append((score, canon_scatterer, nlegs, cluster))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _build_top_path_record(
    *,
    canon_scatterer: tuple[str, ...],
    nlegs: int,
    cluster: Sequence[Any],
    chi_g: np.ndarray,
    k_ref: np.ndarray,
    fourier_params: dict[str, Any] | None,
    used_keys: set[str],
) -> dict[str, Any]:
    """Format one ranked path cluster into an EnsembleWriter `/top_paths` record."""
    ft_p = xftf_arrays(k_ref, chi_g, fourier_params, warn_uncovered=False)
    mean_reff = float(np.mean([float(_attr_or_key(p, "r_eff", 0.0)) for p in cluster]))
    angles = [
        float(a)
        for p in cluster
        if (a := _attr_or_key(p, "angle")) is not None and np.isfinite(float(a))
    ]
    mean_angle = float(np.mean(angles)) if angles else None
    key = make_path_key(
        "-".join(canon_scatterer),
        nlegs,
        mean_reff,
        r_bin=0.15,
        angle=mean_angle,
    )
    if key in used_keys:
        key = f"{key}_{len(used_keys)}"
    used_keys.add(key)

    return {
        "path_key": key,
        "scatterer": "-".join(canon_scatterer),
        "nlegs": nlegs,
        "r_eff": mean_reff,
        "degeneracy": float(
            np.mean([float(_attr_or_key(p, "degeneracy", 0.0)) for p in cluster])
        ),
        "cw_ratio": float(
            np.mean([float(_attr_or_key(p, "cw_ratio", 0.0)) for p in cluster])
        ),
        "sig2": float(np.mean([float(_attr_or_key(p, "sig2", 0.0)) for p in cluster])),
        "count": len(cluster),
        "angle": mean_angle,
        "chi": chi_g,
        "chir_mag": ft_p["chir_mag"],
    }


def export_ensemble(
    results_h5: Path | str,
    ensemble_path: Path | str | None = None,
    *,
    fourier_params: dict[str, Any] | None = None,
    top_n_paths: int = 25,
    radius: float | None = None,
    chunk_size: int = 8192,
    cluster_mode: str = "running_mean",
) -> Path:
    """Export a format-4 ensemble archive from a pipeline ``results.h5`` store.

    Reads per-task ``site_results`` and (when present) ``path_results`` from an
    :class:`~md_exafs.hdf5_store.ExafsHDF5Store` file and writes a format-4
    ``ensemble_results.h5`` via :class:`~md_exafs.hdf5.EnsembleWriter` with
    ``format_version=4``, ``fourier_params``, ``radius``, per-k ``n_contributors``,
    and ensemble-weighted ``/top_paths``.

    Args:
        results_h5: Path to the pipeline ``results.h5`` HDF5 store.
        ensemble_path: Output path for the format-4 ensemble archive. Defaults
            to ``ensemble_results.h5`` in the same directory as ``results_h5``.
        fourier_params: Optional Fourier-transform overrides. When ``None``,
            parameters are extracted from ``meta/feff_config`` in ``results_h5``.
        top_n_paths: Maximum number of top path groups to store in ``/top_paths``.
        radius: Optional FEFF cluster radius override (Å). Defaults to the value
            stored in ``meta/feff_config``.
        chunk_size: Number of ``path_results/chi`` rows read per chunk.
        cluster_mode: How FEFF path samples are grouped into ``/top_paths``:
            ``"running_mean"`` (historical; caps group width, so one broad
            shell can be split) or ``"valley"`` (split only at density valleys
            between shells). Affects path attribution only.

    Returns:
        Resolved path to the written format-4 ensemble archive.
    """
    import json

    import h5py  # type: ignore[import-untyped]

    from .debye_waller import CLUSTER_MODES

    if cluster_mode not in CLUSTER_MODES:
        raise ValueError(
            f"cluster_mode must be one of {CLUSTER_MODES}, got {cluster_mode!r}"
        )

    src_path = Path(results_h5).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Pipeline results HDF5 not found: {src_path}")

    out_path = (
        Path(ensemble_path).resolve()
        if ensemble_path is not None
        else src_path.parent / "ensemble_results.h5"
    )

    with h5py.File(src_path, "r", locking=False) as f:
        if "site_results" not in f:
            raise ValueError(
                f"{src_path} does not contain a 'site_results' group "
                "(expected a pipeline ExafsHDF5Store file)."
            )

        cfg: dict[str, Any] = {}
        if "meta" in f and "feff_config" in f["meta"].attrs:
            raw_cfg = f["meta"].attrs["feff_config"]
            if isinstance(raw_cfg, bytes):
                raw_cfg = raw_cfg.decode("utf-8")
            cfg = json.loads(str(raw_cfg))

        resolved_radius: float | None = (
            float(radius)
            if radius is not None
            else (float(cfg["radius"]) if cfg.get("radius") is not None else None)
        )

        ft_use: dict[str, Any] = {}
        for key in ("kmin", "kmax", "kweight", "dk", "dk2", "window"):
            if cfg.get(key) is not None:
                ft_use[key] = cfg[key]
        if cfg.get("rmax") is not None:
            ft_use["rmax"] = cfg["rmax"]
        elif cfg.get("rmax_out") is not None:
            ft_use["rmax"] = cfg["rmax_out"]
        if fourier_params:
            ft_use.update(fourier_params)

        sr = f["site_results"]
        k_grid = np.asarray(sr["k_grid"], dtype=np.float64)
        chi_mat = np.asarray(sr["chi"], dtype=np.float64)
        frame_indices = np.asarray(sr["frame_index"], dtype=np.int64)
        site_indices = np.asarray(sr["site_index"], dtype=np.int64)
        if "success" in sr:
            ok_mask = np.asarray(sr["success"], dtype=bool)
        else:
            ok_mask = np.ones(len(frame_indices), dtype=bool)

        all_chis: list[np.ndarray] = []
        frame_chis: dict[int, list[np.ndarray]] = {}
        site_chis: dict[int, list[np.ndarray]] = {}
        for i in np.flatnonzero(ok_mask):
            chi_row = chi_mat[int(i)]
            all_chis.append(chi_row)
            frame_chis.setdefault(int(frame_indices[int(i)]), []).append(chi_row)
            site_chis.setdefault(int(site_indices[int(i)]), []).append(chi_row)

        if not all_chis:
            raise ValueError(f"No successful site_results found in {src_path}")

        overall = average_chi_arrays([k_grid] * len(all_chis), all_chis)
        k_ref = overall.k
        ft_overall = xftf_arrays(k_ref, overall.mean, ft_use or None)

        n_tasks = len(all_chis)
        top_evaluated: list[dict[str, Any]] = []
        if "path_results" in f and "chi" in f["path_results"]:
            pr = f["path_results"]
            n_rows = int(pr["chi"].shape[0])
            if n_rows > 0:
                k_paths = (
                    np.asarray(pr["k_grid_paths"], dtype=np.float64)
                    if "k_grid_paths" in pr
                    else k_grid
                )
                r_eff_arr = np.asarray(pr["r_eff"], dtype=np.float64)
                nlegs_arr = np.asarray(pr["nlegs"], dtype=np.int64)
                deg_arr = np.asarray(pr["degeneracy"], dtype=np.float64)
                cw_arr = np.asarray(pr["cw_ratio"], dtype=np.float64)
                angle_arr = (
                    np.asarray(pr["angle"], dtype=np.float64)
                    if "angle" in pr
                    else np.full(n_rows, np.nan)
                )
                raw_scat = pr["scatterer"][:]
                scat_list = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in raw_scat
                ]

                meta_rows = [
                    {
                        "row_idx": i,
                        "scatterer": scat_list[i],
                        "nlegs": int(nlegs_arr[i]),
                        "r_eff": float(r_eff_arr[i]),
                        "degeneracy": float(deg_arr[i]),
                        "cw_ratio": float(cw_arr[i]),
                        "sig2": 0.0,
                        "angle": (
                            float(angle_arr[i]) if np.isfinite(angle_arr[i]) else None
                        ),
                    }
                    for i in range(n_rows)
                ]

                scored = _rank_path_clusters(
                    meta_rows, n_tasks=n_tasks, cluster_mode=cluster_mode
                )
                kept_clusters = scored[:top_n_paths]

                row_to_cluster = np.full(n_rows, -1, dtype=np.int32)
                for c_idx, (_score, _sc, _nl, cluster) in enumerate(kept_clusters):
                    for item in cluster:
                        row_to_cluster[int(item["row_idx"])] = c_idx

                chi_sums = np.zeros(
                    (len(kept_clusters), k_paths.size), dtype=np.float64
                )
                chi_ds = pr["chi"]
                for start in range(0, n_rows, chunk_size):
                    end = min(start + chunk_size, n_rows)
                    c_ids = row_to_cluster[start:end]
                    active = np.flatnonzero(c_ids >= 0)
                    if active.size == 0:
                        continue
                    block = np.asarray(chi_ds[start:end], dtype=np.float64)
                    block = np.nan_to_num(block, nan=0.0)
                    for local_idx in active:
                        chi_sums[int(c_ids[local_idx])] += block[int(local_idx)]

                same_grid = k_paths.shape == k_ref.shape and np.allclose(
                    k_paths, k_ref, rtol=0.0, atol=1e-9
                )
                denom = overall.n_contributors
                safe_denom = np.where(denom > 0, denom, 1)
                used_keys: set[str] = set()
                for c_idx, (_score, canon_scatterer, nlegs, cluster) in enumerate(
                    kept_clusters
                ):
                    raw_sum = chi_sums[c_idx]
                    if not same_grid:
                        raw_sum = np.nan_to_num(
                            resample_chi(k_paths, raw_sum, k_ref), nan=0.0
                        )
                    chi_g = np.where(denom > 0, raw_sum / safe_denom, 0.0)
                    top_evaluated.append(
                        _build_top_path_record(
                            canon_scatterer=canon_scatterer,
                            nlegs=nlegs,
                            cluster=cluster,
                            chi_g=chi_g,
                            k_ref=k_ref,
                            fourier_params=ft_use or None,
                            used_keys=used_keys,
                        )
                    )

    with EnsembleWriter(
        out_path,
        k_grid=k_ref,
        fourier_params=ft_use or None,
        radius=resolved_radius,
    ) as writer:
        writer.set_overall_average(
            k_ref,
            overall.mean,
            chi_std=overall.std,
            ft_results=ft_overall,
            n_contributors=overall.n_contributors,
        )
        for f_idx, c_list in sorted(frame_chis.items()):
            avg = average_chi_arrays([k_ref] * len(c_list), c_list)
            writer.add_frame_average(
                f_idx, k_ref, avg.mean, n_contributors=avg.n_contributors
            )
        for s_idx, c_list in sorted(site_chis.items()):
            avg = average_chi_arrays([k_ref] * len(c_list), c_list)
            writer.add_site_average(
                s_idx, k_ref, avg.mean, n_contributors=avg.n_contributors
            )
        writer.add_top_paths(top_evaluated, n_tasks=n_tasks)

    logger.info(f"Exported format-4 ensemble archive to {out_path}")
    return out_path


__all__ = [
    "DEFAULT_K_GRID",
    "FeffTask",
    "BatchExecutor",
    "export_ensemble",
    "merge_shards",
    "parse_task_outputs",
    "collect_task_into_shard",
    "write_batch_shard",
]
