"""Refactored three-stage EXAFS processing architecture.

This module implements a clean separation of concerns:
1. Input generation - Create all FEFF input files
2. FEFF execution - Run calculations in parallel
3. Result processing - Load, average, and plot results
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import copy2
from typing import Any, NamedTuple

import numpy as np
from ase import Atoms
from ase.geometry import wrap_positions
from larch import Group

from .feff_utils import (
    FeffConfig,
    WindowType,
    cleanup_feff_output,
    generate_multi_site_feff_inputs,
    normalize_absorbers,
    run_multi_site_feff_calculations,
    validate_absorber_indices,
)

logger = logging.getLogger(__name__)

#: Column header FEFF writes at the top of chi.dat, reproduced when a spectrum
#: is restored from a cache rather than produced by a FEFF run.
CHI_DAT_HEADER = "#       k          chi          mag           phase @#"

__all__ = [
    "FeffTask",
    "FeffBatch",
    "InputGenerator",
    "FeffExecutor",
    "ResultProcessor",
    "PipelineProcessor",
    "LoadedResults",
    "fourier_config_from_hdf5",
    "load_results_from_hdf5",
    "write_results_to_hdf5",
]


@dataclass
class FeffTask:
    """Represents a single FEFF calculation task."""

    input_file: Path
    site_index: int
    frame_index: int = 0
    absorber_element: str = ""
    #: Structure to generate ``feff.inp`` from on demand.  Only set for lazily
    #: planned tasks, whose input file is written just before the task runs.
    structure: Atoms | None = None

    @property
    def feff_dir(self) -> Path:
        """Directory containing the FEFF input file."""
        return self.input_file.parent

    @property
    def task_id(self) -> str:
        """Unique identifier for this task."""
        return f"frame_{self.frame_index:04d}_site_{self.site_index:04d}"

    def materialize(self, config: FeffConfig) -> Path:
        """Ensure ``feff.inp`` exists, generating it from the structure if needed.

        Args:
            config: Configuration to generate the input with.  For a batch that
                reuses precomputed potentials this **must** be the path-only
                config (``CONTROL 0 0 0 1 1 1``), i.e. ``FeffBatch.config``.

        Returns:
            Path to the (now existing) ``feff.inp``.

        Raises:
            FileNotFoundError: If the file is absent and no structure was
                recorded to regenerate it from.
        """
        if not self.input_file.exists():
            if self.structure is None:
                raise FileNotFoundError(
                    f"FEFF input file {self.input_file} does not exist "
                    "and no structure provided."
                )
            from .feff_utils import generate_pymatgen_input

            self.feff_dir.mkdir(parents=True, exist_ok=True)
            generate_pymatgen_input(
                self.structure,
                self.site_index,
                self.feff_dir,
                config,
            )
        return self.input_file


@dataclass
class FeffBatch:
    """Collection of FEFF tasks to be executed.

    This may contain a pre-compute task list that will generate potentials
    to be re-used by subsequent tasks.
    """

    tasks: list[FeffTask]
    output_dir: Path
    config: FeffConfig
    precompute_tasks: list[FeffTask] = None
    #: True when the tasks were only planned, not written to disk.  Their
    #: directories and ``feff.inp`` files are materialized one streaming chunk
    #: at a time, which is what keeps the on-disk directory count bounded.
    lazy: bool = False

    def get_precompute_tasks(self) -> list[FeffTask]:
        """Get pre-compute tasks, if any."""
        return self.precompute_tasks if self.precompute_tasks else []

    def get_tasks_by_frame(self) -> dict[int, list[FeffTask]]:
        """Group tasks by frame index."""
        frames = {}
        for task in self.tasks:
            if task.frame_index not in frames:
                frames[task.frame_index] = []
            frames[task.frame_index].append(task)
        return frames

    def get_tasks_by_site(self) -> dict[int, list[FeffTask]]:
        """Group tasks by site index."""
        sites = {}
        for task in self.tasks:
            if task.site_index not in sites:
                sites[task.site_index] = []
            sites[task.site_index].append(task)
        return sites


def build_exafs_group(
    k: np.ndarray,
    chi: np.ndarray,
    task: FeffTask,
    fourier_params: dict,
) -> Group:
    """Build a Fourier-transformed larch group for one site's chi(k).

    The transform uses the *full* configured Fourier parameters rather than
    ``kweight`` alone: leaving a spectrum on larch's built-in window/kmin/kmax
    defaults would transform it differently from the averages, silently
    ignoring the configured window, kmin, kmax and dk.

    Args:
        k: Photoelectron wavenumber grid.
        chi: chi(k) on that grid.  Complex input is reduced to its real part.
        task: Task the spectrum belongs to, used for the group's metadata.
        fourier_params: ``FeffConfig.fourier_params`` for the active config.

    Returns:
        A larch Group carrying ``k``, ``chi``, the r-space transform, and the
        frame/site/absorber metadata the averaging stage groups by.
    """
    from larch.xafs import xftf

    group = Group()
    group.k = np.asarray(k, dtype=np.float64)
    group.chi = np.asarray(np.real(chi), dtype=np.float64)
    xftf(group, **fourier_params)
    group.site_idx = task.site_index
    group.frame_idx = task.frame_index
    group.absorber_element = task.absorber_element
    group.task_id = task.task_id
    return group


class InputGenerator:
    """Stage A: Generate FEFF input files for all tasks."""

    def __init__(self, config: FeffConfig):
        """Initialize the input generator.

        Args:
            config: FEFF configuration object
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate_single_site_inputs(
        self,
        structure: Atoms,
        absorber: str | int | list[int],
        output_dir: Path,
        frame_index: int = 0,
    ) -> FeffBatch:
        """Generate inputs for single or multiple sites in one structure.

        Args:
            structure: ASE Atoms object
            absorber: Absorber specification (element symbol, index, or list of indices)
            output_dir: Base output directory
            frame_index: Frame index for trajectory processing

        Returns:
            FeffBatch containing all tasks for this structure
        """
        # Normalize absorber specification
        absorber_indices = normalize_absorbers(structure, absorber)

        # Generate inputs
        input_files = generate_multi_site_feff_inputs(
            atoms=structure,
            absorber_indices=absorber_indices,
            base_output_dir=output_dir,
            config=self.config,
        )

        # Create tasks
        tasks = []
        absorber_element = structure[absorber_indices[0]].symbol
        for i, input_file in enumerate(input_files):
            task = FeffTask(
                input_file=input_file.resolve(),  # Ensure absolute path
                site_index=absorber_indices[i],
                frame_index=frame_index,
                absorber_element=absorber_element,
            )
            tasks.append(task)

        return FeffBatch(tasks=tasks, output_dir=output_dir, config=self.config)

    def generate_trajectory_inputs(
        self,
        structures: list[Atoms],
        absorber: str | int | list[int],
        output_dir: Path,
        precompute_potentials: bool = False,
        precompute_potentials_structure: Atoms = None,
        input_progress_callback: callable = None,
        lazy: bool = False,
    ) -> FeffBatch:
        """Generate inputs for trajectory with multiple frames.

        Args:
            structures: List of ASE Atoms objects (trajectory frames)
            absorber: Absorber specification
            output_dir: Base output directory
            precompute_potentials: Whether to precompute potentials
            precompute_potentials_structure: Structure to use for
                                            precomputing potentials.
                                            If None, uses the average
                                            structure for all frames.
            input_progress_callback: Optional callback (completed, total)
                                     invoked after each frame's inputs are
                                     generated.
            lazy: If True, defer writing feff.inp and creating directories until
                  tasks are executed in chunks, bounding on-disk directory count.

        Returns:
            FeffBatch containing all tasks for all frames,
              with optional precompute tasks
        """
        precompute_tasks = []
        precompute_output_dir = None

        if precompute_potentials:
            # Determine structure for pre-computing potentials
            if precompute_potentials_structure is None:
                # Compute average structure
                self.logger.info(
                    "Computing average structure for pre-computing potentials"
                )
                precompute_potentials_structure = average_structure(structures)

            self.logger.info("Pre-computing potentials for trajectory")
            # Create pre-compute tasks
            precompute_output_dir = output_dir / "precomputed_potentials"
            precompute_output_dir.mkdir(parents=True, exist_ok=True)

            # Write structure used for pre-computing potentials
            structure_file = precompute_output_dir / "precompute_structure.extxyz"
            precompute_potentials_structure.write(structure_file)

            # Create config for precomputing (only potentials, no paths)
            # CONTROL: ipot=1, ixsph=1, ifms=1, ipaths=0, igenfmt=0, iff2x=0
            precompute_config = FeffConfig(**vars(self.config))
            precompute_config.control = "1 1 1 0 0 0"

            # Temporarily swap config to generate precompute inputs
            original_config = self.config
            self.config = precompute_config

            try:
                # Normalize absorber for the precompute structure
                absorber_indices = normalize_absorbers(
                    precompute_potentials_structure, absorber
                )

                # Generate FEFF inputs for precompute
                input_files = generate_multi_site_feff_inputs(
                    atoms=precompute_potentials_structure,
                    absorber_indices=absorber_indices,
                    base_output_dir=precompute_output_dir,
                    config=precompute_config,
                )

                # Create precompute tasks
                absorber_element = precompute_potentials_structure[
                    absorber_indices[0]
                ].symbol
                for i, input_file in enumerate(input_files):
                    task = FeffTask(
                        input_file=input_file.resolve(),
                        site_index=absorber_indices[i],
                        frame_index=-1,  # Special marker for precompute tasks
                        absorber_element=absorber_element,
                    )
                    precompute_tasks.append(task)

                self.logger.info(
                    f"Created {len(precompute_tasks)} precompute tasks for potentials"
                )
            finally:
                # Restore original config
                self.config = original_config

            # Now prepare config for main tasks (paths only, reuse potentials)
            # CONTROL: ipot=0, ixsph=0, ifms=0, ipaths=1, igenfmt=1, iff2x=1
            main_config = FeffConfig(**vars(self.config))
            main_config.control = "0 0 0 1 1 1"

            self.logger.info(
                "Generating main FEFF tasks to re-use pre-computed potentials"
            )
        else:
            # No precompute - use original config
            main_config = self.config

        # Generate tasks for all frames
        all_tasks: list[FeffTask] = []

        # Temporarily swap to main config if we're precomputing
        if precompute_potentials:
            original_config = self.config
            self.config = main_config

        n_frames = len(structures)
        if input_progress_callback:
            input_progress_callback(0, n_frames)
        try:
            # frame_idx is the 0-based index into `structures` as returned
            # by ase.io.read(), NOT the original trajectory frame number.
            # If the user passed e.g. {"index": "0::50"}, frame_0000 →
            # trajectory frame 0, frame_0001 → trajectory frame 50, etc.
            # TODO: store the original trajectory index alongside frame_idx
            # so that directory/HDF5 labels can be matched back to the
            # source trajectory without ambiguity.
            for frame_idx, structure in enumerate(structures):
                frame_dir = output_dir / f"frame_{frame_idx:04d}"
                if lazy:
                    absorber_indices = normalize_absorbers(structure, absorber)
                    # Validate up front, exactly as the eager path does via
                    # generate_multi_site_feff_inputs, so a bad absorber
                    # specification fails now rather than mid-run.
                    validate_absorber_indices(structure, absorber_indices)
                    absorber_element = structure[absorber_indices[0]].symbol
                    all_tasks.extend(
                        FeffTask(
                            input_file=(
                                frame_dir / f"site_{site_idx:04d}" / "feff.inp"
                            ).resolve(),
                            site_index=site_idx,
                            frame_index=frame_idx,
                            absorber_element=absorber_element,
                            structure=structure,
                        )
                        for site_idx in absorber_indices
                    )
                else:
                    frame_batch = self.generate_single_site_inputs(
                        structure=structure,
                        absorber=absorber,
                        output_dir=frame_dir,
                        frame_index=frame_idx,
                    )
                    all_tasks.extend(frame_batch.tasks)
                if input_progress_callback:
                    input_progress_callback(frame_idx + 1, n_frames)
        finally:
            if precompute_potentials:
                # Restore original config
                self.config = original_config

        return FeffBatch(
            tasks=all_tasks,
            output_dir=output_dir,
            # The config the *main* tasks were planned with, which for a
            # precompute run is the path-only variant (CONTROL 0 0 0 1 1 1).
            # A lazy batch writes its feff.inp files from this later on, so
            # handing back self.config here would silently recompute the
            # potentials the run is meant to be reusing.
            config=main_config,
            precompute_tasks=precompute_tasks if precompute_potentials else None,
            lazy=lazy,
        )


class FeffExecutor:
    """Stage B: Execute FEFF calculations in parallel with caching support."""

    def __init__(
        self,
        max_workers: int | None = None,
        cache_dir: Path | None = None,
        force_recalculate: bool = False,
        hdf5_store=None,
    ):
        """Initialize the FEFF executor.

        Args:
            max_workers: Maximum number of parallel workers
            cache_dir: Directory for caching results
            force_recalculate: Whether to force recalculation
            hdf5_store: Optional :class:`~larch_cli_wrapper.hdf5_store.ExafsHDF5Store`
                for writing per-site chi(k) and path contributions incrementally.
        """
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.force_recalculate = force_recalculate
        self.hdf5_store = hdf5_store
        self.loaded_groups: dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"FEFF caching enabled: {self.cache_dir}")

    def _get_feff_input_hash(self, feff_input_file: Path) -> str:
        """Generate cache key from FEFF input file content."""
        import hashlib

        try:
            content = feff_input_file.read_text()
            # Remove timestamps and other variable content, focus on calculation
            # parameters
            lines = []
            for line in content.split("\n"):
                line = line.strip()
                # Skip comment lines and empty lines that don't affect calculation
                if line and not line.startswith("*") and not line.startswith("#"):
                    lines.append(line)

            stable_content = "\n".join(lines)
            return hashlib.sha256(stable_content.encode()).hexdigest()[:16]
        except (OSError, ValueError) as e:
            self.logger.warning(f"Failed to hash FEFF input {feff_input_file}: {e}")
            # Fallback to file modification time and size
            stat = feff_input_file.stat()
            fallback_content = f"{stat.st_mtime}_{stat.st_size}"
            return hashlib.sha256(fallback_content.encode()).hexdigest()[:16]

    def _load_cached_result(self, cache_key: str) -> tuple[any, any] | None:
        """Load cached FEFF result if available."""
        if not self.cache_dir or self.force_recalculate:
            return None

        from .cache_utils import load_from_cache

        return load_from_cache(cache_key, self.cache_dir, self.force_recalculate)

    def _save_to_cache(self, cache_key: str, chi: any, k: any) -> None:
        """Save FEFF result to cache."""
        if not self.cache_dir:
            return

        from .cache_utils import save_to_cache

        save_to_cache(cache_key, chi, k, self.cache_dir)

    def _resolve_path_workers(self, n_dirs: int) -> int:
        """Mirror the FEFF worker count for parallel path parsing.

        Uses ``self.max_workers`` when set (the same value passed to the FEFF
        calculations), otherwise falls back to the identical default formula
        used by ``feff_utils._run_parallel``: ``min(n, cpu//2, 4)``.
        """
        if self.max_workers is not None:
            n_workers = self.max_workers
        else:
            cpu_count = os.cpu_count() or 4
            n_workers = min(n_dirs, max(1, cpu_count // 2), 4)
        return max(1, min(n_workers, n_dirs))

    def _parse_paths_parallel(
        self,
        dirs: list[tuple[str, Path]],
        parallel: bool,
        progress_callback: callable = None,
    ) -> dict[str, list]:
        """Parse feffNNNN.dat path files across directories.

        Reading path contributions relies on larch's ``FeffDatFile`` text
        parsing, which is CPU/GIL-bound, so this is offloaded to a
        ``ProcessPoolExecutor``.  The returned path-contribution dicts contain
        only numpy arrays and scalars, so they pickle cleanly across processes.
        HDF5 writing remains serial in the caller.

        Args:
            dirs: ``[(task_id, feff_dir), ...]`` for successful calculations.
            parallel: Whether parallel execution is enabled for this batch.
            progress_callback: Optional callback ``(completed, total)`` invoked
                as each directory's path files finish parsing.

        Returns:
            ``{task_id: [path_contribution_dict, ...]}``.
        """
        from .hdf5_store import _read_path_contributions_from_dir

        max_paths = self.hdf5_store.max_paths

        if not dirs:
            return {}

        total = len(dirs)
        if progress_callback:
            progress_callback(0, total)

        def _serial() -> dict[str, list]:
            out: dict[str, list] = {}
            for done, (task_id, feff_dir) in enumerate(dirs, start=1):
                out[task_id] = _read_path_contributions_from_dir(
                    feff_dir, max_paths=max_paths
                )
                if progress_callback:
                    progress_callback(done, total)
            return out

        n_workers = self._resolve_path_workers(total)
        if not parallel or n_workers <= 1 or total == 1:
            return _serial()

        from concurrent.futures import ProcessPoolExecutor, as_completed

        self.logger.info(
            f"Reading path contributions from {total} directories "
            f"using {n_workers} worker processes"
        )
        results: dict[str, list] = {}
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                future_to_task = {
                    executor.submit(
                        _read_path_contributions_from_dir, feff_dir, max_paths
                    ): task_id
                    for task_id, feff_dir in dirs
                }
                done = 0
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    results[task_id] = future.result()
                    done += 1
                    if progress_callback:
                        progress_callback(done, total)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                f"Parallel path parsing failed ({type(exc).__name__}: {exc}); "
                "falling back to serial reading."
            )
            return _serial()
        return results

    # Essential potential files produced by the precompute run
    # (CONTROL 1 1 1 0 0 0) that each main path-only task (CONTROL 0 0 0 1 1 1)
    # needs to reuse the potentials.
    _POTENTIAL_FILES = (
        "phase.pad",
        "pot.pad",
        "xsect.json",
        "xsph.json",
        "genfmt.json",
        "ff2x.json",
        "geom.json",
        "atoms.json",
        "pot.json",
        "global.json",
        "path.json",
        "libpotph.json",
        "POTENTIALS",
    )

    # Subset of ``_POTENTIAL_FILES`` without which a path-only run (CONTROL
    # 0 0 0 1 1 1) cannot start at all: FEFF aborts with
    # "#ERROR STOP cannot find phase.pad in rdxsph".  A site missing either of
    # these has no usable potentials, regardless of what the precompute run
    # reported.
    _REQUIRED_POTENTIAL_FILES = ("phase.pad", "pot.pad")

    def _place_potential_file(self, src: Path, dst: Path, mode: str) -> bool:
        """Place a single potential file at ``dst`` using copy/hardlink/symlink.

        Falls back to copying if linking is unsupported (e.g. cross-device
        hardlink).  Returns True on success.
        """
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "hardlink":
                try:
                    os.link(src, dst)
                    return True
                except OSError:
                    copy2(src, dst)  # cross-device or FS without hardlinks
                    return True
            elif mode == "symlink":
                os.symlink(src, dst)
                return True
            else:  # "copy"
                copy2(src, dst)
                return True
        except OSError as e:
            self.logger.error(f"Failed to place {src.name} -> {dst}: {e}")
            return False

    def _sites_with_potentials(
        self, batch: FeffBatch, candidate_sites: set[int]
    ) -> set[int]:
        """Return the subset of ``candidate_sites`` with usable potentials.

        Potentials are precomputed per absorber site and are entirely
        independent of one another, so this is checked site by site: a site is
        usable only if its precompute directory holds every file in
        ``_REQUIRED_POTENTIAL_FILES``.  This catches both sites whose FEFF
        precompute run failed and sites that reported success but whose output
        went missing afterwards.
        """
        precompute_dir = batch.output_dir / "precomputed_potentials"
        usable = set()
        for site_index in candidate_sites:
            site_dir = precompute_dir / f"site_{site_index:04d}"
            missing = [
                filename
                for filename in self._REQUIRED_POTENTIAL_FILES
                if not (site_dir / filename).exists()
            ]
            if missing:
                self.logger.warning(
                    f"Site {site_index}: precomputed potentials unusable, "
                    f"missing {', '.join(missing)} in {site_dir}"
                )
            else:
                usable.add(site_index)
        return usable

    def _distribute_potentials(
        self,
        batch: FeffBatch,
        parallel: bool = True,
        progress_callback: callable = None,
        valid_sites: set[int] | None = None,
        tasks: list[FeffTask] | None = None,
    ) -> int:
        """Distribute precomputed potential files to main task directories.

        For large trajectories this places tens of thousands of small files, so
        it is parallelized (I/O bound) and reports progress.  The placement
        mechanism is controlled by ``config.potential_link_mode``
        (``"copy"``/``"hardlink"``/``"symlink"``); linking is far faster and
        uses no extra disk because the potentials are read-only during
        path-only FEFF runs.

        Args:
            batch: Batch whose tasks should receive the potentials.
            parallel: Whether to place files using a thread pool.
            progress_callback: Optional callback ``(completed, total)``.
            valid_sites: If given, only tasks whose ``site_index`` is in this
                set are provisioned.  Potentials are per-site and independent,
                so a site whose precompute run failed must not block the sites
                that succeeded.  ``None`` provisions every task.
            tasks: Subset of tasks to provision.  Defaults to ``batch.tasks``.
                Streaming runs pass one chunk at a time so that only the task
                directories that currently exist on disk are populated.

        Returns the number of files successfully placed.
        """
        mode = getattr(batch.config, "potential_link_mode", "copy") or "copy"
        precompute_dir = batch.output_dir / "precomputed_potentials"
        tasks = [
            task
            for task in (batch.tasks if tasks is None else tasks)
            if valid_sites is None or task.site_index in valid_sites
        ]
        n_tasks = len(tasks)

        self.logger.info(
            f"Distributing precomputed potential files to {n_tasks} task "
            f"directories (mode={mode})"
        )

        # Build the list of (src, dst) placements up front.
        placements: list[tuple[Path, Path]] = []
        for task in tasks:
            precompute_site_dir = precompute_dir / f"site_{task.site_index:04d}"
            if not precompute_site_dir.exists():
                self.logger.warning(
                    "Precompute directory not found for "
                    f"site {task.site_index}: {precompute_site_dir}"
                )
                continue
            task.feff_dir.mkdir(parents=True, exist_ok=True)
            for filename in self._POTENTIAL_FILES:
                src = precompute_site_dir / filename
                if src.exists():
                    placements.append((src, task.feff_dir / filename))
                else:
                    self.logger.warning(f"Required file not found: {src}")

        total = len(placements)
        if progress_callback:
            progress_callback(0, total)
        if total == 0:
            return 0

        def _place(item: tuple[Path, Path]) -> bool:
            src, dst = item
            return self._place_potential_file(src, dst, mode)

        files_placed = 0
        # Copies to distinct destinations are independent; a thread pool gives a
        # large speedup for this I/O-bound work.  Linking is so fast that serial
        # placement is already cheap.
        use_parallel = parallel and mode == "copy" and total > 1
        if use_parallel:
            from concurrent.futures import ThreadPoolExecutor

            workers = self.max_workers or min(total, max(1, (os.cpu_count() or 4)), 8)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for done, ok in enumerate(executor.map(_place, placements), start=1):
                    files_placed += int(ok)
                    if progress_callback:
                        progress_callback(done, total)
        else:
            for done, item in enumerate(placements, start=1):
                files_placed += int(_place(item))
                if progress_callback:
                    progress_callback(done, total)

        self.logger.info(
            f"Placed {files_placed} of {total} potential files across "
            f"{n_tasks} main tasks (mode={mode})"
        )
        return files_placed

    @staticmethod
    def _spectrum_rejection_reason(
        k: np.ndarray | None, chi: np.ndarray | None
    ) -> str | None:
        """Describe why a chi(k) spectrum is unusable, or ``None`` if it is fine.

        FEFF can exit successfully and still leave behind a spectrum that
        cannot be averaged -- an empty or truncated ``chi.dat``, a run that
        diverged to NaN/Inf, or an all-zero spectrum from a site whose
        potentials were missing.  Folding such a spectrum into the ensemble
        average corrupts it silently, so a rejected result is reported as a
        failure rather than merely kept out of the cleanup list.

        Args:
            k: Photoelectron wavenumber grid, or ``None`` if unreadable.
            chi: chi(k) on that grid, or ``None`` if unreadable.

        Returns:
            A short human-readable reason, or ``None`` when the spectrum is
            usable.
        """
        if k is None or chi is None:
            return "no k/chi arrays could be read"
        if len(k) == 0:
            return "empty k grid"
        if len(k) != len(chi):
            return f"k/chi length mismatch ({len(k)} vs {len(chi)})"
        if not (np.isfinite(k).all() and np.isfinite(chi).all()):
            return "k or chi contains NaN/Inf"
        if np.all(chi == 0.0):
            return "chi is identically zero"
        return None

    def _retain_task_dir(
        self, task: FeffTask, feff_dir: Path, reason: str, cleanup: bool
    ) -> None:
        """Keep a task's directory for inspection, minus its per-path files.

        Unbounded growth in the failure case is exactly what streaming exists
        to avoid, so the feffNNNN.dat files go even here; ``feff.inp``,
        ``chi.dat`` and the logs -- what a post-mortem actually needs -- stay.

        Args:
            task: Task whose directory is being held back.
            feff_dir: The directory itself.
            reason: Why it is being retained, for the log line.
            cleanup: Whether per-path file cleanup is enabled at all.
        """
        self.logger.warning(
            f"{task.task_id}: {reason} Keeping {feff_dir} on disk for inspection."
        )
        if cleanup:
            cleanup_feff_output(feff_dir, keep_essential=False)

    def _cleanup_task_dir(self, task: FeffTask) -> None:
        """Remove a task's scratch directory, and its frame directory if empty.

        Args:
            task: Task whose ``feff_dir`` is no longer needed because its
                results are committed to the HDF5 archive.
        """
        feff_dir = task.feff_dir
        if feff_dir.exists():
            try:
                shutil.rmtree(feff_dir)
            except OSError as e:
                self.logger.warning(
                    f"Could not remove scratch directory {feff_dir}: {e}"
                )
            else:
                self.logger.debug(f"Removed scratch directory: {feff_dir}")

        # Remove the parent frame_XXXX directory once its last site is gone.
        frame_dir = feff_dir.parent
        if frame_dir.name.startswith("frame_"):
            try:
                frame_dir.rmdir()
            except OSError:
                pass  # Still holds other sites, or already gone.
            else:
                self.logger.debug(f"Removed empty frame directory: {frame_dir}")

    def execute_batch(
        self,
        batch: FeffBatch,
        parallel: bool = True,
        progress_callback: callable = None,
        hdf5_progress_callback: callable = None,
        path_read_progress_callback: callable = None,
        chunk_progress_callback: callable = None,
        copy_progress_callback: callable = None,
    ) -> dict[str, bool]:
        """Execute all FEFF calculations in a batch with caching support.

        Args:
            batch: FeffBatch to execute
            parallel: Whether to use parallel execution
            progress_callback: Optional callback function called with (completed, total)
            hdf5_progress_callback: Optional callback called with (completed, total)
                when writing HDF5 results.
            path_read_progress_callback: Optional callback called with
                (completed, total) as per-path feffNNNN.dat files are read from
                each calculation directory.
            chunk_progress_callback: Optional callback called with
                (chunk_index, n_chunks) at the start of each streaming chunk
                (only when the work is split into more than one chunk).
            copy_progress_callback: Optional callback called with
                (completed, total) while distributing precomputed potential
                files into each frame's task directory.

        Returns:
            Dict mapping task_id to success status

        Raises:
            RuntimeError: If potentials were precomputed for every site but
                none of them are usable, so no task could possibly succeed.

        Note:
            When precomputing potentials, tasks belonging to sites whose
            precompute run failed are removed from ``batch.tasks`` (and
            reported as failed) so that the remaining sites still run.
        """
        self.logger.info(
            f"Executing {len(batch.tasks)} FEFF calculations"
            f"{' in parallel' if parallel and len(batch.tasks) > 1 else ''}"
            f"{' with caching' if self.cache_dir else ''}"
        )

        # Tasks dropped because their site has no usable precomputed potentials.
        skipped_task_results: dict[str, bool] = {}
        # Sites with usable precomputed potentials, for deferred (lazy)
        # distribution in the streaming loop.  ``None`` when not precomputing.
        potential_sites: set[int] | None = None

        # Execute precompute tasks first if they exist
        if batch.precompute_tasks:
            self.logger.info(
                f"Executing {len(batch.precompute_tasks)} pre-compute "
                f"FEFF calculations for potentials"
            )

            # Execute precompute without caching
            precompute_input_files = [
                task.input_file for task in batch.precompute_tasks
            ]
            precompute_results = run_multi_site_feff_calculations(
                input_files=precompute_input_files,
                cleanup=batch.config.cleanup_feff_files,
                parallel=parallel,
                max_workers=self.max_workers,
                progress_callback=None,  # Don't report precompute progress separately
                require_chi=False,  # Precompute: phase.pad+pot.pad only, not chi.dat
            )

            # Resolve the outcome per absorber site.  ``precompute_results`` is
            # returned in the same order as ``input_files``, i.e. the order of
            # ``batch.precompute_tasks``.
            all_sites = {task.site_index for task in batch.precompute_tasks}
            reported_ok = {
                task.site_index
                for task, (_feff_dir, success) in zip(
                    batch.precompute_tasks, precompute_results, strict=False
                )
                if success
            }
            # Trust the files on disk, not just the reported exit status.
            usable_sites = self._sites_with_potentials(batch, reported_ok)
            unusable_sites = all_sites - usable_sites

            if unusable_sites:
                self.logger.error(
                    f"Precompute failed for {len(unusable_sites)} of "
                    f"{len(all_sites)} sites: {sorted(unusable_sites)}. "
                    "These sites cannot reuse potentials and will be skipped."
                )
            else:
                self.logger.info("All precompute tasks completed successfully")

            # Sites are independent: provision every site that succeeded, even
            # when others failed.  Doing this all-or-nothing used to leave every
            # task directory without phase.pad/pot.pad, so *all* path-only runs
            # aborted with "cannot find phase.pad in rdxsph".
            #
            # A lazy batch has no task directories yet, so distribution is
            # deferred to the streaming loop, which provisions one chunk at a
            # time immediately after materializing it.
            potential_sites = usable_sites
            if not batch.lazy:
                self._distribute_potentials(
                    batch,
                    parallel=parallel,
                    progress_callback=copy_progress_callback,
                    valid_sites=usable_sites,
                )

            if unusable_sites:
                skipped = [
                    task for task in batch.tasks if task.site_index in unusable_sites
                ]
                runnable = [
                    task
                    for task in batch.tasks
                    if task.site_index not in unusable_sites
                ]
                if batch.tasks and not runnable:
                    raise RuntimeError(
                        "Potential precomputation failed for every absorber "
                        f"site ({sorted(unusable_sites)}); no trajectory frame "
                        "can reuse potentials. Aborting before running "
                        f"{len(batch.tasks)} doomed FEFF calculations."
                    )
                self.logger.warning(
                    f"Skipping {len(skipped)} of {len(batch.tasks)} tasks "
                    f"belonging to sites {sorted(unusable_sites)}; the ensemble "
                    f"average will be built from the remaining "
                    f"{len(usable_sites)} sites."
                )
                # Mark skipped tasks as failed so the result stage ignores them.
                for task in skipped:
                    skipped_task_results[task.task_id] = False
                batch.tasks = runnable

        total_tasks = len(batch.tasks)
        completed_tasks = 0
        self.loaded_groups.clear()

        # Scratch eviction needs somewhere durable to evict *to*, so it is only
        # ever attempted when results are being committed to an HDF5 archive.
        clean_scratch_requested = (
            batch.config.clean_scratch
            and batch.config.cleanup_feff_files
            and self.hdf5_store is not None
        )

        # Initialize progress
        if progress_callback:
            progress_callback(completed_tasks, total_tasks)

        # Check cache for each task and separate cached vs. uncached
        cached_results = {}
        tasks_to_run = []

        for task in batch.tasks:
            # ── Priority 1: HDF5 cache (when --hdf5 is active) ──────────────
            if self.hdf5_store and not self.force_recalculate:
                if self.hdf5_store.has_site_result(task.frame_index, task.site_index):
                    try:
                        g = self.hdf5_store.load_site_as_group(
                            task.frame_index, task.site_index
                        )
                        # The archive stores chi(k) only, so the r-space
                        # transform has to be redone here -- the cached group is
                        # otherwise indistinguishable from a freshly computed
                        # one right up until something asks it for `r`.
                        self.loaded_groups[task.task_id] = build_exafs_group(
                            g.k, g.chi, task, batch.config.fourier_params
                        )
                        if not clean_scratch_requested:
                            task.feff_dir.mkdir(parents=True, exist_ok=True)
                            chi_file = task.feff_dir / "chi.dat"
                            # Write minimal chi.dat so ResultProcessor can read it
                            data = np.column_stack(
                                [g.k, g.chi, np.zeros_like(g.k), np.zeros_like(g.k)]
                            )
                            np.savetxt(
                                chi_file, data, header=CHI_DAT_HEADER, fmt="%.8e"
                            )
                        cached_results[task.task_id] = True
                        completed_tasks += 1
                        if progress_callback:
                            progress_callback(completed_tasks, total_tasks)
                        self.logger.debug(f"HDF5 cache hit for {task.task_id}")
                        continue  # skip pkl cache and FEFF run
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warning(
                            f"HDF5 cache read failed for {task.task_id}: {exc}"
                        )
                        # fall through to FEFF run

            # ── Priority 2: pkl cache (only when HDF5 is not active) ────────
            # The key is the hash of feff.inp, so it is computed whenever the
            # cache is in play -- including under --force-recalculate, where it
            # is needed to *refresh* the stale entry rather than read it.
            cache_key = None
            if not self.hdf5_store and self.cache_dir:
                task.materialize(batch.config)
                cache_key = self._get_feff_input_hash(task.input_file)
                cached_data = self._load_cached_result(cache_key)
                if cached_data is not None:
                    k, chi = cached_data
                    self.logger.debug(f"Pkl cache hit for {task.task_id}")
                    try:
                        self.loaded_groups[task.task_id] = build_exafs_group(
                            k, chi, task, batch.config.fourier_params
                        )

                        task.feff_dir.mkdir(parents=True, exist_ok=True)
                        chi_file = task.feff_dir / "chi.dat"
                        if chi_file.exists():
                            self.logger.warning(
                                f"chi.dat already exists for {task.task_id} "
                                f"({chi_file}), overwriting from cache"
                            )
                        mag = np.abs(chi)
                        phase = np.angle(chi)
                        chi_real = np.real(chi) if np.iscomplexobj(chi) else chi
                        data = np.column_stack([k, chi_real, mag, phase])
                        np.savetxt(chi_file, data, header=CHI_DAT_HEADER, fmt="%.8e")
                        cached_results[task.task_id] = True
                        completed_tasks += 1
                        if progress_callback:
                            progress_callback(completed_tasks, total_tasks)
                        continue
                    except (OSError, ValueError, TypeError) as e:
                        self.logger.warning(
                            f"Failed to restore pkl cache for {task.task_id}: {e}"
                        )
                        # fall through to FEFF run

            tasks_to_run.append((task, cache_key))

        # Log cache statistics
        n_cached = len(cached_results)
        n_to_run = len(tasks_to_run)
        if n_cached > 0:
            self.logger.info(
                f"Found {n_cached} cached results, running {n_to_run} new calculations"
            )

        # Execute uncached calculations in disk-bounded chunks.
        #
        # Rather than running *every* remaining task before any of its per-path
        # feffNNNN.dat files are read and deleted, we process the tasks in
        # chunks: run one chunk of FEFF calculations, parse their path files,
        # write the results to HDF5, then immediately delete the path files
        # before moving on to the next chunk.  This keeps peak on-disk usage
        # bounded to roughly one chunk of calculation directories, which is
        # essential for large trajectories whose accumulated feffNNNN.dat files
        # would otherwise fill the disk.
        if tasks_to_run:
            store_paths = bool(self.hdf5_store and self.hdf5_store.store_paths)

            chunk_size = getattr(batch.config, "stream_chunk_size", None)
            if not chunk_size or chunk_size <= 0:
                chunk_size = len(tasks_to_run)

            total_to_run = len(tasks_to_run)
            n_cached_initial = len(cached_results)
            n_feff_done = 0  # FEFF calcs completed in already-finished chunks
            n_paths_read = 0  # path directories parsed in finished chunks

            def feff_progress_callback(feff_completed: int, feff_total: int):
                """Update overall progress across all chunks."""
                if progress_callback:
                    current_completed = n_cached_initial + n_feff_done + feff_completed
                    progress_callback(current_completed, total_tasks)

            def path_progress(done: int, total: int):
                if path_read_progress_callback:
                    path_read_progress_callback(n_paths_read + done, total_to_run)

            n_chunks = (total_to_run + chunk_size - 1) // chunk_size
            if n_chunks > 1:
                self.logger.info(
                    f"Streaming {total_to_run} calculations in {n_chunks} chunks "
                    f"of up to {chunk_size} (path files parsed and deleted per "
                    f"chunk to bound disk usage)"
                )

            for chunk_idx, chunk_start in enumerate(
                range(0, total_to_run, chunk_size), start=1
            ):
                chunk = tasks_to_run[chunk_start : chunk_start + chunk_size]

                if batch.lazy:
                    # Write this chunk's directories and feff.inp files.  Note
                    # that ``batch.config`` is the config the tasks were planned
                    # with -- for a precompute run that is the path-only config
                    # (CONTROL 0 0 0 1 1 1), without which the run would
                    # recompute the potentials it is supposed to be reusing.
                    for task, _ in chunk:
                        task.materialize(batch.config)
                    if potential_sites is not None:
                        self._distribute_potentials(
                            batch,
                            parallel=parallel,
                            progress_callback=copy_progress_callback,
                            valid_sites=potential_sites,
                            tasks=[t for t, _ in chunk],
                        )

                input_files = [task.input_file for task, _ in chunk]

                if n_chunks > 1:
                    self.logger.info(
                        f"Streaming chunk {chunk_idx}/{n_chunks} "
                        f"({len(chunk)} calculations)"
                    )
                    if chunk_progress_callback:
                        chunk_progress_callback(chunk_idx, n_chunks)

                results = run_multi_site_feff_calculations(
                    input_files=input_files,
                    # Disable auto-cleanup when keep_path_files is set; we clean
                    # up manually after reading the per-path feffNNNN.dat files.
                    cleanup=batch.config.cleanup_feff_files
                    and not batch.config.keep_path_files,
                    parallel=parallel,
                    max_workers=self.max_workers,
                    progress_callback=feff_progress_callback,
                )
                n_feff_done += len(chunk)

                # Pre-parse per-path feffNNNN.dat files for this chunk in
                # parallel processes.  The larch FeffDatFile parse is the
                # GIL-bound bottleneck, so it is offloaded to a process pool;
                # the HDF5 write below stays serial in the main thread.
                parsed_paths: dict[str, list] = {}
                if store_paths:
                    from .feff_utils import get_feff_numbered_files

                    dirs_to_parse = [
                        (task.task_id, feff_dir)
                        for (task, _cache_key), (feff_dir, success) in zip(
                            chunk, results, strict=False
                        )
                        if success
                    ]
                    parsed_paths = self._parse_paths_parallel(
                        dirs_to_parse,
                        parallel,
                        progress_callback=path_progress,
                    )
                    n_paths_read += len(dirs_to_parse)

                # Process results, read chi(k), recompute path chi on the grid,
                # delete the per-path files, and stage the HDF5 write.
                pending_hdf5_writes: list[dict] = []
                validated_tasks_this_chunk: list[FeffTask] = []
                for (task, cache_key), (feff_dir, success) in zip(
                    chunk, results, strict=False
                ):
                    if not success:
                        cached_results[task.task_id] = False
                        continue

                    from .feff_utils import read_feff_output

                    # Two independent questions, deliberately kept apart:
                    #   reject -- the spectrum itself cannot be trusted, so it
                    #     must not reach the archive or the ensemble average;
                    #   retain -- something about the run warrants a look, so
                    #     its directory survives the chunk.
                    # A rejected spectrum is always retained; the converse does
                    # not hold, because a perfectly good chi(k) whose path files
                    # would not parse is still a perfectly good chi(k).
                    cleanup = batch.config.cleanup_feff_files
                    try:
                        k, chi = read_feff_output(feff_dir)
                    except (OSError, ValueError, TypeError) as e:
                        self._retain_task_dir(
                            task,
                            feff_dir,
                            f"discarding result -- could not read FEFF output ({e}).",
                            cleanup,
                        )
                        cached_results[task.task_id] = False
                        continue

                    reject_reason = self._spectrum_rejection_reason(k, chi)
                    if reject_reason is not None:
                        self._retain_task_dir(
                            task,
                            feff_dir,
                            f"discarding result -- {reject_reason}.",
                            cleanup,
                        )
                        cached_results[task.task_id] = False
                        continue

                    retain_reason: str | None = None
                    path_contributions: list | None = None
                    if store_paths:
                        from .hdf5_store import recompute_path_chi_on_grid

                        n_feff_files = len(get_feff_numbered_files(feff_dir))
                        _parsed = parsed_paths.get(task.task_id, [])
                        if n_feff_files == 0:
                            retain_reason = (
                                "store_paths=True but no feffNNNN.dat files were "
                                "found. Was FEFF run with CONTROL …1 1 1?"
                            )
                        elif not _parsed:
                            retain_reason = (
                                f"{n_feff_files} feffNNNN.dat files found but none "
                                "could be parsed. Check larch FeffDatFile "
                                "compatibility."
                            )
                        # Optionally prune negligible paths at write time to
                        # shrink the results file.
                        _min_cw = batch.config.store_min_cw_ratio
                        if _min_cw is not None:
                            _parsed = [
                                pc
                                for pc in _parsed
                                if pc.get("cw_ratio", 100.0) >= _min_cw
                            ]
                        path_contributions = [
                            recompute_path_chi_on_grid(pc, k) for pc in _parsed
                        ]

                    if cache_key:
                        self._save_to_cache(cache_key, chi, k)
                        self.logger.debug(f"Cached result for {task.task_id}")

                    self.loaded_groups[task.task_id] = build_exafs_group(
                        k, chi, task, batch.config.fourier_params
                    )

                    if self.hdf5_store:
                        pending_hdf5_writes.append(
                            {
                                "frame_index": task.frame_index,
                                "site_index": task.site_index,
                                "k": np.asarray(k),
                                "chi": np.asarray(np.real(chi)),
                                "absorber_element": task.absorber_element,
                                "success": True,
                                "path_contributions": path_contributions,
                            }
                        )

                    cached_results[task.task_id] = True

                    if retain_reason is not None:
                        # The spectrum is archived as usual; only the directory
                        # sticks around, so the path problem can be looked into
                        # without losing the calculation.
                        self._retain_task_dir(task, feff_dir, retain_reason, cleanup)
                        continue

                    # Directories that survive the chunk need their per-path
                    # files removed here; the ones about to be evicted do not,
                    # since the whole tree goes.
                    if cleanup and not clean_scratch_requested:
                        cleanup_feff_output(feff_dir)

                    validated_tasks_this_chunk.append(task)

                # Write this chunk's results to HDF5 before running the next
                # chunk.  Repeated calls append/overwrite rows incrementally.
                hdf5_write_success = True
                if self.hdf5_store and pending_hdf5_writes:
                    if hdf5_progress_callback:
                        hdf5_progress_callback(0, 1)
                    try:
                        self.hdf5_store.write_site_results_batch(pending_hdf5_writes)
                    except Exception as exc:  # noqa: BLE001
                        import traceback

                        hdf5_write_success = False
                        self.logger.error(
                            f"HDF5 batch write failed: {exc}\n" + traceback.format_exc()
                        )
                    if hdf5_progress_callback:
                        hdf5_progress_callback(1, 1)

                # Evict this chunk's scratch directories, but only once their
                # results are safely committed: the archive is now the sole
                # copy, so a failed write must leave the directories alone.
                if clean_scratch_requested:
                    if hdf5_write_success:
                        for task in validated_tasks_this_chunk:
                            self._cleanup_task_dir(task)
                    elif batch.config.cleanup_feff_files:
                        # Retained for recovery, but not at full size.
                        for task in validated_tasks_this_chunk:
                            cleanup_feff_output(task.feff_dir, keep_essential=False)

        # Skipped tasks are merged in last so they never inflate the progress
        # accounting above, which is based on ``len(batch.tasks)``.
        return {**skipped_task_results, **cached_results}


class ResultProcessor:
    """Stage C: Process FEFF results, average, and create output."""

    def __init__(self, config: FeffConfig):
        """Initialize the result processor.

        Args:
            config: FEFF configuration object
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def load_successful_results(
        self,
        batch: FeffBatch,
        task_results: dict[str, bool],
        loaded_groups: dict[str, Group] | None = None,
        hdf5_store: Any | None = None,
    ) -> dict[str, Group]:
        """Load EXAFS data from successful calculations.

        Args:
            batch: Original FeffBatch
            task_results: Results from FeffExecutor
            loaded_groups: Optional pre-loaded in-memory dictionary of Larch Groups
            hdf5_store: Optional ExafsHDF5Store to load results from

        Returns:
            Dict mapping task_id to Larch Group
        """
        from .feff_utils import read_feff_output

        groups: dict[str, Group] = {}

        for task in batch.tasks:
            if not task_results.get(task.task_id, False):
                continue

            # 1. Check in-memory pre-loaded groups
            if loaded_groups and task.task_id in loaded_groups:
                groups[task.task_id] = loaded_groups[task.task_id]
                continue

            # 2. Check HDF5 store if provided
            if hdf5_store is not None:
                try:
                    grp = hdf5_store.load_site_as_group(
                        task.frame_index, task.site_index
                    )
                    # Only chi(k) is persisted, so the r-space transform is
                    # redone here with the configured parameters.
                    groups[task.task_id] = build_exafs_group(
                        grp.k, grp.chi, task, self.config.fourier_params
                    )
                    continue
                except (KeyError, OSError, ValueError) as e:
                    self.logger.debug(
                        f"Could not load site from HDF5 for {task.task_id}: {e}"
                    )

            # 3. Disk fallback (legacy mode when scratch directories are kept)
            try:
                k, chi = read_feff_output(task.feff_dir)
                groups[task.task_id] = build_exafs_group(
                    k, chi, task, self.config.fourier_params
                )

            except (OSError, ValueError, KeyError) as e:
                self.logger.warning(f"Failed to load results for {task.task_id}: {e}")

        self.logger.info(f"Loaded {len(groups)} successful EXAFS spectra")
        return groups

    def create_frame_averages(
        self,
        groups: dict[str, Group],
        batch: FeffBatch,
        # weights: list[float] | None = None,
    ) -> dict[int, Group]:
        """Create an averaged group for each frame.

        i.e. Average over all sites within each frame.

        Args:
            groups: Dict of groups
            batch: Original FeffBatch
            # weights: Optional weights for averaging: not implemented yet

        Returns:
            Dict mapping frame_index to averaged Group
        """
        from .exafs_data import create_averaged_group

        frames = batch.get_tasks_by_frame()
        frame_averages = {}

        for frame_idx, frame_tasks in frames.items():
            # Get successful groups for this frame
            frame_groups = []
            for task in frame_tasks:
                if task.task_id in groups:
                    frame_groups.append(groups[task.task_id])

            if frame_groups:
                if len(frame_groups) == 1:
                    avg_group = frame_groups[0]
                else:
                    avg_group = create_averaged_group(
                        frame_groups, self.config.fourier_params
                    )

                # Add metadata
                avg_group.frame_idx = frame_idx
                avg_group.is_average = True
                avg_group.average_type = "frame"
                avg_group.n_components = len(frame_groups)

                frame_averages[frame_idx] = avg_group

        return frame_averages

    def create_site_averages(
        self,
        groups: dict[str, Group],
        batch: FeffBatch,
        # weights: list[float] | None = None,
    ) -> dict[int, Group]:
        """Create site-averaged groups (average over frames for each site).

        Args:
            groups: Dict of groups
            batch: Original FeffBatch
            # weights: Optional weights for averaging: not implemented yet

        Returns:
            Dict mapping site_index to averaged Group (averaged over frames)
        """
        from .exafs_data import create_averaged_group

        # Get all tasks organized by site
        sites = batch.get_tasks_by_site()
        site_averages = {}

        for site_idx, site_tasks in sites.items():
            # Get successful groups for this site across all frames
            site_groups = []
            for task in site_tasks:
                if task.task_id in groups:
                    site_groups.append(groups[task.task_id])

            if site_groups:
                if len(site_groups) == 1:
                    avg_group = site_groups[0]
                else:
                    avg_group = create_averaged_group(
                        site_groups, self.config.fourier_params
                    )

                # Add metadata
                avg_group.site_idx = site_idx
                avg_group.is_average = True
                avg_group.average_type = "site"
                avg_group.n_components = len(site_groups)

                site_averages[site_idx] = avg_group

        return site_averages

    def create_overall_average(
        self,
        all_groups: list[Group],
        # weights: list[float] | None = None,
    ) -> Group | None:
        """Create overall average across all frames.

        Args:
            all_groups: List of all groups (one for each frame_site calculation)
            # weights: Optional weights for averaging (not implemented yet)

        Returns:
            Overall averaged Group or None if no data
        """
        from .exafs_data import create_averaged_group

        if not all_groups:
            return None

        if len(all_groups) == 1:
            avg_group = all_groups[0]
        else:
            avg_group = create_averaged_group(all_groups, self.config.fourier_params)

        # Add metadata
        avg_group.is_average = True
        avg_group.average_type = "overall"
        avg_group.n_components = len(all_groups)

        return avg_group


class PipelineProcessor:
    """Unified processor using the three-stage approach."""

    def __init__(
        self,
        config: FeffConfig,
        max_workers: int | None = None,
        cache_dir: Path | None = None,
        force_recalculate: bool = False,
        hdf5_path: Path | None = None,
    ):
        """Initialize the pipeline processor.

        Args:
            config: FEFF configuration object
            max_workers: Maximum number of parallel workers
            cache_dir: Directory for caching results
            force_recalculate: Whether to force recalculation
            hdf5_path: Optional path for HDF5 output file. When set, all
                per-site chi(k) data and aggregates are written into a single
                HDF5 file instead of scattered ASCII files.  If the file
                already exists it will be appended to (new frames will be added).
        """
        self.config = config
        self.hdf5_path = hdf5_path

        # Create HDF5 store if requested (closed again before executor starts)
        self._hdf5_store = None
        if hdf5_path is not None:
            from .hdf5_store import ExafsHDF5Store

            self._hdf5_store = ExafsHDF5Store(
                hdf5_path,
                config=config,
                store_paths=config.keep_path_files,
                dedup_k=True,
                max_paths=config.max_paths,
                store_path_params=config.store_path_params,
            )

        self.input_generator = InputGenerator(config)
        self.feff_executor = FeffExecutor(
            max_workers=max_workers,
            cache_dir=cache_dir,
            force_recalculate=force_recalculate,
            hdf5_store=self._hdf5_store,
        )
        self.result_processor = ResultProcessor(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process_trajectory(
        self,
        structures: list[Atoms],
        absorber: str | int | list[int],
        output_dir: Path,
        parallel: bool = True,
        progress_callback: callable = None,
        hdf5_progress_callback: callable = None,
        path_read_progress_callback: callable = None,
        chunk_progress_callback: callable = None,
        copy_progress_callback: callable = None,
        input_progress_callback: callable = None,
        precompute_potentials: bool = False,
        precompute_potentials_structure: Atoms = None,
        # site_weights: list[float] | None = None, # Not implemented yet
        # frame_weights: list[float] | None = None, # Not implemented yet
    ) -> tuple[Group, dict[int, Group], dict[int, Group], list[Group]]:
        """Process a trajectory with the three-stage approach.

        Args:
            structures: List of ASE Atoms objects (trajectory frames)
            absorber: Absorber specification
            output_dir: Base output directory
            parallel: Whether to use parallel execution
            progress_callback: Optional callback function
            hdf5_progress_callback: Optional callback function for HDF5 path writes
            path_read_progress_callback: Optional callback for reading per-path
                feffNNNN.dat files (completed, total directories)
            chunk_progress_callback: Optional callback (chunk_index, n_chunks)
                invoked at the start of each disk-bounded streaming chunk
            copy_progress_callback: Optional callback (completed, total) while
                distributing precomputed potential files to task directories
            input_progress_callback: Optional callback (completed, total) while
                generating FEFF input files for each frame (Stage A)
            precompute_potentials: Whether to precompute potentials once and reuse
            precompute_potentials_structure: Structure to use for
                                             precompute (defaults to average)

        Returns:
            Tuple of (overall_average, frame_averages, actual_site_averages,
                     individual_groups)
        """
        # Stage A: Plan (or write) the inputs for all frames.  Planning them
        # lazily only pays off when the directories are going to be evicted
        # again chunk by chunk, which needs an HDF5 archive to evict into.
        use_lazy = (
            self.config.clean_scratch
            and self._hdf5_store is not None
            and self.config.cleanup_feff_files
        )
        batch = self.input_generator.generate_trajectory_inputs(
            structures=structures,
            absorber=absorber,
            output_dir=output_dir,
            precompute_potentials=precompute_potentials,
            precompute_potentials_structure=precompute_potentials_structure,
            input_progress_callback=input_progress_callback,
            lazy=use_lazy,
        )

        # Stage B: Execute all FEFF calculations
        task_results = self.feff_executor.execute_batch(
            batch,
            parallel=parallel,
            progress_callback=progress_callback,
            hdf5_progress_callback=hdf5_progress_callback
            if not self._hdf5_store or not self._hdf5_store.store_paths
            else None,
            path_read_progress_callback=path_read_progress_callback,
            chunk_progress_callback=chunk_progress_callback,
            copy_progress_callback=copy_progress_callback,
        )

        # Stage C: Process results
        groups = self.result_processor.load_successful_results(
            batch,
            task_results,
            loaded_groups=self.feff_executor.loaded_groups,
            hdf5_store=self._hdf5_store,
        )
        frame_averages = self.result_processor.create_frame_averages(
            groups, batch
        )  # These are frame averages
        site_averages = self.result_processor.create_site_averages(
            groups, batch
        )  # These are site averages
        overall_average = self.result_processor.create_overall_average(
            list(groups.values())
        )

        # Write aggregates to HDF5 store if one is active
        if self._hdf5_store is not None:
            try:
                if overall_average is not None:
                    self._hdf5_store.write_overall_average(
                        overall_average, n_components=len(groups)
                    )
                for frame_idx, grp in frame_averages.items():
                    self._hdf5_store.write_frame_average(
                        frame_idx,
                        grp,
                        n_components=getattr(grp, "n_components", 1),
                    )
                for site_idx, grp in site_averages.items():
                    self._hdf5_store.write_site_average(
                        site_idx,
                        grp,
                        n_components=getattr(grp, "n_components", 1),
                    )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"HDF5 aggregate write failed: {exc}")

            # Second HDF5 file: MD-averaged path contributions
            if self._hdf5_store.store_paths:
                try:
                    self._write_averaged_paths(
                        overall_average=overall_average,
                        site_averages=site_averages,
                        fourier_params=self.config.fourier_params,
                        hdf5_progress_callback=hdf5_progress_callback,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(f"Averaged paths write failed: {exc}")

        # Collect individual groups for plotting
        individual_groups = list(groups.values())

        return overall_average, frame_averages, site_averages, individual_groups

    def _write_averaged_paths(
        self,
        overall_average: Group | None,
        site_averages: dict[int, Group],
        fourier_params: dict,
        hdf5_progress_callback: callable = None,
    ) -> None:
        """Compute MD-averaged path contributions and write to a second HDF5 file."""
        from .exafs_data import aggregate_store_paths
        from .hdf5_store import AveragedPathsStore

        if self._hdf5_store is None or self.hdf5_path is None:
            return

        avg_path = self.hdf5_path.with_name(self.hdf5_path.stem + "_averaged_paths.h5")

        # Count successful spectra for correct weighting, then close the writer
        # handle *before* aggregating: the streaming aggregator opens the same
        # results file read-only from multiple worker processes, which conflicts
        # with an open ``'a'``-mode writer handle (HDF5 file locking / same-file
        # re-open).  Nothing writes to this store again after this point.
        n_total_overall = sum(1 for _ in self._hdf5_store.iter_site_results())
        self._hdf5_store.close()

        store = AveragedPathsStore(avg_path, source_h5_path=self.hdf5_path)

        # Phase 1: aggregate all stored path contributions.  This reads every
        # path row back from HDF5, clusters into physical path populations and
        # averages them, using a vectorized, process-parallel streaming
        # aggregator (see :func:`aggregate_store_paths`) that is dramatically
        # faster than per-row Python accumulation for large trajectories.
        overall_paths_all, per_site_paths = aggregate_store_paths(
            self.hdf5_path,
            fourier_params,
            max_workers=self.feff_executor.max_workers,
            progress_callback=hdf5_progress_callback,
        )

        def _add_contribution_pct(contribs, total_group, n_total) -> None:
            if not contribs or n_total <= 0:
                return
            k_ref = np.asarray(total_group.k, dtype=np.float64)
            total_norm = np.trapezoid(np.abs(total_group.chi), k_ref)
            if total_norm > 0:
                for pc in contribs.values():
                    weight = pc.n_samples / n_total
                    pc.contribution_pct = float(
                        np.trapezoid(np.abs(pc.chi * weight), pc.k) / total_norm * 100
                    )

        # Phase 2: finalize (cluster) and write the averaged paths per group.
        total_steps = len(site_averages) + (1 if overall_average is not None else 0)
        current_step = 0
        if hdf5_progress_callback:
            hdf5_progress_callback(current_step, total_steps, "writing")

        if overall_average is not None:
            overall_paths = overall_paths_all
            _add_contribution_pct(overall_paths, overall_average, n_total_overall)
            store.write_average(
                "overall_average",
                overall_average,
                overall_paths,
                n_total=n_total_overall,
            )
            current_step += 1
            if hdf5_progress_callback:
                hdf5_progress_callback(current_step, total_steps, "writing")

        for site_idx, site_group in site_averages.items():
            site_paths = per_site_paths.get(site_idx, {})
            n_total_site = (
                max((pc.n_samples for pc in site_paths.values()), default=1)
                if site_paths
                else 1
            )
            _add_contribution_pct(site_paths, site_group, n_total_site)
            store.write_average(
                f"site_averages/site_{site_idx:04d}", site_group, site_paths
            )
            current_step += 1
            if hdf5_progress_callback:
                hdf5_progress_callback(current_step, total_steps, "writing")

        store.close()
        self.logger.info(f"Wrote averaged paths store: {avg_path}")

    def get_cache_info(self) -> dict:
        """Get cache information."""
        cache_dir = self.feff_executor.cache_dir
        if not cache_dir or not cache_dir.exists():
            return {"enabled": False, "cache_dir": None, "files": 0, "size_mb": 0.0}

        cache_files = list(cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "enabled": True,
            "cache_dir": str(cache_dir),
            "files": len(cache_files),
            "size_mb": total_size / (1024 * 1024),
        }

    def clear_cache(self) -> int:
        """Clear all cache files.

        Returns:
            Number of files cleared
        """
        cache_dir = self.feff_executor.cache_dir
        if not cache_dir or not cache_dir.exists():
            return 0

        cache_files = list(cache_dir.glob("*.pkl"))
        cleared_count = 0

        for cache_file in cache_files:
            try:
                cache_file.unlink()
                cleared_count += 1
            except OSError:
                self.logger.warning(f"Failed to delete cache file: {cache_file}")

        self.logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count

    def get_diagnostics(self) -> dict:
        """Get system diagnostics."""
        import platform
        import sys

        cache_info = self.get_cache_info()
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cache_enabled": cache_info["enabled"],
            "cache_dir": cache_info["cache_dir"],
            "cache_files": cache_info["files"],
            "cache_size_mb": cache_info["size_mb"],
        }


class LoadedResults(NamedTuple):
    """Spectra and averages reconstructed from a results HDF5 archive."""

    groups: dict[str, Group]
    frame_averages: dict[int, Group]
    site_averages: dict[int, Group]
    overall_average: Group | None
    path_contributions: dict


def write_results_to_hdf5(
    hdf5_path: Path,
    config: FeffConfig,
    groups: dict[str, Group],
    frame_averages: dict[int, Group],
    site_averages: dict[int, Group],
    overall_average: Group | None,
) -> Path:
    """Write per-site spectra and averages to a results HDF5 archive.

    This is the inverse of :func:`load_results_from_hdf5`, and the two are
    meant to stay a matched pair: anything written here must be readable back.
    The archive is the durable artefact of a run -- the scratch FEFF
    directories can be deleted afterwards, and the spectra can still be
    re-analysed with different Fourier settings.

    The config is recorded in the file's metadata, so a later reload knows the
    k-range and window the spectra were produced with.

    Path contributions are not written; they are only available when the run
    kept per-path files, which the staged notebook workflow does not do.

    Args:
        hdf5_path: Destination file.  Created if absent, appended to if not.
        config: The configuration the spectra were produced with.
        groups: Per-site larch groups, keyed by task id.
        frame_averages: Frame-averaged groups, keyed by frame index.
        site_averages: Site-averaged groups, keyed by site index.
        overall_average: The ensemble average, if one could be formed.

    Returns:
        The path written to, for convenience in reporting.
    """
    from .hdf5_store import ExafsHDF5Store

    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with ExafsHDF5Store(hdf5_path, config=config, store_paths=False) as store:
        store.write_site_results_batch(
            [
                {
                    "frame_index": group.frame_idx,
                    "site_index": group.site_idx,
                    "k": np.asarray(group.k),
                    "chi": np.asarray(group.chi),
                    "absorber_element": getattr(group, "absorber_element", ""),
                    "success": True,
                    "path_contributions": None,
                }
                for group in groups.values()
            ]
        )
        if overall_average is not None:
            store.write_overall_average(overall_average, n_components=len(groups))
        for frame_idx, group in frame_averages.items():
            store.write_frame_average(
                frame_idx, group, n_components=getattr(group, "n_components", 1)
            )
        for site_idx, group in site_averages.items():
            store.write_site_average(
                site_idx, group, n_components=getattr(group, "n_components", 1)
            )
    return hdf5_path


#: FeffConfig fields that control the Fourier transform, and nothing else.
#: These are the only settings that can be changed when re-analysing a finished
#: run, since chi(k) is already fixed by the FEFF calculations.
_FOURIER_FIELDS = (
    "kmin",
    "kmax",
    "kweight",
    "dk",
    "dk2",
    "window",
    "with_phase",
    "rmax_out",
    "nfft",
    "kstep",
)


def fourier_config_from_hdf5(
    hdf5_path: Path, base: FeffConfig | None = None
) -> FeffConfig:
    """Return ``base`` with the FT parameters recorded in a results archive.

    Every run stores the config it used, so the archive knows the k-range and
    window its spectra were produced with.  Re-analysing with different values
    is legitimate, but it should be a deliberate act: silently falling back to
    library defaults can quietly change the answer, since a too-low ``kmin``
    admits the non-physical low-k divergence of chi(k) and inflates chi(R).

    Only the Fourier fields are taken from the file.  The rest of the stored
    config describes how the FEFF calculations were run and cannot be replayed.

    Args:
        hdf5_path: A results HDF5 file written by the pipeline.
        base: Config supplying every non-Fourier field.  Defaults to
            ``FeffConfig()``.

    Returns:
        A new config.  If the file records no usable Fourier settings, ``base``
        is returned unchanged.
    """
    from .hdf5_store import ExafsHDF5Store

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    base = base or FeffConfig()
    with ExafsHDF5Store(hdf5_path, mode="r") as store:
        stored = store.read_metadata().get("feff_config")
    if not isinstance(stored, dict):
        return base

    overrides = {f: stored[f] for f in _FOURIER_FIELDS if f in stored}
    if "window" in overrides:
        try:
            overrides["window"] = WindowType(overrides["window"])
        except ValueError:
            del overrides["window"]
    return replace(base, **overrides) if overrides else base


def load_results_from_hdf5(
    hdf5_path: Path,
    config: FeffConfig | None = None,
    *,
    want_paths: bool = False,
    min_cw_ratio: float | None = None,
    output_dir: Path | None = None,
) -> LoadedResults:
    """Rebuild EXAFS spectra and averages from an existing results HDF5 file.

    Only chi(k) is persisted per site, so every Fourier transform is redone
    here using ``config.fourier_params``.  That is what makes it possible to
    re-analyse a finished run with different FT settings (window, kmin, kmax,
    dk, kweight) without re-running a single FEFF calculation.

    Args:
        hdf5_path: A ``results.h5`` written by the pipeline with HDF5 output
            enabled.
        config: Configuration supplying the Fourier transform parameters.
            Defaults to the parameters recorded in the file itself, so that
            re-analysis reproduces the original run unless asked otherwise.
        want_paths: Also re-aggregate the stored per-path contributions.
            Requires the original run to have kept path files.
        min_cw_ratio: Drop aggregated paths whose curved-wave amplitude ratio
            is below this threshold.  ``None`` keeps all of them.
        output_dir: Directory recorded on the synthesised batch, for
            bookkeeping only; nothing is written to it.

    Returns:
        A :class:`LoadedResults` holding the per-site groups, the frame, site
        and overall averages, and the aggregated path contributions.

    Raises:
        FileNotFoundError: If ``hdf5_path`` does not exist.
        ValueError: If the file holds no per-site spectra.
    """
    from larch.xafs import xftf

    from .hdf5_store import ExafsHDF5Store

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    if config is None:
        config = fourier_config_from_hdf5(hdf5_path)

    groups: dict[str, Group] = {}
    tasks: list[FeffTask] = []
    path_contributions: dict = {}

    with ExafsHDF5Store(hdf5_path, mode="r") as store:
        for site_result in store.iter_site_results():
            fidx = site_result.frame_index
            sidx = site_result.site_index

            group = Group()
            group.k = np.array(site_result.k)
            group.chi = np.array(site_result.chi)
            # (Re)apply the Fourier transform with the active parameters.
            xftf(group, **config.fourier_params)
            group.frame_idx = fidx
            group.site_idx = sidx
            group.absorber_element = site_result.absorber_element
            task_id = f"frame_{fidx:04d}_site_{sidx:04d}"
            group.task_id = task_id
            groups[task_id] = group
            tasks.append(
                FeffTask(
                    input_file=Path("/dev/null"),
                    site_index=sidx,
                    frame_index=fidx,
                    absorber_element=group.absorber_element,
                )
            )

        if want_paths:
            try:
                from .exafs_data import PathAggregator, filter_path_contributions

                aggregator = PathAggregator()
                n_path_records = 0
                for (
                    path_key,
                    info,
                    frame_idx,
                    site_idx,
                ) in store.iter_path_contributions():
                    info = dict(info)
                    info["frame_index"] = frame_idx
                    info["site_index"] = site_idx
                    aggregator.add({path_key: info})
                    n_path_records += 1
                if n_path_records > 0:
                    path_contributions = aggregator.finalize(config.fourier_params)
                    if min_cw_ratio is not None:
                        path_contributions = filter_path_contributions(
                            path_contributions, min_cw_ratio=min_cw_ratio
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not load path contributions: {exc}")

    if not groups:
        raise ValueError(
            f"No per-site data found in {hdf5_path}. "
            "Was the pipeline run with HDF5 output enabled?"
        )

    batch = FeffBatch(
        tasks=tasks, output_dir=output_dir or hdf5_path.parent, config=config
    )
    processor = ResultProcessor(config)

    # Recomputed from the per-site groups rather than read back from the
    # store's own /aggregates. Both go through average_chi_arrays, and the
    # groups are already in memory, so the stored copy saves one nanmean and
    # costs the guarantee that what is returned as `groups` is what the
    # averages were taken over. There is no staleness marker to check it
    # against.
    return LoadedResults(
        groups=groups,
        frame_averages=processor.create_frame_averages(groups, batch),
        site_averages=processor.create_site_averages(groups, batch),
        overall_average=processor.create_overall_average(list(groups.values())),
        path_contributions=path_contributions,
    )


def average_structure(structures: list[Atoms]) -> Atoms:
    """Compute the average atomic positions across an ASE trajectory.

    This function averages the atomic positions of multiple ASE Atoms objects,
    correctly handling periodic boundary conditions.

    It checks that all structures have the same atom types, number of atoms,
    cell, and PBC settings.

    It assumes that there are no large diffusive motions that
    would cause atoms to jump > half the cell length from the initial frame.

    Args:
        structures : list[ase.Atoms]
            List of ASE Atoms objects (e.g. frames from a trajectory).
            All structures must have the same atom types, number of atoms,
            cell, and PBC.

    Returns:
        avg_structure : ase.Atoms
            A new ASE Atoms object containing the averaged structure.
    """
    if not structures:
        raise ValueError("No structures provided")

    ref_structure = structures[0]
    n_atoms = len(ref_structure)
    ref_symbols = ref_structure.get_chemical_symbols()
    pbc = ref_structure.get_pbc()
    cell = ref_structure.get_cell()

    # Consistency checks
    for i, s in enumerate(structures):
        if len(s) != n_atoms:
            raise ValueError(f"Structure {i} has {len(s)} atoms, expected {n_atoms}")
        if s.get_chemical_symbols() != ref_symbols:
            raise ValueError(f"Structure {i} has different atom types")
        if not np.allclose(s.get_cell(), cell, atol=1e-10):
            raise ValueError(f"Structure {i} has a different cell")
        if not np.array_equal(s.get_pbc(), pbc):
            raise ValueError(f"Structure {i} has different PBC")

    # Copy reference as output container
    avg_structure = ref_structure.copy()

    # Convert all positions to fractional coordinates
    frac_coords = np.array([s.get_scaled_positions() for s in structures])

    # Unwrap fractional coordinates along trajectory to remove jumps
    unwrapped_frac = np.zeros_like(frac_coords)
    for atom_idx in range(n_atoms):
        for coord_idx in range(3):
            if pbc[coord_idx]:  # Only unwrap in periodic directions
                unwrapped_frac[:, atom_idx, coord_idx] = np.unwrap(
                    frac_coords[:, atom_idx, coord_idx], period=1.0
                )
            else:
                unwrapped_frac[:, atom_idx, coord_idx] = frac_coords[
                    :, atom_idx, coord_idx
                ]

    # Average in fractional coordinates
    avg_frac = np.mean(unwrapped_frac, axis=0)

    # Convert back to Cartesian coordinates using modern ASE Cell API
    avg_positions = cell.cartesian_positions(avg_frac)

    # Wrap positions back into unit cell
    avg_positions = wrap_positions(avg_positions, cell, pbc=pbc)

    avg_structure.set_positions(avg_positions)
    return avg_structure
