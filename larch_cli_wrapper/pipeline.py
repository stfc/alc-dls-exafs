"""Refactored three-stage EXAFS processing architecture.

This module implements a clean separation of concerns:
1. Input generation - Create all FEFF input files
2. FEFF execution - Run calculations in parallel
3. Result processing - Load, average, and plot results
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from larch import Group

from .feff_utils import FeffConfig

logger = logging.getLogger(__name__)

__all__ = [
    "FeffTask",
    "FeffBatch",
    "InputGenerator",
    "FeffExecutor",
    "ResultProcessor",
    "PipelineProcessor",
]


@dataclass
class FeffTask:
    """Represents a single FEFF calculation task."""

    input_file: Path
    site_index: int
    frame_index: int = 0
    absorber_element: str = ""

    @property
    def feff_dir(self) -> Path:
        """Directory containing the FEFF input file."""
        return self.input_file.parent

    @property
    def task_id(self) -> str:
        """Unique identifier for this task."""
        return f"frame_{self.frame_index:04d}_site_{self.site_index:04d}"


@dataclass
class FeffBatch:
    """Collection of FEFF tasks to be executed."""

    tasks: list[FeffTask]
    output_dir: Path
    config: FeffConfig

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
        from .feff_utils import generate_multi_site_feff_inputs, normalize_absorbers

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
    ) -> FeffBatch:
        """Generate inputs for trajectory with multiple frames.

        Args:
            structures: List of ASE Atoms objects (trajectory frames)
            absorber: Absorber specification
            output_dir: Base output directory

        Returns:
            FeffBatch containing all tasks for all frames
        """
        all_tasks = []

        for frame_idx, structure in enumerate(structures):
            frame_dir = output_dir / f"frame_{frame_idx:04d}"
            frame_batch = self.generate_single_site_inputs(
                structure=structure,
                absorber=absorber,
                output_dir=frame_dir,
                frame_index=frame_idx,
            )
            all_tasks.extend(frame_batch.tasks)

        return FeffBatch(tasks=all_tasks, output_dir=output_dir, config=self.config)


class FeffExecutor:
    """Stage B: Execute FEFF calculations in parallel with caching support."""

    def __init__(
        self,
        max_workers: int | None = None,
        cache_dir: Path | None = None,
        force_recalculate: bool = False,
    ):
        """Initialize the FEFF executor.

        Args:
            max_workers: Maximum number of parallel workers
            cache_dir: Directory for caching results
            force_recalculate: Whether to force recalculation
        """
        self.max_workers = max_workers
        self.cache_dir = cache_dir
        self.force_recalculate = force_recalculate
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

    def execute_batch(
        self,
        batch: FeffBatch,
        parallel: bool = True,
        progress_callback: callable = None,
    ) -> dict[str, bool]:
        """Execute all FEFF calculations in a batch with caching support.

        Args:
            batch: FeffBatch to execute
            parallel: Whether to use parallel execution
            progress_callback: Optional callback function called with (completed, total)

        Returns:
            Dict mapping task_id to success status
        """
        self.logger.info(
            f"Executing {len(batch.tasks)} FEFF calculations"
            f"{' in parallel' if parallel and len(batch.tasks) > 1 else ''}"
            f"{' with caching' if self.cache_dir else ''}"
        )

        total_tasks = len(batch.tasks)
        completed_tasks = 0

        # Initialize progress
        if progress_callback:
            progress_callback(completed_tasks, total_tasks)

        # Check cache for each task and separate cached vs. uncached
        cached_results = {}
        tasks_to_run = []

        for task in batch.tasks:
            cache_key = self._get_feff_input_hash(task.input_file)
            cached_data = self._load_cached_result(cache_key)

            if cached_data is not None:
                chi, k = cached_data
                self.logger.debug(f"Cache hit for {task.task_id}")

                # Write cached data directly to chi.dat using FEFF format
                try:
                    # Ensure output directory exists
                    task.feff_dir.mkdir(parents=True, exist_ok=True)
                    chi_file = task.feff_dir / "chi.dat"

                    # If it already exists, raise a warning
                    if chi_file.exists():
                        self.logger.warning(
                            f"chi.dat already exists for {task.task_id} ({chi_file}), "
                            f"overwriting from cache"
                        )

                    # Create FEFF-style format with k, chi, mag, phase columns
                    mag = np.abs(chi)
                    phase = np.angle(chi)

                    # Create data array in FEFF format: k, chi (real), mag, phase
                    chi_real = np.real(chi) if np.iscomplexobj(chi) else chi
                    data = np.column_stack([k, chi_real, mag, phase])

                    # Write with FEFF-style header
                    header = "#       k          chi          mag           phase @#"
                    np.savetxt(chi_file, data, header=header, fmt="%.8e")

                    cached_results[task.task_id] = True
                    completed_tasks += 1

                    # Report progress for cached result
                    if progress_callback:
                        progress_callback(completed_tasks, total_tasks)

                except (OSError, ValueError, TypeError) as e:
                    self.logger.warning(
                        f"Failed to write cached result for {task.task_id}: {e}"
                    )
                    # If writing cached result fails, add to tasks_to_run
                    tasks_to_run.append((task, cache_key))
            else:
                tasks_to_run.append((task, cache_key))

        # Log cache statistics
        n_cached = len(cached_results)
        n_to_run = len(tasks_to_run)
        if n_cached > 0:
            self.logger.info(
                f"Found {n_cached} cached results, running {n_to_run} new calculations"
            )

        # Execute uncached calculations
        if tasks_to_run:
            from .feff_utils import run_multi_site_feff_calculations

            input_files = [task.input_file for task, _ in tasks_to_run]

            # Create wrapper callback for FEFF calculations
            def feff_progress_callback(feff_completed: int, feff_total: int):
                """Update overall progress based on FEFF calculation progress."""
                nonlocal completed_tasks
                if progress_callback:
                    # Update completed_tasks to current cached + completed FEFF
                    # calculations
                    current_completed = len(cached_results) + feff_completed
                    progress_callback(current_completed, total_tasks)

            results = run_multi_site_feff_calculations(
                input_files=input_files,
                cleanup=batch.config.cleanup_feff_files,
                parallel=parallel,
                max_workers=self.max_workers,
                progress_callback=feff_progress_callback,
                force_recalculate=self.force_recalculate,
            )

            # Update completed_tasks to final count (will be used for final processing)
            completed_tasks = len(cached_results) + len(tasks_to_run)

            # Process results and update cache
            for (task, cache_key), (feff_dir, success) in zip(
                tasks_to_run, results, strict=False
            ):
                if success:
                    # Save successful result to cache
                    try:
                        from .feff_utils import read_feff_output

                        chi, k = read_feff_output(feff_dir)
                        self._save_to_cache(cache_key, chi, k)
                        self.logger.debug(f"Cached result for {task.task_id}")
                    except (OSError, ValueError, TypeError) as e:
                        self.logger.warning(
                            f"Failed to cache result for {task.task_id}: {e}"
                        )

                cached_results[task.task_id] = success

        return cached_results


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
    ) -> dict[str, Group]:
        """Load EXAFS data from successful calculations.

        Args:
            batch: Original FeffBatch
            task_results: Results from FeffExecutor

        Returns:
            Dict mapping task_id to Larch Group
        """
        from .feff_utils import read_feff_output

        groups = {}

        for task in batch.tasks:
            if task_results.get(task.task_id, False):
                try:
                    chi, k = read_feff_output(task.feff_dir)

                    # Create larch group
                    from larch import Group
                    from larch.xafs import xftf

                    group = Group()
                    group.k = k
                    group.chi = chi

                    # Apply Fourier transform
                    xftf(k, chi, group=group, kweight=self.config.kweight)

                    # Add metadata
                    group.site_idx = task.site_index
                    group.frame_idx = task.frame_index
                    group.absorber_element = task.absorber_element
                    group.task_id = task.task_id

                    groups[task.task_id] = group

                except (OSError, ValueError, KeyError) as e:
                    self.logger.warning(
                        f"Failed to load results for {task.task_id}: {e}"
                    )

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
    ):
        """Initialize the pipeline processor.

        Args:
            config: FEFF configuration object
            max_workers: Maximum number of parallel workers
            cache_dir: Directory for caching results
            force_recalculate: Whether to force recalculation
        """
        self.config = config
        self.input_generator = InputGenerator(config)
        self.feff_executor = FeffExecutor(
            max_workers=max_workers,
            cache_dir=cache_dir,
            force_recalculate=force_recalculate,
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
        # site_weights: list[float] | None = None, # Not implemented yet
        # frame_weights: list[float] | None = None, # Not implemented yet
    ) -> tuple[Group, dict[int, Group], dict[int, Group], list[Group]]:
        """Process a trajectory with the three-stage approach.

        Returns:
            Tuple of (overall_average, frame_averages, actual_site_averages,
                     individual_groups)
        """
        # Stage A: Generate inputs for all frames
        batch = self.input_generator.generate_trajectory_inputs(
            structures=structures,
            absorber=absorber,
            output_dir=output_dir,
        )

        # Stage B: Execute all FEFF calculations
        task_results = self.feff_executor.execute_batch(
            batch, parallel=parallel, progress_callback=progress_callback
        )

        # Stage C: Process results
        groups = self.result_processor.load_successful_results(batch, task_results)
        frame_averages = self.result_processor.create_frame_averages(
            groups, batch
        )  # These are frame averages
        site_averages = self.result_processor.create_site_averages(
            groups, batch
        )  # These are site averages
        overall_average = self.result_processor.create_overall_average(
            list(groups.values())
        )

        # Collect individual groups for plotting
        individual_groups = list(groups.values())

        return overall_average, frame_averages, site_averages, individual_groups

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
