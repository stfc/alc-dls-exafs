"""Streamlined Larch Wrapper - Enhanced EXAFS processing with caching.

Enhanced EXAFS processing with memory management, and cross-platform robustness.
"""

import gc
import logging
import multiprocessing as mp
import os
import sys
import threading
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from larch import Group
from larch.xafs import xftf

from .cache_utils import get_cache_key, load_from_cache, save_to_cache
from .feff_utils import (
    PRESETS,
    FeffConfig,
    generate_feff_input,
    read_feff_output,
    run_feff_calculation,
)

# Required dependencies
try:
    from ase import Atoms
    from ase.io import read as ase_read
except ImportError:
    raise ImportError("ASE is required. Install with: pip install ase") from None


# ================== EXCEPTIONS ==================
class EXAFSProcessingError(Exception):
    """Base exception for EXAFS processing errors."""

    pass


class FEFFCalculationError(EXAFSProcessingError):
    """FEFF calculation failed."""

    pass


class StructureValidationError(EXAFSProcessingError):
    """Invalid structure data."""

    pass


# ================== ENUMS ==================
class ProcessingMode(Enum):
    """Enumeration of processing modes for EXAFS data."""

    SINGLE_FRAME = "single_frame"
    TRAJECTORY = "trajectory"


# ================== PROGRESS REPORTING ==================
class ProgressReporter(Protocol):
    """Protocol for progress reporting during processing operations."""

    def update(self, current: int, total: int, description: str) -> None:
        """Update progress with current status."""
        ...

    def set_description(self, description: str) -> None:
        """Set the current operation description."""
        ...

    def close(self) -> None:
        """Close and clean up the progress reporter."""
        ...


class TQDMReporter:
    """Progress reporter using tqdm for console progress bars."""

    def __init__(self, total_frames: int, initial_description: str = "Starting"):
        """Initialize the TQDM progress reporter."""
        self.total = total_frames
        self.current = 0
        self.desc = initial_description
        self.pbar = None
        try:
            from tqdm import tqdm

            self.pbar = tqdm(total=total_frames, desc=initial_description)
        except ImportError:
            # Fallback to simple terminal output
            self.pbar = None
            print(f"Starting: {initial_description} [0/{total_frames}]", flush=True)

    def update(self, current: int, total: int, description: str) -> None:
        """Update the progress bar with current status."""
        if self.pbar:
            self.pbar.n = current
            self.pbar.desc = description
            self.pbar.refresh()
        else:
            percent = (current / total) * 100
            print(f"\r[{percent:6.2f}%] {description}", end="", flush=True)
            if current == total:
                print()

    def set_description(self, description: str) -> None:
        """Set the current operation description."""
        if self.pbar:
            self.pbar.set_description(description)
        else:
            self.desc = description

    def close(self) -> None:
        """Close and clean up the progress bar."""
        if self.pbar:
            self.pbar.close()


class CallbackReporter:
    """Reporter that bridges to CLI progress callbacks."""

    def __init__(self, callback: Callable[[int, int, str], None]):
        """Initialize the callback reporter."""
        self.callback = callback

    def update(self, current: int, total: int, description: str) -> None:
        """Update progress via callback function."""
        self.callback(current, total, description)

    def set_description(self, description: str) -> None:
        """Set description (not used in callback implementation)."""
        pass  # Not used in current implementation

    def close(self) -> None:
        """Close callback reporter (nothing to clean up)."""
        pass  # Nothing to clean up


# ================== PROCESSING RESULT ==================
@dataclass
class FrameProcessingResult:
    """Result of processing a single frame with error information."""

    chi: np.ndarray | None = None
    k: np.ndarray | None = None
    error: str | None = None
    frame_idx: int | None = None


@dataclass
class ProcessingResult:
    """Container for EXAFS processing results and metadata.

    Stores the processed EXAFS data along with plotting information,
    processing mode, and caching statistics.
    """

    exafs_group: Group
    plot_paths: dict[str, Path]  # e.g., {"pdf": ..., "svg": ..., "png": ...}
    processing_mode: ProcessingMode
    nframes: int = 1
    individual_frame_groups: list[Group] | None = None
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def k(self) -> np.ndarray:
        """K-space values from the EXAFS data."""
        return self.exafs_group.k

    @property
    def chi(self) -> np.ndarray:
        """Chi(k) values from the EXAFS data."""
        return self.exafs_group.chi

    @property
    def r(self) -> np.ndarray:
        """R-space values from the Fourier transform."""
        return self.exafs_group.r

    @property
    def chir_mag(self) -> np.ndarray:
        """Magnitude of chi(R) from the Fourier transform."""
        return self.exafs_group.chir_mag

    def save(self, output_dir: Path) -> Path:
        """Save EXAFS data to text files.

        Args:
            output_dir: Directory to save the data files

        Returns:
            Path to the output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_dir / "exafs_data.dat",
            np.column_stack((self.k, self.chi)),
            header="k chi",
        )
        np.savetxt(
            output_dir / "ft_data.dat",
            np.column_stack((self.r, self.chir_mag)),
            header="R chir_mag",
        )
        return output_dir


# ================== PARALLEL PROCESSING ==================
class ParallelProcessor:
    """Enhanced parallel processing manager with robust error handling."""

    # Class-level constants for timeout configuration
    DEFAULT_TASK_TIMEOUT = 180  # seconds - timeout for individual FEFF calculations
    DEFAULT_SHUTDOWN_TIMEOUT = 15  # seconds - timeout for pool shutdown

    def __init__(self, n_workers: int | None = None, timeout: float | None = None):
        """Initialize the parallel processor.

        Args:
            n_workers: Number of worker processes. If None, uses optimal count.
            timeout: Timeout in seconds for individual tasks. None = uses default.
        """
        self.n_workers = self._determine_optimal_workers(n_workers)
        self.timeout = timeout or self.DEFAULT_TASK_TIMEOUT
        self.logger = logging.getLogger(__name__)

    def _determine_optimal_workers(self, n_workers: int | None) -> int:
        """Determine optimal number of workers based on system and constraints."""
        if n_workers is not None:
            return max(1, n_workers)

        # Get CPU count
        cpu_count = mp.cpu_count()

        # Platform-specific optimizations
        if sys.platform.startswith("win"):
            # Windows: more conservative due to process overhead
            optimal = max(1, cpu_count // 2)
        elif sys.platform.startswith("darwin"):
            # macOS: moderate scaling
            optimal = max(1, min(cpu_count - 1, 8))
        else:
            # Linux/Unix: more aggressive scaling
            optimal = max(1, min(cpu_count, 12))

        # Memory consideration: limit based on available memory
        try:
            import psutil

            # Rough estimate: 500MB per worker for FEFF calculations
            available_gb = psutil.virtual_memory().available / (1024**3)
            memory_limit = max(1, int(available_gb // 0.5))
            optimal = min(optimal, memory_limit)
        except ImportError:
            pass  # psutil not available, use CPU-based estimate

        return optimal

    @contextmanager
    def process_pool(self):
        """Enhanced context manager for multiprocessing pool with better cleanup."""
        pool = None
        try:
            # Use spawn method for better isolation (especially important on
            # macOS/Windows)
            ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else mp

            pool = ctx.Pool(
                self.n_workers,
                initializer=self._worker_init,
                initargs=(logging.getLogger().level,),
                # Restart workers after 10 tasks to prevent memory leaks
                maxtasksperchild=10,
            )

            self.logger.info(f"Started parallel pool with {self.n_workers} workers")
            yield pool

        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
            raise
        finally:
            if pool:
                try:
                    # Graceful shutdown with timeout
                    pool.close()

                    # Implement timeout for pool.join() using threading
                    join_thread = threading.Thread(target=pool.join)
                    join_thread.start()
                    join_thread.join(timeout=self.DEFAULT_SHUTDOWN_TIMEOUT)

                    if join_thread.is_alive():
                        self.logger.warning(
                            f"Pool shutdown timed out after "
                            f"{self.DEFAULT_SHUTDOWN_TIMEOUT}s, forcing termination"
                        )
                        pool.terminate()
                        join_thread.join(timeout=5)  # Give terminate a few seconds
                        if join_thread.is_alive():
                            self.logger.error(
                                "Pool termination also timed out, processes may "
                                "still be running"
                            )
                    else:
                        self.logger.info("Parallel pool closed successfully")

                except (OSError, RuntimeError) as e:
                    self.logger.warning(f"Error during pool shutdown: {e}")
                    # Force termination if graceful shutdown fails
                    try:
                        pool.terminate()
                        # Use threading for terminate join as well
                        join_thread = threading.Thread(target=pool.join)
                        join_thread.start()
                        join_thread.join(timeout=5)
                    except (OSError, RuntimeError):
                        # Best effort cleanup - if we can't clean up properly, log
                        # it but don't crash the application
                        self.logger.warning("Best effort pool cleanup failed")

    def process_with_timeout(self, pool, func, tasks, timeout=None):
        """Process tasks with timeout and error handling."""
        timeout = timeout or self.timeout

        if timeout:
            # Use imap_unordered for better timeout handling
            try:
                results = []
                for result in pool.imap(func, tasks):
                    results.append(result)
                return results
            except mp.TimeoutError:
                self.logger.error(f"Task timed out after {timeout} seconds")
                raise
        else:
            # No timeout - use regular imap
            return list(pool.imap(func, tasks))

    @staticmethod
    def _worker_init(log_level: int) -> None:
        """Enhanced worker initialization with better resource management."""
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format="[Worker-%(process)d] [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,  # Override any existing logging config
        )

        # Platform-specific optimizations
        if sys.platform.startswith("linux"):
            try:
                # Set process priority to be nice to other processes
                os.nice(1)
            except (OSError, AttributeError):
                pass

        # Memory management: force garbage collection
        import gc

        gc.collect()

        # Set up signal handling for graceful shutdown
        try:
            import signal

            def signal_handler(signum, frame):
                logging.getLogger().info(
                    f"Worker received signal {signum}, shutting down gracefully"
                )
                sys.exit(0)

            signal.signal(signal.SIGTERM, signal_handler)
        except (ImportError, AttributeError):
            pass  # Signal handling not available on this platform


# ================== MAIN WRAPPER ==================
class LarchWrapper:
    """Main wrapper class for Larch EXAFS processing with caching and parallel."""

    def __init__(
        self,
        verbose: bool = True,
        cleanup_on_exit: bool = True,
        cache_dir: Path | None = None,
    ):
        """Initialize the LarchWrapper.

        Args:
            verbose: Enable verbose logging
            cleanup_on_exit: Clean up temporary files on exit
            cache_dir: Directory for caching. If None, no caching is used.
        """
        self.cleanup_on_exit = cleanup_on_exit
        self._temp_files: list[Path] = []
        self._temp_dirs: list[Path] = []
        self.logger = self._setup_logger(verbose)
        self.parallel_processor = ParallelProcessor()
        # Add caching capability
        self.cache_dir = (
            (cache_dir or Path.home() / ".larch_cache")
            if cache_dir is not None
            else None
        )
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

    def _setup_logger(self, verbose: bool) -> logging.Logger:
        logger = logging.getLogger("larch_wrapper")
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        logger.handlers.clear()
        if verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def __enter__(self) -> "LarchWrapper":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        if self.cleanup_on_exit:
            self.cleanup_temp_files()
            self.cleanup_temp_dirs()
        if exc_type is not None:
            self.logger.error(f"Error during processing: {exc_val}")

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files[:]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                self._temp_files.remove(temp_file)
            except (OSError, PermissionError) as e:
                self.logger.warning(f"Could not remove temp file {temp_file}: {e}")
        gc.collect()

    def cleanup_temp_dirs(self) -> None:
        """Clean up temporary directories."""
        import shutil

        for temp_dir in self._temp_dirs[:]:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                self._temp_dirs.remove(temp_dir)
            except (OSError, PermissionError) as e:
                self.logger.warning(f"Could not remove temp dir {temp_dir}: {e}")

    # ================== CACHING METHODS ==================
    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.cache_dir or not self.cache_dir.exists():
            self.logger.info("No cache directory to clear")
            return
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            self.logger.info(f"Cleared {len(cache_files)} cache files")
        except (OSError, PermissionError) as e:
            self.logger.warning(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the cache."""
        if not self.cache_dir:
            return {"enabled": False}

        if not self.cache_dir.exists():
            return {
                "enabled": True,
                "cache_dir": str(self.cache_dir),
                "files": 0,
                "size_mb": 0,
            }

        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "files": len(cache_files),
            "size_mb": round(total_size / (1024 * 1024), 2),
        }

    def run_feff(self, feff_dir: Path, config: FeffConfig | None = None) -> bool:
        """Run FEFF calculation - now uses utility function."""
        cleanup = config.cleanup_feff_files if config else True
        result = run_feff_calculation(
            feff_dir, verbose=self.logger.level <= logging.INFO, cleanup=cleanup
        )
        if result:
            self.logger.info("FEFF calculation completed")
        else:
            self.logger.error("FEFF calculation failed")

        return result

    def process_feff_output(self, feff_dir: Path, config: FeffConfig) -> Group:
        """Process FEFF output - now uses utility function for reading."""
        chi, k = read_feff_output(feff_dir)

        # Create Larch group and apply Fourier transform
        g = Group()
        g.k = k
        g.chi = chi
        xftf(g, **config.fourier_params)
        return g

    def process_trajectory_feff_outputs(
        self,
        frame_dirs: list[Path],
        output_dir: Path,
        config: FeffConfig,
        show_plot: bool = False,
        plot_style: str = "publication",
        plot_individual_frames: bool = False,
        chi_weighting: str = "chi",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingResult:
        """Process trajectory FEFF outputs from frame_* subdirectories.

        Args:
            frame_dirs: List of sorted frame_* directories
            output_dir: Directory for output files
            config: FEFF configuration parameters
            show_plot: Display plots after processing
            plot_style: Plot style ('publication' or 'presentation')
            plot_individual_frames: Create plots for each frame
            chi_weighting: Chi weighting for plots ('chi', 'k2chi', 'k3chi')
            progress_callback: Callback function for progress updates

        Returns:
            ProcessingResult containing averaged EXAFS data and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chi_list = []
        individual_groups = []
        k_ref = None
        total_frames = len(frame_dirs)

        # Initialize reporter
        reporter: ProgressReporter
        if progress_callback is None:
            reporter = TQDMReporter(total_frames, "Processing trajectory FEFF outputs")
        else:
            reporter = CallbackReporter(progress_callback)

        try:
            for i, frame_dir in enumerate(frame_dirs):
                # Update progress
                reporter.update(i, total_frames, f"Processing {frame_dir.name}")

                chi_file = frame_dir / "chi.dat"
                if not chi_file.exists():
                    self.logger.warning(f"No chi.dat found in {frame_dir}, skipping")
                    continue

                try:
                    # Read FEFF output for this frame
                    chi, k = read_feff_output(frame_dir)

                    # Set reference k-grid from first successful frame
                    if k_ref is None:
                        k_ref = k.copy()

                    # Interpolate to common k-grid if needed
                    if not np.array_equal(k, k_ref):
                        chi_interp = np.interp(k_ref, k, chi, left=0, right=0)
                    else:
                        chi_interp = chi

                    chi_list.append(chi_interp)

                    # Create frame group for individual plotting
                    frame_group = Group()
                    frame_group.k = k_ref.copy()
                    frame_group.chi = chi_interp
                    xftf(frame_group, **config.fourier_params)
                    individual_groups.append(frame_group)

                except (OSError, ValueError, RuntimeError, KeyError) as e:
                    # OSError: file I/O, ValueError: data processing errors,
                    # RuntimeError: calculation failures, KeyError: missing data keys
                    self.logger.error(f"Error processing {frame_dir}: {e}")
                    continue

            # Finalize progress
            reporter.update(total_frames, total_frames, "Completing processing")

            if not chi_list:
                raise FEFFCalculationError("No valid FEFF outputs found in trajectory")

            # Average the chi data
            chi_avg = np.mean(chi_list, axis=0)

            # Create averaged EXAFS group
            avg_group = Group()
            avg_group.k = k_ref
            avg_group.chi = chi_avg
            xftf(avg_group, **config.fourier_params)

            # Generate plots
            plot_paths = self.plot_results(
                avg_group,
                output_dir,
                "EXAFS_FT_trajectory",
                show_plot=show_plot,
                plot_style=plot_style,
                absorber="auto",  # Could be extracted from FEFF input if needed
                edge=config.edge,
                individual_frames=individual_groups if plot_individual_frames else None,
                chi_weighting=chi_weighting,
            )

            return ProcessingResult(
                exafs_group=avg_group,
                plot_paths=plot_paths,
                processing_mode=ProcessingMode.TRAJECTORY,
                nframes=len(chi_list),
                individual_frame_groups=individual_groups,
            )

        finally:
            reporter.close()

    def plot_results(
        self,
        exafs_group: Group,
        output_dir: Path,
        filename_base: str = "EXAFS_FT",
        show_plot: bool = False,
        plot_style: str = "publication",
        absorber: str = "X",
        edge: str = "K",
        individual_frames: list[Group] | None = None,
        show_individual_legend: bool = True,
        max_individual_frames: int = 100,
        chi_weighting: str = "chi",
    ) -> dict[str, Path]:
        """Generate plots for EXAFS results with marimo-style formatting.

        Args:
            exafs_group: Larch Group containing processed EXAFS data
            output_dir: Directory to save plots
            filename_base: Base filename for plot files
            show_plot: Whether to display plots interactively
            plot_style: Plot style configuration
            absorber: Absorbing atom symbol for annotation
            edge: Absorption edge for annotation
            individual_frames: Optional list of individual trajectory frames
            show_individual_legend: Whether to show legend for individual frames
            max_individual_frames: Maximum number of individual frames to plot
            chi_weighting: Chi weighting ('chi', 'k2chi', 'k3chi')

        Returns:
            Dictionary mapping format names to output file paths
        """
        from pathlib import Path as PathlibPath

        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get the style file path
        styles_dir = PathlibPath(__file__).parent / "styles"

        # Style configurations with external style files
        styles: dict[str, dict[str, Any]] = {
            "publication": {
                "style_file": styles_dir / "exafs_publication.mplstyle",
                "figsize": (12, 5),
            },
            "presentation": {
                "style_file": styles_dir / "exafs_presentation.mplstyle",
                "figsize": (10, 4),
            },
        }

        style_config = styles.get(plot_style, styles["publication"])
        style_file_path = style_config["style_file"]

        # Check if style file exists and warn if not
        if not style_file_path.exists():
            self.logger.warning(
                f"Style file {style_file_path} not found. "
                "Using matplotlib default style."
            )

        # Store original matplotlib settings
        original_params = plt.rcParams.copy()

        try:
            # Apply style if file exists
            if style_file_path.exists():
                plt.style.use(str(style_file_path))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style_config["figsize"])

            # Compute weighted chi based on chi_weighting
            if chi_weighting == "k2chi":
                chi_weighted = exafs_group.chi * exafs_group.k**2
                ylabel = r"$k^{2}\chi(k)$"
                title = r"EXAFS $k^{2}\chi(k)$"
            elif chi_weighting == "k3chi":
                chi_weighted = exafs_group.chi * exafs_group.k**3
                ylabel = r"$k^{3}\chi(k)$"
                title = r"EXAFS $k^{3}\chi(k)$"
            else:  # default "chi"
                chi_weighted = exafs_group.chi
                ylabel = r"$\chi(k)$"
                title = r"EXAFS $\chi(k)$"

            # === Left plot: χ(k) in k-space ===
            # Plot individual frames first (background)
            if individual_frames and show_individual_legend:
                frames_to_plot = individual_frames[:max_individual_frames]
                for i, frame in enumerate(frames_to_plot):
                    if chi_weighting == "k2chi":
                        chi_frame_weighted = frame.chi * frame.k**2
                    elif chi_weighting == "k3chi":
                        chi_frame_weighted = frame.chi * frame.k**3
                    else:
                        chi_frame_weighted = frame.chi
                    ax1.plot(
                        frame.k,
                        chi_frame_weighted,
                        color="gray",
                        alpha=0.3,
                        linewidth=1,
                        label="Individual frames" if i == 0 else "",
                        zorder=1,
                    )

            # Add standard deviation envelope if available
            if (
                hasattr(exafs_group, "chi_std")
                and exafs_group.chi_std is not None
                and individual_frames
            ):
                if chi_weighting == "k2chi":
                    chi_std_weighted = exafs_group.chi_std * exafs_group.k**2
                elif chi_weighting == "k3chi":
                    chi_std_weighted = exafs_group.chi_std * exafs_group.k**3
                else:
                    chi_std_weighted = exafs_group.chi_std
                ax1.fill_between(
                    exafs_group.k,
                    chi_weighted - chi_std_weighted,
                    chi_weighted + chi_std_weighted,
                    alpha=0.1,
                    color="black",
                    label="±1σ",
                    zorder=2,
                )
                main_label = f"{ylabel} Average ± σ"
            else:
                main_label = ylabel

            # Plot main spectrum
            ax1.plot(
                exafs_group.k,
                chi_weighted,
                color="black",
                linewidth=2.5,
                label=main_label,
                zorder=3,
            )

            ax1.set_xlabel(r"$k [\text{Å}^{-1}]$")
            ax1.set_ylabel(ylabel)
            ax1.set_title(title)

            if individual_frames or (
                hasattr(exafs_group, "chi_std") and exafs_group.chi_std is not None
            ):
                ax1.legend(loc="upper right")

            # === Right plot: |χ(R)| Fourier Transform ===
            # Plot individual frames first (background)
            if individual_frames and show_individual_legend:
                frames_to_plot = individual_frames[:max_individual_frames]
                for i, frame in enumerate(frames_to_plot):
                    ax2.plot(
                        frame.r,
                        frame.chir_mag,
                        color="gray",
                        alpha=0.3,
                        linewidth=1,
                        label="Individual frames" if i == 0 else "",
                        zorder=1,
                    )

            # Plot main spectrum
            main_ft_label = "|χ(R)| Average" if individual_frames else "|χ(R)|"
            ax2.plot(
                exafs_group.r,
                exafs_group.chir_mag,
                color="black",
                linewidth=2.5,
                label=main_ft_label,
                zorder=3,
            )

            ax2.set_xlabel("R [Å]")
            ax2.set_ylabel("|χ(R)|")
            ax2.set_title("Fourier Transform |χ(R)|")

            if individual_frames:
                ax2.legend(loc="center right")

            # Add annotation similar to marimo (position it to avoid legend overlap)
            annotation_text = f"{absorber} {edge} edge"
            ax2.text(
                0.02,
                0.98,
                annotation_text,
                transform=ax2.transAxes,
                fontsize=16,
                ha="left",
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 1,
                },
                zorder=10,
            )

            # Apply marimo-style axis formatting
            for ax in [ax1, ax2]:
                # Make all spines visible with proper thickness
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_color("black")

                # Mirror ticks on all sides (like marimo)
                ax.tick_params(
                    top=True,
                    right=True,
                    which="major",
                    length=6,
                    width=2,
                    color="black",
                )
                ax.tick_params(
                    top=True,
                    right=True,
                    which="minor",
                    length=3,
                    width=1,
                    color="black",
                )

            plt.tight_layout()

            # Save plots in multiple formats
            outputs = {}
            for fmt in ["pdf", "svg", "png"]:
                path = output_dir / f"{filename_base}.{fmt}"
                # Use DPI from matplotlib settings (set by style file)
                current_dpi = plt.rcParams.get("savefig.dpi", 300)
                fig.savefig(
                    path,
                    format=fmt,
                    dpi=current_dpi,
                    bbox_inches="tight",
                    transparent=(fmt == "svg"),
                )
                outputs[fmt] = path

            if show_plot:
                plt.show()
            else:
                plt.close(fig)

        finally:
            # Restore original matplotlib settings
            plt.rcParams.clear()
            plt.rcParams.update(original_params)

        return outputs

    def _process_single_frame(
        self, atoms: Atoms, absorber: str, output_dir: Path, config: FeffConfig
    ) -> FrameProcessingResult | None:
        """Process a single frame with caching support - now uses utility functions."""
        # Check cache first if caching is enabled
        if self.cache_dir and not config.force_recalculate:
            cache_key = get_cache_key(atoms, absorber, config)
            cached_result = load_from_cache(
                cache_key, self.cache_dir, config.force_recalculate
            )
            if cached_result is not None:
                chi, k = cached_result
                return FrameProcessingResult(chi=chi, k=k)

        try:
            # Use utility functions
            generate_feff_input(atoms, absorber, output_dir, config)

            if not run_feff_calculation(
                output_dir, verbose=self.logger.level <= logging.INFO
            ):
                return FrameProcessingResult(
                    error="FEFF calculation failed", chi=None, k=None
                )

            chi, k = read_feff_output(output_dir)

            # Save to cache if caching is enabled
            if self.cache_dir:
                cache_key = get_cache_key(atoms, absorber, config)
                save_to_cache(cache_key, chi, k, self.cache_dir)

            return FrameProcessingResult(chi=chi, k=k)

        except (OSError, RuntimeError, ValueError) as e:
            error_msg = f"Frame processing failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return FrameProcessingResult(error=error_msg, chi=None, k=None)

    def _process_structures_chunked(
        self,
        structures: list[Atoms],
        absorber: str,
        output_dir: Path,
        config: FeffConfig,
        reporter: ProgressReporter,
        plot_individual_frames: bool = False,
        chi_weighting: str = "chi",
    ) -> ProcessingResult:
        chi_list = []
        individual_groups = []
        k_ref = None  # Will store k-grid from first successful frame
        total_frames = len(structures)
        cache_hits = 0
        cache_misses = 0

        # Process frames using enhanced parallel framework
        if config.parallel and total_frames > 1:
            n_workers = config.n_workers or self.parallel_processor.n_workers
            # Platform-specific warnings and adjustments
            if sys.platform.startswith("win") and n_workers > 4:
                self.logger.warning(
                    f"Reducing workers from {n_workers} to 4 on Windows for stability"
                )
                n_workers = 4
        else:
            # Sequential processing
            n_workers = 1

        # Update parallel processor configuration
        processor = ParallelProcessor(n_workers=n_workers)  # Uses DEFAULT_TASK_TIMEOUT

        self.logger.info(
            f"Processing {total_frames} frames with {n_workers} worker"
            f"{'s' if n_workers > 1 else ''} "
            f"({'parallel' if n_workers > 1 else 'sequential'})"
        )

        try:
            with processor.process_pool() as pool:
                # Create worker tasks - include cache information
                tasks = [
                    (i, atoms, absorber, output_dir, config, False, self.cache_dir)
                    for i, atoms in enumerate(structures)
                ]

                # Process with enhanced error handling and timeout
                if n_workers == 1:
                    # Sequential processing - direct function calls for better debugging
                    results_list = []
                    for task in tasks:
                        result = self._process_frame_worker(task)
                        results_list.append(result)

                        # Update progress after each frame
                        if not result.error:
                            reporter.update(
                                len([r for r in results_list if not r.error]),
                                total_frames,
                                f"Processed frame {result.frame_idx}",
                            )
                else:
                    # Parallel processing with timeout and error handling
                    try:
                        results_iter = pool.imap(self._process_frame_worker, tasks)
                        results_list = []
                        failed_frames = 0
                        max_failures = max(
                            1, total_frames // 2
                        )  # Allow up to 50% failures

                        for _result_idx, frame_result in enumerate(results_iter):
                            results_list.append(frame_result)

                            # Track failures and check for early termination
                            if frame_result.error:
                                failed_frames += 1
                                self.logger.warning(
                                    f"Frame {frame_result.frame_idx} failed: "
                                    f"{frame_result.error}"
                                )

                                # Early termination if too many failures
                                if failed_frames > max_failures:
                                    self.logger.error(
                                        f"Too many failures ({failed_frames}/"
                                        f"{len(results_list)}), terminating processing"
                                    )
                                    raise RuntimeError(
                                        f"Processing terminated: {failed_frames} "
                                        f"frames failed out of {len(results_list)} "
                                        f"processed"
                                    )

                            # Update progress after each completed frame
                            if not frame_result.error:
                                successful_frames = len(
                                    [r for r in results_list if not r.error]
                                )
                                reporter.update(
                                    successful_frames,
                                    total_frames,
                                    f"Processed frame {frame_result.frame_idx}",
                                )

                    except (RuntimeError, ValueError, OSError) as e:
                        self.logger.error(f"Parallel processing failed: {e}")
                        # Fall back to sequential processing
                        self.logger.info("Falling back to sequential processing...")
                        results_list = []
                        for task in tasks:
                            result = self._process_frame_worker(task)
                            results_list.append(result)

                # Process results
                for frame_result in results_list:
                    if frame_result.error:
                        self.logger.error(
                            f"Frame {frame_result.frame_idx} failed: "
                            f"{frame_result.error}"
                        )
                        continue

                    # Skip frames with invalid data
                    if frame_result.k is None or frame_result.chi is None:
                        self.logger.warning(
                            f"Frame {frame_result.frame_idx} "
                            "has invalid data (k or chi is None), skipping"
                        )
                        continue

                    # Set reference k-grid from first successful frame
                    if k_ref is None:
                        k_ref = frame_result.k.copy()

                    # Interpolate to common k-grid if needed
                    if not np.array_equal(frame_result.k, k_ref):
                        chi_interp = np.interp(
                            k_ref, frame_result.k, frame_result.chi, left=0, right=0
                        )
                    else:
                        chi_interp = frame_result.chi

                    chi_list.append(chi_interp)

                    # Create frame group for plotting
                    frame_group = Group()
                    frame_group.k = k_ref.copy()
                    frame_group.chi = chi_interp
                    xftf(frame_group, **config.fourier_params)
                    individual_groups.append(frame_group)

                    # Track cache statistics
                    if hasattr(frame_result, "from_cache") and frame_result.from_cache:
                        cache_hits += 1
                    else:
                        cache_misses += 1

        except (OSError, RuntimeError, ValueError) as e:
            self.logger.error(f"Frame processing failed: {e}")
            raise RuntimeError(f"Failed to process trajectory frames: {e}") from e

        # Final validation
        if not chi_list:
            raise RuntimeError(
                "No frames were processed successfully. Cannot generate result."
            )

        if k_ref is None:
            raise RuntimeError(
                "No valid k-space data found in any frame. Cannot generate result."
            )

        # Always use k_ref (defined from first success)
        avg_chi = np.mean(np.array(chi_list), axis=0)
        result_group = Group()
        result_group.k = k_ref
        result_group.chi = avg_chi
        result_group.chi_std = np.std(np.array(chi_list), axis=0)

        # Apply FT to averaged spectrum
        xftf(result_group, **config.fourier_params)

        # Generate plots
        plot_paths = self.plot_results(
            result_group,
            output_dir,
            "trajectory_avg_EXAFS_FT",
            show_plot=False,
            plot_style="publication",
            absorber=absorber,
            edge=config.edge,
            individual_frames=individual_groups if plot_individual_frames else None,
            chi_weighting=chi_weighting,
        )

        return ProcessingResult(
            exafs_group=result_group,
            plot_paths=plot_paths,
            processing_mode=ProcessingMode.TRAJECTORY,
            nframes=len(chi_list),
            individual_frame_groups=individual_groups,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

    @staticmethod
    def _run_feff_with_timeout(frame_dir, config, timeout=300):
        """Run FEFF calculation with timeout to prevent hanging processes."""
        import threading

        result = [None]
        exception = [None]
        completed = [False]

        def run_feff():
            try:
                result[0] = run_feff_calculation(
                    frame_dir, verbose=False, cleanup=config.cleanup_feff_files
                )
                completed[0] = True
            except (RuntimeError, ValueError, OSError) as e:
                exception[0] = e
                completed[0] = True

        thread = threading.Thread(target=run_feff)
        thread.daemon = True  # Ensure thread doesn't keep process alive
        thread.start()

        # Wait for completion or timeout
        thread.join(timeout=timeout)

        if not completed[0]:
            # Thread is still running - timeout occurred
            logger = logging.getLogger(__name__)
            logger.error(f"FEFF calculation in {frame_dir} timed out after {timeout}s")
            return False

        if exception[0]:
            raise exception[0]

        return result[0] if result[0] is not None else False

    @staticmethod
    def _process_frame_worker(frame_data: tuple[Any, ...]) -> FrameProcessingResult:
        """Enhanced worker function with better resource management.

        Processes a single frame with comprehensive error reporting and resource
        cleanup.
        """
        import gc
        import time

        start_time = time.time()
        frame_idx = None

        try:
            from .cache_utils import get_cache_key, load_from_cache, save_to_cache
            from .feff_utils import (
                generate_feff_input,
                read_feff_output,
            )

            frame_idx, atoms, absorber, output_base, config, is_single, cache_dir = (
                frame_data
            )

            # Worker process logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Processing frame {frame_idx} (PID: {os.getpid()})")

            # Check cache first if caching is enabled
            from_cache = False
            if cache_dir and not config.force_recalculate:
                cache_key = get_cache_key(atoms, absorber, config)
                cached_result = load_from_cache(
                    cache_key, cache_dir, config.force_recalculate
                )
                if cached_result is not None:
                    chi, k = cached_result
                    from_cache = True
                    logger.debug(f"Frame {frame_idx} loaded from cache")
                    result = FrameProcessingResult(chi=chi, k=k, frame_idx=frame_idx)
                    result.from_cache = from_cache
                    return result

            # Setup frame directory
            if is_single:
                frame_dir = Path(output_base)
            else:
                frame_dir = Path(output_base) / f"frame_{frame_idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)

            # Use utility functions for all FEFF operations with progress logging
            logger.debug(f"Frame {frame_idx}: Generating FEFF input")
            try:
                generate_feff_input(atoms, absorber, frame_dir, config)
            except Exception as e:
                raise RuntimeError(
                    f"FEFF input generation failed for frame {frame_idx}: {e}"
                ) from e

            logger.debug(f"Frame {frame_idx}: Running FEFF calculation")
            try:
                # Run FEFF calculation with timeout protection
                feff_success = LarchWrapper._run_feff_with_timeout(
                    frame_dir,
                    config,
                    timeout=300,  # 5 minute timeout per frame
                )
                if not feff_success:
                    # Check for specific error indicators
                    log_file = frame_dir / "feff.log"
                    error_details = ""
                    if log_file.exists():
                        try:
                            log_content = log_file.read_text(
                                encoding="utf-8", errors="replace"
                            )
                            # Extract last few lines for error context
                            log_lines = log_content.split("\n")[-10:]
                            error_details = " FEFF log excerpt:\n" + "\n".join(
                                log_lines
                            )
                        except (OSError, UnicodeDecodeError) as read_error:
                            error_details = f" (could not read FEFF log: {read_error})"

                    raise RuntimeError(
                        f"FEFF calculation failed for frame {frame_idx}.{error_details}"
                    )
            except RuntimeError:
                raise  # Re-raise RuntimeError as-is
            except (OSError, UnicodeDecodeError, ValueError) as e:
                raise RuntimeError(
                    f"FEFF calculation error for frame {frame_idx}: {e}"
                ) from e

            logger.debug(f"Frame {frame_idx}: Reading FEFF output")
            try:
                chi, k = read_feff_output(frame_dir)
            except (OSError, UnicodeDecodeError, ValueError) as e:
                # Check what output files exist for debugging
                output_files = list(frame_dir.glob("*.dat"))
                file_info = (
                    f"Available .dat files: {[f.name for f in output_files]}"
                    if output_files
                    else "No .dat files found"
                )
                raise RuntimeError(
                    f"FEFF output reading failed for frame {frame_idx}: {e}. "
                    f"{file_info}"
                ) from e

            # Validate output data
            if chi is None or k is None:
                raise ValueError(
                    f"Invalid FEFF output for frame {frame_idx}: chi or k is None"
                )

            if len(chi) == 0 or len(k) == 0:
                raise ValueError(f"Empty FEFF output for frame {frame_idx}")

            # Save to cache if caching is enabled
            if cache_dir:
                cache_key = get_cache_key(atoms, absorber, config)
                save_to_cache(cache_key, chi, k, cache_dir)
                logger.debug(f"Frame {frame_idx}: Saved to cache")

            # Force garbage collection to prevent memory buildup
            gc.collect()

            elapsed = time.time() - start_time
            logger.debug(f"Frame {frame_idx} completed in {elapsed:.2f}s")

            result = FrameProcessingResult(chi=chi, k=k, frame_idx=frame_idx)
            result.from_cache = from_cache
            return result

        except KeyboardInterrupt:
            # Handle graceful shutdown
            logger = logging.getLogger(__name__)
            logger.info(f"Frame {frame_idx} interrupted by user")
            return FrameProcessingResult(
                error="Interrupted by user", frame_idx=frame_idx
            )

        except (RuntimeError, ValueError, OSError) as e:
            # Enhanced error reporting
            logger = logging.getLogger(__name__)
            error_msg = f"Frame {frame_idx}: {type(e).__name__}: {str(e)}"

            # Add timing information if available
            if start_time:
                elapsed = time.time() - start_time
                error_msg += f" (failed after {elapsed:.2f}s)"

            # Add traceback for debugging
            if logger.isEnabledFor(logging.DEBUG):
                error_msg += f"\nTraceback:\n{traceback.format_exc()}"

            logger.error(error_msg)

            # Force garbage collection even on error
            gc.collect()

            return FrameProcessingResult(error=error_msg, frame_idx=frame_idx)

    def process(
        self,
        structure: Path | str | Atoms | list[Atoms],
        absorber: str,
        output_dir: Path,
        config: FeffConfig | None = None,
        trajectory: bool = False,
        show_plot: bool = False,
        plot_individual_frames: bool = False,
        frame_index: int | None = None,
        plot_style: str = "publication",
        chi_weighting: str = "chi",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingResult:
        """Process a structure or trajectory to generate EXAFS data.

        Args:
            structure: Structure file path, Atoms object, or trajectory file
            absorber: Absorbing atom symbol (e.g., 'Fe', 'Cu')
            output_dir: Directory for output files
            config: FEFF configuration parameters
            trajectory: Process as trajectory (multiple frames)
            show_plot: Display plots after processing
            plot_individual_frames: Create plots for each frame
            frame_index: Process specific frame (if None, processes all)
            plot_style: Plot style ('publication' or 'presentation')
            chi_weighting: Chi weighting for plots ('chi', 'k2chi', 'k3chi')
            progress_callback: Callback function for progress updates

        Returns:
            ProcessingResult containing EXAFS data and metadata
        """
        config = config or FeffConfig()
        output_dir = Path(output_dir).resolve()

        if isinstance(structure, Atoms):
            if trajectory:
                raise ValueError("trajectory mode not supported for single Atoms")
            result = self._process_single_frame(structure, absorber, output_dir, config)
            if result is None or result.error or result.chi is None or result.k is None:
                error_msg = (
                    result.error if result is not None else "Processing returned None"
                )
                raise FEFFCalculationError(
                    f"Single frame processing failed: {error_msg}"
                )

            group = Group()
            group.k = result.k
            group.chi = result.chi
            xftf(group, **config.fourier_params)

            plot_paths = self.plot_results(
                group,
                output_dir,
                "EXAFS_FT",
                show_plot=show_plot,
                plot_style=plot_style,
                absorber=absorber,
                edge=config.edge,
                chi_weighting=chi_weighting,
            )
            return ProcessingResult(
                exafs_group=group,
                plot_paths=plot_paths,
                processing_mode=ProcessingMode.SINGLE_FRAME,
            )
        elif isinstance(structure, list) and all(
            isinstance(s, Atoms) for s in structure
        ):
            if not trajectory and len(structure) > 1:
                raise ValueError("List of Atoms requires trajectory=True")
            if frame_index is not None:
                raise ValueError("frame_index not supported for list of Atoms")
            structures = structure
        elif isinstance(structure, str | Path):
            # Handle file-based input
            ase_index = self._construct_ase_index(
                trajectory, frame_index, config.sample_interval
            )
            structures = ase_read(str(structure), index=ase_index)
            if not isinstance(structures, list):
                structures = [structures]

        # Initialize reporter with proper fallback behavior
        reporter: ProgressReporter
        if progress_callback is None:
            reporter = TQDMReporter(len(structures), "Starting")
        else:
            reporter = CallbackReporter(progress_callback)

        try:
            return self._process_structures_chunked(
                structures,
                absorber,
                output_dir,
                config,
                reporter,
                plot_individual_frames,
                chi_weighting,
            )
        finally:
            reporter.close()

    def _construct_ase_index(
        self, trajectory: bool, frame_index: int | None, sample_interval: int
    ) -> str:
        """Construct ASE index string for reading trajectory frames.

        Args:
            trajectory: Whether processing as trajectory
            frame_index: Specific frame index (if any)
            sample_interval: Sampling interval for trajectory

        Returns:
            ASE index string for frame selection
        """
        if frame_index is not None:
            return str(frame_index)
        elif trajectory:
            return f"::{sample_interval}" if sample_interval > 1 else ":"
        else:
            return "-1"

    def get_diagnostics(self) -> dict[str, Any]:
        """Get system and wrapper diagnostics information.

        Returns:
            Dictionary containing system info, dependencies, and cache status
        """
        cache_info = self.get_cache_info()
        return {
            "system": {"platform": os.name, "python": sys.version.split()[0]},
            "dependencies": {
                "ase": True,
                "larch": True,
            },
            "presets": list(PRESETS.keys()),
            "cache": {
                "enabled": cache_info["enabled"],
                "files": cache_info.get("files", 0),
                "size_mb": cache_info.get("size_mb", 0),
            },
        }

    def print_diagnostics(self) -> None:
        """Print formatted diagnostics information to console."""
        diag = self.get_diagnostics()
        print("=" * 50)
        print("LARCH WRAPPER DIAGNOSTICS")
        print("=" * 50)
        print(
            f"System: {diag['system']['platform']} | Python: {diag['system']['python']}"
        )
        print("Dependencies: ASE ✓ ")
        print(
            f"Cache: {'Enabled' if diag['cache']['enabled'] else 'Disabled'} | "
            f"{diag['cache']['files']} files | {diag['cache']['size_mb']} MB"
        )
        print(f"Available presets: {', '.join(diag['presets'])}")
        print("=" * 50)
