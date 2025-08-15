"""Streamlined Larch Wrapper - Enhanced EXAFS processing with caching, memory management, and cross-platform robustness"""

import gc
import logging
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
from larch import Group
from larch.xafs import xftf

from .cache_utils import get_cache_key, load_from_cache, save_to_cache
from .feff_utils import generate_feff_input, run_feff_calculation, read_feff_output, generate_larixite_input, generate_pymatgen_input, FeffConfig, PRESETS

# Required dependencies
try:
    from ase import Atoms
    from ase.io import read as ase_read
    from ase.io import write as ase_write
except ImportError:
    raise ImportError("ASE is required. Install with: pip install ase")

# Optional dependencies
try:
    import numba
    @numba.jit(nopython=True, cache=True)
    def _fast_average(data: np.ndarray) -> np.ndarray:
        return np.mean(data, axis=0)

    @numba.jit(nopython=True, cache=True)
    def _fast_std(data: np.ndarray) -> np.ndarray:
        return np.std(data, axis=0)
    NUMBA_AVAILABLE = True
except ImportError:
    def _fast_average(data: np.ndarray) -> np.ndarray:
        return np.mean(data, axis=0)

    def _fast_std(data: np.ndarray) -> np.ndarray:
        return np.std(data, axis=0)
    NUMBA_AVAILABLE = False




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
    SINGLE_FRAME = "single_frame"
    TRAJECTORY = "trajectory"
    AVERAGE = "average"



# ================== PROGRESS REPORTING ==================
class ProgressReporter(Protocol):
    def update(self, current: int, total: int, description: str) -> None: ...
    def set_description(self, description: str) -> None: ...
    def close(self) -> None: ...

class TQDMReporter:
    def __init__(self):
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
            self.pbar = None
        except ImportError:
            self.tqdm = None
            self.pbar = None

    def update(self, current: int, total: int, description: str) -> None:
        if self.tqdm is None:
            return
        if self.pbar is None:
            self.pbar = self.tqdm(total=total, desc=description)
        self.pbar.n = current
        self.pbar.set_description(description)
        self.pbar.refresh()

    def close(self) -> None:
        if self.pbar:
            self.pbar.close()

class SimpleReporter:
    def update(self, current: int, total: int, description: str) -> None:
        percent = (current / total) * 100
        print(f"\r[{percent:6.2f}%] {description}", end="", flush=True)
        if current == total:
            print()

    def close(self) -> None:
        print()

class CallbackReporter:
    """Reporter that bridges to CLI progress callbacks."""
    def __init__(self, callback: Callable[[int, int, str], None]):
        self.callback = callback

    def update(self, current: int, total: int, description: str) -> None:
        self.callback(current, total, description)

    def set_description(self, description: str) -> None:
        pass  # Not used in current implementation

    def close(self) -> None:
        pass  # Nothing to clean up


# ================== PROCESSING RESULT ==================
@dataclass
class ProcessingResult:
    exafs_group: Group
    plot_paths: Dict[str, Path]  # e.g., {"pdf": ..., "svg": ..., "png": ...}
    processing_mode: ProcessingMode
    nframes: int = 1
    individual_frame_groups: Optional[List[Group]] = None

    @property
    def k(self) -> np.ndarray:
        return self.exafs_group.k

    @property
    def chi(self) -> np.ndarray:
        return self.exafs_group.chi

    @property
    def r(self) -> np.ndarray:
        return self.exafs_group.r

    @property
    def chir_mag(self) -> np.ndarray:
        return self.exafs_group.chir_mag

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "exafs_data.dat", np.column_stack((self.k, self.chi)), header="k chi")
        np.savetxt(output_dir / "ft_data.dat", np.column_stack((self.r, self.chir_mag)), header="R chir_mag")
        return output_dir


# ================== CACHING ==================
# Caching functionality is now integrated directly into LarchWrapper


# ================== PARALLEL PROCESSING ==================
class ParallelProcessor:
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or min(mp.cpu_count() - 1, 8)

    @contextmanager
    def process_pool(self):
        pool = None
        try:
            pool = mp.Pool(self.n_workers, initializer=self._worker_init, initargs=(logging.getLogger().level,))
            yield pool
        finally:
            if pool:
                pool.close()
                pool.join()

    @staticmethod
    def _worker_init(log_level):
        logging.basicConfig(level=log_level)


# ================== MAIN WRAPPER ==================
class LarchWrapper:
    def __init__(self, verbose: bool = True, cleanup_on_exit: bool = True, cache_dir: Optional[Path] = None):
        self.cleanup_on_exit = cleanup_on_exit
        self._temp_files = []
        self._temp_dirs = []
        self.logger = self._setup_logger(verbose)
        self.parallel_processor = ParallelProcessor()
        # Add caching capability
        self.cache_dir = (cache_dir or Path.home() / ".larch_cache") if cache_dir is not None else None
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            self.cleanup_temp_files()
            self.cleanup_temp_dirs()
        if exc_type is not None:
            self.logger.error(f"Error during processing: {exc_val}")
        return False

    def cleanup_temp_files(self):
        for temp_file in self._temp_files[:]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                self._temp_files.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not remove temp file {temp_file}: {e}")
        gc.collect()

    def cleanup_temp_dirs(self):
        import shutil
        for temp_dir in self._temp_dirs[:]:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                self._temp_dirs.remove(temp_dir)
            except Exception as e:
                self.logger.warning(f"Could not remove temp dir {temp_dir}: {e}")

    # ================== CACHING METHODS ==================
    def clear_cache(self):
        """Clear all cached results."""
        if not self.cache_dir or not self.cache_dir.exists():
            self.logger.info("No cache directory to clear")
            return
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            self.logger.info(f"Cleared {len(cache_files)} cache files")
        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        if not self.cache_dir:
            return {"enabled": False}
        
        if not self.cache_dir.exists():
            return {"enabled": True, "cache_dir": str(self.cache_dir), "files": 0, "size_mb": 0}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        
        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "files": len(cache_files),
            "size_mb": round(total_size / (1024 * 1024), 2)
        }

    def generate_feff_input(self, structure: Union[Path, Atoms], absorber: Union[str, int],
                        output_dir: Path, config: FeffConfig) -> Path:
        """Generate FEFF input files - now uses utility functions."""
        if isinstance(structure, Atoms):
            # Use utility function directly for Atoms objects
            return generate_feff_input(structure, absorber, output_dir, config)
        else:
            # For file-based structures, read first then use utility
            structure_path = Path(structure).resolve()
            atoms = ase_read(str(structure_path))
            return generate_feff_input(atoms, absorber, output_dir, config)

    def _generate_with_larixite(self, structure_path: Path, absorber: Union[str, int],
                            output_dir: Path, config: FeffConfig) -> Path:
        """Legacy method - now redirects to utility function."""
        atoms = ase_read(str(structure_path))
        return generate_larixite_input(atoms, absorber, output_dir, config)

    def _generate_with_pymatgen(self, structure_path: Path, absorber: Union[str, int],
                            output_dir: Path, config: FeffConfig) -> Path:
        """Legacy method - now redirects to utility function."""
        atoms = ase_read(str(structure_path))  
        return generate_pymatgen_input(atoms, absorber, output_dir, config)

    def run_feff(self, feff_dir: Path, config: Optional[FeffConfig] = None) -> bool:
        """Run FEFF calculation - now uses utility function."""
        result = run_feff_calculation(feff_dir, verbose=self.logger.level <= logging.INFO)
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
        xftf(g, kweight=config.kweight, window=config.window, dk=config.dk,
            kmin=config.kmin, kmax=config.kmax)
        return g


    def plot_results(self, exafs_group: Group, output_dir: Path,
                     filename_base: str = "EXAFS_FT", show_plot: bool = False,
                     plot_style: str = "publication", absorber: str = "X", edge: str = "K",
                     individual_frames: Optional[List[Group]] = None,
                     show_individual_legend: bool = True,
                     max_individual_frames: int = 100) -> Dict[str, Path]:
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
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        import matplotlib.pyplot as plt
        from pathlib import Path as PathlibPath

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get the style file path
        styles_dir = PathlibPath(__file__).parent / "styles"
        
        # Style configurations with external style files
        styles = {
            "publication": {
                "style_file": styles_dir / "exafs_publication.mplstyle",
                "figsize": (12, 5)
            },
            "presentation": {
                "style_file": styles_dir / "exafs_presentation.mplstyle",
                "figsize": (10, 4)
            },
        }
        
        style_config = styles.get(plot_style, styles["publication"])
        
        # Store original matplotlib settings
        original_params = plt.rcParams.copy()
        
        try:
            # Apply marimo-style using external style file
            if style_config["style_file"].exists():
                plt.style.use(str(style_config["style_file"]))
            else:
                self.logger.warning(f"Style file {style_config['style_file']} not found, using default style")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=style_config["figsize"])
            
            # === Left plot: χ(k) in k-space ===
            # Plot individual frames first (background)
            if individual_frames and show_individual_legend:
                frames_to_plot = individual_frames[:max_individual_frames]
                for i, frame in enumerate(frames_to_plot):
                    ax1.plot(frame.k, frame.chi, 
                            color='gray', alpha=0.3, linewidth=1,
                            label='Individual frames' if i == 0 else '', zorder=1)
            
            # Add standard deviation envelope if available
            if hasattr(exafs_group, 'chi_std') and exafs_group.chi_std is not None and individual_frames:
                ax1.fill_between(exafs_group.k, 
                               exafs_group.chi - exafs_group.chi_std,
                               exafs_group.chi + exafs_group.chi_std,
                               alpha=0.1, color='black', label='±1σ', zorder=2)
                main_label = "χ(k) Average ± σ"
            else:
                main_label = "χ(k)"
            
            # Plot main spectrum
            ax1.plot(exafs_group.k, exafs_group.chi, 
                    color='black', linewidth=2.5, label=main_label, zorder=3)

            ax1.set_xlabel(r"k [Å$^{-1}$]")
            ax1.set_ylabel(r"χ(k)")
            ax1.set_title(r"EXAFS χ(k)")

            if individual_frames or (hasattr(exafs_group, 'chi_std') and exafs_group.chi_std is not None):
                ax1.legend(loc='upper right')
            
            # === Right plot: |χ(R)| Fourier Transform ===
            # Plot individual frames first (background)
            if individual_frames and show_individual_legend:
                frames_to_plot = individual_frames[:max_individual_frames]
                for i, frame in enumerate(frames_to_plot):
                    ax2.plot(frame.r, frame.chir_mag, 
                            color='gray', alpha=0.3, linewidth=1,
                            label='Individual frames' if i == 0 else '', zorder=1)
            
            # Plot main spectrum
            main_ft_label = "|χ(R)| Average" if individual_frames else "|χ(R)|"
            ax2.plot(exafs_group.r, exafs_group.chir_mag, 
                    color='black', linewidth=2.5, label=main_ft_label, zorder=3)
            
            ax2.set_xlabel("R [Å]")
            ax2.set_ylabel("|χ(R)|")
            ax2.set_title("Fourier Transform |χ(R)|")
            
            if individual_frames:
                ax2.legend(loc='center right')
            
            # Add annotation similar to marimo (position it to avoid legend overlap)
            annotation_text = f"{absorber} {edge} edge"
            ax2.text(0.02, 0.98, annotation_text, transform=ax2.transAxes, 
                    fontsize=16, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                             edgecolor='black', linewidth=1), zorder=10)
            
            # Apply marimo-style axis formatting
            for ax in [ax1, ax2]:
                # Make all spines visible with proper thickness
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_color('black')
                
                # Mirror ticks on all sides (like marimo)
                ax.tick_params(top=True, right=True, which='major', 
                              length=6, width=2, color='black')
                ax.tick_params(top=True, right=True, which='minor', 
                              length=3, width=1, color='black')
            
            plt.tight_layout()
            
            # Save plots in multiple formats
            outputs = {}
            for fmt in ["pdf", "svg", "png"]:
                path = output_dir / f"{filename_base}.{fmt}"
                # Use DPI from matplotlib settings (set by style file)
                current_dpi = plt.rcParams.get('savefig.dpi', 300)
                fig.savefig(path, format=fmt, dpi=current_dpi,
                           bbox_inches="tight", transparent=(fmt == "svg"))
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

    def _process_single_frame(self, atoms: Atoms, absorber: str, output_dir: Path,
                            config: FeffConfig) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Process a single frame with caching support - now uses utility functions."""
        # Check cache first if caching is enabled
        if self.cache_dir and not config.force_recalculate:
            cache_key = get_cache_key(atoms, absorber, config)
            cached_result = load_from_cache(cache_key, self.cache_dir, config.force_recalculate)
            if cached_result is not None:
                return cached_result
        
        try:
            # Use utility functions
            generate_feff_input(atoms, absorber, output_dir, config)
            
            if not run_feff_calculation(output_dir, verbose=self.logger.level <= logging.INFO):
                return None
            
            chi, k = read_feff_output(output_dir)
            
            # Save to cache if caching is enabled
            if self.cache_dir:
                cache_key = get_cache_key(atoms, absorber, config)
                save_to_cache(cache_key, chi, k, self.cache_dir)
            
            return chi, k
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            return None

    def _process_structures_chunked(self, structures: List[Atoms], absorber: str, output_dir: Path,
                                config: FeffConfig, reporter: ProgressReporter, 
                                plot_individual_frames: bool = False) -> ProcessingResult:
        chi_list = []
        individual_groups = []
        k_ref = None  # Will store k-grid from first successful frame
        total_frames = len(structures)
        
        # Process frames
        if config.parallel and total_frames > 1:
            n_workers = config.n_workers or min(mp.cpu_count(), total_frames)
            self.logger.info(f"Using {n_workers} parallel workers")
            
            try:
                with mp.Pool(n_workers) as pool:
                    # Create worker tasks - include cache information
                    tasks = [(i, atoms, absorber, output_dir, config, False, self.cache_dir) for i, atoms in enumerate(structures)]
                    
                    # Use imap for incremental progress updates
                    results_iter = pool.imap(self._process_frame_worker, tasks)
                    
                    # Process results as they complete
                    for result_idx, (chi, k, frame_idx, error) in enumerate(results_iter):
                        if error:
                            self.logger.error(f"Frame {frame_idx} failed: {error}")
                            continue
                        
                        if k_ref is None:
                            k_ref = k.copy()
                        
                        # Interpolate to common k-grid if needed
                        if not np.array_equal(k, k_ref):
                            chi = np.interp(k_ref, k, chi, left=0, right=0)
                        
                        chi_list.append(chi)
                        
                        # Create frame group for plotting
                        frame_group = Group()
                        frame_group.k = k_ref.copy()
                        frame_group.chi = chi
                        xftf(frame_group, kweight=config.kweight, window=config.window, 
                             dk=config.dk, kmin=config.kmin, kmax=config.kmax)
                        individual_groups.append(frame_group)
                        
                        # Update progress after each completed frame
                        reporter.update(len(chi_list), total_frames, f"Processed {len(chi_list)}/{total_frames}")
                    
            except Exception as e:
                self.logger.error(f"Parallel processing failed: {e}, falling back to sequential")
                config.parallel = False
        
        # Sequential processing (or fallback from parallel)
        if not config.parallel or total_frames == 1:
            self.logger.info(f"Processing {total_frames} frames sequentially")
            processed = 0

            for i, atoms in enumerate(structures):
                frame_dir = output_dir / f"frame_{i:04d}"
                frame_dir.mkdir(exist_ok=True)

                result = self._process_single_frame(atoms, absorber, frame_dir, config)
                if result is not None:
                    chi, k = result

                    # Set k_ref from first successful frame
                    if k_ref is None:
                        k_ref = k.copy()

                    # Interpolate to common grid if needed
                    if not np.array_equal(k, k_ref):
                        chi_interp = np.interp(k_ref, k, chi, left=0, right=0)
                    else:
                        chi_interp = chi

                    # Add interpolated chi to the list for averaging
                    chi_list.append(chi_interp)

                    # Create group with common k-grid
                    frame_group = Group()
                    frame_group.k = k_ref.copy()
                    frame_group.chi = chi_interp
                    xftf(frame_group, kweight=config.kweight, window=config.window,
                        dk=config.dk, kmin=config.kmin, kmax=config.kmax)
                    individual_groups.append(frame_group)

                processed += 1
                reporter.update(processed, total_frames, f"Processed {processed}/{total_frames}")

        # Final validation
        if not chi_list:
            raise RuntimeError("No frames were processed successfully. Cannot generate result.")

        # Always use k_ref (defined from first success)
        avg_chi = _fast_average(np.array(chi_list))
        result_group = Group()
        result_group.k = k_ref
        result_group.chi = avg_chi
        result_group.chi_std = _fast_std(np.array(chi_list))

        # Apply FT to averaged spectrum
        xftf(result_group, kweight=config.kweight, window=config.window,
            dk=config.dk, kmin=config.kmin, kmax=config.kmax)

        # Generate plots
        plot_paths = self.plot_results(
            result_group, output_dir, "trajectory_avg_EXAFS_FT",
            show_plot=False, plot_style="publication", absorber=absorber, edge=config.edge,
            individual_frames=individual_groups if plot_individual_frames else None
        )

        return ProcessingResult(
            exafs_group=result_group,
            plot_paths=plot_paths,
            processing_mode=ProcessingMode.TRAJECTORY,
            nframes=len(chi_list),
            individual_frame_groups=individual_groups
        )

    @staticmethod
    def _process_frame_worker(frame_data: tuple) -> tuple:
        """Worker function for parallel frame processing - now uses utility functions."""
        try:
            from .feff_utils import generate_feff_input, run_feff_calculation, read_feff_output
            from .cache_utils import get_cache_key, load_from_cache, save_to_cache
            
            frame_idx, atoms, absorber, output_base, config, is_single, cache_dir = frame_data
            
            # Check cache first if caching is enabled
            if cache_dir and not config.force_recalculate:
                cache_key = get_cache_key(atoms, absorber, config)
                cached_result = load_from_cache(cache_key, cache_dir, config.force_recalculate)
                if cached_result is not None:
                    chi, k = cached_result
                    return chi, k, frame_idx, None
            
            # Setup frame directory
            if is_single:
                frame_dir = Path(output_base)
            else:
                frame_dir = Path(output_base) / f"frame_{frame_idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
            
            # Use utility functions for all FEFF operations
            generate_feff_input(atoms, absorber, frame_dir, config)
            
            if not run_feff_calculation(frame_dir, verbose=False):
                raise RuntimeError("FEFF calculation failed")
            
            chi, k = read_feff_output(frame_dir)
            
            # Save to cache if caching is enabled
            if cache_dir:
                cache_key = get_cache_key(atoms, absorber, config)
                save_to_cache(cache_key, chi, k, cache_dir)
            
            return chi, k, frame_idx, None
            
        except Exception as e:
            return None, None, frame_idx, str(e)

    def process(self, structure: Union[Path, str, Atoms], absorber: str, output_dir: Path,
                config: Optional[FeffConfig] = None, trajectory: bool = False,
                show_plot: bool = False, plot_individual_frames: bool = False,
                frame_index: Optional[int] = None,
                progress_callback: Optional[Callable[[int, int, str], None]] = None) -> ProcessingResult:
        config = config or FeffConfig()
        output_dir = Path(output_dir).resolve()

        if isinstance(structure, Atoms):
            if trajectory:
                raise ValueError("trajectory mode not supported for single Atoms")
            result = self._process_single_frame(structure, absorber, output_dir, config)
            if result is None:
                raise FEFFCalculationError("Single frame processing failed")
            chi, k = result
            group = Group()
            group.k = k
            group.chi = chi
            xftf(group, kweight=config.kweight, window=config.window, dk=config.dk,
                 kmin=config.kmin, kmax=config.kmax)
            plot_paths = self.plot_results(group, output_dir, "EXAFS_FT", show_plot=show_plot, 
                                          absorber=absorber, edge=config.edge)
            return ProcessingResult(exafs_group=group, plot_paths=plot_paths,
                                  processing_mode=ProcessingMode.SINGLE_FRAME)

        # Handle file-based input
        ase_index = self._construct_ase_index(trajectory, frame_index, config.sample_interval)
        structures = ase_read(str(structure), index=ase_index)
        if not isinstance(structures, list):
            structures = [structures]

        reporter = TQDMReporter() if progress_callback is None else CallbackReporter(progress_callback)
        reporter.update(0, len(structures), "Starting")

        return self._process_structures_chunked(structures, absorber, output_dir, config, reporter, plot_individual_frames)

    def _construct_ase_index(self, trajectory: bool, frame_index: Optional[int], sample_interval: int) -> str:
        if frame_index is not None:
            return str(frame_index)
        elif trajectory:
            return f"::{sample_interval}" if sample_interval > 1 else ":"
        else:
            return "-1"

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "system": {"platform": os.name, "python": sys.version.split()[0]},
            "dependencies": {
                "ase": True,
                "larch": True,
                "numba": NUMBA_AVAILABLE,
            },
            "presets": list(PRESETS.keys()),
        }

    def print_diagnostics(self):
        diag = self.get_diagnostics()
        print("=" * 50)
        print("LARCH WRAPPER DIAGNOSTICS")
        print("=" * 50)
        print(f"System: {diag['system']['platform']} | Python: {diag['system']['python']}")
        print(f"Dependencies: ASE ✓ | Numba {'✓' if diag['dependencies']['numba'] else '✗'}")
        print(f"Available presets: {', '.join(diag['presets'])}")
        print("=" * 50)