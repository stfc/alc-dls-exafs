"""Streamlined Larch Wrapper - Simplified EXAFS processing with parallel support and cross-platform compatibility"""

import gc
import logging
import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from larch import Group
from larch.io import read_ascii
from larch.xafs import xftf
from larch.xafs.feffrunner import feff8l, find_exe
from larixite import cif2feffinp

# Required dependencies
try:
    from ase import Atoms
    from ase.io import read as ase_read
    from ase.io import write as ase_write
except ImportError:
    raise ImportError("ASE is required. Install with: pip install ase")

# Optional dependencies with fallbacks
try:
    from pymatgen.core import Structure
    from pymatgen.io.feff.sets import MPEXAFSSet, MPXANESSet
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

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

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configuration presets
PRESETS = {
    "quick": {"spectrum_type": "EXAFS", "edge": "K", "radius": 8.0, "kmin": 2, "kmax": 12},
    "publication": {"spectrum_type": "EXAFS", "edge": "K", "radius": 12.0, "kmin": 3, "kmax": 18, "dk": 4},
    # "xanes": {"spectrum_type": "XANES", "edge": "K", "radius": 8.0, "user_tag_settings": {"SCF": "7.0", "FMS": "9.0"}},
}


class ProcessingMode(Enum):
    """Processing mode for input data."""
    SINGLE_FRAME = "single_frame"
    TRAJECTORY = "trajectory"

    AVERAGE = "average"

class SpectrumType(str, Enum):
    """Valid spectrum types."""
    EXAFS = "EXAFS"
    XANES = "XANES"
    DANES = "DANES"
    XMCD = "XMCD"
    ELNES = "ELNES"
    EXELFS = "EXELFS"
    FPRIME = "FPRIME"
    NRIXS = "NRIXS"
    XES = "XES"

class EdgeType(str, Enum):
    """Valid absorption edges."""
    K = "K"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"

@dataclass
class FeffConfig:
    """Streamlined configuration for FEFF calculations."""
    # FEFF calculation parameters
    spectrum_type: str = "EXAFS"
    edge: str = "K"
    radius: float = 10.0
    method: str = "auto"  # "auto", "larixite", or "pymatgen"
    user_tag_settings: Dict[str, str] = field(default_factory=dict)
    
    # Fourier transform parameters
    kweight: int = 2
    window: str = "hanning"
    dk: float = 1.0
    kmin: float = 2.0
    kmax: float = 14.0
    
    # Processing options
    parallel: bool = False
    n_workers: Optional[int] = None
    sample_interval: int = 1
    
    @classmethod
    def from_preset(cls, preset_name: str) -> "FeffConfig":
        """Create configuration from preset."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
        
        preset = PRESETS[preset_name].copy()
        user_tag_settings = preset.pop("user_tag_settings", {})
        return cls(user_tag_settings=user_tag_settings, **preset)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FeffConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for configuration files")
        
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        
        user_tag_settings = params.pop("user_tag_settings", {})
        return cls(user_tag_settings=user_tag_settings, **params)

@dataclass
class ProcessingResult:
    """Result of EXAFS processing operation."""
    exafs_group: Group
    plot_paths: Tuple[Path, Path]  # (pdf_path, svg_path)
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
    
    def save(self, output_dir: Path):
        """Save processing results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        np.savetxt(output_dir / "exafs_data.dat", np.column_stack((self.k, self.chi)), header="k chi")
        np.savetxt(output_dir / "ft_data.dat", np.column_stack((self.r, self.chir_mag)), header="R chir_mag")
        return output_dir


class LarchWrapper:
    """Streamlined EXAFS processing wrapper with parallel support and cross-platform compatibility."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the wrapper."""
        self.logger = self._setup_logger(verbose)
        self._temp_files = []
    
    def _setup_logger(self, verbose: bool) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("larch_wrapper")
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        logger.handlers = []
        
        if verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_temp_files()
    
    def __del__(self):
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Clean up temporary files and free memory."""
        for temp_file in self._temp_files[:]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                self._temp_files.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not remove temp file {temp_file}: {e}")
        gc.collect()

    def _validate_inputs(self, structure_path: Path, absorber: str):
        """Validate input parameters."""
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file {structure_path} not found")
        if structure_path.stat().st_size == 0:
            raise ValueError(f"Structure file {structure_path} is empty")
        
        # Validate absorber (try as int first, then as element symbol)
        try:
            return int(absorber)  # Site index
        except ValueError:
            if len(absorber) > 2:
                self.logger.warning(f"Absorber '{absorber}' may not be valid element symbol")
            return absorber.capitalize()
    
    def _create_temp_cif(self, structure_path: Path) -> Path:
        """Convert structure to CIF format if needed."""
        if structure_path.suffix.lower() == ".cif":
            return structure_path
        
        try:
            atoms = ase_read(str(structure_path))
            temp_fd, temp_path = tempfile.mkstemp(suffix=".cif", prefix="larch_temp_")
            os.close(temp_fd)
            temp_cif = Path(temp_path)
            self._temp_files.append(temp_cif)
            ase_write(str(temp_cif), atoms, format="cif")
            return temp_cif
        except Exception as e:
            raise RuntimeError(f"Failed to convert structure to CIF: {e}")
    
    def generate_feff_input(self, structure: Union[Path, Atoms], absorber: Union[str, int], 
                           output_dir: Path, config: FeffConfig) -> Path:
        """Generate FEFF input file.
        
        Args:
            structure: Path to structure file or ASE Atoms object
            absorber: Absorbing atom symbol or index
            output_dir: Output directory for FEFF input
            config: FEFF configuration settings
            
        Returns:
            Path to generated FEFF input file
        """
        # Handle ASE Atoms input
        if isinstance(structure, Atoms):
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                ase_write(tmp.name, structure, format="cif")
                structure_path = Path(tmp.name)
                self._temp_files.append(structure_path)
        else:
            structure_path = Path(structure).resolve()
        
        absorber = self._validate_inputs(structure_path, str(absorber))
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-select method
        if config.method == "auto":
            config.method = "pymatgen" if config.spectrum_type == "XANES" and PYMATGEN_AVAILABLE else "larixite"
        
        # Generate input based on method
        if config.method == "pymatgen" and PYMATGEN_AVAILABLE:
            return self._generate_with_pymatgen(structure_path, absorber, output_dir, config)
        else:
            return self._generate_with_larixite(structure_path, absorber, output_dir, config)
    
    def _generate_with_larixite(self, structure_path: Path, absorber: Union[str, int], 
                               output_dir: Path, config: FeffConfig) -> Path:
        """Generate FEFF input using larixite."""
        cif_path = self._create_temp_cif(structure_path)
        inp = cif2feffinp(cif_path, absorber)
        input_file = output_dir / "feff.inp"
        input_file.write_text(inp)
        return input_file
    
    def _generate_with_pymatgen(self, structure_path: Path, absorber: Union[str, int], 
                               output_dir: Path, config: FeffConfig) -> Path:
        """Generate FEFF input using pymatgen."""
        try:
            structure = Structure.from_file(str(structure_path))
        except Exception:
            # Fallback to ASE conversion
            cif_path = self._create_temp_cif(structure_path)
            structure = Structure.from_file(str(cif_path))
        
        # Remove unsupported settings
        user_settings = config.user_tag_settings.copy()
        if "COREHOLE" in user_settings:
            self.logger.warning("COREHOLE not supported in FEFF8L, removing")
            del user_settings["COREHOLE"]
        
        # Select appropriate set
        if config.spectrum_type == "EXAFS":
            feff_set = MPEXAFSSet(
                absorbing_atom=absorber, structure=structure, edge=config.edge,
                radius=config.radius, user_tag_settings=user_settings
            )
        elif config.spectrum_type == "XANES":
            feff_set = MPXANESSet(
                absorbing_atom=absorber, structure=structure, edge=config.edge,
                radius=config.radius, user_tag_settings=user_settings
            )
        else:
            raise ValueError(f"Unsupported spectrum type: {config.spectrum_type}")
        
        feff_set.write_input(str(output_dir))
        return output_dir / "feff.inp"
    
    def run_feff(self, feff_dir: Path, config: Optional[FeffConfig] = None) -> bool:
        """Run FEFF calculation with cross-platform stdout handling.
        
        Args:
            feff_dir: Directory containing FEFF input file
            config: FEFF configuration (optional, only used for logging context)
            
        Returns:
            True if FEFF calculation succeeded, False otherwise
        """
        feff_dir = Path(feff_dir)
        input_path = feff_dir / "feff.inp"
        log_path = feff_dir / "feff.log"
        
        if not input_path.exists():
            raise FileNotFoundError(f"FEFF input file {input_path} not found")
        
        # Cross-platform stdout capture
        original_stdout = sys.stdout
        
        try:
            self.logger.info(f"Running FEFF calculation (log: {log_path})")
            
            # Safe text-only logger for cross-platform compatibility
            class SafeLogger:
                def __init__(self, log_file):
                    self.log_file = log_file
                    self.encoding = 'utf-8'
                    
                def write(self, text):
                    if isinstance(text, bytes):
                        text = text.decode('utf-8', errors='replace')
                    # Filter control characters for Windows compatibility
                    clean_text = ''.join(c if ord(c) >= 32 or c in '\n\t\r' else '?' for c in text)
                    self.log_file.write(clean_text)
                    self.log_file.flush()
                
                def flush(self):
                    self.log_file.flush()
            
            with open(log_path, 'w', encoding='utf-8', errors='replace') as log_file:
                sys.stdout = SafeLogger(log_file)
                result = feff8l(folder=str(feff_dir), feffinp="feff.inp", verbose=True)
            
            # Verify output
            chi_file = feff_dir / "chi.dat"
            if not chi_file.exists():
                self.logger.error(f"FEFF output missing: {chi_file}")
                return False
            
            self.logger.info("FEFF calculation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"FEFF calculation failed: {e}")
            return False
        finally:
            sys.stdout = original_stdout
    
    def process_feff_output(self, feff_dir: Path, config: FeffConfig) -> Group:
        """Process FEFF output and perform Fourier transform.
        
        Args:
            feff_dir: Directory containing FEFF output files
            config: Configuration with Fourier transform parameters
            
        Returns:
            Larch Group containing processed EXAFS data with Fourier transform
        """
        chi_file = feff_dir / "chi.dat"
        if not chi_file.exists():
            raise FileNotFoundError(f"FEFF output {chi_file} not found")
        
        feff_data = read_ascii(str(chi_file))
        
        # Create group and perform FT
        g = Group()
        g.k = feff_data.k
        g.chi = feff_data.chi
        
        xftf(g, kweight=config.kweight, window=config.window, dk=config.dk,
             kmin=config.kmin, kmax=config.kmax)
        
        return g
    
    def process_trajectory_output(self, trajectory_dir: Path, config: FeffConfig, 
                                plot_individual_frames: bool = False, 
                                show_plot: bool = False,
                                output_dir: Optional[Path] = None) -> ProcessingResult:
        """Process FEFF outputs from a trajectory and create averaged result.
        
        Args:
            trajectory_dir: Directory containing frame_XXXX subdirectories with FEFF outputs
            config: Configuration with Fourier transform parameters
            plot_individual_frames: Whether to plot individual frames alongside averaged result
            show_plot: Whether to display plots interactively
            output_dir: Directory to save plots (defaults to trajectory_dir)
            
        Returns:
            ProcessingResult with averaged EXAFS data and individual frames (if requested)
        """
        trajectory_dir = Path(trajectory_dir)
        output_dir = Path(output_dir) if output_dir else trajectory_dir
        
        # Find frame directories
        frame_dirs = sorted([d for d in trajectory_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('frame_')])
        
        if not frame_dirs:
            raise ValueError(f"No frame directories found in {trajectory_dir}")
        
        self.logger.info(f"Processing {len(frame_dirs)} trajectory frames")
        
        # Process each frame
        chi_list = []
        individual_frame_groups = [] if plot_individual_frames else None
        k_ref = None
        
        for frame_dir in frame_dirs:
            try:
                # Process this frame's FEFF output
                frame_group = self.process_feff_output(frame_dir, config)
                
                # Store k-grid reference from first frame
                if k_ref is None:
                    k_ref = frame_group.k.copy()
                
                # Interpolate to common k-grid if needed
                chi = frame_group.chi
                if not np.array_equal(frame_group.k, k_ref):
                    chi = np.interp(k_ref, frame_group.k, chi, left=0, right=0)
                
                chi_list.append(chi)
                
                # Store individual frame if requested
                if plot_individual_frames:
                    # Create frame group with common k-grid
                    frame_plot_group = Group()
                    frame_plot_group.k = k_ref.copy()
                    frame_plot_group.chi = chi
                    # Apply Fourier transform
                    xftf(frame_plot_group, kweight=config.kweight, window=config.window,
                         dk=config.dk, kmin=config.kmin, kmax=config.kmax)
                    individual_frame_groups.append(frame_plot_group)
                
                self.logger.info(f"Processed {frame_dir.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {frame_dir.name}: {e}")
                continue
        
        if not chi_list:
            raise RuntimeError("No valid frames processed")
        
        # Create averaged result
        result_group = Group()
        result_group.k = k_ref
        result_group.chi = _fast_average(np.array(chi_list))
        result_group.chi_std = _fast_std(np.array(chi_list))
        
        # Apply Fourier transform to averaged result
        xftf(result_group, kweight=config.kweight, window=config.window,
             dk=config.dk, kmin=config.kmin, kmax=config.kmax)
        
        # Generate plots
        frames_to_plot = individual_frame_groups if plot_individual_frames else None
        plot_paths = self.plot_results(result_group, output_dir, "trajectory_output_EXAFS_FT", 
                                     show_plot, individual_frames=frames_to_plot)
        
        return ProcessingResult(
            exafs_group=result_group,
            plot_paths=plot_paths,
            processing_mode=ProcessingMode.TRAJECTORY,
            nframes=len(chi_list),
            individual_frame_groups=individual_frame_groups
        )
    
    def plot_results(self, exafs_group: Group, output_dir: Path, 
                     filename_base: str = "EXAFS_FT", show_plot: bool = False,
                     individual_frames: Optional[List[Group]] = None,
                     max_frames_to_plot: int = 10) -> Tuple[Path, Path]:
        """Generate plots for EXAFS results.
        
        Args:
            exafs_group: Larch Group containing EXAFS data with Fourier transform (main/averaged spectrum)
            output_dir: Directory to save plots
            filename_base: Base filename for plot files (default: "EXAFS_FT")
            show_plot: Whether to display plots interactively
            individual_frames: Optional list of Group objects from individual trajectory frames
            max_frames_to_plot: Maximum number of individual frames to plot (default: 10)
            
        Returns:
            Tuple of (PDF path, SVG path) for generated plots
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot individual frames first (so they appear in background)
        if individual_frames:
            n_frames_to_plot = min(len(individual_frames), max_frames_to_plot)
            self.logger.info(f"Plotting {n_frames_to_plot} individual frames (out of {len(individual_frames)} total)")
            
            # Sample frames evenly if we have more than max_frames_to_plot
            if len(individual_frames) > max_frames_to_plot:
                frame_indices = np.linspace(0, len(individual_frames)-1, max_frames_to_plot, dtype=int)
                frames_to_plot = [individual_frames[i] for i in frame_indices]
            else:
                frames_to_plot = individual_frames
            
            # Plot individual frames with light transparency
            for i, frame_group in enumerate(frames_to_plot):
                ax.plot(frame_group.r, frame_group.chir_mag, 
                       color='lightblue', alpha=0.3, linewidth=0.8, 
                       label='Individual frames' if i == 0 else '')
        
        # Plot the main/averaged spectrum prominently
        main_label = "Averaged spectrum" if individual_frames else "EXAFS spectrum"
        ax.plot(exafs_group.r, exafs_group.chir_mag, linewidth=3, color='darkblue', label=main_label)
        
        # Add standard deviation envelope if available
        if hasattr(exafs_group, 'chir_mag_std') and individual_frames:
            ax.fill_between(exafs_group.r, 
                           exafs_group.chir_mag - exafs_group.chir_mag_std,
                           exafs_group.chir_mag + exafs_group.chir_mag_std,
                           alpha=0.2, color='darkblue', label='±1 std')
        
        ax.set_xlabel(r"R ($\AA$)", fontsize=12)
        ax.set_ylabel(r"$|\chi(R)|$", fontsize=12)
        
        # Update title based on what we're plotting
        if individual_frames:
            title = f"Fourier Transform of EXAFS: Average + {len(individual_frames)} Frames"
        else:
            title = "Fourier Transform of EXAFS"
        ax.set_title(title, fontsize=14)
        
        ax.grid(True, alpha=0.3)
        
        # Add legend if we have individual frames
        if individual_frames:
            ax.legend(fontsize=10)
        
        # Save plots
        pdf_path = output_dir / f"{filename_base}.pdf"
        svg_path = output_dir / f"{filename_base}.svg"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(svg_path, format="svg", transparent=True, bbox_inches="tight")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return pdf_path, svg_path
    
    def process(self, structure: Union[Path, str, Atoms], absorber: str, output_dir: Path,
                config: Optional[FeffConfig] = None, trajectory: bool = False, 
                show_plot: bool = False, plot_individual_frames: bool = False,
                frame_index: Optional[int] = None,
                progress_callback: Optional[Callable[[int, int, str], None]] = None) -> ProcessingResult:
        """Main processing method for single frames or trajectories.
        
        Args:
            structure: Path to structure file or ASE Atoms object
            absorber: Absorbing atom symbol or index
            output_dir: Output directory for results
            config: Processing configuration (uses defaults if None)
            trajectory: Whether input is a trajectory (multiple frames)
            show_plot: Whether to display plots interactively
            plot_individual_frames: Whether to plot individual trajectory frames (trajectory only)
            frame_index: Optional specific frame index to process (for trajectories)
            progress_callback: Optional callback for progress updates
            
        Returns:
            ProcessingResult with EXAFS data and plot paths
        """
        config = config or FeffConfig()
        output_dir = Path(output_dir).resolve()
        
        # Handle ASE Atoms input directly
        if isinstance(structure, Atoms):
            if frame_index is not None:
                raise ValueError("frame_index cannot be used with ASE Atoms objects")
            if trajectory:
                raise ValueError("trajectory mode cannot be used with single ASE Atoms objects")
            
            if progress_callback:
                progress_callback(0, 1, "Processing single structure")
            
            return self._process_single(structure, absorber, output_dir, config, show_plot)
        
        # Construct ASE index string based on parameters
        ase_index = self._construct_ase_index(trajectory, frame_index, config.sample_interval)
        
        # Read structures using constructed index
        try:
            structures = ase_read(str(structure), index=ase_index)
        except Exception as e:
            raise RuntimeError(f"Failed to read structure from {structure}: {e}")
        
        # Determine processing mode and frame count
        if isinstance(structures, list):
            n_frames = len(structures)
            is_trajectory = True
        else:
            structures = [structures]  # Convert single structure to list for consistency
            n_frames = 1
            is_trajectory = False
        
        # Initialize progress callback with frame count
        if progress_callback:
            if frame_index is not None:
                progress_callback(0, 1, f"Processing frame {frame_index}")
            else:
                progress_callback(0, n_frames, f"Starting processing of {n_frames} frames")
        
        try:
            # Use unified processing method
            return self._process_structures(structures, absorber, output_dir, config, 
                                          show_plot, plot_individual_frames, progress_callback)
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
    
    def _construct_ase_index(self, trajectory: bool, frame_index: Optional[int], sample_interval: int) -> str:
        """Construct ASE index string based on processing parameters.
        
        Args:
            trajectory: Whether processing as trajectory
            frame_index: Specific frame index (if any)
            sample_interval: Sampling interval for trajectories
            
        Returns:
            ASE index string for ase_read
        """
        if frame_index is not None:
            # Specific frame
            return str(frame_index)
        elif trajectory:
            # Trajectory with sampling
            if sample_interval == 1:
                return ":"  # All frames
            else:
                return f"::{sample_interval}"  # Every nth frame
        else:
            # Single structure (last frame by default)
            return "-1"
    
    def _process_structures(self, structures: Union[Atoms, List[Atoms]], absorber: str, output_dir: Path,
                           config: FeffConfig, show_plot: bool, plot_individual_frames: bool = False,
                           progress_callback: Optional[Callable[[int, int, str], None]] = None) -> ProcessingResult:
        """Unified method to process single structure or trajectory."""
        # Normalize input to list
        if isinstance(structures, Atoms):
            structures = [structures]
            is_single = True
        else:
            is_single = False
        
        n_frames = len(structures)
        self.logger.info(f"Processing {n_frames} frame{'s' if n_frames > 1 else ''}")
        
        # Setup progress tracking
        completed = 0
        def update_progress(desc: str):
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed, n_frames, desc)
        
        # Process frames
        chi_list = []
        individual_frame_groups = []
        k_ref = None
        
        # Use parallel processing for multiple frames if enabled
        if config.parallel and n_frames > 1:
            n_workers = config.n_workers or min(mp.cpu_count(), n_frames)
            self.logger.info(f"Using {n_workers} parallel workers")
            
            try:
                with mp.Pool(n_workers) as pool:
                    # Create worker tasks
                    tasks = [(i, atoms, absorber, output_dir, config, is_single) for i, atoms in enumerate(structures)]
                    results = pool.map(self._process_frame_worker, tasks)
                
                # Process results
                for chi, k, frame_idx, error in results:
                    if error:
                        self.logger.error(f"Frame {frame_idx} failed: {error}")
                        update_progress(f"Failed frame {frame_idx+1}/{n_frames}")
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
                    individual_frame_groups.append(frame_group)
                    
                    update_progress(f"Processed frame {frame_idx+1}/{n_frames}")
                    
            except Exception as e:
                self.logger.error(f"Parallel processing failed: {e}, falling back to sequential")
                config.parallel = False
        
        # Sequential processing (or fallback from parallel)
        if not config.parallel or n_frames == 1:
            for i, atoms in enumerate(structures):
                # For single frames, use main output dir; for trajectories, use frame subdirs
                if is_single:
                    frame_dir = output_dir
                else:
                    frame_dir = output_dir / f"frame_{i:04d}"
                    frame_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Process individual frame
                    self.generate_feff_input(atoms, absorber, frame_dir, config)
                    
                    if not self.run_feff(frame_dir, config):
                        raise RuntimeError("FEFF calculation failed")
                    
                    # Process FEFF output
                    frame_group = self.process_feff_output(frame_dir, config)
                    
                    chi = frame_group.chi
                    k = frame_group.k
                    
                    if k_ref is None:
                        k_ref = k.copy()
                    
                    # Interpolate if needed
                    if not np.array_equal(k, k_ref):
                        chi = np.interp(k_ref, k, chi, left=0, right=0)
                        # Update frame_group with interpolated data
                        frame_group.k = k_ref.copy()
                        frame_group.chi = chi
                        # Reapply Fourier transform with interpolated data
                        xftf(frame_group, kweight=config.kweight, window=config.window,
                             dk=config.dk, kmin=config.kmin, kmax=config.kmax)
                    
                    chi_list.append(chi)
                    individual_frame_groups.append(frame_group)
                    
                    update_progress(f"Processed frame {i+1}/{n_frames}")
                    
                except Exception as e:
                    self.logger.error(f"Frame {i} failed: {str(e)}")
                    update_progress(f"Failed frame {i+1}/{n_frames}")
                    continue
        
        if not chi_list:
            raise RuntimeError("No valid frames processed")
        
        # Handle single frame vs trajectory results
        if n_frames == 1:
            # Single frame - return the frame directly
            result_group = individual_frame_groups[0]
            processing_mode = ProcessingMode.SINGLE_FRAME
            filename_base = "EXAFS_FT"
            individual_frames_for_plot = None
        else:
            # Multiple frames - create averaged result
            result_group = Group()
            result_group.k = k_ref
            result_group.chi = _fast_average(np.array(chi_list))
            result_group.chi_std = _fast_std(np.array(chi_list))
            
            # Apply Fourier transform to averaged result
            xftf(result_group, kweight=config.kweight, window=config.window, 
                 dk=config.dk, kmin=config.kmin, kmax=config.kmax)
            
            processing_mode = ProcessingMode.TRAJECTORY
            filename_base = "trajectory_avg_EXAFS_FT"
            individual_frames_for_plot = individual_frame_groups if plot_individual_frames else None
        
        # Generate plots
        plot_paths = self.plot_results(result_group, output_dir, filename_base, show_plot, 
                                     individual_frames=individual_frames_for_plot)
        
        return ProcessingResult(
            exafs_group=result_group,
            plot_paths=plot_paths,
            processing_mode=processing_mode,
            nframes=n_frames,
            individual_frame_groups=individual_frame_groups if n_frames > 1 else None
        )
    
    @staticmethod
    def _process_frame_worker(frame_data: tuple) -> tuple:
        """Worker function for parallel frame processing."""
        try:
            frame_idx, atoms, absorber, output_base, config, is_single = frame_data
            
            # Setup frame directory
            if is_single:
                frame_dir = Path(output_base)
            else:
                frame_dir = Path(output_base) / f"frame_{frame_idx:04d}"
                frame_dir.mkdir(parents=True, exist_ok=True)
            
            # Create wrapper instance for this worker
            wrapper = LarchWrapper(verbose=False)
            
            # Process frame directly
            wrapper.generate_feff_input(atoms, absorber, frame_dir, config)
            
            if not wrapper.run_feff(frame_dir, config):
                raise RuntimeError("FEFF calculation failed")
            
            # Process FEFF output
            frame_group = wrapper.process_feff_output(frame_dir, config)
            wrapper.cleanup_temp_files()
            
            return frame_group.chi, frame_group.k, frame_idx, None
            
        except Exception as e:
            return None, None, frame_idx, str(e)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get system diagnostics."""
        return {
            "system": {"platform": os.name, "python": sys.version.split()[0]},
            "dependencies": {
                "ase": True,
                "pymatgen": PYMATGEN_AVAILABLE,
                "larch": True,
                "numba": NUMBA_AVAILABLE,
            },
            "feff": {"executables": self._get_feff_executables()},
            "presets": list(PRESETS.keys()),
        }
    
    def _get_feff_executables(self) -> Dict[str, str]:
        """Get available FEFF executables."""
        executables = ["feff8l", "feff6l", "feff"]
        available = {}
        for exe in executables:
            try:
                path = find_exe(exe)
                if path:
                    available[exe] = str(path)
            except Exception:
                pass
        return available
    
    def print_diagnostics(self):
        """Print diagnostic information."""
        diag = self.get_diagnostics()
        print("=" * 50)
        print("LARCH WRAPPER DIAGNOSTICS")
        print("=" * 50)
        print(f"System: {diag['system']['platform']} | Python: {diag['system']['python']}")
        print(f"Dependencies: ASE ✓ | Pymatgen {'✓' if diag['dependencies']['pymatgen'] else '✗'} | Numba {'✓' if diag['dependencies']['numba'] else '✗'}")
        print(f"FEFF executables: {len(diag['feff']['executables'])} found")
        print(f"Available presets: {', '.join(diag['presets'])}")
        print("=" * 50)
