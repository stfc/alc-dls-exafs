"""Larch Wrapper - Simplified API for EXAFS processing with clear configuration and results"""

import gc
import logging
import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from larch import Group
from larch.io import read_ascii
from larch.xafs import xftf
from larch.xafs.feffrunner import feff8l, find_exe
from larixite import cif2feffinp

# Optional dependencies
try:
    from ase import Atoms
    from ase.io import read as ase_read
    from ase.io import write as ase_write
except ImportError:
    raise ImportError("ASE is required for this package. Install with: pip install ase")

try:
    from pymatgen.core import Structure
    from pymatgen.io.feff.sets import MPEXAFSSet, MPXANESSet
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

# Optional performance optimization
try:
    import numba
    NUMBA_AVAILABLE = True
    
    @numba.jit(nopython=True, cache=True)
    def _fast_average_chi(chi_arrays: np.ndarray) -> np.ndarray:
        """Fast averaging for trajectory processing using numba."""
        return np.mean(chi_arrays, axis=0)
    
    @numba.jit(nopython=True, cache=True)
    def _fast_std_chi(chi_arrays: np.ndarray) -> np.ndarray:
        """Fast standard deviation calculation using numba."""
        return np.std(chi_arrays, axis=0)
except ImportError:
    NUMBA_AVAILABLE = False
    def _fast_average_chi(chi_arrays: np.ndarray) -> np.ndarray:
        """Fallback averaging without numba."""
        return np.mean(chi_arrays, axis=0)
    
    def _fast_std_chi(chi_arrays: np.ndarray) -> np.ndarray:
        """Fallback standard deviation without numba."""
        return np.std(chi_arrays, axis=0)

# YAML support for parameter sets
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Constants for supported spectrum types
SPECTRUM_TYPES = {
    "EXAFS", "XANES", "DANES", "XMCD", "ELNES", "EXELFS", "FPRIME", "NRIXS", "XES"
}

# Configuration presets for different use cases
PRESETS = {
    "quick": {
        "spectrum_type": "EXAFS",
        "edge": "K",
        "radius": 8.0,
        "kmin": 2,
        "kmax": 12,
        "dk": 1,
        "kweight": 2,
    },
    "publication": {
        "spectrum_type": "EXAFS",
        "edge": "K",
        "radius": 12.0,
        "kmin": 3,
        "kmax": 18,
        "dk": 4,
        "kweight": 2,
        "user_tag_settings": {"NLEG": "8", "CRITERIA": "curved"},
    },
    "xanes": {
        "spectrum_type": "XANES",
        "edge": "K",
        "radius": 8.0,
        "user_tag_settings": {"SCF": "7.0", "FMS": "9.0"},
    },
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
    """Configuration for FEFF calculations and processing.
    
    This class encapsulates all parameters needed for FEFF calculations and EXAFS processing.
    It provides a clean way to manage configuration without overwhelming method signatures.
    """
    # FEFF calculation parameters
    spectrum_type: SpectrumType = SpectrumType.EXAFS
    edge: EdgeType = EdgeType.K
    radius: float = 10.0
    method: Literal["auto", "larixite", "pymatgen"] = "auto"
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
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string inputs to enums if needed
        if isinstance(self.spectrum_type, str):
            try:
                self.spectrum_type = SpectrumType(self.spectrum_type.upper())
            except ValueError:
                valid_types = ", ".join([t.value for t in SpectrumType])
                raise ValueError(f"Invalid spectrum type: {self.spectrum_type}. Valid options: {valid_types}")
        
        if isinstance(self.edge, str):
            try:
                self.edge = EdgeType(self.edge.upper())
            except ValueError:
                valid_edges = ", ".join([e.value for e in EdgeType])
                raise ValueError(f"Invalid edge: {self.edge}. Valid options: {valid_edges}")
    
    @classmethod
    def from_preset(cls, preset_name: str) -> "FeffConfig":
        """Create configuration from a named preset."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {', '.join(PRESETS.keys())}")
        
        preset = PRESETS[preset_name].copy()
        user_tag_settings = preset.pop("user_tag_settings", {})
        return cls(
            user_tag_settings=user_tag_settings,
            **{k: v for k, v in preset.items() if k in cls.__dataclass_fields__}
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FeffConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for configuration files. Install with: pip install pyyaml")
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file {yaml_path} not found")
        
        with open(yaml_path) as f:
            yaml_params = yaml.safe_load(f)
        
        # Convert YAML parameters to FeffConfig
        user_tag_settings = yaml_params.pop("user_tag_settings", {})
        return cls(
            user_tag_settings=user_tag_settings,
            **{k: v for k, v in yaml_params.items() if k in cls.__dataclass_fields__}
        )
    
    def to_yaml(self, yaml_path: Path, description: Optional[str] = None):
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for configuration files. Install with: pip install pyyaml")
        
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary for YAML serialization
        config_dict = {
            "spectrum_type": self.spectrum_type.value,
            "edge": self.edge.value,
            "radius": self.radius,
            "kweight": self.kweight,
            "window": self.window,
            "dk": self.dk,
            "kmin": self.kmin,
            "kmax": self.kmax,
            "method": self.method,
            "user_tag_settings": self.user_tag_settings
        }
        
        with open(yaml_path, "w") as f:
            if description:
                f.write(f"# {description}\n")
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

@dataclass
class ProcessingResult:
    """Result of an EXAFS processing operation.
    
    This class provides a clean interface to access processing results without
    having to understand the underlying Larch Group structure.
    """
    exafs_group: Group
    plot_paths: Tuple[Path, Path]  # (pdf_path, svg_path)
    processing_mode: ProcessingMode
    nframes: int = 1
    
    @property
    def k(self) -> np.ndarray:
        """Wave number array."""
        return self.exafs_group.k
    
    @property
    def chi(self) -> np.ndarray:
        """EXAFS function χ(k)."""
        return self.exafs_group.chi
    
    @property
    def r(self) -> np.ndarray:
        """Real space distance array from Fourier transform."""
        return self.exafs_group.r
    
    @property
    def chir_mag(self) -> np.ndarray:
        """Magnitude of Fourier transform χ(R)."""
        return self.exafs_group.chir_mag
    
    @property
    def is_averaged(self) -> bool:
        """Whether this result represents an average of multiple frames."""
        return self.processing_mode == ProcessingMode.AVERAGE or self.nframes > 1
    
    def save(self, output_dir: Path):
        """Save processing results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save EXAFS data
        np.savetxt(
            output_dir / "exafs_data.dat",
            np.column_stack((self.k, self.chi)),
            header="k  chi"
        )
        
        # Save Fourier transform data
        np.savetxt(
            output_dir / "ft_data.dat",
            np.column_stack((self.r, self.chir_mag)),
            header="R  chir_mag"
        )
        
        return output_dir

class LarchWrapper:
    """Simplified wrapper for EXAFS processing with Larch.
    
    This class provides a clean, intuitive API for EXAFS processing with:
    - Unified processing method for all use cases
    - Dedicated configuration object
    - Structured results
    - Clear error handling
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the LarchWrapper.
        
        Args:
            verbose: Whether to show detailed logging
        """
        self.logger = self._setup_logger(verbose)
        self._temp_files = []
    
    def _setup_logger(self, verbose: bool) -> logging.Logger:
        """Set up logging for the wrapper."""
        logger = logging.getLogger("larch_wrapper")
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Clear any existing handlers to prevent duplicates
        logger.handlers = []
        
        if verbose:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_files()
    
    def __del__(self):
        """Clean up temporary files."""
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Remove any temporary files created during processing and free memory."""
        # Remove temporary files - iterate over copy to avoid modification during iteration
        for temp_file in self._temp_files[:]:  # Create a copy of the list
            try:
                if temp_file.exists():
                    temp_file.unlink()
                self._temp_files.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not remove temp file {temp_file}: {e}")
        
        # Force garbage collection
        gc.collect()
    
    def _validate_structure_file(self, structure_path: Path):
        """Validate structure file exists and is readable."""
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file {structure_path} not found")
        if not structure_path.is_file():
            raise ValueError(f"{structure_path} is not a file")
        if structure_path.stat().st_size == 0:
            raise ValueError(f"Structure file {structure_path} is empty")
    
    def _validate_absorber(self, absorber: str) -> Union[str, int]:
        """Validate and normalize absorber specification."""
        # Try to convert to integer (site index)
        try:
            site_index = int(absorber)
            self.logger.debug(f"Using absorber as site index: {site_index}")
            return site_index
        except ValueError:
            # Keep as element symbol
            if len(absorber) > 2:
                self.logger.warning(
                    f"Absorber '{absorber}' may not be a valid element symbol"
                )
            return absorber.capitalize()
    
    def _create_temp_cif(self, structure_path: Path) -> Path:
        """Create a temporary CIF file from any ASE-readable structure."""
        structure_path = Path(structure_path).resolve()
        if structure_path.suffix.lower() == ".cif":
            return structure_path
        
        try:
            atoms = ase_read(str(structure_path))
            temp_fd, temp_path = tempfile.mkstemp(suffix=".cif", prefix="larch_temp_")
            os.close(temp_fd)
            temp_cif = Path(temp_path).resolve()
            self._temp_files.append(temp_cif)
            ase_write(str(temp_cif), atoms, format="cif")
            return temp_cif
        except Exception as e:
            raise RuntimeError(f"Failed to convert structure to CIF: {e}") from e
    
    def _load_structure_with_pymatgen(self, structure_path: Path) -> Structure:
        """Load structure using pymatgen with comprehensive ASE fallback for format conversion."""
        if not PYMATGEN_AVAILABLE:
            raise ImportError("pymatgen is required for this functionality")
        
        try:
            # Try direct pymatgen loading first
            return Structure.from_file(str(structure_path))
        except Exception as e:
            self.logger.debug(
                f"Direct pymatgen loading failed ({e}), trying ASE conversion..."
            )
            try:
                # Fallback 1: convert to CIF with ASE, then load with pymatgen
                cif_path = self._create_temp_cif(structure_path)
                return Structure.from_file(str(cif_path))
            except Exception as e2:
                self.logger.debug(
                    f"CIF conversion failed ({e2}), trying direct ASE-to-pymatgen..."
                )
                try:
                    # Fallback 2: Use ASE to pymatgen adaptor directly
                    from pymatgen.io.ase import AseAtomsAdaptor
                    atoms = ase_read(str(structure_path))
                    adaptor = AseAtomsAdaptor()
                    return adaptor.get_structure(atoms)
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to load structure with pymatgen using all methods:\n"
                        f"  Direct: {e}\n"
                        f"  CIF conversion: {e2}\n"
                        f"  ASE adaptor: {e3}"
                    ) from e3
    
    def _generate_feff_input(
        self,
        structure: Union[Path, str, Atoms],
        absorber: Union[str, int],
        output_dir: Path,
        config: FeffConfig
    ) -> Path:
        """Generate FEFF input file based on configuration."""
        # Handle direct ASE atoms input
        if isinstance(structure, Atoms):
            # Create temporary CIF from atoms
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                ase_write(tmp.name, structure, format="cif")
                structure_path = Path(tmp.name)
                self._temp_files.append(structure_path)
        else:
            structure_path = Path(structure).resolve()
        
        # Validate inputs
        self._validate_structure_file(structure_path)
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-select method based on requirements and availability
        if config.method == "auto":
            if config.spectrum_type in [SpectrumType.XANES, SpectrumType.DANES, SpectrumType.XMCD] and PYMATGEN_AVAILABLE:
                config.method = "pymatgen"
                self.logger.debug("Auto-selected pymatgen method for advanced spectrum type")
            else:
                config.method = "larixite"
                self.logger.debug("Auto-selected larixite method for EXAFS spectrum")
        
        # Generate FEFF input based on selected method
        if config.method == "pymatgen":
            if not PYMATGEN_AVAILABLE:
                raise ImportError("pymatgen is required for this method but not installed")
            return self._generate_feff_with_pymatgen(
                structure_path,
                absorber,
                output_dir,
                config
            )
        else:  # larixite
            return self._generate_feff_with_larixite(
                structure_path, absorber, output_dir, config
            )
    
    def _generate_feff_with_larixite(
        self,
        structure_path: Path,
        absorber: Union[str, int],
        output_dir: Path,
        config: FeffConfig
    ) -> Path:
        """Generate FEFF input using larixite (CIF-based)."""
        # Convert to CIF if needed
        if structure_path.suffix.lower() != ".cif":
            self.logger.info(f"Converting {structure_path} to CIF format...")
            cif_path = self._create_temp_cif(structure_path)
        else:
            cif_path = structure_path
        
        # Generate FEFF input
        inp = cif2feffinp(cif_path, absorber)
        input_file_path = output_dir / "feff.inp"
        input_file_path.write_text(inp)
        return input_file_path
    
    def _generate_feff_with_pymatgen(
        self,
        structure_path: Path,
        absorber: Union[str, int],
        output_dir: Path,
        config: FeffConfig
    ) -> Path:
        """Generate FEFF input using pymatgen."""
        structure = self._load_structure_with_pymatgen(structure_path)
        
        # Remove COREHOLE if present in user_tag_settings (not supported in FEFF8L)
        user_tag_settings = config.user_tag_settings.copy()
        if "COREHOLE" in user_tag_settings:
            self.logger.warning("COREHOLE setting is not supported in FEFF8L, removing it")
            del user_tag_settings["COREHOLE"]
        
        # Select appropriate FEFF input set
        if config.spectrum_type == SpectrumType.EXAFS:
            feff_set = MPEXAFSSet(
                absorbing_atom=absorber,
                structure=structure,
                edge=config.edge.value,
                radius=config.radius,
                user_tag_settings=user_tag_settings or {},
            )
        elif config.spectrum_type == SpectrumType.XANES:
            feff_set = MPXANESSet(
                absorbing_atom=absorber,
                structure=structure,
                edge=config.edge.value,
                radius=config.radius,
                user_tag_settings=user_tag_settings or {},
            )
        else:
            raise ValueError(f"Unsupported spectrum type: {config.spectrum_type}")
        
        feff_set.write_input(str(output_dir))
        return output_dir / "feff.inp"
    
    def _run_feff(self, feff_dir: Path, config: FeffConfig) -> bool:
        """Run FEFF calculation with stdout logged safely."""
        feff_dir = Path(feff_dir).resolve()
        input_path = feff_dir / "feff.inp"
        log_path = feff_dir / "feff.log"
        
        if not input_path.exists():
            raise FileNotFoundError(f"FEFF input file {input_path} not found")
        
        # Store original stdout
        original_stdout = sys.stdout
        
        try:
            self.logger.info(f"Starting FEFF calculation (output logged to {log_path})")
            
            # Simple text-only wrapper for logging
            with open(log_path, 'w', encoding='utf-8', errors='replace') as log_file:
                class TextOnlyLogger:
                    def __init__(self):
                        self.encoding = 'utf-8'
                        self.errors = 'replace'
                    
                    def write(self, text):
                        if isinstance(text, bytes):
                            text = text.decode('utf-8', errors='replace')
                        # Filter out problematic control characters
                        clean_text = ''.join(c if ord(c) >= 32 or c in '\n\t\r' else '?' for c in text)
                        log_file.write(clean_text)
                        log_file.flush()
                    
                    def flush(self):
                        log_file.flush()
                
                # Redirect stdout to our safe logger
                sys.stdout = TextOnlyLogger()
                
                # Run FEFF with verbose output (now safely captured)
                result = feff8l(
                    folder=str(feff_dir),
                    feffinp="feff.inp",
                    verbose=True
                )
            
            # Check if expected output files were created
            chi_file = feff_dir / "chi.dat"
            if not chi_file.exists():
                self.logger.error(f"FEFF output missing: {chi_file}")
                return False
            
            self.logger.info("FEFF calculation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"FEFF calculation failed: {e}")
            return False
        finally:
            # Always restore original stdout
            sys.stdout = original_stdout

    @staticmethod
    def _process_frame_worker(frame_data: tuple) -> tuple[np.ndarray | None, np.ndarray | None, int, str | None]:
        """Worker function for processing a single trajectory frame in parallel."""
        try:
            frame_index, atoms, absorber, output_base, config_dict = frame_data
            
            # Import required modules in worker process
            import sys
            import tempfile
            from pathlib import Path
            from ase.io import write as ase_write
            
            # Reconstruct config from dictionary
            config = FeffConfig(**config_dict)
            frame_output = Path(output_base) / f"frame_{frame_index:04d}"
            frame_output.mkdir(parents=True, exist_ok=True)
            
            # Save structure to file in the frame directory
            struct_path = frame_output / "structure.cif"
            ase_write(str(struct_path), atoms, format="cif")
            
            # Create a new LarchWrapper instance for this process
            wrapper = LarchWrapper(verbose=False)  # Reduce verbosity in parallel workers
            
            # Set up proper encoding for this worker process
            if sys.stdout.encoding is None or sys.stdout.encoding.lower() == "none":
                class WorkerEncodingWrapper:
                    def __init__(self, stream):
                        self.stream = stream
                        self.encoding = "utf-8"
                        self.errors = "replace"
                    def __getattr__(self, name):
                        return getattr(self.stream, name)
                    def write(self, text):
                        if isinstance(text, bytes):
                            text = text.decode(self.encoding, self.errors)
                        return self.stream.write(text)
                sys.stdout = WorkerEncodingWrapper(sys.stdout)
            
            try:
                frame_result = wrapper._process_single_frame(
                    struct_path, absorber, frame_output, config, show_plot=False
                )
                return frame_result.chi, frame_result.k, frame_index, None
            except Exception as e:
                return None, None, frame_index, str(e)
            finally:
                wrapper.cleanup_temp_files()
        except Exception as e:
            return None, None, frame_index, str(e)
        
    def _process_feff_output(self, feff_dir: Path, config: FeffConfig) -> Group:
        """Process FEFF output with Fourier transform using configuration."""
        chi_file = feff_dir / "chi.dat"
        if not chi_file.exists():
            raise FileNotFoundError(f"FEFF output file {chi_file} not found")
        feff_data = read_ascii(str(chi_file))
        
        # Create larch Group and perform FT with config parameters
        g = Group()
        g.k = feff_data.k
        g.chi = feff_data.chi
        
        xftf(
            g,
            kweight=config.kweight,
            window=config.window,
            dk=config.dk,
            kmin=config.kmin,
            kmax=config.kmax
        )
        
        return g
    
    def _plot_results(
        self,
        exafs_group: Group,
        output_dir: Path,
        filename_base: str,
        show_plot: bool = False
    ) -> Tuple[Path, Path]:
        """Generate plots for EXAFS results."""
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(exafs_group.r, exafs_group.chir_mag, linewidth=2)
        ax.set_xlabel(r"R ($\AA$)", fontsize=12)
        ax.set_ylabel(r"$|\chi(R)|$", fontsize=12)
        ax.set_title("Fourier Transform of EXAFS", fontsize=14)
        ax.grid(True, alpha=0.3)

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

    def generate_inputs(
        self,
        structure: Union[Path, str, Atoms],
        absorber: str,
        output_dir: Path,
        config: Optional[FeffConfig] = None,
        trajectory: bool = False
    ) -> Dict[str, Any]:
        """Generate FEFF inputs for a structure or trajectory.
        
        This method creates FEFF input files and returns information about them.
        It's the first step of the full processing pipeline and can be used standalone.
        
        Args:
            structure: Path to structure file, or ASE Atoms object
            absorber: Absorbing atom symbol or index
            output_dir: Output directory for results
            config: Processing configuration (uses defaults if None)
            trajectory: Whether input is a trajectory (multiple frames)
            
        Returns:
            Dictionary with input generation results including:
            - input_files: List of generated input file paths
            - output_dir: Path to output directory
            - nframes: Number of frames processed
            - processing_mode: ProcessingMode enum value
            
        Raises:
            FileNotFoundError: If structure file doesn't exist
            ValueError: If parameters are invalid
            RuntimeError: If input generation fails
        """
        # Setup configuration
        config = config or FeffConfig()
        output_dir = Path(output_dir).resolve()
        
        # Validate and normalize absorber
        normalized_absorber = self._validate_absorber(absorber)
        
        # Determine processing mode
        processing_mode = ProcessingMode.TRAJECTORY if trajectory else ProcessingMode.SINGLE_FRAME
        self.logger.info(f"Generating FEFF inputs for {processing_mode.value} processing of {absorber}")
        
        input_files = []
        
        try:
            if not trajectory:
                # Single frame input generation
                input_file = self._generate_feff_input(
                    structure, normalized_absorber, output_dir, config
                )
                input_files.append(input_file)
                nframes = 1
                
            else:
                # Trajectory input generation
                trajectory_path = Path(structure).resolve()
                self._validate_structure_file(trajectory_path)
                
                # Read trajectory frames
                structures = ase_read(str(trajectory_path), index=f"::1")
                if not isinstance(structures, list):
                    structures = [structures]
                    
                # Apply sample interval
                if config.sample_interval > 1:
                    structures = structures[::config.sample_interval]
                    
                nframes = len(structures)
                self.logger.info(f"Generating inputs for {nframes} frames with sample interval {config.sample_interval}")
                
                # Create frame-specific directories and inputs
                for i, atoms in enumerate(structures):
                    frame_dir = output_dir / f"frame_{i:04d}"
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save structure to file in the frame directory
                    struct_path = frame_dir / "structure.cif"
                    ase_write(str(struct_path), atoms, format="cif")
                    
                    # Generate FEFF input for this frame
                    input_file = self._generate_feff_input(
                        struct_path, normalized_absorber, frame_dir, config
                    )
                    input_files.append(input_file)
            
            self.logger.info(f"Successfully generated {len(input_files)} FEFF input file(s)")
            
            return {
                'input_files': input_files,
                'output_dir': output_dir,
                'nframes': nframes,
                'processing_mode': processing_mode,
                'config': config
            }
            
        except Exception as e:
            self.logger.error(f"Input generation failed: {str(e)}")
            raise

    def process(
        self,
        structure: Union[Path, str, Atoms],
        absorber: str,
        output_dir: Path,
        config: Optional[FeffConfig] = None,
        trajectory: bool = False,
        show_plot: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ProcessingResult:
        """Process a structure or trajectory for EXAFS analysis.

        This is the main entry point for all processing needs. It handles:
        - Single structure processing
        - Trajectory processing (with averaging)
        - Configuration management
        - Result organization

        Args:
            structure: Path to structure file, or ASE Atoms object
            absorber: Absorbing atom symbol or index
            output_dir: Output directory for results
            config: Processing configuration (uses defaults if None)
            trajectory: Whether input is a trajectory (multiple frames)
            show_plot: Whether to display plots interactively
            progress_callback: Optional callback for progress updates (completed, total, description)

        Returns:
            ProcessingResult containing all relevant data and plot paths

        Raises:
            FileNotFoundError: If structure file doesn't exist
            ValueError: If parameters are invalid
            RuntimeError: If processing fails
        """
        # First, generate inputs (this is always the first step)
        input_info = self.generate_inputs(
            structure=structure,
            absorber=absorber,
            output_dir=output_dir,
            config=config,
            trajectory=trajectory
        )
        
        # Extract information from input generation
        config = input_info['config']
        processing_mode = input_info['processing_mode']
        nframes = input_info['nframes']
        
        try:
            if not trajectory:
                # Process single frame
                return self._process_single_frame_from_inputs(
                    input_info, show_plot
                )
            else:
                # Process trajectory
                return self._process_trajectory_from_inputs(
                    input_info, show_plot, progress_callback
                )

        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise

    def _process_single_frame_from_inputs(
        self,
        input_info: Dict[str, Any],
        show_plot: bool
    ) -> ProcessingResult:
        """Process a single structure frame from pre-generated inputs."""
        output_dir = input_info['output_dir']
        config = input_info['config']
        
        # Run FEFF
        if not self._run_feff(output_dir, config):
            raise RuntimeError("FEFF calculation failed")

        # Process output
        exafs_group = self._process_feff_output(output_dir, config)

        # Generate plots
        plot_paths = self._plot_results(
            exafs_group,
            output_dir,
            f"EXAFS_FT",
            show_plot=show_plot
        )

        return ProcessingResult(
            exafs_group=exafs_group,
            plot_paths=plot_paths,
            processing_mode=ProcessingMode.SINGLE_FRAME,
            nframes=1
        )

    def _process_trajectory_from_inputs(
        self,
        input_info: Dict[str, Any],
        show_plot: bool,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ProcessingResult:
        """Process a trajectory from pre-generated inputs."""
        output_base = input_info['output_dir']
        config = input_info['config']
        nframes = input_info['nframes']
        input_files = input_info['input_files']
        
        self.logger.info(f"Processing {nframes} pre-generated frames")

        # Setup progress tracking
        completed = 0
        
        def update_progress(description: str):
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed, nframes, description)

        # Process frames
        chi_list = []
        k_ref = None

        if config.parallel and nframes > 1:
            # Parallel processing with pre-generated inputs
            n_workers = config.n_workers or min(mp.cpu_count(), nframes)
            self.logger.info(f"Using parallel processing with {n_workers} workers")
            
            # Process frames in parallel using existing directories
            frame_dirs = [output_base / f"frame_{i:04d}" for i in range(nframes)]
            
            try:
                # Use multiprocessing to run FEFF and process outputs
                with mp.Pool(processes=n_workers) as pool:
                    # Create tasks for parallel FEFF runs
                    feff_tasks = [(frame_dir, config) for frame_dir in frame_dirs]
                    feff_results = pool.starmap(self._run_feff_and_process, feff_tasks)
                
                # Process results
                for i, (chi, k, success, error) in enumerate(feff_results):
                    if not success:
                        self.logger.error(f"Frame {i} processing failed: {error}")
                        update_progress(f"Failed frame {i+1}/{nframes}")
                        continue
                    
                    # Initialize reference k-grid
                    if k_ref is None:
                        k_ref = k.copy()
                    
                    # Check k-grid consistency
                    if len(k) != len(k_ref) or not np.allclose(k, k_ref, atol=1e-4):
                        # Interpolate to reference grid
                        chi = np.interp(k_ref, k, chi, left=0, right=0)
                    
                    chi_list.append(chi)
                    update_progress(f"Processed frame {i+1}/{nframes}")
                    
            except Exception as e:
                self.logger.error(f"Parallel processing failed: {e}")
                self.logger.info("Falling back to sequential processing...")
                config.parallel = False  # Disable parallel processing for fallback
        
        if not config.parallel or nframes == 1:
            # Sequential processing
            for i in range(nframes):
                frame_dir = output_base / f"frame_{i:04d}"
                
                try:
                    # Run FEFF and process output
                    if not self._run_feff(frame_dir, config):
                        raise RuntimeError(f"FEFF calculation failed for frame {i}")
                    
                    exafs_group = self._process_feff_output(frame_dir, config)
                    
                    # Store results
                    chi = exafs_group.chi
                    k = exafs_group.k
                    
                    # Initialize reference k-grid
                    if k_ref is None:
                        k_ref = k.copy()
                    
                    # Check k-grid consistency
                    if len(k) != len(k_ref) or not np.allclose(k, k_ref, atol=1e-4):
                        # Interpolate to reference grid
                        chi = np.interp(k_ref, k, chi, left=0, right=0)
                        k = k_ref.copy()
                    
                    chi_list.append(chi)
                    update_progress(f"Processed frame {i+1}/{nframes}")
                except Exception as e:
                    self.logger.error(f"Frame {i} processing failed: {str(e)}")
                    update_progress(f"Failed frame {i+1}/{nframes}")
                    continue
        
        if not chi_list:
            raise RuntimeError("No valid frames processed")
        
        # Create result group
        result = Group()
        result.k = k_ref
        result.chi = _fast_average_chi(np.array(chi_list))
        result.chi_std = _fast_std_chi(np.array(chi_list))
        result.nframes = nframes
        
        # Apply Fourier transform
        xftf(
            result,
            kweight=config.kweight,
            window=config.window,
            dk=config.dk,
            kmin=config.kmin,
            kmax=config.kmax
        )
        
        # Generate plots for averaged data
        plot_paths = self._plot_results(
            result,
            output_base,
            "trajectory_avg_EXAFS_FT",
            show_plot=show_plot
        )
        
        return ProcessingResult(
            exafs_group=result,
            plot_paths=plot_paths,
            processing_mode=ProcessingMode.AVERAGE,
            nframes=nframes
        )

    @staticmethod
    def _run_feff_and_process(frame_dir: Path, config: FeffConfig) -> tuple:
        """Helper method for parallel FEFF execution and processing."""
        try:
            # Create a new wrapper instance for this process
            wrapper = LarchWrapper(verbose=False)
            
            # Run FEFF
            if not wrapper._run_feff(frame_dir, config):
                return None, None, False, "FEFF calculation failed"
            
            # Process output
            exafs_group = wrapper._process_feff_output(frame_dir, config)
            return exafs_group.chi, exafs_group.k, True, None
            
        except Exception as e:
            return None, None, False, str(e)
        finally:
            if 'wrapper' in locals():
                wrapper.cleanup_temp_files()

    def _process_single_frame(
        self,
        structure: Union[Path, str, Atoms],
        absorber: Union[str, int],
        output_dir: Path,
        config: FeffConfig,
        show_plot: bool
    ) -> ProcessingResult:
        """Process a single structure frame."""
        # Generate FEFF input
        feff_input = self._generate_feff_input(structure, absorber, output_dir, config)

        # Run FEFF
        if not self._run_feff(output_dir, config):
            raise RuntimeError("FEFF calculation failed")

        # Process output
        exafs_group = self._process_feff_output(output_dir, config)

        # Generate plots
        plot_paths = self._plot_results(
            exafs_group,
            output_dir,
            f"EXAFS_FT",
            show_plot=show_plot
        )

        return ProcessingResult(
            exafs_group=exafs_group,
            plot_paths=plot_paths,
            processing_mode=ProcessingMode.SINGLE_FRAME,
            nframes=1
        )


    def _process_trajectory(
        self,
        trajectory: Union[Path, str],
        absorber: Union[str, int],
        output_base: Path,
        config: FeffConfig,
        show_plot: bool,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> ProcessingResult:
        """Process a trajectory and return averaged results with proper frame isolation."""
        trajectory_path = Path(trajectory).resolve()
        self._validate_structure_file(trajectory_path)

        # Read trajectory frames
        structures = ase_read(str(trajectory_path), index=f"::1")
        if not isinstance(structures, list):
            structures = [structures]  # Single frame case
        
        # Apply sample interval
        if config.sample_interval > 1:
            structures = structures[::config.sample_interval]

        n_frames = len(structures)
        self.logger.info(f"Processing {n_frames} frames with sample interval {config.sample_interval}")

        # Create output base directory
        output_base = Path(output_base).resolve()
        output_base.mkdir(parents=True, exist_ok=True)

        # Setup progress tracking
        total_frames = len(structures)
        completed = 0
        
        def update_progress(description: str):
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed, total_frames, description)

        # Create frame-specific directories first
        frame_dirs = []
        for i in range(n_frames):
            frame_dir = output_base / f"frame_{i:04d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_dirs.append(frame_dir)

        # Process frames
        chi_list = []
        k_ref = None

        if config.parallel and n_frames > 1:
            # Parallel processing
            n_workers = config.n_workers or min(mp.cpu_count(), n_frames)
            self.logger.info(f"Using parallel processing with {n_workers} workers")
            
            # Convert config to dictionary for serialization
            config_dict = {
                'spectrum_type': config.spectrum_type.value,
                'edge': config.edge.value,
                'radius': config.radius,
                'method': config.method,
                'user_tag_settings': config.user_tag_settings,
                'kweight': config.kweight,
                'window': config.window,
                'dk': config.dk,
                'kmin': config.kmin,
                'kmax': config.kmax,
            }
            
            # Prepare frame data for parallel processing
            frame_data = [
                (i, atoms, absorber, str(output_base), config_dict)
                for i, atoms in enumerate(structures)
            ]
            
            # Process frames in parallel
            try:
                with mp.Pool(processes=n_workers) as pool:
                    results = pool.map(self._process_frame_worker, frame_data)
                
                # Process results
                for chi, k, frame_index, error in results:
                    if error is not None:
                        self.logger.error(f"Frame {frame_index} processing failed: {error}")
                        update_progress(f"Failed frame {frame_index+1}/{total_frames}")
                        continue
                    
                    if chi is None or k is None:
                        self.logger.error(f"Frame {frame_index} returned None results")
                        update_progress(f"Failed frame {frame_index+1}/{total_frames}")
                        continue
                    
                    # Initialize reference k-grid
                    if k_ref is None:
                        k_ref = k.copy()
                    
                    # Check k-grid consistency
                    if len(k) != len(k_ref) or not np.allclose(k, k_ref, atol=1e-4):
                        # Interpolate to reference grid
                        chi = np.interp(k_ref, k, chi, left=0, right=0)
                    
                    chi_list.append(chi)
                    update_progress(f"Processed frame {frame_index+1}/{total_frames}")
                    
            except Exception as e:
                self.logger.error(f"Parallel processing failed: {e}")
                self.logger.info("Falling back to sequential processing...")
                config.parallel = False  # Disable parallel processing for fallback
        
        if not config.parallel or n_frames == 1:
            # Sequential processing (original implementation)
            for i, atoms in enumerate(structures):
                frame_output = frame_dirs[i]
                
                # Save structure to file in the frame directory
                struct_path = frame_output / "structure.cif"
                ase_write(str(struct_path), atoms, format="cif")
                
                # Process frame using its dedicated directory
                try:
                    frame_result = self._process_single_frame(
                        struct_path, absorber, frame_output, config, show_plot=False
                    )
                    
                    # Store results
                    chi = frame_result.chi
                    k = frame_result.k
                    
                    # Initialize reference k-grid
                    if k_ref is None:
                        k_ref = k.copy()
                    
                    # Check k-grid consistency
                    if len(k) != len(k_ref) or not np.allclose(k, k_ref, atol=1e-4):
                        # Interpolate to reference grid
                        chi = np.interp(k_ref, k, chi, left=0, right=0)
                        k = k_ref.copy()
                    
                    chi_list.append(chi)
                    update_progress(f"Processed frame {i+1}/{total_frames}")
                except Exception as e:
                    self.logger.error(f"Frame {i} processing failed: {str(e)}")
                    update_progress(f"Failed frame {i+1}/{total_frames}")
                    continue
        
        if not chi_list:
            raise RuntimeError("No valid frames processed")
        
        # Create result group
        result = Group()
        result.k = k_ref
        result.chi = _fast_average_chi(np.array(chi_list))
        result.chi_std = _fast_std_chi(np.array(chi_list))
        result.nframes = n_frames
        
        # Apply Fourier transform
        xftf(
            result,
            kweight=config.kweight,
            window=config.window,
            dk=config.dk,
            kmin=config.kmin,
            kmax=config.kmax
        )
        
        # Generate plots for averaged data
        plot_paths = self._plot_results(
            result,
            output_base,
            "trajectory_avg_EXAFS_FT",
            show_plot=show_plot
        )
        
        return ProcessingResult(
            exafs_group=result,
            plot_paths=plot_paths,
            processing_mode=ProcessingMode.AVERAGE,
            nframes=n_frames
        )


    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics about the installation."""
        diagnostics = {
            "system_info": {
                "platform": os.name,
                "python_version": sys.version.split()[0],
            },
            "dependencies": {
                "ase_available": True,  # Required dependency
                "pymatgen_available": PYMATGEN_AVAILABLE,
                "larch_available": True,  # If we got here, larch is available
                "numba_available": NUMBA_AVAILABLE,
            },
            "feff_info": {
                "version": self.get_feff_version(),
                "executables": self.get_available_feff_executables(),
            },
            "capabilities": {
                "supported_spectrum_types": [t.value for t in SpectrumType],
                "supported_edges": [e.value for e in EdgeType],
                "default_parameters": {
                    "spectrum_type": SpectrumType.EXAFS.value,
                    "edge": EdgeType.K.value,
                    "radius": 10.0,
                    "kweight": 2,
                    "window": "hanning",
                    "dk": 1.0,
                    "kmin": 2.0,
                    "kmax": 14.0,
                },
                "available_presets": list(PRESETS.keys()),
            }
        }
        
        # Add warnings and recommendations
        warnings = []
        recommendations = []
        
        if not PYMATGEN_AVAILABLE:
            warnings.append(
                "Pymatgen not available - advanced spectrum types (XANES, etc.) may not work optimally"
            )
            recommendations.append("Install pymatgen: pip install pymatgen")
        
        if not NUMBA_AVAILABLE:
            recommendations.append(
                "Install numba for faster trajectory processing: pip install numba"
            )
        
        if not diagnostics["feff_info"]["executables"]:
            warnings.append("No FEFF executables found - FEFF calculations will fail")
            recommendations.append(
                "Check larch installation and ensure FEFF binaries are available"
            )
        
        diagnostics["warnings"] = warnings
        diagnostics["recommendations"] = recommendations
        
        return diagnostics
    
    def print_diagnostics(self):
        """Print a formatted diagnostic report."""
        diagnostics = self.get_diagnostics()
        print("=" * 60)
        print("LARCH WRAPPER DIAGNOSTICS")
        print("=" * 60)
        print("\n🔧 SYSTEM INFO:")
        print(f"   Platform: {diagnostics['system_info']['platform']}")
        print(f"   Python: {diagnostics['system_info']['python_version']}")
        
        print("\n📦 DEPENDENCIES:")
        for dep, available in diagnostics["dependencies"].items():
            status = "✓" if available else "✗"
            print(f"   {status} {dep}: {available}")
        
        print("\n⚡ PERFORMANCE:")
        print(f"   CPU cores: {mp.cpu_count()}")
        print(f"   Parallel processing: ✓ Available")
        print(f"   Numba optimization: {'✓' if NUMBA_AVAILABLE else '✗'}")
        print(f"   Recommended workers: {min(mp.cpu_count(), 4)}")
        
        print("\n⚛️ FEFF INFO:")
        print(f"   Version: {diagnostics['feff_info']['version']}")
        print(f"   Executables found: {len(diagnostics['feff_info']['executables'])}")
        if diagnostics["feff_info"]["executables"]:
            print("   Available executables:")
            for exe in diagnostics["feff_info"]["executables"].keys():
                print(f"     • {exe}")
        
        print("\n⚙️ PRESETS AVAILABLE:")
        for preset in diagnostics["capabilities"]["available_presets"]:
            print(f"   • {preset}")
        
        if diagnostics["warnings"]:
            print("\n⚠️ WARNINGS:")
            for warning in diagnostics["warnings"]:
                print(f"   • {warning}")
        
        if diagnostics["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in diagnostics["recommendations"]:
                print(f"   • {rec}")
        
        print("\n✅ Diagnostics complete!")
        print("=" * 60)
    
    def get_feff_version(self) -> str:
        """Get FEFF version information using available executables."""
        # Try different FEFF executables in order of preference
        feff_executables = ["feff8l_rdinp", "feff6l", "feff8l", "feff"]
        for exe_name in feff_executables:
            try:
                exe_path = find_exe(exe_name)
                if exe_path:
                    return f"{exe_name} (found at {exe_path})"
            except Exception:
                continue
        return "FEFF (no executable found)"
    
    def get_available_feff_executables(self) -> Dict[str, str]:
        """Get a dictionary of available FEFF executables and their paths."""
        feff_executables = [
            "feff8l_rdinp", "feff8l_pot", "feff8l_xsph", "feff8l_pathfinder",
            "feff8l_genfmt", "feff8l_ff2x", "feff8l", "feff6l", "feff"
        ]
        available = {}
        for exe_name in feff_executables:
            try:
                exe_path = find_exe(exe_name)
                if exe_path:
                    available[exe_name] = str(exe_path)
            except Exception:
                pass
        return available