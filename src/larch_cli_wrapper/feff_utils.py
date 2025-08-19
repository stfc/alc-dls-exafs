"""FEFF input generation utilities - Fixed for consistent output between methods."""

import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

# Optional dependencies
try:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.feff.sets import MPEXAFSSet

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from larixite import cif2feffinp

    LARIXITE_AVAILABLE = True
except ImportError:
    LARIXITE_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Configuration presets using larixite defaults
PRESETS = {
    "quick": {
        "spectrum_type": "EXAFS",
        "edge": "K",
        "radius": 8.0,
        "kmin": 2,
        "kmax": 12,
        "kweight": 2,
        "window": "hanning",
        "dk": 1.0,
        "user_tag_settings": {},  # Use larixite defaults
    },
    "publication": {
        "spectrum_type": "EXAFS",
        "edge": "K",
        "radius": 12.0,
        "kmin": 3,
        "kmax": 18,
        "kweight": 2,
        "window": "hanning",
        "dk": 4.0,
        "user_tag_settings": {},  # Use larixite defaults
    },
}


class SpectrumType(str, Enum):
    """Enumeration of supported spectrum types."""

    EXAFS = "EXAFS"
    # XANES = "XANES"
    # DANES = "DANES"
    # XMCD = "XMCD"
    # ELNES = "ELNES"
    # EXELFS = "EXELFS"
    # FPRIME = "FPRIME"
    # NRIXS = "NRIXS"
    # XES = "XES"


class EdgeType(str, Enum):
    """Enumeration of supported absorption edges."""

    K = "K"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M4 = "M4"
    M5 = "M5"


# ================== CONFIGURATION ==================
@dataclass
class FeffConfig:
    """Configuration class for FEFF calculations."""

    spectrum_type: str = "EXAFS"
    edge: str = "K"
    radius: float = 8.0  # cluster size
    method: str = "auto"  # Updated default to auto
    user_tag_settings: dict[str, str] = field(
        default_factory=dict
    )  # Empty by default - use method defaults
    kweight: int = 2
    window: str = "hanning"
    dk: float = 1.0
    kmin: float = 2.0
    kmax: float = 14.0
    parallel: bool = False
    n_workers: int | None = None
    sample_interval: int = 1
    force_recalculate: bool = False

    def __post_init__(self):
        """Post-initialization validation of configuration parameters."""
        self._validate_spectrum_type()
        self._validate_energy_range()
        self._validate_fourier_params()
        self._validate_radius()
        self._validate_method()
        self._validate_n_workers()
        self._validate_sample_interval()
        # No automatic defaults - let each method use its own defaults

    def _validate_spectrum_type(self):
        if self.spectrum_type not in SpectrumType.__members__:
            raise ValueError(f"Invalid spectrum_type: {self.spectrum_type}")

    def _validate_energy_range(self):
        if self.kmin >= self.kmax:
            raise ValueError(f"kmin ({self.kmin}) must be less than kmax ({self.kmax})")
        if self.kmin < 0:
            raise ValueError(f"kmin must be positive, got {self.kmin}")

    def _validate_fourier_params(self):
        if self.dk <= 0:
            raise ValueError(f"dk must be positive, got {self.dk}")
        if not 1 <= self.kweight <= 3:
            logging.warning(f"Unusual kweight value: {self.kweight}")

    def _validate_radius(self):
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

    def _validate_method(self):
        valid_methods = ["auto", "larixite", "pymatgen"]
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method: {self.method}. Valid methods: {valid_methods}"
            )

    def _validate_n_workers(self):
        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError(f"Invalid n_workers: {self.n_workers}")

    def _validate_sample_interval(self):
        """Validate sample_interval parameter."""
        if self.sample_interval < 1:
            raise ValueError(
                f"sample_interval must be >= 1, got {self.sample_interval}"
            )

    @classmethod
    def from_preset(cls, preset_name: str) -> "FeffConfig":
        """Create configuration from a named preset."""
        if preset_name not in PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}"
            )
        preset = PRESETS[preset_name].copy()
        return cls(**preset)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FeffConfig":
        """Load configuration from a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for configuration files")
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        return cls(**params)

    def to_yaml(self, yaml_path: Path):
        """Save configuration to a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for configuration files")
        with open(yaml_path, "w") as f:
            yaml.dump(self.as_dict(), f)

    def as_dict(self) -> dict:
        """Convert configuration to dictionary format."""
        return {
            "spectrum_type": self.spectrum_type,
            "edge": self.edge,
            "radius": self.radius,
            "method": self.method,
            "user_tag_settings": self.user_tag_settings,
            "kweight": self.kweight,
            "window": self.window,
            "dk": self.dk,
            "kmin": self.kmin,
            "kmax": self.kmax,
            "parallel": self.parallel,
            "n_workers": self.n_workers,
            "sample_interval": self.sample_interval,
            "force_recalculate": self.force_recalculate,
        }

    def __repr_json__(self):
        """JSON representation for interactive environments."""
        return json.dumps(self.as_dict(), indent=4)


def validate_absorber(atoms: Atoms, absorber: str | int) -> str:
    """Validate and normalize absorber to element symbol."""
    if isinstance(absorber, int):
        if not 0 <= absorber < len(atoms):
            raise ValueError(f"Absorber index {absorber} out of range")
        return atoms.get_chemical_symbols()[absorber]
    else:
        absorber_element = str(absorber).capitalize()
        symbols = atoms.get_chemical_symbols()
        if absorber_element not in symbols:
            raise ValueError(
                f"Absorber element {absorber_element} not found in structure"
            )
        return absorber_element


def extract_larixite_defaults():
    """Extract default parameters from larixite template for consistency."""
    return {
        "S02": "1.0",
        "EXCHANGE": "0",  # Hedin-Lundqvist
        "CONTROL": "1 1 1 1 1 1",
        "PRINT": "1 0 0 0 0 3",
        "EXAFS": "20",
        "NLEG": "6",
        "SCF": "5.0",
    }


def generate_pymatgen_input(
    atoms: Atoms, absorber: str | int, output_dir: Path, config
) -> Path:
    """Generate FEFF input using pymatgen with larixite-compatible defaults."""
    if not PYMATGEN_AVAILABLE:
        raise ImportError("Pymatgen is required but not available")

    absorber_element = validate_absorber(atoms, absorber)
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(atoms)

    # Start with larixite defaults, then apply user overrides
    larixite_defaults = extract_larixite_defaults()
    user_settings = larixite_defaults.copy()
    user_settings.update(config.user_tag_settings)  # User settings override defaults

    # Apply radius setting
    user_settings["RPATH"] = str(config.radius)

    # Remove problematic settings for FEFF8L compatibility
    user_settings.pop("COREHOLE", None)
    if "_del" not in user_settings:
        user_settings["_del"] = []
    if "COREHOLE" not in user_settings["_del"]:
        user_settings["_del"].append("COREHOLE")

    # Create FEFF set with consistent parameters
    if config.spectrum_type == "EXAFS":
        feff_set = MPEXAFSSet(
            absorbing_atom=absorber_element,
            structure=structure,
            edge=config.edge,
            radius=config.radius,
            user_tag_settings=user_settings,
        )
    else:
        raise ValueError(f"Unsupported spectrum type: {config.spectrum_type}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feff_set.write_input(str(output_dir))

    return output_dir / "feff.inp"


def generate_larixite_input(
    atoms: Atoms, absorber: str | int, output_dir: Path, config
) -> Path:
    """Generate FEFF input using larixite with optional user overrides."""
    if not LARIXITE_AVAILABLE:
        raise ImportError("Larixite is required but not available")

    absorber_element = validate_absorber(atoms, absorber)

    with tempfile.NamedTemporaryFile(
        suffix=".cif", prefix="larch_temp_", delete=False
    ) as tmp:
        temp_cif = Path(tmp.name)

        try:
            ase_write(tmp.name, atoms, format="cif")
            inp = cif2feffinp(temp_cif, absorber=absorber_element, edge=config.edge)

            # Only modify larixite output if user has custom settings
            if config.user_tag_settings:
                # Post-process larixite output to apply custom settings
                inp_lines = inp.split("\n")
                modified_lines = []

                for line in inp_lines:
                    line_stripped = line.strip()

                    # Replace lines based on user settings
                    skip_line = False
                    for tag, value in config.user_tag_settings.items():
                        if line_stripped.startswith(tag):
                            modified_lines.append(f"{tag}       {value}")
                            skip_line = True
                            break

                    if not skip_line:
                        # Apply radius setting to RPATH
                        if line_stripped.startswith("RPATH"):
                            modified_lines.append(f"RPATH     {config.radius}")
                        else:
                            modified_lines.append(line)

                # Ensure all required user tags are present
                present_tags = {
                    line.split()[0]
                    for line in modified_lines
                    if line.strip() and not line.startswith("*")
                }
                for tag, value in config.user_tag_settings.items():
                    if tag not in present_tags:
                        # Insert after EDGE line
                        for i, line in enumerate(modified_lines):
                            if line.startswith("EDGE"):
                                modified_lines.insert(i + 1, f"{tag}       {value}")
                                break

                inp = "\n".join(modified_lines)
            else:
                # Just update RPATH for radius if different from default
                if config.radius != 10.0:  # larixite default
                    inp_lines = inp.split("\n")
                    for i, line in enumerate(inp_lines):
                        if line.strip().startswith("RPATH"):
                            inp_lines[i] = f"RPATH     {config.radius}"
                            break
                    inp = "\n".join(inp_lines)

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_file = output_dir / "feff.inp"
            input_file.write_text(inp)

            return input_file

        finally:
            if temp_cif.exists():
                temp_cif.unlink()


def generate_feff_input(
    atoms: Atoms, absorber: str | int, output_dir: Path, config
) -> Path:
    """Generate FEFF input using the specified method with improved error handling."""
    # Determine method
    if config.method == "auto":
        if LARIXITE_AVAILABLE:
            method = "larixite"
        elif PYMATGEN_AVAILABLE:
            method = "pymatgen"
        else:
            raise ValueError(
                "No FEFF input generation method available. "
                "Please install pymatgen or larixite."
            )
    else:
        method = config.method

    # Validate method availability
    if method == "pymatgen" and not PYMATGEN_AVAILABLE:
        raise ValueError(
            "Pymatgen method requested but pymatgen is not available. "
            "Install with: pip install pymatgen"
        )
    elif method == "larixite" and not LARIXITE_AVAILABLE:
        raise ValueError(
            "Larixite method requested but larixite is not available. "
            "Install with: pip install larixite"
        )

    # Generate input using appropriate method
    try:
        if method == "pymatgen":
            return generate_pymatgen_input(atoms, absorber, output_dir, config)
        elif method == "larixite":
            return generate_larixite_input(atoms, absorber, output_dir, config)
        else:
            raise ValueError(
                f"Unsupported method: {method}. Valid methods: auto, larixite, pymatgen"
            )
    except Exception as e:
        # Log detailed error information
        error_msg = f"FEFF input generation failed with method '{method}': {str(e)}"
        logging.error(error_msg)
        raise


def run_feff_calculation(feff_dir: Path, verbose: bool = False) -> bool:
    """Run FEFF calculation with proper error handling.

    Args:
        feff_dir: Directory containing feff.inp
        verbose: Whether to enable verbose output

    Returns:
        True if calculation succeeded, False otherwise
    """
    import os
    import sys

    from larch.xafs.feffrunner import feff8l

    feff_dir = Path(feff_dir)
    input_path = feff_dir / "feff.inp"
    log_path = feff_dir / "feff.log"

    if not input_path.exists():
        raise FileNotFoundError(f"FEFF input file {input_path} not found")

    # Initialize log file
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"FEFF calculation started at {datetime.now()}\n")
            log_file.write(f"Input file: {input_path}\n")
            log_file.write(f"Working directory: {feff_dir}\n")
            log_file.write("-" * 50 + "\n\n")
    except OSError as log_init_error:
        print(f"Warning: Could not initialize log file: {log_init_error}")

    # Store original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Fix encoding issue for subprocess calls
    if sys.stdout.encoding is None:
        os.environ["PYTHONIOENCODING"] = "utf-8"

    try:
        # Run FEFF calculation
        if verbose:
            print(f"Running FEFF calculation in {feff_dir}")
            result = feff8l(folder=str(feff_dir), feffinp="feff.inp", verbose=True)
        else:
            # Redirect output to log
            try:
                with open(
                    log_path, "a", encoding="utf-8", errors="replace"
                ) as log_file:
                    # Simple file redirection approach
                    sys.stdout = log_file
                    sys.stderr = log_file

                    log_file.write("FEFF8L Output:\n")
                    log_file.write("-" * 20 + "\n")
                    log_file.flush()

                    result = feff8l(
                        folder=str(feff_dir), feffinp="feff.inp", verbose=False
                    )

                    log_file.write(f"\n{'-' * 20}\n")
                    log_file.write(f"FEFF calculation result: {result}\n")

            except (OSError, RuntimeError) as feff_error:
                # If redirection fails, try without it
                result = feff8l(folder=str(feff_dir), feffinp="feff.inp", verbose=False)

                # Log the error
                try:
                    with open(
                        log_path, "a", encoding="utf-8", errors="replace"
                    ) as log_file:
                        log_file.write(f"Error during calculation: {feff_error}\n")
                        log_file.write(
                            f"FEFF result (without output capture): {result}\n"
                        )
                except OSError:
                    # If logging fails, we can still continue
                    pass

        # Check for output files
        chi_file = feff_dir / "chi.dat"
        success = chi_file.exists() and bool(result)

        # Final log entry with comprehensive information
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"\nCalculation completed at {datetime.now()}\n")
                log_file.write(f"Success: {success}\n")

                # List all files created by FEFF
                feff_files = list(feff_dir.glob("*"))
                output_files = [
                    f for f in feff_files if f.name not in ["feff.inp", "feff.log"]
                ]

                if output_files:
                    log_file.write(f"Output files created ({len(output_files)}):\n")
                    for f in sorted(output_files):
                        try:
                            size = f.stat().st_size
                            log_file.write(f"  {f.name} ({size} bytes)\n")
                        except OSError:
                            log_file.write(f"  {f.name}\n")
                else:
                    log_file.write("No output files found\n")

                if chi_file.exists():
                    # Also log some info about the chi.dat file
                    try:
                        with open(chi_file) as chi_f:
                            lines = chi_f.readlines()
                            log_file.write(f"chi.dat contains {len(lines)} lines\n")
                            if lines:
                                log_file.write(f"First line: {lines[0].strip()}\n")
                    except OSError:
                        # If we can't read chi.dat, continue
                        pass
                else:
                    log_file.write("Warning: chi.dat file not found\n")
        except OSError:
            # If logging fails, we can still return the result
            pass

        return success

    except (OSError, RuntimeError, ValueError) as e:
        # Ensure log file exists and log the error
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"\nERROR: {str(e)}\n")
                log_file.write(f"Exception type: {type(e).__name__}\n")
        except OSError:
            # If we can't write to log, at least print the error
            print(f"FEFF calculation failed: {e}")

        return False

    finally:
        # Always restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def read_feff_output(feff_dir: Path):
    """Read FEFF chi.dat output with fallback methods and improved error handling."""
    import numpy as np
    from larch.io import read_ascii

    chi_file = feff_dir / "chi.dat"
    if not chi_file.exists():
        raise FileNotFoundError(f"FEFF output {chi_file} not found")

    try:
        feff_data = read_ascii(str(chi_file))
        return feff_data.chi, feff_data.k
    except (OSError, ValueError, AttributeError) as read_error:
        try:
            data = np.loadtxt(str(chi_file), comments="#", usecols=(0, 1, 2))
            k = data[:, 0]
            chi = data[:, 2]
            return chi, k
        except (OSError, ValueError, IndexError) as fallback_error:
            error_msg = (
                f"Failed to read {chi_file}:\n"
                f"Primary error: {read_error}\n"
                f"Fallback error: {fallback_error}"
            )
            raise Exception(error_msg) from None
