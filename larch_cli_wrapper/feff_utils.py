"""FEFF input generation utilities - Fixed for consistent output between methods."""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.feff.sets import MPEXAFSSet

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Configuration presets using pymatgen defaults
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
        "user_tag_settings": {},  # Use pymatgen defaults
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
        "user_tag_settings": {},  # Use pymatgen defaults
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


class WindowType(str, Enum):
    """Enumeration of supported window types."""

    HANNING = "hanning"  # cosine-squared taper
    PARZEN = "parzen"  # linear taper
    WELCH = "welch"  # quadratic taper
    GAUSSIAN = "gaussian"  # Gaussian (normal) function window
    SINE = "sine"  # sine function window
    KAISER = "kaiser"  # Kaiser-Bessel function-derived window


# ================== CONFIGURATION ==================
@dataclass
class FeffConfig:
    """Configuration class for FEFF calculations."""

    spectrum_type: str = "EXAFS"
    edge: str = "K"
    radius: float = 8.0  # cluster size
    user_tag_settings: dict[str, str] = field(
        default_factory=dict
    )  # Empty by default - use pymatgen defaults
    # FFT parameters for EXAFS transform:
    kmin: float = 2.0  # starting k for FT Window
    kmax: float = 14.0  # ending k for FT Window
    kweight: int = 2  # exponent for weighting spectra by k**kweight
    dk: float = 1.0  # tapering parameter for FT Window
    dk2: float | None = None  # second tapering parameter for FT Window (larch default)
    with_phase: bool = False  # output the phase as well as magnitude, real, imag
    rmax_out: float = 10.0  # highest R for output data (Ang)
    window: WindowType = WindowType.HANNING  # type of window function
    nfft: int | None = None  # value to use for N_fft (None = use larch default: 2048)
    kstep: float | None = (
        None  # value to use for delta_k (k[1]-k[0] Ang^-1) (None = use larch default)
    )
    # Parallel execution settings
    parallel: bool = False
    n_workers: int | None = None
    # Trajectory sampling settings
    sample_interval: int = 1
    # Force recalculation even if output exists
    force_recalculate: bool = False
    # Clean up unnecessary FEFF output files
    cleanup_feff_files: bool = True

    # Get dictionary of the FT parameters
    @property
    def fourier_params(self) -> dict[str, float | int | str]:
        """Return Fourier transform parameters as a dictionary."""
        # Build dict then drop parameters explicitly set to None so that
        # larch's xftf() function can use its internal defaults. Passing
        # nfft=None leads to numpy.zeros(None) -> 0-d array and an IndexError
        # "too many indices" inside xftf_fast.
        params: dict[str, float | int | str | None] = {
            "kmin": self.kmin,
            "kmax": self.kmax,
            "kweight": self.kweight,
            "dk": self.dk,
            "dk2": self.dk2,
            "with_phase": self.with_phase,
            "window": self.window,
            "rmax_out": self.rmax_out,
            "nfft": self.nfft,  # exclude if None
            "kstep": self.kstep,
        }
        return {k: v for k, v in params.items() if v is not None}

    def __post_init__(self) -> None:
        """Post-initialization validation of configuration parameters."""
        self._validate_spectrum_type()
        self._validate_energy_range()
        self._validate_fourier_params()
        self._validate_radius()
        self._validate_n_workers()
        self._validate_sample_interval()

    def _validate_spectrum_type(self) -> None:
        if self.spectrum_type not in SpectrumType.__members__:
            raise ValueError(f"Invalid spectrum_type: {self.spectrum_type}")

    def _validate_energy_range(self) -> None:
        if self.kmin >= self.kmax:
            raise ValueError(f"kmin ({self.kmin}) must be less than kmax ({self.kmax})")
        if self.kmin < 0:
            raise ValueError(f"kmin must be positive, got {self.kmin}")

    def _validate_fourier_params(self) -> None:
        if self.dk <= 0:
            raise ValueError(f"dk must be positive, got {self.dk}")
        if not 1 <= self.kweight <= 3:
            logging.warning(f"Unusual kweight value: {self.kweight}")

    def _validate_radius(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")

    def _validate_n_workers(self) -> None:
        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError(f"Invalid n_workers: {self.n_workers}")

    def _validate_sample_interval(self) -> None:
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
        # Type: ignore for the unpacking since we know the preset structure is correct
        return cls(**preset)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "FeffConfig":
        """Load configuration from a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for configuration files")
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        if not isinstance(params, dict):
            raise ValueError("YAML file must contain a dictionary")
        return cls(**params)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for configuration files")
        with open(yaml_path, "w") as f:
            yaml.dump(self.as_dict(), f)

    def as_dict(self) -> dict[str, object]:
        """Convert configuration to dictionary format."""
        return {
            "spectrum_type": self.spectrum_type,
            "edge": self.edge,
            "radius": self.radius,
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
            "cleanup_feff_files": self.cleanup_feff_files,
        }

    def __repr_json__(self) -> str:
        """JSON representation for interactive environments."""
        return json.dumps(self.as_dict(), indent=4)


def validate_absorber(atoms: Atoms, absorber: str | int) -> str:
    """Validate and normalize absorber to element symbol."""
    if isinstance(absorber, int):
        if not 0 <= absorber < len(atoms):
            raise ValueError(f"Absorber index {absorber} out of range")
        return str(atoms.get_chemical_symbols()[absorber])
    else:
        absorber_element = str(absorber).capitalize()
        symbols = atoms.get_chemical_symbols()
        if absorber_element not in symbols:
            raise ValueError(
                f"Absorber element {absorber_element} not found in structure"
            )
        return absorber_element


def normalize_absorbers(
    atoms: Atoms, absorbers: str | int | list[int] | list[str]
) -> list[int]:
    """Normalize absorber specification to a list of atom indices.

    Args:
        atoms: The atomic structure
        absorbers: Absorber specification:
            - str: Element symbol (e.g., "Fe") - returns indices of all atoms of this
              element
            - int: Single atom index
            - list[int]: List of atom indices
            - list[str]: List of element symbols - returns indices of all matching atoms

    Returns:
        List of atom indices for the absorbing atoms

    Raises:
        ValueError: If absorber specification is invalid
    """
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    if isinstance(absorbers, str):
        # Single element symbol - find all atoms of this element
        element = absorbers.capitalize()
        if element not in symbols:
            raise ValueError(f"Element {element} not found in structure")
        indices = [i for i, sym in enumerate(symbols) if sym == element]
        if not indices:
            raise ValueError(f"No atoms of element {element} found in structure")
        return indices

    elif isinstance(absorbers, int):
        # Single atom index
        if not 0 <= absorbers < n_atoms:
            raise ValueError(
                f"Absorber index {absorbers} out of range (0-{n_atoms - 1})"
            )
        return [absorbers]

    elif isinstance(absorbers, list):
        if not absorbers:
            raise ValueError("Absorber list cannot be empty")

        if all(isinstance(x, int) for x in absorbers):
            # List of indices
            for idx in absorbers:
                if not 0 <= idx < n_atoms:
                    raise ValueError(
                        f"Absorber index {idx} out of range (0-{n_atoms - 1})"
                    )
            return list(absorbers)

        elif all(isinstance(x, str) for x in absorbers):
            # List of element symbols
            indices = []
            for element in absorbers:
                element = element.capitalize()
                if element not in symbols:
                    raise ValueError(f"Element {element} not found in structure")
                element_indices = [i for i, sym in enumerate(symbols) if sym == element]
                indices.extend(element_indices)

            if not indices:
                raise ValueError("No matching atoms found for specified elements")

            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            return unique_indices

        else:
            raise ValueError("Mixed types in absorber list not supported")
    else:
        raise ValueError(f"Invalid absorber type: {type(absorbers)}")


def get_absorber_element_from_index(atoms: Atoms, absorber_index: int) -> str:
    """Get element symbol for a given atom index."""
    if not 0 <= absorber_index < len(atoms):
        raise ValueError(f"Absorber index {absorber_index} out of range")
    return str(atoms.get_chemical_symbols()[absorber_index])


def generate_pymatgen_input(
    atoms: Atoms, absorber: str | int, output_dir: Path, config: FeffConfig
) -> Path:
    """Generate FEFF input using pymatgen."""
    # For backward compatibility, if there are multiple matching absorbers, use the
    # first one
    absorber_indices = normalize_absorbers(atoms, absorber)
    absorber_index = absorber_indices[0]  # Always use the first matching absorber
    absorber_element = get_absorber_element_from_index(atoms, absorber_index)

    # Convert to pymatgen structure
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(atoms)

    # Create FEFF set with user settings
    user_settings = config.user_tag_settings.copy()

    # Apply radius setting
    user_settings["RPATH"] = str(config.radius)

    # Remove problematic settings for FEFF8L compatibility
    user_settings.pop("COREHOLE", None)

    # Ensure _del is a list for removing incompatible keywords
    if "_del" not in user_settings:
        del_list: list[str] = []
    else:
        del_value = user_settings["_del"]
        if isinstance(del_value, str):
            del_list = [del_value]
        elif isinstance(del_value, list):
            del_list = del_value
        else:
            raise ValueError("_del must be a string or list of strings")

    user_settings["_del"] = del_list  # type: ignore[assignment]

    # Add FEFF8L incompatible keywords to the deletion list
    incompatible_keywords = ["COREHOLE", "COREHOLE FSR"]
    for keyword in incompatible_keywords:
        if keyword not in del_list:
            del_list.append(keyword)

    # Create FEFF set
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


def generate_pymatgen_input_multi(
    atoms: Atoms,
    absorbers: str | int | list[int] | list[str],
    output_dir: Path,
    config: FeffConfig,
) -> list[Path]:
    """Generate FEFF input using pymatgen for multiple absorbers.

    Returns a list of output files, one for each absorber.
    Each file is in a directory named with the pattern: {base_name}_site_{index}
    """
    absorber_indices = normalize_absorbers(atoms, absorbers)

    if len(absorber_indices) == 1:
        # Single absorber - use existing function
        return [generate_pymatgen_input(atoms, absorber_indices[0], output_dir, config)]

    # Multiple absorbers - create separate directories for each
    output_files = []
    base_name = output_dir.name

    for absorber_index in absorber_indices:
        site_dir = output_dir.parent / f"{base_name}_site_{absorber_index}"
        result_file = generate_pymatgen_input(atoms, absorber_index, site_dir, config)
        output_files.append(result_file)

    return output_files


def generate_feff_input(
    atoms: Atoms, absorber: str | int, output_dir: Path, config: FeffConfig
) -> Path:
    """Generate FEFF input using pymatgen."""
    return generate_pymatgen_input(atoms, absorber, output_dir, config)


def generate_feff_input_multi(
    atoms: Atoms,
    absorbers: str | int | list[int] | list[str],
    output_dir: Path,
    config: FeffConfig,
) -> list[Path]:
    """Generate FEFF input for multiple absorbers using pymatgen."""
    return generate_pymatgen_input_multi(atoms, absorbers, output_dir, config)


def run_feff_calculation(
    feff_dir: Path, verbose: bool = False, cleanup: bool = True
) -> bool:
    """Run FEFF calculation with simplified error handling.

    Args:
        feff_dir: Directory containing feff.inp
        verbose: Whether to enable verbose output
        cleanup: Whether to clean up unnecessary output files

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

    # Set up encoding environment
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Store original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Create basic log
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"FEFF calculation started at {datetime.now()}\n")
            log_file.write(f"Input file: {input_path}\n")
            log_file.write(f"Working directory: {feff_dir}\n")
            log_file.write("-" * 50 + "\n\n")

        # Run FEFF calculation
        if not verbose:
            # Redirect output to log file
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                sys.stdout = log_file
                sys.stderr = log_file
                result = feff8l(folder=str(feff_dir), feffinp="feff.inp", verbose=False)
        else:
            result = feff8l(folder=str(feff_dir), feffinp="feff.inp", verbose=True)

        # Check success
        chi_file = feff_dir / "chi.dat"
        success = chi_file.exists() and bool(result)

        # Clean up if requested and successful
        if success and cleanup:
            cleanup_feff_output(feff_dir, keep_essential=True)

        # Log final result
        with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"\nCalculation completed at {datetime.now()}\n")
            log_file.write(f"Success: {success}\n")
            if not chi_file.exists():
                log_file.write("Warning: chi.dat file not found\n")

        return success

    except (OSError, RuntimeError, ValueError, UnicodeDecodeError) as e:
        # Log any errors
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"\nERROR: {str(e)}\n")
                log_file.write(f"Exception type: {type(e).__name__}\n")
        except OSError:
            print(f"FEFF calculation failed: {e}")
        return False

    finally:
        # Always restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def get_feff_numbered_files(feff_dir: Path) -> list[Path]:
    """Get all feff####.dat files (any number of digits)."""
    feff_dir = Path(feff_dir)
    if not feff_dir.exists():
        return []

    # Simple regex: feff + digits + .dat (case insensitive)
    pattern = re.compile(r"^feff\d+\.dat$", re.IGNORECASE)

    feff_files = []
    for file_path in feff_dir.iterdir():
        if file_path.is_file() and pattern.match(file_path.name):
            feff_files.append(file_path)

    return feff_files


def cleanup_feff_output(feff_dir: Path, keep_essential: bool = True) -> int:
    """Clean up FEFF output files to save disk space.

    Args:
        feff_dir: Directory containing FEFF output files
        keep_essential: If True, keep only essential files

    Returns:
        Number of files removed
    """
    logger = logging.getLogger("larch_wrapper")

    feff_dir = Path(feff_dir)
    if not feff_dir.exists():
        return 0

    files_removed = 0

    # Get all numbered FEFF files (feff0001.dat, feff12345.dat, etc.)
    feff_files = get_feff_numbered_files(feff_dir)

    # Remove the numbered files
    for feff_file in feff_files:
        try:
            feff_file.unlink()
            files_removed += 1
            logger.debug(f"Removed: {feff_file.name}")
        except OSError as e:
            logger.warning(f"Could not remove {feff_file}: {e}")

    # If keep_essential=True, also remove some other cleanup files
    if keep_essential:
        cleanup_patterns = ["feffrun_*.log", "log*.dat", "misc.dat"]
        for pattern in cleanup_patterns:
            for file_to_remove in feff_dir.glob(pattern):
                try:
                    file_to_remove.unlink()
                    files_removed += 1
                    logger.debug(f"Removed: {file_to_remove.name}")
                except OSError as e:
                    logger.warning(f"Could not remove {file_to_remove}: {e}")

    if files_removed > 0:
        logger.info(f"Removed {files_removed} FEFF files from {feff_dir}")

    return files_removed


def read_feff_output(feff_dir: Path) -> tuple[object, object]:
    """Read FEFF chi.dat output with fallback methods and improved error handling."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("NumPy is required for reading FEFF output") from None

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
            mag = data[:, 2]
            phase = data[:, 3]

            # reconstruct complex chi
            chi = mag * np.exp(1j * phase)
            return chi, k
        except (OSError, ValueError, IndexError) as fallback_error:
            error_msg = (
                f"Failed to read {chi_file}:\n"
                f"Primary error: {read_error}\n"
                f"Fallback error: {fallback_error}"
            )
            raise Exception(error_msg) from None
