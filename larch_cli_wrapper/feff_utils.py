"""FEFF input generation utilities - Fixed for consistent output between methods."""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.feff.sets import MPEXAFSSet

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from larch.io import read_ascii
from larch.xafs.feffrunner import feff8l

# Maximum number of absorber sites to process to avoid excessive computation
LARGE_NUMBER_OF_SITES = 100

# Configuration presets using pymatgen defaults
PRESETS = {
    "quick": {
        "spectrum_type": "EXAFS",
        "edge": "K",
        "radius": 4.0,
        "kmin": 2,
        "kmax": 12,
        "kweight": 2,
        "window": "hanning",
        "dk": 1.0,
        "user_tag_settings": {
            "EXCHANGE": 0,
            "S02": 1.0,
            "PRINT": "1 0 0 0 0 3",
            "NLEG": 6,
            "_del": ["COREHOLE", "COREHOLE FSR", "SCF"],
        },
    },
    "publication": {
        "spectrum_type": "EXAFS",
        "edge": "K",
        "radius": 8.0,
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


__all__ = [
    "LARGE_NUMBER_OF_SITES",
    "PRESETS",
    "FeffConfig",
    "SpectrumType",
    "EdgeType",
    "WindowType",
    "normalize_absorbers",
    "validate_absorber_indices",
    "generate_multi_site_feff_inputs",
    "run_multi_site_feff_calculations",
    "run_feff_calculation",
    "read_feff_output",
    "average_chi_spectra",
    "cleanup_feff_output",
]


# ================== CONFIGURATION ==================
@dataclass
class FeffConfig:
    """Configuration class for FEFF calculations."""

    spectrum_type: str = "EXAFS"
    edge: str = "K"
    radius: float = 4.0  # cluster size (quick preset default)
    user_tag_settings: dict[str, str] = field(
        default_factory=lambda: {
            "EXCHANGE": "0",
            "S02": "1.0",
            "PRINT": "1 0 0 0 0 3",
            "NLEG": "6",
            "_del": ["COREHOLE", "COREHOLE FSR", "SCF"],
        }
    )  # Quick preset defaults
    # FFT parameters for EXAFS transform:
    kmin: float = 2.0  # starting k for FT Window
    kmax: float = 12.0  # ending k for FT Window (quick preset default)
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

    @property
    def feff_params(self) -> dict[str, str | float | dict[str, str]]:
        """Return FEFF calculation parameters as a dictionary.

        These are the parameters that affect the FEFF calculation itself,
        not the analysis/Fourier transform parameters. Used for caching
        to determine when FEFF calculations need to be re-run.
        """
        return {
            "spectrum_type": self.spectrum_type,
            "edge": self.edge,
            "radius": self.radius,
            "user_tag_settings": self.user_tag_settings,
        }

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
            "kmin": self.kmin,
            "kmax": self.kmax,
            "kweight": self.kweight,
            "dk": self.dk,
            "dk2": self.dk2,
            "with_phase": self.with_phase,
            "rmax_out": self.rmax_out,
            "window": self.window,
            "nfft": self.nfft,
            "kstep": self.kstep,
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
            # List of indices - type narrowing
            int_absorbers = absorbers  # Now type checker knows these are all ints
            for idx in int_absorbers:
                if not 0 <= idx < n_atoms:
                    raise ValueError(
                        f"Absorber index {idx} out of range (0-{n_atoms - 1})"
                    )
            return int_absorbers

        elif all(isinstance(x, str) for x in absorbers):
            # List of element symbols - type narrowing
            str_absorbers = absorbers  # Now type checker knows these are all strs
            indices = []
            for element in str_absorbers:
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


def validate_absorber_indices(atoms: Atoms, absorber_indices: list[int]) -> str:
    """Validate that all absorber indices correspond to the same chemical element.

    Args:
        atoms: ASE Atoms object
        absorber_indices: List of absorber atom indices

    Returns:
        Chemical symbol of the absorbing element

    Raises:
        ValueError: If indices correspond to different elements or are invalid
    """
    if not absorber_indices:
        raise ValueError("At least one absorber index must be provided")

    # Check that all indices are valid
    n_atoms = len(atoms)
    for idx in absorber_indices:
        if not 0 <= idx < n_atoms:
            raise ValueError(f"Absorber index {idx} out of range (0-{n_atoms - 1})")

    # Check that all indices correspond to the same element
    symbols = atoms.get_chemical_symbols()
    absorber_element = symbols[absorber_indices[0]]

    for _i, idx in enumerate(absorber_indices):
        if symbols[idx] != absorber_element:
            raise ValueError(
                f"Absorber indices must all correspond to the same element. "
                f"Index {absorber_indices[0]} is {absorber_element} but "
                f"index {idx} is {symbols[idx]}"
            )

    # Log warning if too many sites
    if len(absorber_indices) > LARGE_NUMBER_OF_SITES:
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Number of absorber sites ({len(absorber_indices)}) "
            f"is very large - are you sure this is correct? "
        )

    return absorber_element


def average_chi_spectra(
    k_arrays: list[np.ndarray],
    chi_arrays: list[np.ndarray],
    weights: list[float] | None = None,
    *,
    restrict_to_common_range: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Average χ(k) spectra with optional common-range alignment.

    Args:
        k_arrays: List of k-grids (one per spectrum)
        chi_arrays: List of χ(k) arrays (one per spectrum)
        weights: Optional weights for averaging (``None`` for equal weights)
        restrict_to_common_range: When True, restrict interpolation to the
            overlapping k-range across all spectra using the shortest grid.
            When False, use the first spectrum's grid and zero-pad gaps.

    Returns:
        Tuple of (averaged χ, k_grid)

    Raises:
        ValueError: If inputs are empty, mismatched, or have no overlap when
            ``restrict_to_common_range`` is requested.
    """
    if not k_arrays or not chi_arrays:
        raise ValueError("Empty input arrays provided")

    if len(k_arrays) != len(chi_arrays):
        raise ValueError("Number of k and chi arrays must match")

    if len(k_arrays) == 1:
        # Single spectrum - no averaging needed
        return chi_arrays[0], k_arrays[0]

    if restrict_to_common_range:
        k_min = max(float(k.min()) for k in k_arrays)
        k_max = min(float(k.max()) for k in k_arrays)

        if not np.isfinite(k_min) or not np.isfinite(k_max) or k_min >= k_max:
            raise ValueError("No overlapping k-range found for averaging")

        n_points = min(len(k) for k in k_arrays)
        k_ref = np.linspace(k_min, k_max, n_points)
    else:
        # Set reference k-grid from first spectrum
        k_ref = k_arrays[0].copy()
    chi_list = []

    for _i, (k, chi) in enumerate(zip(k_arrays, chi_arrays, strict=False)):
        # Interpolate to common k-grid if needed
        if not np.array_equal(k, k_ref):
            # Handle complex chi data by interpolating real and imaginary parts
            # separately
            if np.iscomplexobj(chi):
                chi_real = np.interp(k_ref, k, chi.real, left=0, right=0)
                chi_imag = np.interp(k_ref, k, chi.imag, left=0, right=0)
                chi_interp = chi_real + 1j * chi_imag
            else:
                chi_interp = np.interp(k_ref, k, chi, left=0, right=0)
        else:
            chi_interp = chi

        chi_list.append(chi_interp)

    # Apply weights if provided
    if weights is not None:
        if len(weights) != len(chi_list):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of "
                f"spectra ({len(chi_list)})"
            )

        # Normalize weights
        weights_array = np.array(weights)
        weights_array = weights_array / np.sum(weights_array)

        # Weighted average
        chi_avg = np.average(chi_list, axis=0, weights=weights_array)
    else:
        # Simple average
        chi_avg = np.mean(chi_list, axis=0)

    return chi_avg, k_ref


def generate_pymatgen_input(
    atoms: Atoms, absorber_index: int, output_dir: Path, config: FeffConfig
) -> Path:
    """Generate FEFF input using pymatgen for a single absorber site index.

    Args:
        atoms: ASE Atoms object containing the structure
        absorber_index: Index of the absorbing atom (0-based)
        output_dir: Directory where FEFF input files will be written
        config: FEFF configuration object

    Returns:
        Path to the generated feff.inp file
    """
    # Validate absorber index
    if not 0 <= absorber_index < len(atoms):
        raise ValueError(
            f"Absorber index {absorber_index} out of range (0-{len(atoms) - 1})"
        )

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
            absorbing_atom=absorber_index,
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


def run_multi_site_feff_calculations(
    input_files: list[Path],
    cleanup: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
    progress_callback: callable = None,
    timeout: int = 600,
    max_retries: int = 2,
    force_recalculate: bool = False,
) -> list[tuple[Path, bool]]:
    """Run FEFF calculations for multiple sites efficiently.

    Args:
        input_files: List of paths to feff.inp files
        cleanup: Whether to clean up unnecessary FEFF output files
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None for auto)
        progress_callback: Optional callback called with (completed, total)
            after each calculation
        timeout: Timeout per calculation in seconds (default: 600 = 10 minutes)
        max_retries: Maximum number of retry attempts for failed calculations
            (default: 2)
        force_recalculate: Whether to force recalculation even if chi.dat exists

    Returns:
        List of (feff_dir, success) tuples
    """
    import concurrent.futures
    import time

    def run_single_calculation(input_file: Path) -> tuple[Path, bool]:
        """Run FEFF calculation for a single site with retry logic."""
        feff_dir = input_file.parent

        for attempt in range(max_retries + 1):
            try:
                success = run_feff_calculation(
                    feff_dir,
                    verbose=False,
                    cleanup=cleanup,
                    timeout=timeout,
                    force_recalculate=force_recalculate,
                )

                if success:
                    return feff_dir, True

                # If not successful and not the last attempt, wait before retry
                if attempt < max_retries:
                    time.sleep(1 + attempt)  # Progressive backoff: 1s, 2s, 3s...

            except (OSError, RuntimeError):
                # Log the error but continue with retry logic
                if attempt < max_retries:
                    time.sleep(1 + attempt)
                else:
                    # Final attempt failed
                    return feff_dir, False

        return feff_dir, False

    total_tasks = len(input_files)
    completed_count = 0

    if parallel and len(input_files) > 1:
        if max_workers is None:
            # More conservative default: use fewer workers to reduce resource contention
            import os

            cpu_count = os.cpu_count() or 4
            max_workers = min(len(input_files), max(1, cpu_count // 2), 3)

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_input = {
                executor.submit(run_single_calculation, input_file): input_file
                for input_file in input_files
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_input):
                result = future.result()
                results.append(result)
                completed_count += 1

                # Report progress
                if progress_callback:
                    progress_callback(completed_count, total_tasks)

        # Reorder results to match original input order
        input_to_result = {
            future_to_input[future]: future.result() for future in future_to_input
        }
        results = [input_to_result[input_file] for input_file in input_files]
    else:
        results = []
        for input_file in input_files:
            result = run_single_calculation(input_file)
            results.append(result)
            completed_count += 1

            # Report progress for sequential execution
            if progress_callback:
                progress_callback(completed_count, total_tasks)

    return results


def generate_multi_site_feff_inputs(
    atoms: Atoms,
    absorber_indices: list[int],
    base_output_dir: Path,
    config: FeffConfig,
) -> list[Path]:
    """Generate FEFF inputs for multiple absorber sites.

    Creates separate subdirectories for each absorber site using
    the naming pattern: site_XXXX (where XXXX is the zero-padded index).

    Args:
        atoms: ASE Atoms object containing the structure
        absorber_indices: List of absorber atom indices
        base_output_dir: Base directory for outputs (sites will be in subdirs)
        config: FEFF configuration object

    Returns:
        List of paths to generated feff.inp files
    """
    if not absorber_indices:
        raise ValueError("At least one absorber index must be provided")

    # Validate all indices
    for idx in absorber_indices:
        if not 0 <= idx < len(atoms):
            raise ValueError(f"Absorber index {idx} out of range (0-{len(atoms) - 1})")

    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    input_files = []
    for absorber_idx in absorber_indices:
        site_dir = base_output_dir / f"site_{absorber_idx:04d}"
        feff_input_path = generate_pymatgen_input(atoms, absorber_idx, site_dir, config)
        input_files.append(feff_input_path)

    return input_files


def generate_feff_input_multi(
    atoms: Atoms,
    absorbers: str | int | list[int] | list[str],
    output_dir: Path,
    config: FeffConfig,
) -> list[Path]:
    """Generate FEFF input for multiple absorbers.

    Args:
        atoms: ASE Atoms object
        absorbers: Absorber specification (element, index, or list)
        output_dir: Base output directory
        config: FEFF configuration

    Returns:
        List of paths to generated feff.inp files
    """
    absorber_indices = normalize_absorbers(atoms, absorbers)
    return generate_multi_site_feff_inputs(atoms, absorber_indices, output_dir, config)


def run_feff_calculation(
    feff_dir: Path,
    verbose: bool = False,
    cleanup: bool = True,
    timeout: int = 600,
    force_recalculate: bool = False,
) -> bool:
    """Run FEFF calculation with robust encoding handling.

    Args:
        feff_dir: Directory containing feff.inp
        verbose: Whether to enable verbose output
        cleanup: Whether to clean up unnecessary output files
        timeout: Timeout in seconds (default: 600 = 10 minutes)
        force_recalculate: Whether to force recalculation even if chi.dat exists

    Returns:
        True if calculation succeeded, False otherwise
    """
    import os
    import shutil
    import subprocess
    import sys

    feff_dir = Path(feff_dir)
    input_path = feff_dir / "feff.inp"
    log_path = feff_dir / "feff.log"
    chi_file = feff_dir / "chi.dat"

    if not input_path.exists():
        raise FileNotFoundError(f"FEFF input file {input_path} not found")

    # Check if chi.dat already exists and force_recalculate is False
    if not force_recalculate and chi_file.exists():
        # Verify the existing chi.dat file is valid
        try:
            # Use read_feff_output function from this module
            read_feff_output(feff_dir)  # This will raise an exception if invalid

            # Log that we're using existing results
            with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"FEFF calculation skipped at {datetime.now()}\n")
                log_file.write(f"Using existing chi.dat file: {chi_file}\n")
                log_file.write("Use force_recalculate=True to override\n")

            return True

        except (OSError, ValueError, IndexError):
            # Existing chi.dat is invalid, remove it and proceed with calculation
            try:
                chi_file.unlink()
            except OSError:
                pass  # Ignore errors removing invalid file

    try:
        # Create basic log
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"FEFF calculation started at {datetime.now()}\n")
            log_file.write(f"Input file: {input_path}\n")
            log_file.write(f"Working directory: {feff_dir}\n")
            log_file.write("-" * 50 + "\n\n")

        # Try to use larch feff8l if available, but with encoding fix
        try:
            # Use larch's feff8l, but ensure proper encoding

            if not verbose:
                # Don't redirect stdout/stderr for non-verbose mode
                # Instead, let feff8l handle its own output and capture it differently
                os.environ["PYTHONIOENCODING"] = "utf-8"

                # Run with verbose=False but don't redirect streams
                result = feff8l(folder=str(feff_dir), feffinp="feff.inp", verbose=False)
            else:
                # For verbose mode, ensure stdout has encoding
                if not hasattr(sys.stdout, "encoding") or sys.stdout.encoding is None:
                    sys.stdout.encoding = "utf-8"
                result = feff8l(
                    folder=str(feff_dir), feffinp="feff.inp", verbose=verbose
                )

        except Exception as larch_error:
            # If larch fails, try using subprocess as fallback
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"Larch feff8l failed: {larch_error}\n")
                log_file.write("Attempting subprocess fallback...\n")

            # Look for feff executable
            feff_exe = shutil.which("feff8l") or shutil.which("feff")
            if feff_exe is None:
                raise RuntimeError("No FEFF executable found in PATH") from larch_error

            # Validate executable path for security
            feff_exe = Path(feff_exe).resolve()
            if not feff_exe.exists() or not feff_exe.is_file():
                raise RuntimeError(
                    f"FEFF executable not found or not a file: {feff_exe}"
                ) from larch_error

            # Run FEFF via subprocess
            try:
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

                # Run FEFF via subprocess
                # S603: subprocess call is safe here - executable path is validated
                proc = subprocess.run(  # noqa: S603
                    [str(feff_exe)],  # Convert Path to string for subprocess
                    cwd=str(feff_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=timeout,  # Configurable timeout (default 10 minutes)
                    check=False,  # Don't raise on non-zero exit codes
                )

                # Log subprocess output
                with open(
                    log_path, "a", encoding="utf-8", errors="replace"
                ) as log_file:
                    log_file.write(f"FEFF stdout:\n{proc.stdout}\n")
                    if proc.stderr:
                        log_file.write(f"FEFF stderr:\n{proc.stderr}\n")

                result = proc.returncode == 0

            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
            ) as proc_error:
                with open(
                    log_path, "a", encoding="utf-8", errors="replace"
                ) as log_file:
                    log_file.write(f"Subprocess failed: {proc_error}\n")
                result = False

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

    except (
        OSError,
        subprocess.CalledProcessError,
        FileNotFoundError,
        PermissionError,
        TimeoutError,
    ) as e:
        # Log any errors from FEFF execution or file operations
        error_msg = str(e)
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
                log_file.write(f"\nERROR: {error_msg}\n")
                log_file.write(f"Exception type: {type(e).__name__}\n")
        except OSError:
            print(f"FEFF calculation failed: {error_msg}")
        return False


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


def read_feff_output(feff_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read FEFF chi.dat output using larch.

    Args:
        feff_dir: Directory containing FEFF output files

    Returns:
        Tuple of (chi, k) arrays where chi is complex and k is real

    Raises:
        FileNotFoundError: If chi.dat file is not found
        ValueError: If data cannot be parsed or has wrong format
    """
    chi_file = feff_dir / "chi.dat"
    if not chi_file.exists():
        raise FileNotFoundError(f"FEFF output {chi_file} not found")

    try:
        feff_data = read_ascii(str(chi_file))

        if not hasattr(feff_data, "k"):
            raise AttributeError("FEFF data missing required 'k' attribute")

        # FEFF format: reconstruct complex chi from mag and phase if available
        if hasattr(feff_data, "mag") and hasattr(feff_data, "phase"):
            import numpy as np

            chi = feff_data.mag * np.exp(1j * feff_data.phase)
            return chi, feff_data.k

        # Fallback: use chi directly (may be real or complex)
        elif hasattr(feff_data, "chi"):
            return feff_data.chi, feff_data.k

        else:
            raise AttributeError(
                "FEFF data missing chi information. "
                "Expected either (mag, phase) or (chi) columns"
            )

    except (OSError, ValueError, AttributeError) as err:
        raise ValueError(f"Error reading FEFF output {chi_file}: {err}") from err
