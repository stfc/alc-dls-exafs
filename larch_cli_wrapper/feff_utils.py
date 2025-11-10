"""FEFF input generation utilities - Fixed for consistent output between methods."""

import concurrent
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

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

logger = logging.getLogger("larch_wrapper")

# Maximum number of absorber sites to process to avoid excessive computation
LARGE_NUMBER_OF_SITES = 100

# FEFF card fields that can be set to null in YAML to disable them
FEFF_CARD_FIELDS = {
    "control",
    "print",
    "s02",
    "scf",
    "exchange",
    "nleg",
    "exafs",
    "criteria",
}


def _load_presets() -> dict[str, dict[str, Any]]:
    """Load preset configurations from bundled YAML files.

    Returns:
        Dictionary mapping preset names to their configuration dictionaries.
        Falls back to minimal defaults if YAML files cannot be loaded.
    """
    presets = {}
    preset_dir = Path(__file__).parent / "feff_configs"

    if not YAML_AVAILABLE:
        logger.warning("PyYAML not available, using minimal default presets")
        return {
            "quick": {"spectrum_type": "EXAFS", "edge": "K", "radius": 4.0},
            "publication": {"spectrum_type": "EXAFS", "edge": "K", "radius": 8.0},
        }

    if not preset_dir.exists():
        logger.warning(f"Preset directory not found: {preset_dir}")
        return {
            "quick": {"spectrum_type": "EXAFS", "edge": "K", "radius": 4.0},
            "publication": {"spectrum_type": "EXAFS", "edge": "K", "radius": 8.0},
        }

    # Load all YAML files in the preset directory
    for yaml_file in preset_dir.glob("*.yaml"):
        preset_name = yaml_file.stem
        try:
            with open(yaml_file) as f:
                preset_config = yaml.safe_load(f)
            if isinstance(preset_config, dict):
                # Handle FEFF card fields explicitly set to null
                explicit_none_fields = [
                    field.upper()
                    for field in FEFF_CARD_FIELDS
                    if field in preset_config and preset_config[field] is None
                ]

                # If there are fields explicitly set to None, add them to delete_tags
                if explicit_none_fields:
                    existing_delete_tags = preset_config.get("delete_tags", [])
                    if isinstance(existing_delete_tags, str):
                        existing_delete_tags = [existing_delete_tags]
                    elif existing_delete_tags is None:
                        existing_delete_tags = []
                    else:
                        existing_delete_tags = list(existing_delete_tags)

                    # Add explicit None fields to delete list
                    for field in explicit_none_fields:
                        if field not in existing_delete_tags:
                            existing_delete_tags.append(field)

                    preset_config["delete_tags"] = existing_delete_tags

                presets[preset_name] = preset_config
                logger.debug(f"Loaded preset '{preset_name}' from {yaml_file}")
            else:
                logger.warning(f"Invalid preset format in {yaml_file}")
        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Failed to load preset from {yaml_file}: {e}")
    return presets


# Load presets from YAML files at module import time
PRESETS = _load_presets()


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
    "FEFF_CARD_FIELDS",
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
    """Configuration class for FEFF calculations.

    FEFF input "cards" are represented via explicit fields on this class.
    Values are written into feff.inp as space-separated strings. Prefer strings
    for full control; numbers are accepted and normalized. Known tags and their
    meanings:

    - CONTROL ipot ixsph ifms ipaths igenfmt iff2x (Standard)
        Run one or more FEFF program modules. 0 = skip, 1 = run. Modules must run
        sequentially without skipping; e.g. "1 1 1 0 0 1" is invalid. Default is
        "1 1 1 1 1 1" (run all). Sub-groups: ipot -> atomic/pot/screen;
        ifms -> fms/mkgtr; iff2x -> ff2x/sfconv/eels. LDOS is controlled by LDOS
        card, not CONTROL.

    - PRINT ppot pxsph pfms ppaths pgenfmt pff2x (Standard)
        Control print levels (output files) per module. Default is 0 for each.

    - EDGE label s02 (Standard)
        Set the edge by label (K, L1, L2, L3, ...). For very shallow edges (M and
        higher), results are less tested. You may also specify an amplitude
        reduction factor S02 here. In this project, prefer FeffConfig.edge for the
        edge selection and use the S02 card for S02 itself for clarity.

    - SCF rfms1 [lfms1 nscmt ca nmix] (Standard)
        Control self-consistent potentials. Accepts 1–5 tokens
        (rfms1[, lfms1[, nscmt[, ca[, nmix]]]]). Defaults for omitted tokens:
        lfms1=0 (solids), nscmt=30, ca=0.2, nmix=1. rfms1 is the radius (Å) for
        full multiple scattering during SCF; typically ~30 atoms. lfms1=1 is for
        molecules. nscmt is max iterations; ca is initial mixing; nmix number of
        mixing iterations before Broyden.

    - S02 s02 (Standard)
        Amplitude reduction factor S02. If < 0.1, FEFF estimates it from atomic overlap
        integrals. Typically between 0.8 and 1.0. Using this card is clearer than using
        EDGE label s02.

    - EXCHANGE ixc vr0 vi0 [ixc0] (Useful)
        Exchange-correlation model selection and constant shifts: ixc selects the
        model, optional ixc0 for background; vr0 is a Fermi level shift (eV);
        vi0 is an imaginary optical potential (broadening). Defaults: ixc=0
        (Hedin–Lundqvist), vr0=0.0, vi0=0.0.

    - NLEG nleg (Useful)
        Maximum number of legs per scattering path. nleg=2 limits to single scattering.
        Default is 8.
    - EXAFS
        EXAFS card sets the maximum value of k for EXAFS calculations.
        k is set by xkmax, and the default value is 20 A^−1.

    Normalization rules (enforced by set_tag/apply_tags):
        - Tokens are serialized to strings separated by spaces.
        - S02: float >= 0
        - EXAFS: positive integer
        - PRINT/CONTROL: sequences of integers
        - SCF: 1 or 5 numeric tokens; defaults applied when omitted
        - EXCHANGE: free-form tokens (commonly integers/floats), joined
        - NLEG: positive integer
        - Unknown tags: tokens are joined as-is
    """

    spectrum_type: str = "EXAFS"
    edge: str = "K"
    radius: float = 4.0  # cluster size (quick preset default)
    # Explicit FEFF card fields. Values accept strings, numbers, or sequences and
    # will be normalized.
    control: Any | None = None
    print: Any | None = "1 0 0 0 0 3"
    s02: Any | None = 1.0
    scf: Any | None = None
    exchange: Any | None = 0
    nleg: Any | None = 6
    exafs: Any | None = None
    criteria: Any | None = None

    delete_tags: list[str] | str | None = (
        None  # Additional tags to delete (COREHOLE tags always deleted automatically)
    )
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
    def feff_params(self) -> dict[str, str | float | dict[str, str | list[str]]]:
        """Return FEFF calculation parameters as a dictionary.

        These are the parameters that affect the FEFF calculation itself,
        not the analysis/Fourier transform parameters. Used for caching
        to determine when FEFF calculations need to be re-run.
        """
        params: dict[str, str | float | dict[str, str | list[str]]] = {
            "spectrum_type": self.spectrum_type,
            "edge": self.edge,
            "radius": self.radius,
        }
        # Include a stable representation of FEFF cards for cache keys
        cards = self.to_pymatgen_user_tags()
        if cards:
            params.update(cards)
        return params

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
        """Load configuration from a YAML file.

        Expects keys to match FeffConfig field names directly
        (e.g., control, print, s02, scf, exchange, nleg, exafs, delete_tags).

        Fields explicitly set to null in YAML will be added to delete_tags
        to disable those FEFF cards.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for configuration files")
        with open(yaml_path) as f:
            params = yaml.safe_load(f)
        if not isinstance(params, dict):
            raise ValueError("YAML file must contain a dictionary")

        # Track FEFF card fields that are explicitly set to None
        explicit_none_fields = [
            field.upper()
            for field in FEFF_CARD_FIELDS
            if field in params and params[field] is None
        ]

        # If there are fields explicitly set to None, add them to delete_tags
        if explicit_none_fields:
            existing_delete_tags = params.get("delete_tags", [])
            if isinstance(existing_delete_tags, str):
                existing_delete_tags = [existing_delete_tags]
            elif existing_delete_tags is None:
                existing_delete_tags = []
            else:
                existing_delete_tags = list(existing_delete_tags)

            # Add explicit None fields to delete list
            for field in explicit_none_fields:
                if field not in existing_delete_tags:
                    existing_delete_tags.append(field)

            params["delete_tags"] = existing_delete_tags

        return cls(**params)  # type: ignore[arg-type]

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for configuration files")
        with open(yaml_path, "w") as f:
            yaml.dump(self.as_dict(), f)

    def as_dict(self) -> dict[str, object]:
        """Convert configuration to dictionary format.

        This representation combines FEFF parameters (used for caching/input
        generation) with Fourier transform parameters, plus execution flags.
        It leverages the dedicated helpers ``feff_params`` and
        ``fourier_params`` to keep serialization consistent across the codebase.
        """
        data: dict[str, object] = {}

        # Core FEFF calculation parameters (includes spectrum_type, edge, radius
        # and a stable feff_cards representation when applicable)
        data.update(self.feff_params)

        # Fourier-transform and output parameters (only include non-None)
        data.update(self.fourier_params)

        # Execution / runtime flags
        data.update(
            {
                "parallel": self.parallel,
                "n_workers": self.n_workers,
                "sample_interval": self.sample_interval,
                "force_recalculate": self.force_recalculate,
                "cleanup_feff_files": self.cleanup_feff_files,
            }
        )

        return data

    def __repr_json__(self) -> str:
        """JSON representation for interactive environments."""
        return json.dumps(self.as_dict(), indent=4)

    def to_pymatgen_user_tags(self) -> dict[str, str | list[str]]:
        """Build the FEFF card dict for pymatgen from explicit fields only.

        All values are normalized to FEFF string format.
        Returns a dict suitable for MPEXAFSSet(user_tag_settings=...).
        """
        tags: dict[str, str | list[str]] = {}

        field_map: dict[str, Any] = {
            "CONTROL": self.control,
            "PRINT": self.print,
            "S02": self.s02,
            "SCF": self.scf,
            "EXCHANGE": self.exchange,
            "NLEG": self.nleg,
            "EXAFS": self.exafs,
            "CRITERIA": self.criteria,
        }

        # Add normalized explicit fields (skip None values)
        for key, val in field_map.items():
            if val is not None:
                tags[key] = self._normalize_tag(key, val)

        # Build deletion list from explicit delete_tags
        del_list: list[str] = []
        if self.delete_tags:
            if isinstance(self.delete_tags, str):
                del_list.append(self.delete_tags)
            else:
                del_list.extend([str(x) for x in self.delete_tags])

        # Deduplicate preserving order
        if del_list:
            seen: set[str] = set()
            uniq: list[str] = []
            for x in del_list:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            tags["_del"] = uniq

        return tags

    # -------- Normalization helpers ---------
    @staticmethod
    def _to_str_tokens(value: Any) -> list[str]:
        """Convert a scalar or sequence to list[str] tokens for FEFF lines."""
        if isinstance(value, list | tuple):
            return [str(x) for x in value]
        elif isinstance(value, str):
            # Split by whitespace to tokens; keep as-is if already a single token
            return value.split()
        else:
            return [str(value)]

    def _normalize_tag(self, name: str, value: Any) -> str:
        """Normalize a tag value to a valid FEFF string.

        With validation for known tags.
        """
        key = name.strip().upper()
        tokens = self._to_str_tokens(value)

        # Known tag rules
        if key == "S02":
            # amplitude reduction factor; allow float >= 0
            try:
                s02 = float(tokens[0])
            except Exception as e:
                raise ValueError(f"S02 must be a number, got {value!r}") from e
            if s02 < 0:
                raise ValueError(f"S02 must be >= 0, got {s02}")
            return f"{s02}"

        if key == "EXAFS":
            # Number of k-points in EXAFS output or a control parameter;
            # expect a positive integer
            try:
                exafs = int(float(tokens[0]))
            except Exception as e:
                raise ValueError(f"EXAFS must be an integer, got {value!r}") from e
            if exafs <= 0:
                raise ValueError(f"EXAFS must be > 0, got {exafs}")
            return str(exafs)

        if key == "PRINT":
            # sequence of integers
            try:
                ints = [str(int(float(t))) for t in tokens]
            except Exception as e:
                raise ValueError(f"PRINT expects integer tokens, got {value!r}") from e
            return " ".join(ints)

        if key == "CONTROL":
            # sequence of integers (feature toggles)
            try:
                ints = [str(int(float(t))) for t in tokens]
            except Exception as e:
                raise ValueError(
                    f"CONTROL expects integer tokens, got {value!r}"
                ) from e
            return " ".join(ints)

        if key == "SCF":
            # Accept 1–5 tokens. Defaults for optional fields if omitted.
            # SCF rfms1 [lfms1 nscmt ca nmix]
            if len(tokens) < 1 or len(tokens) > 5:
                raise ValueError(
                    f"SCF expects between 1 and 5 tokens, got {len(tokens)}: {value!r}"
                )
            try:
                # rfms1: positive float (radius)
                rfms1 = float(tokens[0])
                if rfms1 <= 0:
                    raise ValueError(f"SCF rfms1 must be > 0, got {rfms1}")

                # Fill trailing with defaults
                lfms1 = int(float(tokens[1])) if len(tokens) >= 2 else 0
                nscmt = int(float(tokens[2])) if len(tokens) >= 3 else 30
                ca = float(tokens[3]) if len(tokens) >= 4 else 0.2
                nmix = int(float(tokens[4])) if len(tokens) >= 5 else 1

                # Validate ranges
                # TODO
                # Format numbers compactly
                def _fmt(x: float) -> str:
                    s = str(float(x))
                    return s.rstrip("0").rstrip(".") if "." in s else str(int(float(x)))

                out = [
                    _fmt(rfms1),
                    str(int(lfms1)),
                    str(int(nscmt)),
                    _fmt(ca),
                    str(int(nmix)),
                ]
            except Exception as e:
                # If parsing above raised a ValueError, re-raise. Otherwise wrap.
                if isinstance(e, ValueError):
                    raise
                raise ValueError(f"Invalid SCF specification: {value!r}") from e
            return " ".join(out)

        if key == "EXCHANGE":
            # Often an integer code or string; accept as-is tokens joined
            return " ".join(tokens)

        if key == "NLEG":
            try:
                nleg = int(float(tokens[0]))
            except Exception as e:
                raise ValueError(f"NLEG must be an integer, got {value!r}") from e
            if nleg <= 0:
                raise ValueError(f"NLEG must be > 0, got {nleg}")
            return str(nleg)

        # Fallback: join tokens with spaces
        return " ".join(tokens)


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

    logger.debug(f"Generating FEFF input for absorber index {absorber_index}")

    # Convert to pymatgen structure
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(atoms)

    # Create FEFF set with user settings
    user_settings = config.to_pymatgen_user_tags()
    user_settings["RPATH"] = str(config.radius)

    logger.debug(f"RPATH set to {config.radius}")

    # Handle _del keyword for removing incompatible tags
    del_value = user_settings.pop("_del", None)
    if del_value is None:
        del_list: list[str] = []
    elif isinstance(del_value, str):
        del_list = [del_value]
    elif isinstance(del_value, list):
        del_list = del_value
    else:
        raise ValueError("_del must be a string or list of strings")

    # Add FEFF8L incompatible keywords to the deletion list
    incompatible_keywords = ["COREHOLE", "COREHOLE FSR"]
    for keyword in incompatible_keywords:
        if keyword not in del_list:
            del_list.append(keyword)
            logger.debug(f"Added {keyword} to deletion list for FEFF8L compatibility")

    user_settings["_del"] = del_list

    # Create FEFF set
    if config.spectrum_type == "EXAFS":
        logger.debug(
            "Creating EXAFS FEFF set with "
            f"edge {config.edge} and radius {config.radius}"
        )
        feff_set = MPEXAFSSet(
            absorbing_atom=absorber_index,
            structure=structure,
            edge=config.edge,
            radius=config.radius,
            user_tag_settings=user_settings,
        )
    else:
        raise ValueError(f"Unsupported spectrum type: {config.spectrum_type}")

    # Always use absolute output_dir to avoid relative path issues
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory created: {output_dir}")

    # Some external writers may change the process CWD. Guard and restore.
    original_cwd = os.getcwd()
    try:
        logger.info(f"Writing FEFF input to {output_dir}")
        feff_set.write_input(str(output_dir))
        logger.info("FEFF input generated successfully")
    finally:
        try:
            os.chdir(original_cwd)
        except OSError as e:
            # Best-effort restore; directory may have been deleted
            logger.warning(
                f"Failed to restore working directory to {original_cwd}: {e}"
            )

    return output_dir / "feff.inp"


def run_multi_site_feff_calculations(
    input_files: list[Path],
    cleanup: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    timeout: int = 600,
    max_retries: int = 2,
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

    Returns:
        List of (feff_dir, success) tuples in the same order as input_files
    """
    if not input_files:
        return []

    def run_with_retry(input_file: Path) -> tuple[Path, bool]:
        """Run FEFF calculation with retry logic."""
        feff_dir = input_file.parent

        for attempt in range(max_retries + 1):
            try:
                success = run_feff_calculation(
                    feff_dir,
                    verbose=False,
                    cleanup=cleanup,
                    timeout=timeout,
                )
                if success:
                    return feff_dir, True

                # Progressive backoff before retry (except on last attempt)
                if attempt < max_retries:
                    time.sleep(1 + attempt)

            except (OSError, RuntimeError, TimeoutError):
                # On final attempt, return failure
                if attempt == max_retries:
                    return feff_dir, False
                # Otherwise, wait and retry
                time.sleep(1 + attempt)

        return feff_dir, False

    # Determine if parallel execution is beneficial
    use_parallel = parallel and len(input_files) > 1

    if use_parallel:
        return _run_parallel(
            input_files, run_with_retry, max_workers, progress_callback
        )
    else:
        return _run_sequential(input_files, run_with_retry, progress_callback)


def _run_parallel(
    input_files: list[Path],
    task_fn: Callable[[Path], tuple[Path, bool]],
    max_workers: int | None,
    progress_callback: Callable[[int, int], None] | None,
) -> list[tuple[Path, bool]]:
    """Execute tasks in parallel using ThreadPoolExecutor."""
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = min(len(input_files), max(1, cpu_count // 2), 4)

    total_tasks = len(input_files)
    completed_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and maintain order mapping
        future_to_input = {
            executor.submit(task_fn, input_file): input_file
            for input_file in input_files
        }

        # Collect results as they complete (for progress tracking)
        input_to_result = {}
        for future in concurrent.futures.as_completed(future_to_input):
            input_file = future_to_input[future]
            result = future.result()
            input_to_result[input_file] = result

            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total_tasks)

    # Return results in original input order
    return [input_to_result[input_file] for input_file in input_files]


def _run_sequential(
    input_files: list[Path],
    task_fn: Callable[[Path], tuple[Path, bool]],
    progress_callback: Callable[[int, int], None] | None,
) -> list[tuple[Path, bool]]:
    """Execute tasks sequentially."""
    total_tasks = len(input_files)
    results = []

    for idx, input_file in enumerate(input_files, start=1):
        result = task_fn(input_file)
        results.append(result)

        if progress_callback:
            progress_callback(idx, total_tasks)

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
    # Reuse common validation (ensures indices are valid
    # and correspond to the same element)
    _ = validate_absorber_indices(atoms, absorber_indices)

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
    verbose: bool = True,
    cleanup: bool = True,
    timeout: int = 600,
) -> bool:
    """Run FEFF calculation with robust error handling.

    Args:
        feff_dir: Directory containing feff.inp
        verbose: Whether to enable verbose output
        cleanup: Whether to clean up unnecessary output files
        timeout: Timeout in seconds (default: 600 = 10 minutes)

    Returns:
        True if calculation succeeded, False otherwise
    """
    feff_dir = Path(feff_dir)
    input_path = feff_dir / "feff.inp"
    log_path = feff_dir / "feff.log"
    chi_file = feff_dir / "chi.dat"
    phase_file = feff_dir / "phase.pad"
    pot_file = feff_dir / "pot.pad"

    if not input_path.exists():
        raise FileNotFoundError(f"FEFF input file {input_path} not found")

    try:
        # Initialize log file
        _write_log_header(log_path, input_path, feff_dir)

        # Run FEFF calculation
        feff_command = ["feff8l"]
        if verbose:
            feff_command.append("-v")

        _run_feff_subprocess(feff_command, feff_dir, log_path, timeout)

        # Check if calculation produced expected output files
        success = _check_output_files(chi_file, phase_file, pot_file)

        # Clean up if requested and successful
        if success and cleanup:
            cleanup_feff_output(feff_dir, keep_essential=True)

        # Log results
        _write_log_footer(log_path, success, chi_file, phase_file, pot_file)

        return success

    except subprocess.TimeoutExpired as e:
        _handle_timeout_error(log_path, timeout, e)
        return False
    except (subprocess.CalledProcessError, OSError, PermissionError) as e:
        _handle_general_error(log_path, e)
        return False


def _write_log_header(log_path: Path, input_path: Path, feff_dir: Path) -> None:
    """Write initial log header."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"FEFF calculation started at {datetime.now()}\n")
        log_file.write(f"Input file: {input_path}\n")
        log_file.write(f"Working directory: {feff_dir}\n")
        log_file.write("-" * 50 + "\n\n")


def _run_feff_subprocess(
    command: list[str], cwd: Path, log_path: Path, timeout: int
) -> None:
    """Execute FEFF subprocess and capture output."""
    # Simple safety check for command
    if not command or not isinstance(command, list):
        raise ValueError("Invalid FEFF command")
    # Ensure all elements are strings
    if not all(isinstance(arg, str) for arg in command):
        raise ValueError("FEFF command arguments must be strings")

    with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"Running command: {' '.join(command)}\n\n")
        subprocess.run(  # noqa: S603
            command,
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
        )


def _check_output_files(chi_file: Path, phase_file: Path, pot_file: Path) -> bool:
    """Check if expected output files exist.

    Success criteria: chi.dat exists OR both phase.pad and pot.pad exist.
    """
    return chi_file.exists() or (phase_file.exists() and pot_file.exists())


def _write_log_footer(
    log_path: Path,
    success: bool,
    chi_file: Path,
    phase_file: Path,
    pot_file: Path,
) -> None:
    """Write final log summary."""
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"\nCalculation completed at {datetime.now()}\n")
        log_file.write(f"Success: {success}\n")
        log_file.write("\nOutput files check:\n")
        log_file.write(f"  chi.dat: {'✓' if chi_file.exists() else '✗'}\n")
        log_file.write(f"  phase.pad: {'✓' if phase_file.exists() else '✗'}\n")
        log_file.write(f"  pot.pad: {'✓' if pot_file.exists() else '✗'}\n")

        if not success:
            log_file.write(
                "\nWarning: Neither chi.dat nor both phase.pad and pot.pad found\n"
            )


def _handle_timeout_error(log_path: Path, timeout: int, error: Exception) -> None:
    """Handle subprocess timeout errors."""
    error_msg = (
        f"FEFF command timed out after {timeout}s. "
        "Consider increasing the timeout or inspecting the input."
    )
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"\nERROR: {error_msg}\n")
            log_file.write(f"Exception: {error}\n")
    except OSError:
        print(f"FEFF calculation failed: {error_msg}")


def _handle_general_error(log_path: Path, error: Exception) -> None:
    """Handle general errors during FEFF execution."""
    error_msg = f"{type(error).__name__}: {error}"
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"\nERROR: {error_msg}\n")
    except OSError:
        print(f"FEFF calculation failed: {error_msg}")


def get_feff_numbered_files(feff_dir: Path, base_string="feff") -> list[Path]:
    """Get all feff####.dat files (any number of digits)."""
    feff_dir = Path(feff_dir)
    if not feff_dir.exists():
        return []

    # Simple regex: feff + digits + .dat (case insensitive)
    pattern = re.compile(rf"^{base_string}\d+\.dat$", re.IGNORECASE)

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

        # >>> Change begins: use the 'chi' column (2nd column) as the observable χ(k)
        if hasattr(feff_data, "chi"):
            k = np.asarray(feff_data.k)
            chi = np.asarray(feff_data.chi)  # real χ(k) = |χ| sin(phase)
            return chi, k
        # Fallbacks (rare): if 'chi' missing, reconstruct, then take its imaginary part
        if hasattr(feff_data, "mag") and hasattr(feff_data, "phase"):
            k = np.asarray(feff_data.k)
            chi = np.asarray(feff_data.mag) * np.sin(np.asarray(feff_data.phase))
            return chi, k
        raise AttributeError("FEFF data missing 'chi' (and mag/phase) in chi.dat")

    except (OSError, ValueError, AttributeError) as err:
        raise ValueError(f"Error reading FEFF output {chi_file}: {err}") from err
