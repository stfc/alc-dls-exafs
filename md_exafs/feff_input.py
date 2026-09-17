"""FEFF input card generation and configuration management.

Provides unified FeffConfig dataclass and build_feff_inp function (ADR 0004, ADR 0006).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.feff.sets import MPEXAFSSet

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger("md_exafs.feff_input")

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
    """Load preset configurations from bundled YAML files."""
    presets = {}
    preset_dir = Path(__file__).parent / "feff_configs"

    if not YAML_AVAILABLE or not preset_dir.exists():
        return {
            "quick": {"spectrum_type": "EXAFS", "edge": "K", "radius": 4.0},
            "publication": {"spectrum_type": "EXAFS", "edge": "K", "radius": 8.0},
        }

    for yaml_file in preset_dir.glob("*.yaml"):
        preset_name = yaml_file.stem
        try:
            with open(yaml_file) as f:
                preset_config = yaml.safe_load(f)
            if isinstance(preset_config, dict):
                explicit_none_fields = [
                    field.upper()
                    for field in FEFF_CARD_FIELDS
                    if field in preset_config and preset_config[field] is None
                ]
                if explicit_none_fields:
                    existing_delete_tags = preset_config.get("delete_tags", [])
                    if isinstance(existing_delete_tags, str):
                        existing_delete_tags = [existing_delete_tags]
                    elif existing_delete_tags is None:
                        existing_delete_tags = []
                    else:
                        existing_delete_tags = list(existing_delete_tags)

                    for fld in explicit_none_fields:
                        if fld not in existing_delete_tags:
                            existing_delete_tags.append(fld)

                    preset_config["delete_tags"] = existing_delete_tags

                presets[preset_name] = preset_config
        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Failed to load preset from {yaml_file}: {e}")
    return presets


PRESETS = _load_presets()


class SpectrumType(str, Enum):
    """Enumeration of supported spectrum types."""

    EXAFS = "EXAFS"


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

    HANNING = "hanning"
    PARZEN = "parzen"
    WELCH = "welch"
    GAUSSIAN = "gaussian"
    SINE = "sine"
    KAISER = "kaiser"


def _to_str_tokens(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(x) for x in value]
    return value.split() if isinstance(value, str) else [str(value)]


def normalize_tag(name: str, value: Any) -> str:
    """Normalize a FEFF card value to the space-separated string FEFF expects.

    Public API: front-ends that accept user-supplied card values (e.g. the
    ``aiida-feff`` ``FeffParameters`` node) call this so that validation and
    formatting of a card is identical whether the value reaches FEFF through
    this library or through a wrapper.

    Args:
        name: FEFF card name, case-insensitive (e.g. ``"s02"``, ``"SCF"``).
        value: Scalar, string, or sequence of tokens.

    Returns:
        The card's value rendered as a single space-separated string.

    Raises:
        ValueError: If the value is outside the range FEFF accepts for that card.
    """
    key = name.strip().upper()
    tokens = _to_str_tokens(value)

    if key == "S02":
        s02 = float(tokens[0])
        if s02 < 0:
            raise ValueError(f"S02 must be >= 0, got {s02}")
        return str(s02)

    if key == "EXAFS":
        exafs = int(float(tokens[0]))
        if exafs <= 0:
            raise ValueError(f"EXAFS must be > 0, got {exafs}")
        return str(exafs)

    if key in ("PRINT", "CONTROL"):
        try:
            return " ".join(str(int(float(t))) for t in tokens)
        except Exception as exc:
            raise ValueError(f"{key} expects integer tokens, got {value!r}") from exc

    if key == "SCF":
        scf_max_tokens = 5
        if not (1 <= len(tokens) <= scf_max_tokens):
            raise ValueError(
                f"SCF expects 1–{scf_max_tokens} tokens, got {len(tokens)}: {value!r}"
            )

        def _fmt(x: float) -> str:
            s = str(float(x))
            return s.rstrip("0").rstrip(".") if "." in s else str(int(float(x)))

        rfms1 = float(tokens[0])
        if rfms1 <= 0:
            raise ValueError(f"SCF rfms1 must be > 0, got {rfms1}")
        lfms1 = int(float(tokens[1])) if len(tokens) >= 2 else 0
        nscmt = int(float(tokens[2])) if len(tokens) >= 3 else 30
        ca = float(tokens[3]) if len(tokens) >= 4 else 0.2
        nmix = int(float(tokens[4])) if len(tokens) >= 5 else 1
        return " ".join([_fmt(rfms1), str(lfms1), str(nscmt), _fmt(ca), str(nmix)])

    if key == "NLEG":
        nleg = int(float(tokens[0]))
        if nleg <= 0:
            raise ValueError(f"NLEG must be > 0, got {nleg}")
        return str(nleg)

    if key == "EXCHANGE":
        if len(tokens) == 1 and tokens[0] == "0":
            return "0 0 0"
        return " ".join(tokens)

    return " ".join(tokens)


@dataclass
class FeffConfig:
    """Unified configuration class for FEFF calculations and analysis."""

    spectrum_type: str = "EXAFS"
    edge: str = "K"
    radius: float = 4.0
    exclude_hydrogen: bool = False

    # Explicit FEFF card fields
    control: Any | None = None
    print: Any | None = "1 0 0 0 0 3"
    s02: Any | None = 1.0
    scf: Any | None = None
    exchange: Any | None = 0
    nleg: Any | None = 6
    exafs: Any | None = None
    criteria: Any | None = None
    delete_tags: list[str] | str | None = None

    # Fourier transform parameters
    kmin: float = 2.0
    kmax: float = 12.0
    kweight: int = 2
    dk: float = 1.0
    dk2: float | None = None
    with_phase: bool = False
    rmax_out: float = 10.0
    window: WindowType | str = WindowType.HANNING
    nfft: int | None = None
    kstep: float | None = None

    # Execution settings
    parallel: bool = False
    n_workers: int | None = None
    sample_interval: int = 1
    force_recalculate: bool = False
    cleanup_feff_files: bool = True
    keep_path_files: bool = False
    max_paths: int | None = None
    stream_chunk_size: int | None = 256
    potential_link_mode: str = "symlink"
    store_path_params: bool = False
    store_min_cw_ratio: float | None = None

    @property
    def fourier_params(self) -> dict[str, Any]:
        """Return Fourier transform parameters as a dictionary."""
        params: dict[str, Any] = {
            "kmin": self.kmin,
            "kmax": self.kmax,
            "kweight": self.kweight,
            "dk": self.dk,
            "dk2": self.dk2,
            "with_phase": self.with_phase,
            "window": self.window.value
            if isinstance(self.window, WindowType)
            else self.window,
            "rmax_out": self.rmax_out,
            "nfft": self.nfft,
            "kstep": self.kstep,
        }
        return {k: v for k, v in params.items() if v is not None}

    @property
    def feff_params(self) -> dict[str, Any]:
        """Return FEFF calculation parameters affecting the calculation itself."""
        params: dict[str, Any] = {
            "spectrum_type": self.spectrum_type,
            "edge": self.edge,
            "radius": self.radius,
            "exclude_hydrogen": self.exclude_hydrogen,
        }
        cards = self.to_pymatgen_user_tags()
        if cards:
            params.update(cards)
        return params

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.kmin >= self.kmax:
            raise ValueError(f"kmin ({self.kmin}) must be less than kmax ({self.kmax})")
        if self.kmin < 0:
            raise ValueError(f"kmin must be positive, got {self.kmin}")
        if self.dk <= 0:
            raise ValueError(f"dk must be positive, got {self.dk}")
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got {self.radius}")
        if self.n_workers is not None and self.n_workers <= 0:
            raise ValueError(f"Invalid n_workers: {self.n_workers}")
        if self.sample_interval < 1:
            raise ValueError(
                f"sample_interval must be >= 1, got {self.sample_interval}"
            )

    def to_pymatgen_user_tags(self) -> dict[str, Any]:
        """Build user_tag_settings dict for pymatgen MPEXAFSSet."""
        user_tags: dict[str, Any] = {}
        cards = {
            "CONTROL": self.control,
            "PRINT": self.print,
            "S02": self.s02,
            "SCF": self.scf,
            "EXCHANGE": self.exchange,
            "NLEG": self.nleg,
            "EXAFS": self.exafs,
            "CRITERIA": self.criteria,
        }
        for name, val in cards.items():
            if val is not None:
                user_tags[name] = normalize_tag(name, val)

        del_list: list[str] = []
        if isinstance(self.delete_tags, str):
            del_list = [self.delete_tags]
        elif isinstance(self.delete_tags, list):
            del_list = list(self.delete_tags)

        # Incompatible with FEFF8L
        for card in ("COREHOLE", "COREHOLE FSR"):
            if card not in del_list:
                del_list.append(card)

        if del_list:
            user_tags["_del"] = del_list
        return user_tags

    @classmethod
    def from_preset(cls, preset_name: str) -> FeffConfig:
        """Create configuration from a named preset."""
        if preset_name not in PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}"
            )
        return cls(**PRESETS[preset_name])

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> FeffConfig:
        """Load configuration from a YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load configuration from YAML.")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid configuration file {yaml_path}")
        return cls(**data)


def make_potentials_feff_config(base_config: FeffConfig) -> FeffConfig:
    """Create a FeffConfig configured to compute potentials only (CONTROL 1 1 1 0 0 0)."""
    cfg = copy.deepcopy(base_config)
    cfg.control = "1 1 1 0 0 0"
    cfg.print = "1 0 0 0 0 0"
    return cfg


def make_paths_feff_config(base_config: FeffConfig) -> FeffConfig:
    """Create a FeffConfig configured to reuse potentials and compute paths (CONTROL 0 0 0 1 1 1)."""
    cfg = copy.deepcopy(base_config)
    cfg.control = "0 0 0 1 1 1"
    return cfg


def build_feff_inp(
    structure: Atoms | Structure,
    config: FeffConfig | None = None,
    absorber_idx: int = 0,
    output_path: Path | str | None = None,
) -> str:
    """Generate the full text of feff.inp for a structure and absorbing site.

    Automatically prunes COREHOLE tags for FEFF8L compatibility.

    Args:
        structure: ASE Atoms or pymatgen Structure object.
        config: FeffConfig configuration (defaults to standard quick preset).
        absorber_idx: 0-based index of the absorbing atom.
        output_path: Optional path or directory to write feff.inp to.

    Returns:
        The content of the generated feff.inp as a string.
    """
    if config is None:
        config = FeffConfig()

    # Convert ASE Atoms to pymatgen Structure if necessary
    if isinstance(structure, Atoms):
        structure = structure.copy()
        if structure.cell is None or structure.cell.volume == 0:
            structure.center(vacuum=10.0)

        if config.exclude_hydrogen:
            non_h = [atom.index for atom in structure if atom.symbol != "H"]
            if absorber_idx not in non_h:
                raise ValueError(
                    f"Absorber index {absorber_idx} is a hydrogen atom "
                    "which is excluded by exclude_hydrogen=True."
                )
            absorber_idx = non_h.index(absorber_idx)
            structure = structure[non_h]
        pmg_structure = AseAtomsAdaptor().get_structure(structure)
    elif isinstance(structure, Structure):
        pmg_structure = structure.copy()
        if config.exclude_hydrogen:
            symbols = [site.species_string for site in pmg_structure.sites]
            non_h = [i for i, sym in enumerate(symbols) if sym != "H"]
            if absorber_idx not in non_h:
                raise ValueError(
                    f"Absorber index {absorber_idx} is a hydrogen atom "
                    "which is excluded by exclude_hydrogen=True."
                )
            absorber_idx = non_h.index(absorber_idx)
            pmg_structure.remove_sites(
                [i for i, sym in enumerate(symbols) if sym == "H"]
            )
    else:
        raise TypeError(
            f"Expected ASE Atoms or pymatgen Structure, got {type(structure)}"
        )

    if not (0 <= absorber_idx < len(pmg_structure)):
        raise ValueError(
            f"Absorber index {absorber_idx} out of range (0-{len(pmg_structure) - 1})"
        )

    user_settings = config.to_pymatgen_user_tags()
    user_settings["RPATH"] = str(config.radius)

    feff_set = MPEXAFSSet(
        absorbing_atom=absorber_idx,
        structure=pmg_structure,
        edge=config.edge,
        radius=config.radius,
        user_tag_settings=user_settings,
    )
    all_input = feff_set.all_input()
    blocks = [
        str(all_input[k])
        for k in ["HEADER", "PARAMETERS", "POTENTIALS", "ATOMS"]
        if k in all_input
    ]
    inp_content = "\n\n".join(blocks) + "\n"

    if output_path is not None:
        target = Path(output_path)
        if target.is_dir() or not target.suffix:
            target.mkdir(parents=True, exist_ok=True)
            target = target / "feff.inp"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(inp_content, encoding="utf-8")

    return inp_content


__all__ = [
    "FeffConfig",
    "SpectrumType",
    "EdgeType",
    "WindowType",
    "PRESETS",
    "FEFF_CARD_FIELDS",
    "make_potentials_feff_config",
    "make_paths_feff_config",
    "build_feff_inp",
    "normalize_tag",
]
