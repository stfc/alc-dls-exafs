"""Larch Wrapper - A lightweight wrapper around larch for EXAFS processing"""

import gc
import logging
import multiprocessing as mp
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from larch import Group
from larch.io import read_ascii
from larch.xafs import xftf
from larch.xafs.feffrunner import feff8l, find_exe
from larixite import cif2feffinp

# YAML support for parameter sets
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
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
    from pymatgen.io.xyz import XYZ

    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
# Optional performance optimization
try:
    import numba

    NUMBA_AVAILABLE = True

    @numba.jit(nopython=True, cache=True)
    def _fast_average_chi(chi_arrays):
        """Fast averaging for trajectory processing using numba."""
        return np.mean(chi_arrays, axis=0)

    @numba.jit(nopython=True, cache=True)
    def _fast_std_chi(chi_arrays):
        """Fast standard deviation calculation using numba."""
        return np.std(chi_arrays, axis=0)

except ImportError:
    NUMBA_AVAILABLE = False

    def _fast_average_chi(chi_arrays):
        """Fallback averaging without numba."""
        return np.mean(chi_arrays, axis=0)

    def _fast_std_chi(chi_arrays):
        """Fallback standard deviation without numba."""
        return np.std(chi_arrays, axis=0)


def _process_frame_worker(
    frame_data: tuple,
) -> tuple[np.ndarray | None, np.ndarray | None, int, str | None]:
    """Worker function for processing a single trajectory frame in parallel.

    Args:
        frame_data: Tuple containing (frame_index, atoms, absorber, output_base, kwargs)

    Returns:
        Tuple of (chi, k, frame_index, error_message)
    """
    try:
        frame_index, atoms, absorber, output_base, kwargs = frame_data
        # Import required modules in worker process
        import tempfile
        from pathlib import Path

        from ase.io import write as ase_write

        # Create temporary structure file
        with tempfile.TemporaryDirectory() as tmpdir:
            struct_path = Path(tmpdir) / f"frame_{frame_index}.cif"
            ase_write(str(struct_path), atoms, format="cif")
            frame_output = Path(output_base) / f"frame_{frame_index:04d}"
            # Create a new LarchWrapper instance for this process
            wrapper = LarchWrapper(
                verbose=False
            )  # Reduce verbosity in parallel workers
            try:
                exafs_group, _ = wrapper.run_full_pipeline(
                    structure_path=struct_path,
                    absorber=absorber,
                    output_dir=frame_output,
                    show_plot=False,
                    **kwargs,
                )
                return exafs_group.chi, exafs_group.k, frame_index, None
            except Exception as e:
                return None, None, frame_index, str(e)
            finally:
                wrapper.cleanup_temp_files()
    except Exception as e:
        return None, None, frame_index, str(e)


# Constants for supported spectrum types and default parameters
ADVANCED_SPECTRUM_TYPES = {
    "XANES",
    "DANES",
    "XMCD",
    "ELNES",
    "EXELFS",
    "FPRIME",
    "NRIXS",
    "XES",
}
BASIC_SPECTRUM_TYPES = {"EXAFS"}
ALL_SPECTRUM_TYPES = ADVANCED_SPECTRUM_TYPES | BASIC_SPECTRUM_TYPES

# Default parameters for different operations
DEFAULT_FEFF_PARAMS = {"spectrum_type": "EXAFS", "edge": "K", "radius": 10.0}
DEFAULT_FT_PARAMS = {"kweight": 2, "window": "hanning", "dk": 1, "kmin": 2, "kmax": 14}

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


def load_config_parameter_set(config_path: Path) -> dict[str, Any]:
    """Load FEFF parameters from a configuration file in YAML format.

    Args:
        config_path: Path to configuration parameter file
    Returns:
        Dictionary with FEFF parameters and wrapper configuration
    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If YAML format is invalid
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for configuration parameter sets. Install with: pip install pyyaml"
        )
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration parameter file {config_path} not found")
    try:
        with open(config_path) as f:
            yaml_params = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format in {config_path}: {e}")
    # Convert pymatgen YAML format to wrapper parameters
    wrapper_params = _convert_pymatgen_yaml_to_wrapper_params(yaml_params)
    return wrapper_params


def _convert_pymatgen_yaml_to_wrapper_params(yaml_params: dict) -> dict[str, Any]:
    """Convert pymatgen-style YAML parameters to wrapper parameters.

    Args:
        yaml_params: Raw parameters from YAML file
    Returns:
        Dictionary with wrapper-compatible parameters
    """
    wrapper_params = {}
    user_tag_settings = {}
    # Map common FEFF parameters
    feff_param_mapping = {
        "EDGE": "edge",
        "RPATH": "radius",
        "S02": ("user_tag_settings", "S02"),
        "SCF": ("user_tag_settings", "SCF"),
        "CONTROL": ("user_tag_settings", "CONTROL"),
        "PRINT": ("user_tag_settings", "PRINT"),
        "COREHOLE": ("user_tag_settings", "COREHOLE"),
        "EXAFS": ("user_tag_settings", "EXAFS"),
        "XANES": ("user_tag_settings", "XANES"),
        "POLARIZATION": ("user_tag_settings", "POLARIZATION"),
        "NLEG": ("user_tag_settings", "NLEG"),
        "CRITERIA": ("user_tag_settings", "CRITERIA"),
        "FMS": ("user_tag_settings", "FMS"),
        "LDOS": ("user_tag_settings", "LDOS"),
        "RECIPROCAL": ("user_tag_settings", "RECIPROCAL"),
    }
    # Process YAML parameters
    for yaml_key, yaml_value in yaml_params.items():
        if yaml_key in feff_param_mapping:
            mapping = feff_param_mapping[yaml_key]
            if isinstance(mapping, tuple):
                # Goes into user_tag_settings
                if mapping[0] == "user_tag_settings":
                    # Convert value to string for FEFF
                    if isinstance(yaml_value, (list, tuple)):
                        user_tag_settings[mapping[1]] = " ".join(map(str, yaml_value))
                    else:
                        user_tag_settings[mapping[1]] = str(yaml_value)
            else:
                # Direct parameter mapping
                if yaml_key == "EDGE":
                    wrapper_params["edge"] = str(yaml_value)
                elif yaml_key == "RPATH":
                    wrapper_params["radius"] = float(yaml_value)
        else:
            # Unknown parameter - add to user_tag_settings
            if isinstance(yaml_value, (list, tuple)):
                user_tag_settings[yaml_key] = " ".join(map(str, yaml_value))
            else:
                user_tag_settings[yaml_key] = str(yaml_value)
    # Determine spectrum type from parameters
    if "XANES" in yaml_params or "FMS" in yaml_params:
        wrapper_params["spectrum_type"] = "XANES"
    elif "EXAFS" in yaml_params:
        wrapper_params["spectrum_type"] = "EXAFS"
    else:
        wrapper_params["spectrum_type"] = "EXAFS"  # Default
    if user_tag_settings:
        wrapper_params["user_tag_settings"] = user_tag_settings
    return wrapper_params


def save_config_parameter_set(
    parameters: dict[str, Any], config_path: Path, description: str = None
) -> None:
    """Save wrapper parameters to a configuration file in YAML format.

    Args:
        parameters: Wrapper parameters dictionary
        config_path: Path to save configuration file
        description: Optional description comment
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for configuration parameter sets. Install with: pip install pyyaml"
        )
    # Convert wrapper parameters back to pymatgen YAML format
    yaml_params = _convert_wrapper_params_to_yaml(parameters)
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        if description:
            f.write(f"# {description}\n")
        yaml.dump(yaml_params, f, default_flow_style=False, sort_keys=True)


def _convert_wrapper_params_to_yaml(parameters: dict[str, Any]) -> dict:
    """Convert wrapper parameters back to pymatgen YAML format."""
    yaml_params = {}
    # Extract user_tag_settings first
    user_tags = parameters.get("user_tag_settings", {})
    # Add user tags directly
    for key, value in user_tags.items():
        if " " in str(value):
            # Convert space-separated strings back to lists for some parameters
            if key in ["CONTROL", "PRINT", "POLARIZATION"]:
                yaml_params[key] = value.split()
            else:
                yaml_params[key] = value
        else:
            yaml_params[key] = value
    # Add direct mappings
    if "edge" in parameters:
        yaml_params["EDGE"] = parameters["edge"]
    if "radius" in parameters:
        yaml_params["RPATH"] = parameters["radius"]
    # Add spectrum-specific parameters
    spectrum_type = parameters.get("spectrum_type", "EXAFS")
    if spectrum_type == "EXAFS":
        yaml_params.setdefault("EXAFS", 20)
    elif spectrum_type == "XANES":
        yaml_params.setdefault("FMS", 9.0)
    return yaml_params


class ProgressReporter:
    """Thread-safe progress reporter for trajectory processing."""

    def __init__(self, total_frames: int = None):
        self.completed = 0
        self.total = total_frames
        self.description = "Processing frames..."
        self.lock = mp.Lock()

    def update(
        self, completed: int = None, total: int = None, description: str = None
    ) -> None:
        """Update progress with thread safety."""
        with self.lock:
            if total is not None:
                self.total = total
            if description is not None:
                self.description = description
            if completed is not None:
                if self.total is None:
                    # Indeterminate progress
                    self.completed = completed
                else:
                    # Determinate progress - clamp to total
                    self.completed = min(completed, self.total)

    def get_progress(self) -> tuple[int, int, str]:
        """Get current progress state."""
        with self.lock:
            return self.completed, self.total, self.description


class LarchWrapper:
    """A wrapper class for common larch EXAFS operations."""

    def __init__(self, verbose: bool = True):
        self.last_exafs_group = None
        self._temp_files = []
        self.config = DEFAULT_FEFF_PARAMS.copy()
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        # Log FEFF version information
        try:
            feff_version = self.get_feff_version()
            self.logger.info(f"Using FEFF version: {feff_version}")
        except Exception:
            self.logger.warning("Could not determine FEFF version")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_files()

    def __del__(self):
        """Clean up temporary files."""
        self.cleanup_temp_files()

    def _validate_dependencies(self, method: str, spectrum_type: str):
        """Validate that required dependencies are available."""
        if method == "pymatgen" and not PYMATGEN_AVAILABLE:
            raise ImportError("pymatgen is required for this method but not installed")
        if spectrum_type.upper() in ADVANCED_SPECTRUM_TYPES and not PYMATGEN_AVAILABLE:
            self.logger.warning(
                f"Advanced spectrum type {spectrum_type} works best with pymatgen"
            )

    def _validate_structure_file(self, structure_path: Path):
        """Validate structure file exists and is readable."""
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file {structure_path} not found")
        if not structure_path.is_file():
            raise ValueError(f"{structure_path} is not a file")
        if structure_path.stat().st_size == 0:
            raise ValueError(f"Structure file {structure_path} is empty")

    def cleanup_temp_files(self):
        """Remove any temporary files created during processing and free memory."""
        # Clear references to large data structures
        self.last_exafs_group = None
        # Remove temporary files - iterate over copy to avoid modification during iteration
        for temp_file in self._temp_files[:]:  # Create a copy of the list
            try:
                if temp_file.exists():
                    temp_file.unlink()  # For compatibility with older Python versions
                self._temp_files.remove(temp_file)  # Remove from original list
            except Exception as e:
                self.logger.warning(f"Could not remove temp file {temp_file}: {e}")
                # Don't remove from list if deletion failed - try again later
        # Force garbage collection
        gc.collect()

    def get_feff_version(self) -> str:
        """Get FEFF version information using available executables."""
        # Try different FEFF executables in order of preference
        feff_executables = ["feff8l_rdinp", "feff6l", "feff8l", "feff"]
        for exe_name in feff_executables:
            try:
                exe_path = find_exe(exe_name)
                if exe_path:
                    # Try to run the executable with version flag or help
                    import subprocess

                    try:
                        # Try --version flag first
                        result = subprocess.run(
                            [str(exe_path), "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if result.stdout and "feff" in result.stdout.lower():
                            return f"FEFF ({result.stdout.strip().split()[0]})"
                    except (
                        subprocess.TimeoutExpired,
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                    ):
                        pass
                    # If version flag doesn't work, just report that we found the executable
                    return f"FEFF ({exe_name} executable found)"
            except Exception:
                continue
        return "FEFF (no executable found)"

    def get_available_feff_executables(self) -> dict[str, str]:
        """Get a dictionary of available FEFF executables and their paths."""
        feff_executables = [
            "feff8l_rdinp",
            "feff8l_pot",
            "feff8l_xsph",
            "feff8l_pathfinder",
            "feff8l_genfmt",
            "feff8l_ff2x",
            "feff8l",
            "feff6l",
            "feff",
        ]
        available = {}
        for exe_name in feff_executables:
            try:
                exe_path = find_exe(exe_name)
                if exe_path:
                    available[exe_name] = str(exe_path)
            except Exception as e:
                self.logger.debug(f"Could not find {exe_name}: {e}")
        return available

    def run_diagnostics(self) -> dict[str, Any]:
        """Run comprehensive diagnostics to help troubleshoot installation issues.

        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            "system_info": {
                "platform": sys.platform,
                "python_version": sys.version,
            },
            "dependencies": {
                "ase_available": True,  # Required dependency
                "pymatgen_available": PYMATGEN_AVAILABLE,
                "larch_available": True,  # If we got here, larch is available
                "numba_available": NUMBA_AVAILABLE,  # Performance optimization
            },
            "feff_info": {
                "version": self.get_feff_version(),
                "executables": self.get_available_feff_executables(),
            },
            "capabilities": self.get_capabilities(),
            "supported_formats": "Any ASE-readable format (CIF, XYZ, etc.)",
            "presets": {name: PRESETS[name] for name in PRESETS.keys()},
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
        if len(diagnostics["feff_info"]["executables"]) == 0:
            warnings.append("No FEFF executables found - FEFF calculations will fail")
            recommendations.append(
                "Check larch installation and ensure FEFF binaries are available"
            )
        elif len(diagnostics["feff_info"]["executables"]) < 6:
            warnings.append(
                "Some FEFF executables missing - some functionality may be limited"
            )
        diagnostics["warnings"] = warnings
        diagnostics["recommendations"] = recommendations
        return diagnostics

    def print_diagnostics(self):
        """Print a formatted diagnostic report."""
        diagnostics = self.run_diagnostics()
        print("=" * 60)
        print("LARCH WRAPPER DIAGNOSTICS")
        print("=" * 60)
        print("\n\U0001f5a5\ufe0f SYSTEM INFO:")
        print(f"   Platform: {diagnostics['system_info']['platform']}")
        print(f"   Python: {diagnostics['system_info']['python_version'].split()[0]}")
        print("\n\U0001f4e6 DEPENDENCIES:")
        for dep, available in diagnostics["dependencies"].items():
            status = "\u2705" if available else "\u274c"
            print(f"   {status} {dep}: {available}")
        print("\n\u26a1 PERFORMANCE:")
        print(f"   CPU cores: {mp.cpu_count()}")
        print("   Parallel processing: \u2705 Available")
        if NUMBA_AVAILABLE:
            print("   Numba optimization: \u2705 Available")
        else:
            print("   Numba optimization: \u274c Not available")
        print("\n\u269b\ufe0f FEFF INFO:")
        print(f"   Version: {diagnostics['feff_info']['version']}")
        print(f"   Executables found: {len(diagnostics['feff_info']['executables'])}")
        if diagnostics["feff_info"]["executables"]:
            print("   Available executables:")
            for exe in diagnostics["feff_info"]["executables"].keys():
                print(f"     \U0001f539 {exe}")
        print("\n\u2699\ufe0f PRESETS AVAILABLE:")
        for preset in diagnostics["presets"].keys():
            print(f"   \U0001f539 {preset}")
        if diagnostics["warnings"]:
            print("\n\u26a0\ufe0f WARNINGS:")
            for warning in diagnostics["warnings"]:
                print(f"   \u26a0\ufe0f {warning}")
        if diagnostics["recommendations"]:
            print("\n\U0001f4a1 RECOMMENDATIONS:")
            for rec in diagnostics["recommendations"]:
                print(f"   \U0001f4a1 {rec}")
        print("\n\u2705 Diagnostics complete!")
        print("=" * 60)

    def _validate_parameters(self, spectrum_type: str, edge: str):
        """Validate spectrum type and edge parameters."""
        spectrum_type = spectrum_type.upper()
        if spectrum_type not in ALL_SPECTRUM_TYPES:
            raise ValueError(
                f"Invalid spectrum type '{spectrum_type}'. Valid options: {', '.join(sorted(ALL_SPECTRUM_TYPES))}"
            )
        supported_edges = ["K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"]
        if edge not in supported_edges:
            raise ValueError(
                f"Unsupported edge '{edge}'. Valid options: {', '.join(supported_edges)}"
            )

    def _validate_absorber(self, absorber: str, structure_path: Path = None):
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

    def use_preset(self, preset_name: str):
        """Apply a configuration preset."""
        if preset_name not in PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available presets: {', '.join(PRESETS.keys())}"
            )
        self.config.update(PRESETS[preset_name])
        self.logger.info(f"Applied preset '{preset_name}'")

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
            raise RuntimeError(f"Failed to convert structure to CIF: {e}")

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
                    )

    def get_capabilities(self) -> dict[str, Any]:
        """Get information about available capabilities."""
        return {
            "ase_available": True,  # ASE is now required
            "pymatgen_available": PYMATGEN_AVAILABLE,
            "numba_available": NUMBA_AVAILABLE,
            "parallel_processing": True,
            "max_workers": mp.cpu_count(),
            "supported_spectrum_types": list(ALL_SPECTRUM_TYPES),
            "advanced_types_need_pymatgen": list(ADVANCED_SPECTRUM_TYPES),
            "supported_edges": ["K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"],
            "default_feff_parameters": DEFAULT_FEFF_PARAMS.copy(),
            "default_ft_parameters": DEFAULT_FT_PARAMS.copy(),
            "available_presets": list(PRESETS.keys()),
            "trajectory_support": True,
            "trajectory_optimization": NUMBA_AVAILABLE,
            "trajectory_parallel_processing": True,
            "cross_platform": True,
            "feff_version": self.get_feff_version(),
            "available_feff_executables": self.get_available_feff_executables(),
        }

    def load_config_parameters(self, config_path: Path) -> dict[str, Any]:
        """Load FEFF parameters from a configuration file and update configuration.

        Args:
            config_path: Path to configuration parameter file
        Returns:
            Dictionary with loaded parameters
        """
        parameters = load_config_parameter_set(config_path)
        self.config.update(parameters)
        self.logger.info(f"Loaded configuration parameter set from {config_path}")
        return parameters

    def save_config_parameters(
        self, config_path: Path, parameters: dict | None = None, description: str = None
    ) -> None:
        """Save current or provided parameters to a configuration file.

        Args:
            config_path: Path to save configuration file
            parameters: Parameters to save (uses current config if None)
            description: Optional description comment
        """
        params_to_save = parameters if parameters is not None else self.config
        save_config_parameter_set(params_to_save, config_path, description)
        self.logger.info(f"Saved configuration parameter set to {config_path}")

    def generate_feff_input(
        self,
        structure_path: Path | None = None,
        absorber: str = None,
        output_dir: Path = None,
        filename: str = "feff.inp",
        atoms: Atoms | None = None,
        spectrum_type: str = "EXAFS",
        edge: str = "K",
        radius: float = 10.0,
        user_tag_settings: dict | None = None,
        method: str = "auto",
        config_parameters: Path | None = None,
    ) -> Path:
        """Generate FEFF input file using the best available method.

        Args:
            structure_path: Path to structure file (optional if atoms provided)
            absorber: Absorbing atom symbol or site index
            output_dir: Directory to save FEFF input
            filename: Name of the output FEFF input file
            atoms: ASE Atoms object (alternative to structure_path)
            spectrum_type: Type of spectrum ('EXAFS', 'XANES', etc.)
            edge: Absorption edge
            radius: Cluster radius in Angstroms
            user_tag_settings: Custom FEFF parameters
            method: Method preference ('auto', 'larixite', 'pymatgen')
            config_parameters: Path to configuration parameter file (overrides other parameters)

        Returns:
            Path to generated FEFF input file
        """
        # Handle direct ASE atoms input
        if atoms is not None:
            # Create temporary CIF from atoms
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                ase_write(tmp.name, atoms, format="cif")
                structure_path = Path(tmp.name)
                self._temp_files.append(structure_path)
        # Load configuration parameters if provided (takes precedence)
        if config_parameters is not None:
            config_params = load_config_parameter_set(config_parameters)
            self.logger.info(f"Using configuration parameters from {config_parameters}")
            # Override parameters with configuration values
            spectrum_type = config_params.get("spectrum_type", spectrum_type)
            edge = config_params.get("edge", edge)
            radius = config_params.get("radius", radius)
            # Merge user_tag_settings
            config_user_tags = config_params.get("user_tag_settings", {})
            if user_tag_settings:
                merged_tags = {
                    **config_user_tags,
                    **user_tag_settings,
                }  # CLI args take precedence
            else:
                merged_tags = config_user_tags
            user_tag_settings = merged_tags if merged_tags else user_tag_settings
        # Validate inputs
        if structure_path is None:
            raise ValueError("Either structure_path or atoms must be provided")
        structure_path = Path(structure_path).resolve()
        output_dir = Path(output_dir).resolve()
        self._validate_structure_file(structure_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        spectrum_type = spectrum_type.upper()
        self._validate_parameters(spectrum_type, edge)
        absorber = self._validate_absorber(absorber, structure_path)
        # Auto-select method based on requirements and availability
        if method == "auto":
            method = self._select_method(spectrum_type)
        # Validate dependencies for chosen method
        self._validate_dependencies(method, spectrum_type)
        if method == "pymatgen":
            return self._generate_feff_with_pymatgen(
                structure_path,
                absorber,
                output_dir,
                spectrum_type,
                edge,
                radius,
                user_tag_settings,
                filename,
            )
        else:  # larixite
            return self._generate_feff_with_larixite(
                structure_path, absorber, output_dir, filename
            )

    def _select_method(self, spectrum_type: str) -> str:
        """Select the best method for FEFF input generation."""
        if spectrum_type in ADVANCED_SPECTRUM_TYPES:
            if PYMATGEN_AVAILABLE:
                self.logger.info(
                    f"Using pymatgen for {spectrum_type} spectrum generation"
                )
                return "pymatgen"
            else:
                self.logger.warning(
                    f"pymatgen not available for {spectrum_type}, falling back to larixite"
                )
        self.logger.info(
            "Using larixite for FEFF input generation (better larch compatibility)"
        )
        return "larixite"

    def _generate_feff_with_larixite(
        self,
        structure_path: Path,
        absorber: str,
        output_dir: Path,
        filename: str = "feff.inp",
    ) -> Path:
        """Generate FEFF input using larixite (CIF-based)."""
        # Convert to CIF if needed
        if structure_path.suffix.lower() != ".cif":
            self.logger.info(f"Converting {structure_path} to CIF format...")
            cif_name = structure_path.stem + "_converted.cif"
            cif_path = output_dir / cif_name
            temp_cif = self._create_temp_cif(structure_path)
            # Copy to persistent location
            cif_path.write_text(temp_cif.read_text())
            self.logger.info(f"Structure converted to CIF: {cif_path}")
        else:
            cif_path = structure_path

        # Generate FEFF input
        inp = cif2feffinp(cif_path, absorber)
        input_file_path = output_dir / filename
        input_file_path.write_text(inp)
        return input_file_path

    def _generate_feff_with_pymatgen(
        self,
        structure_path: Path,
        absorber: str,
        output_dir: Path,
        spectrum_type: str,
        edge: str,
        radius: float,
        user_tag_settings: dict | None,
        filename: str = "feff.inp",
    ) -> Path:
        """Generate FEFF input using pymatgen."""
        structure = self._load_structure_with_pymatgen(structure_path)

        # Remove COREHOLE if present in user_tag_settings
        if user_tag_settings and "COREHOLE" in user_tag_settings:
            self.logger.warning(
                "COREHOLE setting is not supported in FEFF8L, removing it"
            )
            user_tag_settings.pop("COREHOLE")

        # Select appropriate FEFF input set
        if spectrum_type == "EXAFS":
            feff_set = MPEXAFSSet(
                absorbing_atom=absorber,
                structure=structure,
                edge=edge,
                radius=radius,
                user_tag_settings=user_tag_settings or {},
            )
        elif spectrum_type == "XANES":
            feff_set = MPXANESSet(
                absorbing_atom=absorber,
                structure=structure,
                edge=edge,
                radius=radius,
                user_tag_settings=user_tag_settings or {},
            )
        else:
            raise ValueError(f"Unsupported spectrum type: {spectrum_type}")
        feff_set.write_input(str(output_dir))
        return output_dir / filename

    def run_feff(
        self,
        feff_directory: Path,
        input_filename: str = "feff.inp",
        verbose: bool = True,
    ) -> bool:
        """Run FEFF calculation with improved error handling and cross-platform support."""
        feff_directory = Path(feff_directory).resolve()
        input_path = feff_directory / input_filename
        if not input_path.exists():
            raise FileNotFoundError(f"FEFF input file {input_path} not found")
        # Store original stdout for restoration
        original_stdout = sys.stdout
        try:
            # Fix stdout encoding issues, especially on Windows
            if (
                sys.platform.startswith("win")
                or sys.stdout.encoding is None
                or str(sys.stdout.encoding).lower() == "none"
            ):
                self.logger.debug("Fixing stdout encoding for FEFF execution")

                # Create a wrapper for stdout with proper encoding
                class EncodingWrapper:
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

                sys.stdout = EncodingWrapper(sys.stdout)
            # Run FEFF calculation
            self.logger.info("Starting FEFF calculation...")
            result = feff8l(
                folder=str(feff_directory.resolve()),
                feffinp=input_filename,
                verbose=verbose,
            )
            # Check for errors in FEFF output
            if isinstance(result, str):
                result_lower = result.lower()
                if any(
                    error_word in result_lower
                    for error_word in ["error", "fatal", "abort", "failed"]
                ):
                    self.logger.error("FEFF terminated with errors")
                    if verbose:
                        self.logger.error(f"FEFF output: {result}")
                    return False
            # Check if expected output files were created
            chi_file = feff_directory / "chi.dat"
            if not chi_file.exists():
                self.logger.error("FEFF did not produce expected output file 'chi.dat'")
                return False
            self.logger.info("FEFF calculation completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"FEFF calculation failed: {e}")
            if verbose:
                import traceback

                traceback.print_exc()
            return False
        finally:
            # Always restore original stdout
            sys.stdout = original_stdout

    def process_feff_output(self, feff_directory: Path, **ft_kwargs) -> Group:
        """Process FEFF output and perform Fourier transform."""
        chi_file = feff_directory / "chi.dat"
        if not chi_file.exists():
            raise FileNotFoundError(f"FEFF output file {chi_file} not found")
        feff_data = read_ascii(str(chi_file))
        # Create larch Group and perform FT with default parameters
        g = Group()
        g.k = feff_data.k
        g.chi = feff_data.chi
        # Merge default and user parameters
        ft_params = {**DEFAULT_FT_PARAMS, **ft_kwargs}
        xftf(g, **ft_params)
        self.last_exafs_group = g
        return g

    def read_chi_dat(self, feff_directory: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read chi.dat file from FEFF output directory
        Args:
            feff_directory: Path to FEFF output directory
        Returns:
            Tuple of (k, chi) arrays
        """
        chi_file = feff_directory / "chi.dat"
        if not chi_file.exists():
            raise FileNotFoundError(f"chi.dat not found in {feff_directory}")
        data = read_ascii(str(chi_file))
        return data.k, data.chi

    def average_feff_runs(
        self, feff_dirs: list[Path], force: bool = False, ft_params: dict | None = None
    ) -> Group:
        """Average multiple FEFF runs from existing directories
        Args:
            feff_dirs: List of paths to FEFF run directories
            force: Re-run FEFF calculations if True
            ft_params: Parameters for Fourier transform (optional)

        Returns:
            Group containing averaged EXAFS data
        """
        chi_list = []
        k_list = []
        k_ref = None
        for i, path in enumerate(feff_dirs):
            self.logger.info(
                f"Processing FEFF directory {i+1}/{len(feff_dirs)}: {path.name}"
            )
            # Run FEFF if needed
            if force or not (path / "chi.dat").exists():
                if not self.run_feff(path):
                    self.logger.warning(
                        f"Skipping directory {path.name} (FEFF run failed)"
                    )
                    continue
            try:
                k, chi = self.read_chi_dat(path)
            except Exception as e:
                self.logger.error(f"Error reading chi.dat from {path}: {e}")
                continue
            # Initialize reference k-grid
            if k_ref is None:
                k_ref = k.copy()
                self.logger.debug(f"Reference k-grid: {len(k_ref)} points")
            # Check k-grid consistency
            if len(k) != len(k_ref) or not np.allclose(k, k_ref, atol=1e-4):
                self.logger.warning(f"k-grid mismatch in {path.name}")
                self.logger.debug(
                    f"Original k-range: {k.min():.2f}-{k.max():.2f} �\u207b�"
                )
                self.logger.debug(
                    f"Reference k-range: {k_ref.min():.2f}-{k_ref.max():.2f} �\u207b�"
                )
                # Interpolate to reference grid
                chi = np.interp(k_ref, k, chi, left=0, right=0)
                k = k_ref.copy()
            chi_list.append(chi)
            k_list.append(k)
        if not chi_list:
            raise RuntimeError("No valid FEFF outputs found")
        # Convert to arrays for vectorized operations
        chi_array = np.array(chi_list)
        k_array = k_list[0]
        # Create result group
        result = Group()
        result.k = k_array
        result.chi = np.mean(chi_array, axis=0)
        result.chi_std = np.std(chi_array, axis=0, ddof=1)
        result.chi_individual = chi_array
        result.nframes = len(chi_list)
        # Apply Fourier transform if requested
        if ft_params:
            self.logger.info("Performing Fourier transform on averaged spectrum")
            ft_params = {**DEFAULT_FT_PARAMS, **ft_params}
            xftf(result, **ft_params)
        return result

    def plot_fourier_transform(
        self,
        exafs_group: Group | None = None,
        output_dir: Path = Path("."),
        filename_base: str = "EXAFS_FT",
        show_plot: bool = True,
    ) -> tuple[Path, Path]:
        """Plot Fourier transform of EXAFS data."""
        if exafs_group is None:
            if self.last_exafs_group is None:
                raise ValueError(
                    "No EXAFS data available. Run process_feff_output first."
                )
            exafs_group = self.last_exafs_group
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(exafs_group.r, exafs_group.chir_mag, linewidth=2)
        ax.set_xlabel(r"R ($\AA$)", fontsize=12)
        ax.set_ylabel(r"$\vert\chi(\mathrm{R})\vert$", fontsize=12)
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

    def run_full_pipeline(
        self,
        structure_path: Path,
        absorber: str,
        output_dir: Path = Path("outputs/feff_pipeline"),
        **kwargs,
    ) -> tuple[Group, tuple[Path, Path]]:
        """Run the complete FEFF + EXAFS processing pipeline."""
        # Separate plotting and FEFF/FT parameters
        plot_output_dir = kwargs.pop("plot_output_dir", output_dir)
        show_plot = kwargs.pop("show_plot", False)
        # Extract FT parameters
        ft_params = {k: v for k, v in kwargs.items() if k in DEFAULT_FT_PARAMS.keys()}
        # Extract FEFF parameters
        feff_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "spectrum_type",
                "edge",
                "radius",
                "user_tag_settings",
                "method",
                "filename",
            ]
        }
        self.logger.info(f"Generating FEFF input for {absorber} in {structure_path}")
        input_file = self.generate_feff_input(
            structure_path, absorber, output_dir, **feff_params
        )
        self.logger.info(f"FEFF input written to: {input_file}")
        self.logger.info("Running FEFF calculation...")
        if not self.run_feff(output_dir):
            raise RuntimeError("FEFF calculation failed")
        self.logger.info("Processing FEFF output...")
        exafs_group = self.process_feff_output(output_dir, **ft_params)
        self.logger.info("Generating plots...")
        plot_paths = self.plot_fourier_transform(
            exafs_group,
            plot_output_dir,
            f"{structure_path.stem}_{absorber}_EXAFS_FT",
            show_plot=show_plot,
        )
        self.logger.info(
            f"Pipeline complete! Plots saved to: {plot_paths[0]} and {plot_paths[1]}"
        )
        return exafs_group, plot_paths

    def process_trajectory(
        self,
        trajectory_path: Path,
        absorber: str,
        output_base: Path,
        sample_interval: int = 1,
        parallel: bool = False,
        n_workers: int | None = None,
        progress_callback: callable | None = None,
        config_parameters: Path | None = None,
        **kwargs,
    ) -> Group:
        """Process MD trajectory for time-averaged EXAFS with optional parallel processing."""
        trajectory_path = Path(trajectory_path).resolve()
        output_base = Path(output_base).resolve()
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Trajectory file {trajectory_path} not found")
        self.logger.info(f"Processing trajectory: {trajectory_path}")
        self.logger.info(f"Sample interval: {sample_interval}")
        self.logger.info(f"Parallel processing: {parallel}")

        # Load configuration parameters if provided
        if config_parameters is not None:
            config_params = load_config_parameter_set(config_parameters)
            self.logger.info(f"Using configuration parameters from {config_parameters}")
            # Merge configuration parameters with kwargs (kwargs take precedence)
            merged_kwargs = {**config_params, **kwargs}
            kwargs = merged_kwargs

        try:
            # Read trajectory frames
            structures = ase_read(str(trajectory_path), index=f"::{sample_interval}")
            if not isinstance(structures, list):
                structures = [structures]  # Single frame case
            n_frames = len(structures)
            self.logger.info(f"Processing {n_frames} frames")

            # Determine optimal number of workers
            if parallel and n_frames > 1:
                n_workers = n_workers or min(mp.cpu_count(), n_frames, 8)
                self.logger.info(f"Using {n_workers} workers for parallel processing")
            else:
                n_workers = 1
                self.logger.info("Using sequential processing")

            # Create output directory
            output_base.mkdir(parents=True, exist_ok=True)

            # Generate FEFF inputs for all frames first
            feff_dirs = self._generate_feff_inputs(
                structures, absorber, output_base, kwargs
            )

            # Process the FEFF directories with appropriate method
            if n_workers > 1 and len(feff_dirs) > 1:
                return self._process_feff_dirs_parallel(
                    feff_dirs, absorber, kwargs, progress_callback, n_workers
                )
            else:
                return self._process_feff_dirs_sequential(
                    feff_dirs, absorber, kwargs, progress_callback
                )

        except Exception as e:
            raise RuntimeError(f"Trajectory processing failed: {e}")

    def _generate_feff_inputs(
        self, structures: list[Atoms], absorber: str, output_base: Path, kwargs: dict
    ) -> list[Path]:
        """Generate FEFF input directories for trajectory frames."""
        feff_dirs = []

        # Extract relevant parameters
        feff_params = {
            k: v
            for k, v in kwargs.items()
            if k in ["spectrum_type", "edge", "radius", "user_tag_settings", "method"]
        }

        for i, atoms in enumerate(structures):
            frame_output = output_base / f"frame_{i:04d}"
            frame_output.mkdir(parents=True, exist_ok=True)

            # Save structure to file
            struct_path = frame_output / "structure.cif"
            ase_write(str(struct_path), atoms, format="cif")

            # Generate FEFF input
            self.generate_feff_input(
                absorber=absorber,
                output_dir=frame_output,
                structure_path=struct_path,
                **feff_params,
            )
            feff_dirs.append(frame_output)

        return feff_dirs

    def _process_feff_dirs_parallel(
        self,
        feff_dirs: list[Path],
        absorber: str,
        kwargs: dict,
        progress_callback: callable | None,
        n_workers: int,
    ) -> Group:
        """Process FEFF directories in parallel using the existing frame worker"""
        # Extract FT parameters
        ft_params = {k: v for k, v in kwargs.items() if k in DEFAULT_FT_PARAMS.keys()}
        ft_params_final = {**DEFAULT_FT_PARAMS, **ft_params}

        chi_list = []
        k_ref = None

        # Prepare frame data for processing
        frame_data = []
        for i, feff_dir in enumerate(feff_dirs):
            # Extract structure from the directory
            struct_path = feff_dir / "structure.cif"
            if not struct_path.exists():
                continue

            try:
                from ase.io import read as ase_read

                atoms = ase_read(str(struct_path))
                frame_data.append((i, atoms, absorber, feff_dir.parent, kwargs))
            except Exception as e:
                self.logger.warning(f"Could not read structure for {feff_dir}: {e}")
                continue

        if not frame_data:
            raise RuntimeError("No valid frame data for processing")

        # Create a progress reporter if callback is provided
        progress_reporter = None
        if progress_callback:
            progress_reporter = ProgressReporter(total_frames=len(frame_data))

        # Process frames with the existing worker
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_frame_worker, data): data
                for data in frame_data
            }

            # Update initial progress
            if progress_reporter:
                progress_reporter.update(
                    completed=0,
                    total=len(frame_data),
                    description="Processing trajectory frames...",
                )
                if progress_callback:
                    progress_callback(*progress_reporter.get_progress())

            # Process completed futures
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                chi, k, frame_index, error = result

                if error:
                    self.logger.warning(
                        f"Frame {frame_index} processing failed: {error}"
                    )
                    continue

                # Initialize reference k-grid
                if k_ref is None:
                    k_ref = k.copy()

                # Check k-grid consistency
                if len(k) != len(k_ref) or not np.allclose(k, k_ref, atol=1e-4):
                    # Interpolate to reference grid
                    chi = np.interp(k_ref, k, chi, left=0, right=0)
                    k = k_ref.copy()

                chi_list.append(chi)

                # Update progress
                if progress_reporter:
                    progress_reporter.update(
                        completed=i + 1,
                        total=len(frame_data),
                        description=f"Processing frame {i+1}/{len(frame_data)}",
                    )
                    if progress_callback:
                        progress_callback(*progress_reporter.get_progress())

        if not chi_list:
            raise RuntimeError("No valid FEFF outputs found")

        # Create result group
        result = Group()
        result.k = k_ref
        result.chi = _fast_average_chi(np.array(chi_list))
        result.chi_std = _fast_std_chi(np.array(chi_list))
        result.nframes = len(chi_list)

        # Apply Fourier transform
        self.logger.info("Performing Fourier transform on averaged spectrum")
        xftf(result, **ft_params_final)

        return result

    def _process_feff_dirs_sequential(
        self,
        feff_dirs: list[Path],
        absorber: str,
        kwargs: dict,
        progress_callback: callable | None,
    ) -> Group:
        """Process FEFF directories sequentially using the same pattern as parallel"""
        # Extract FT parameters
        ft_params = {k: v for k, v in kwargs.items() if k in DEFAULT_FT_PARAMS.keys()}
        ft_params_final = {**DEFAULT_FT_PARAMS, **ft_params}

        chi_list = []
        k_ref = None

        # Create a progress reporter if callback is provided
        progress_reporter = None
        if progress_callback:
            progress_reporter = ProgressReporter(total_frames=len(feff_dirs))

        for i, feff_dir in enumerate(feff_dirs):
            if progress_reporter:
                progress_reporter.update(
                    completed=i,
                    total=len(feff_dirs),
                    description=f"Processing frame {i+1}/{len(feff_dirs)}",
                )
                if progress_callback:
                    progress_callback(*progress_reporter.get_progress())

            # Create frame data for the worker
            struct_path = feff_dir / "structure.cif"
            if not struct_path.exists():
                continue

            try:
                from ase.io import read as ase_read

                atoms = ase_read(str(struct_path))
                frame_data = (i, atoms, absorber, feff_dir.parent, kwargs)
                chi, k, _, error = _process_frame_worker(frame_data)

                if error:
                    self.logger.warning(f"Frame {i} processing failed: {error}")
                    continue

                # Initialize reference k-grid
                if k_ref is None:
                    k_ref = k.copy()

                # Check k-grid consistency
                if len(k) != len(k_ref) or not np.allclose(k, k_ref, atol=1e-4):
                    # Interpolate to reference grid
                    chi = np.interp(k_ref, k, chi, left=0, right=0)
                    k = k_ref.copy()

                chi_list.append(chi)
            except Exception as e:
                self.logger.error(f"Error processing frame {i}: {e}")
                continue

        if not chi_list:
            raise RuntimeError("No valid FEFF outputs found")

        # Create result group
        result = Group()
        result.k = k_ref
        result.chi = _fast_average_chi(np.array(chi_list))
        result.chi_std = _fast_std_chi(np.array(chi_list))
        result.nframes = len(chi_list)

        # Apply Fourier transform
        self.logger.info("Performing Fourier transform on averaged spectrum")
        xftf(result, **ft_params_final)

        return result

    def quick_exafs(
        self, structure_path: Path, absorber: str, output_dir: Path | None = None
    ) -> tuple[Group, tuple[Path, Path]]:
        """Quick EXAFS processing with sensible defaults.

        Args:
            structure_path: Path to structure file
            absorber: Absorbing atom symbol
            output_dir: Output directory (creates sensible default if None)

        Returns:
            Tuple of (EXAFS Group, (PDF plot path, SVG plot path))
        """
        if output_dir is None:
            output_dir = Path(f"exafs_output_{structure_path.stem}_{absorber}")
        return self.run_full_pipeline(
            structure_path=structure_path,
            absorber=absorber,
            output_dir=output_dir,
            show_plot=False,
        )

    def analyze_structure(
        self,
        structure_path: Path,
        absorber: str,
        spectrum_types: list = None,
        output_base_dir: Path | None = None,
    ) -> dict[str, tuple[Group, tuple[Path, Path]]]:
        """Analyze a structure with multiple spectrum types for comparison.

        Args:
            structure_path: Path to structure file
            absorber: Absorbing atom symbol
            spectrum_types: List of spectrum types to analyze (default: ['EXAFS', 'XANES'])
            output_base_dir: Base output directory
        Returns:
            Dictionary mapping spectrum type to (EXAFS Group, plot paths)
        """
        if spectrum_types is None:
            spectrum_types = ["EXAFS"]
            if PYMATGEN_AVAILABLE:
                spectrum_types.append("XANES")
        if output_base_dir is None:
            output_base_dir = Path(f"analysis_{structure_path.stem}_{absorber}")
        results = {}
        for spectrum_type in spectrum_types:
            self.logger.info(
                f"Analyzing {spectrum_type} for {absorber} in {structure_path}"
            )
            output_dir = output_base_dir / spectrum_type.lower()
            try:
                results[spectrum_type] = self.run_full_pipeline(
                    structure_path=structure_path,
                    absorber=absorber,
                    output_dir=output_dir,
                    spectrum_type=spectrum_type,
                    show_plot=False,
                )
            except Exception as e:
                self.logger.error(f"Failed to analyze {spectrum_type}: {e}")
        return results
