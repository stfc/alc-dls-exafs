"""MD-EXAFS Core Engine - Trajectory analysis and path-resolved EXAFS reconstruction."""

from pathlib import Path

from .constants import ETOK, HBAR2_OVER_2M_EV_ANGSTROM2
from .exafs_data import PathAggregator, PathContribution
from .feff_input import FeffConfig, build_feff_inp, normalize_tag
from .hdf5 import (
    ArchiveReader,
    BatchShardReader,
    BatchShardWriter,
    EnsembleReader,
    EnsembleWriter,
    ExafsHDF5Store,
)
from .paths import PathResult, make_path_key, path_chi
from .potentials import PotentialsManager
from .selection import (
    parse_frame_slice,
    resolve_frame_absorbers,
    resolve_trajectory_absorbers,
)
from .spectra import (
    ChiAverage,
    average_chi_arrays,
    format_chi_ascii,
    resample_chi,
    xftf_arrays,
)

__version__ = "0.3.0"

# Default cache directory for FEFF calculations and results
DEFAULT_CACHE_DIR = Path.home() / ".larch_cache"

__all__ = [
    "__version__",
    "DEFAULT_CACHE_DIR",
    "HBAR2_OVER_2M_EV_ANGSTROM2",
    "ETOK",
    "FeffConfig",
    "build_feff_inp",
    "normalize_tag",
    "PathResult",
    "path_chi",
    "make_path_key",
    "xftf_arrays",
    "average_chi_arrays",
    "ChiAverage",
    "resample_chi",
    "format_chi_ascii",
    "resolve_frame_absorbers",
    "resolve_trajectory_absorbers",
    "parse_frame_slice",
    "PotentialsManager",
    "BatchShardWriter",
    "BatchShardReader",
    "EnsembleWriter",
    "EnsembleReader",
    "ArchiveReader",
    "ExafsHDF5Store",
    "PathAggregator",
    "PathContribution",
]
