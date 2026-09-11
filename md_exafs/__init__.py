"""MD-EXAFS Core - Foundational engine for ensemble-averaged EXAFS spectroscopy."""

from pathlib import Path

__version__ = "0.2.0"

DEFAULT_CACHE_DIR = Path.home() / ".larch_cache"

from .exafs_data import PathAggregator, PathContribution, make_path_key  # noqa: E402
from .hdf5_store import ExafsHDF5Store  # noqa: E402

__all__ = [
    "__version__",
    "DEFAULT_CACHE_DIR",
    "ExafsHDF5Store",
    "PathContribution",
    "PathAggregator",
    "make_path_key",
]
