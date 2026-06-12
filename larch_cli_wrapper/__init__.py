"""Larch CLI Wrapper - A lightweight wrapper around larch for EXAFS processing."""

from pathlib import Path

__version__ = "0.1.0"
__all__ = [
    "LarchWrapper",
    "DEFAULT_CACHE_DIR",
    # HDF5 storage
    "ExafsHDF5Store",
    # Path contribution data structures
    "PathContribution",
    "PathAggregator",
    "make_path_key",
]

# Default cache directory for FEFF calculations and results
DEFAULT_CACHE_DIR = Path.home() / ".larch_cache"

# Public re-exports (imported lazily to avoid heavy import costs at package load)
from .exafs_data import PathAggregator, PathContribution, make_path_key  # noqa: E402
from .hdf5_store import ExafsHDF5Store  # noqa: E402
