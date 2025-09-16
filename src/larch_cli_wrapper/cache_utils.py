"""Cache utilities for EXAFS processing pipeline."""

import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Any

from ase import Atoms


def get_structure_hash(atoms: Atoms) -> str:
    """Generate a hash for atomic structure for caching purposes."""
    return hashlib.md5(  # noqa: S324
        str(atoms.get_positions()).encode() + str(atoms.get_chemical_symbols()).encode()
    ).hexdigest()[:16]


def get_cache_key(atoms: Atoms, absorber: str, config: Any) -> str:
    """Generate a cache key for given structure, absorber, and configuration."""
    structure_hash = get_structure_hash(atoms)
    config_str = (
        f"{config.spectrum_type}_{config.edge}_{config.radius}_"
        f"{config.kmin}_{config.kmax}_{config.kweight}_{config.window}_{config.dk}"
    )
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]  # noqa: S324
    return f"{structure_hash}_{absorber}_{config_hash}"


def load_from_cache(
    cache_key: str, cache_dir: str | Path | None, force_recalculate: bool = False
) -> tuple[Any, Any] | None:
    """Load cached results if available and not forcing recalculation.

    Args:
        cache_key: Unique identifier for the cached data
        cache_dir: Directory where cache files are stored
        force_recalculate: If True, ignore existing cache and return None

    Returns:
        Tuple of (chi, k) arrays if cache hit, None if cache miss or error

    Note:
        The returned arrays are typically numpy arrays or lists of floats
        representing the EXAFS chi(k) data and k-space values.
    """
    if not cache_dir:
        return None
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    if cache_file.exists() and not force_recalculate:
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)  # noqa: S301 - controlled usage
            return cached_data["chi"], cached_data["k"]
        except (EOFError, pickle.UnpicklingError, KeyError):
            try:
                cache_file.unlink()
            except OSError:
                # If we can't delete the cache file, just let the user know
                warnings.warn(
                    f"Failed to delete cache file {cache_file}",
                    UserWarning,
                    stacklevel=2,
                )
    return None


def save_to_cache(
    cache_key: str, chi: Any, k: Any, cache_dir: str | Path | None
) -> None:
    """Save processing results to cache for future use.

    Args:
        cache_key: Unique identifier for the cached data
        chi: EXAFS chi(k) data (typically numpy array or list of floats)
        k: k-space values (typically numpy array or list of floats)
        cache_dir: Directory where cache files are stored

    Note:
        Silently continues on cache write failures with a warning.
        The chi and k parameters should be serializable numeric data.
    """
    if not cache_dir:
        return
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached_data = {"chi": chi, "k": k}
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)
    except OSError:
        # If we can't write to cache, continue without caching but inform the user
        warnings.warn(
            f"Failed to write cache file {cache_file}", UserWarning, stacklevel=2
        )
