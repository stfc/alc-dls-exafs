"""Cache utilities for EXAFS processing pipeline."""

import hashlib
import pickle


def get_structure_hash(atoms):
    """Generate a hash for atomic structure for caching purposes."""
    return hashlib.md5(  # noqa: S324
        str(atoms.get_positions()).encode() + str(atoms.get_chemical_symbols()).encode()
    ).hexdigest()[:16]


def get_cache_key(atoms, absorber, config):
    """Generate a cache key for given structure, absorber, and configuration."""
    structure_hash = get_structure_hash(atoms)
    config_str = (
        f"{config.spectrum_type}_{config.edge}_{config.radius}_"
        f"{config.kmin}_{config.kmax}_{config.kweight}_{config.window}_{config.dk}"
    )
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]  # noqa: S324
    return f"{structure_hash}_{absorber}_{config_hash}"


def load_from_cache(cache_key, cache_dir, force_recalculate=False):
    """Load cached results if available and not forcing recalculation."""
    if not cache_dir:
        return None
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists() and not force_recalculate:
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)  # noqa: S301 - controlled usage
            return cached_data["chi"], cached_data["k"]
        except (EOFError, pickle.UnpicklingError, KeyError):
            try:
                cache_file.unlink()
            except OSError:
                pass
    return None


def save_to_cache(cache_key, chi, k, cache_dir):
    """Save processing results to cache for future use."""
    if not cache_dir:
        return
    cache_file = cache_dir / f"{cache_key}.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached_data = {"chi": chi, "k": k}
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cached_data, f)
    except OSError:
        # If we can't write to cache, continue without caching
        pass
