import hashlib
import pickle
from pathlib import Path

def get_structure_hash(atoms):
    return hashlib.md5(
        str(atoms.get_positions()).encode() +
        str(atoms.get_chemical_symbols()).encode()
    ).hexdigest()[:16]

def get_cache_key(atoms, absorber, config):
    structure_hash = get_structure_hash(atoms)
    config_str = f"{config.spectrum_type}_{config.edge}_{config.radius}_{config.kmin}_{config.kmax}_{config.kweight}_{config.window}_{config.dk}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
    return f"{structure_hash}_{absorber}_{config_hash}"

def load_from_cache(cache_key, cache_dir, force_recalculate=False):
    if not cache_dir:
        return None
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists() and not force_recalculate:
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['chi'], cached_data['k']
        except Exception:
            try:
                cache_file.unlink()
            except:
                pass
    return None

def save_to_cache(cache_key, chi, k, cache_dir):
    if not cache_dir:
        return
    try:
        cache_file = cache_dir / f"{cache_key}.pkl"
        cached_data = {'chi': chi, 'k': k}
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
    except Exception:
        pass