"""HDF5-backed store for EXAFS pipeline results.

Provides incremental, compressed storage of per-site chi(k) and optional
per-path contributions, replacing thousands of individual ASCII files with a
single HDF5 file.  All writes are serialised through a threading.Lock so the
store is safe to call from the main thread while worker results are collected
sequentially (h5py does not support concurrent multi-process writes).

HDF5 layout
-----------
results.h5
├── meta/           attrs: feff_config (JSON), created_at, updated_at,
│   │                      version, store_paths
│   ├── k_grid_sites   float64[N]   shared k grid for all per-site chi arrays
│   │                               (present only when dedup_k=True; individual
│   │                                site "k" datasets are then omitted)
│   ├── k_grid_paths   float64[M]   shared k grid for per-path chi arrays
│   │                               (present only when dedup_k=True and
│   │                                store_paths=True)
│   └── k_grid_params  float64[P]   shared native FEFF coarse k grid for raw
│                                   path parameters amp/pha/lam/rep
│
├── frames/
│   └── frame_0000/ ... frame_NNNN/
│       └── sites/
│           └── site_XXXX/
│               ├── k          float64[N]   (omitted when dedup_k=True)
│               ├── chi        float64[N]   gzip-compressed
│               │              attrs: site_index, frame_index,
│               │                     absorber_element, success
│               └── paths/     (only present when store_paths=True)
│                   └── path_NNNN/
│                       ├── k      float64[N]   (omitted when dedup_k=True)
│                       ├── chi    float64[N]   χ(k) on the fine k grid
│                       ├── amp    float64[P]   FEFF scattering amplitude
│                       ├── pha    float64[P]   total scattering phase shift
│                       ├── lam    float64[P]   mean free path
│                       ├── rep    float64[P]   real part of momentum
│                       ├── k_param float64[P] (omitted when dedup_k=True)
│                       └── attrs: r_eff, nlegs, degeneracy, scatterer,
│                                  cw_ratio
│
└── aggregates/
    ├── overall_average/   k, chi, r, chir_mag, chir_re, chir_im
    │                      attrs: n_components, average_type
    ├── frame_averages/
    │   └── frame_XXXX/    same arrays + attrs (frame_index, n_components)
    └── site_averages/
        └── site_XXXX/     same arrays + attrs (site_index, n_components)
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


# ---------------------------------------------------------------------------
# One-time patch for pyshortcuts / larch encoding issue
# ---------------------------------------------------------------------------
# pyshortcuts.str2bytes / bytes2str call bytes(s, sys.stdout.encoding) at
# *call time*, so the encoding can be None even if it was valid at import.
# We apply a safe UTF-8 fallback lazily (once, right before FeffPathGroup
# is first instantiated) to avoid races with larch's own import order.
def _safe_str2bytes(s: str | bytes) -> bytes:
    if isinstance(s, bytes):
        return s
    return s.encode("utf-8")


def _safe_bytes2str(s: bytes | str) -> str:
    if isinstance(s, str):
        return s
    return s.decode("utf-8")


_FEFF_ENCODING_PATCHED = False


def _ensure_feff_encoding_patch() -> None:
    """Patch pyshortcuts and larch.utils.strutils with UTF-8-safe str2bytes.

    Applied unconditionally once, right before FeffPathGroup is constructed.
    Running after larch is imported ensures all module-level references are
    updated before they are used.
    """
    global _FEFF_ENCODING_PATCHED
    if _FEFF_ENCODING_PATCHED:
        return
    _FEFF_ENCODING_PATCHED = True
    try:
        import pyshortcuts

        pyshortcuts.str2bytes = _safe_str2bytes
        pyshortcuts.bytes2str = _safe_bytes2str
    except ImportError:
        pass
    try:
        import larch.utils.strutils as _strutils

        _strutils.str2bytes = _safe_str2bytes
        _strutils.bytes2str = _safe_bytes2str
    except ImportError:
        pass


if TYPE_CHECKING:
    from larch import Group

    from .exafs_data import EXAFSDataCollection, PathContribution


from .feff_utils import FeffConfig  # noqa: E402

logger = logging.getLogger(__name__)

_STORE_VERSION = "1.0"
_COMPRESS = {"compression": "gzip", "compression_opts": 1}


# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SiteResult:
    """Lightweight container returned by iter_site_results()."""

    frame_index: int
    site_index: int
    absorber_element: str
    success: bool
    k: np.ndarray
    chi: np.ndarray


# ---------------------------------------------------------------------------
# Main store class
# ---------------------------------------------------------------------------


class ExafsHDF5Store:
    """Incremental HDF5 store for EXAFS pipeline results.

    Parameters
    ----------
    path:
        Path to the HDF5 file.  The file is created if it does not exist.
    config:
        ``FeffConfig`` used for this run.  Serialised to JSON and stored as
        metadata.  Pass ``None`` to skip (e.g. when opening an existing file
        for reading).
    store_paths:
        Whether to store per-path chi(k) contributions alongside each site.
    mode:
        ``'a'`` (default) to open for appending / creating, ``'r'`` for
        read-only access.
    """

    def __init__(
        self,
        path: Path,
        config: FeffConfig | None = None,
        store_paths: bool = False,
        dedup_k: bool = True,
        mode: str = "a",
        max_paths: int | None = None,
    ) -> None:
        """Open or create the HDF5 file at *path*."""
        import h5py

        self.path = Path(path)
        self.store_paths = store_paths
        self.dedup_k = dedup_k
        self.max_paths = max_paths if max_paths is not None else (config.max_paths if config is not None else None)
        self._lock = threading.Lock()
        self._h5 = h5py.File(self.path, mode)

        if mode == "a":
            self._init_metadata(config)
        else:
            meta = self._h5.get("meta")
            if meta is not None and "max_paths" in meta.attrs:
                self.max_paths = int(meta.attrs["max_paths"])

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ExafsHDF5Store:
        """Return self to support use as a context manager."""
        return self

    def __exit__(self, *_) -> None:
        """Close the HDF5 file on context manager exit."""
        self.close()

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        try:
            self._h5.flush()
            self._h5.close()
        except Exception:  # noqa: BLE001, S110
            pass

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_metadata(self, config: FeffConfig | None) -> None:
        meta = self._h5.require_group("meta")
        now = datetime.now(timezone.utc).isoformat()
        meta.attrs["version"] = _STORE_VERSION
        # Set created_at only on first creation; always record last updated_at.
        if "created_at" not in meta.attrs:
            meta.attrs["created_at"] = now
        meta.attrs["updated_at"] = now
        if config is not None:
            try:
                from dataclasses import asdict

                meta.attrs["feff_config"] = json.dumps(asdict(config), default=str)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not serialise FeffConfig to HDF5 meta: {exc}")
        meta.attrs["store_paths"] = self.store_paths
        if self.max_paths is not None:
            meta.attrs["max_paths"] = self.max_paths

        # Pre-create top-level groups
        self._h5.require_group("frames")
        self._h5.require_group("aggregates")

    # ------------------------------------------------------------------
    # Writing – per-site results
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Writing – aggregates
    # ------------------------------------------------------------------

    def write_aggregate(
        self,
        group_path: str,
        k: np.ndarray,
        chi: np.ndarray,
        r: np.ndarray,
        chir_mag: np.ndarray,
        chir_re: np.ndarray | None = None,
        chir_im: np.ndarray | None = None,
        **attrs,
    ) -> None:
        """Write an averaged (aggregate) group.

        Parameters
        ----------
        group_path:
            HDF5 path relative to root, e.g.
            ``'aggregates/overall_average'`` or
            ``'aggregates/frame_averages/frame_0000'``.
        """
        with self._lock:
            grp = self._h5.require_group(group_path)
            for name in ("k", "chi", "r", "chir_mag", "chir_re", "chir_im"):
                if name in grp:
                    del grp[name]

            grp.create_dataset("k", data=np.asarray(k, dtype=np.float64), **_COMPRESS)
            grp.create_dataset(
                "chi", data=np.asarray(chi, dtype=np.float64), **_COMPRESS
            )
            grp.create_dataset("r", data=np.asarray(r, dtype=np.float64), **_COMPRESS)
            grp.create_dataset(
                "chir_mag", data=np.asarray(chir_mag, dtype=np.float64), **_COMPRESS
            )
            if chir_re is not None:
                grp.create_dataset(
                    "chir_re",
                    data=np.asarray(chir_re, dtype=np.float64),
                    **_COMPRESS,
                )
            if chir_im is not None:
                grp.create_dataset(
                    "chir_im",
                    data=np.asarray(chir_im, dtype=np.float64),
                    **_COMPRESS,
                )
            for k_attr, v_attr in attrs.items():
                grp.attrs[k_attr] = v_attr

    def write_overall_average(self, group: Group, n_components: int) -> None:
        """Convenience wrapper for the overall average group."""
        self._write_larch_group(
            "aggregates/overall_average",
            group,
            extra_attrs={"n_components": n_components, "average_type": "overall"},
        )

    def write_frame_average(
        self, frame_index: int, group: Group, n_components: int
    ) -> None:
        """Convenience wrapper for a frame-averaged group."""
        self._write_larch_group(
            f"aggregates/frame_averages/frame_{frame_index:04d}",
            group,
            extra_attrs={
                "n_components": n_components,
                "average_type": "frame",
                "frame_index": frame_index,
            },
        )

    def write_site_average(
        self, site_index: int, group: Group, n_components: int
    ) -> None:
        """Convenience wrapper for a site-averaged group."""
        self._write_larch_group(
            f"aggregates/site_averages/site_{site_index:04d}",
            group,
            extra_attrs={
                "n_components": n_components,
                "average_type": "site",
                "site_index": site_index,
            },
        )

    def _write_larch_group(
        self, grp_path: str, group: Group, extra_attrs: dict | None = None
    ) -> None:
        """Write a Larch Group containing k, chi, r, chir_* arrays."""
        chir_re = getattr(group, "chir_re", None)
        chir_im = getattr(group, "chir_im", None)
        self.write_aggregate(
            grp_path,
            k=group.k,
            chi=group.chi,
            r=group.r,
            chir_mag=group.chir_mag,
            chir_re=chir_re,
            chir_im=chir_im,
            **(extra_attrs or {}),
        )

    # ------------------------------------------------------------------
    # Reading – iterating site results
    # ------------------------------------------------------------------

    def iter_site_results(self) -> Iterator[SiteResult]:
        """Iterate over all successfully written site results."""
        site_res = self._h5.get("site_results")
        if site_res is None or "chi" not in site_res:
            return
        k = np.array(site_res["k_grid"])
        frame_indices = np.array(site_res["frame_index"])
        site_indices = np.array(site_res["site_index"])
        absorber_elements = [s.decode('utf-8') if isinstance(s, bytes) else s for s in site_res["absorber_element"]]
        successes = np.array(site_res["success"])
        chis = site_res["chi"]
        for i in range(len(frame_indices)):
            if successes[i]:
                yield SiteResult(
                    frame_index=int(frame_indices[i]),
                    site_index=int(site_indices[i]),
                    absorber_element=absorber_elements[i],
                    success=True,
                    k=k,
                    chi=np.array(chis[i]),
                )

    def has_site_result(self, frame_index: int, site_index: int) -> bool:
        """Return True if a successful result for this site is already stored."""
        site_res = self._h5.get("site_results")
        if site_res is None or "chi" not in site_res:
            return False
        frame_indices = np.array(site_res["frame_index"])
        site_indices = np.array(site_res["site_index"])
        successes = np.array(site_res["success"])
        matches = (frame_indices == frame_index) & (site_indices == site_index)
        if not np.any(matches):
            return False
        idx = np.where(matches)[0][0]
        return bool(successes[idx])

    def load_site_as_group(self, frame_index: int, site_index: int) -> Group:
        """Load a single site result as a Larch Group."""
        from larch import Group

        site_res = self._h5.get("site_results")
        if site_res is None or "chi" not in site_res:
            raise KeyError("No results found in site_results")
        frame_indices = np.array(site_res["frame_index"])
        site_indices = np.array(site_res["site_index"])
        matches = (frame_indices == frame_index) & (site_indices == site_index)
        if not np.any(matches):
            raise KeyError(f"site result for frame={frame_index}, site={site_index} not found")
        idx = np.where(matches)[0][0]
        
        g = Group()
        g.k = np.array(site_res["k_grid"])
        g.chi = np.array(site_res["chi"][idx])
        g.frame_idx = frame_index
        g.site_idx = site_index
        absorber = site_res["absorber_element"][idx]
        g.absorber_element = absorber.decode('utf-8') if isinstance(absorber, bytes) else absorber
        g.task_id = f"frame_{frame_index:04d}_site_{site_index:04d}"
        return g

    # ------------------------------------------------------------------
    # Reading – path contributions
    # ------------------------------------------------------------------

    def iter_path_contributions(self) -> Iterator[tuple[str, dict, int, int]]:
        """Iterate over all stored path contributions."""
        from .exafs_data import make_path_key

        path_res = self._h5.get("path_results")
        if path_res is None or "chi" not in path_res:
            return

        pk = np.array(path_res["k_grid_paths"])
        pk_param = np.array(path_res["k_grid_params"]) if "k_grid_params" in path_res else None

        frame_indices = np.array(path_res["frame_index"])
        site_indices = np.array(path_res["site_index"])
        path_indices = np.array(path_res["path_index"])
        r_effs = np.array(path_res["r_eff"])
        nlegs_arr = np.array(path_res["nlegs"])
        degeneracies = np.array(path_res["degeneracy"])
        scatterers = [s.decode('utf-8') if isinstance(s, bytes) else s for s in path_res["scatterer"]]
        cw_ratios = np.array(path_res["cw_ratio"])

        chis = path_res["chi"]
        amps = path_res.get("amp")
        phas = path_res.get("pha")
        lams = path_res.get("lam")
        reps = path_res.get("rep")

        for i in range(len(frame_indices)):
            frame_index = int(frame_indices[i])
            site_index = int(site_indices[i])
            r_eff = float(r_effs[i])
            nlegs = int(nlegs_arr[i])
            scatterer = scatterers[i]
            path_key = make_path_key(scatterer, nlegs, r_eff)

            info = {
                "k": pk,
                "chi": np.array(chis[i]),
                "r_eff": r_eff,
                "nlegs": nlegs,
                "degeneracy": float(degeneracies[i]),
                "scatterer": scatterer,
                "cw_ratio": float(cw_ratios[i]),
            }
            if amps is not None:
                info["amp"] = np.array(amps[i])
            if phas is not None:
                info["pha"] = np.array(phas[i])
            if lams is not None:
                info["lam"] = np.array(lams[i])
            if reps is not None:
                info["rep"] = np.array(reps[i])
            if pk_param is not None:
                info["k_param"] = pk_param

            yield path_key, info, frame_index, site_index

    # ------------------------------------------------------------------
    # Reading – aggregates
    # ------------------------------------------------------------------

    def load_aggregates(self) -> EXAFSDataCollection | None:
        """Load all aggregate groups into an EXAFSDataCollection.

        Returns None if no aggregates have been written yet.
        """
        from .exafs_data import EXAFSDataCollection

        agg = self._h5.get("aggregates")
        if agg is None:
            return None

        collection = EXAFSDataCollection()

        # Overall average
        if "overall_average" in agg:
            collection.overall_average = self._read_larch_group(agg["overall_average"])

        # Frame averages
        if "frame_averages" in agg:
            for fname in sorted(agg["frame_averages"].keys()):
                frame_grp = agg["frame_averages"][fname]
                g = self._read_larch_group(frame_grp)
                fidx = int(frame_grp.attrs.get("frame_index", fname.split("_")[1]))
                collection.frame_averages[fidx] = g

        # Site averages
        if "site_averages" in agg:
            for sname in sorted(agg["site_averages"].keys()):
                site_grp = agg["site_averages"][sname]
                g = self._read_larch_group(site_grp)
                sidx = int(site_grp.attrs.get("site_index", sname.split("_")[1]))
                collection.site_averages[sidx] = g

        return collection

    def _read_larch_group(self, h5grp) -> Group:
        from larch import Group

        g = Group()
        g.k = np.array(h5grp["k"])
        g.chi = np.array(h5grp["chi"])
        g.r = np.array(h5grp["r"])
        g.chir_mag = np.array(h5grp["chir_mag"])
        if "chir_re" in h5grp:
            g.chir_re = np.array(h5grp["chir_re"])
        if "chir_im" in h5grp:
            g.chir_im = np.array(h5grp["chir_im"])
        _attr_renames = {"frame_index": "frame_idx", "site_index": "site_idx"}
        for attr_name in ("frame_index", "site_index", "n_components", "average_type"):
            if attr_name in h5grp.attrs:
                setattr(
                    g,
                    _attr_renames.get(attr_name, attr_name),
                    h5grp.attrs[attr_name],
                )
        return g

    # ------------------------------------------------------------------
    # Batch writing per-site and path results (extremely fast)
    # ------------------------------------------------------------------

    def write_site_results_batch(
        self,
        results_list: list[dict],
    ) -> None:
        """Write a batch of site results and path contributions in a single HDF5 write operation.

        This avoids the massive performance penalties of resizing datasets one-by-one.
        """
        if not results_list:
            return

        with self._lock:
            grp_site = self._h5.require_group("site_results")
            
            # 1. Determine common k_grid from the first result
            ref_k = np.asarray(results_list[0]["k"], dtype=np.float64)
            if "k_grid" not in grp_site:
                grp_site.create_dataset("k_grid", data=ref_k, **_COMPRESS)
                stored_k = ref_k
            else:
                stored_k = np.array(grp_site["k_grid"])
            
            # 2. Extract and reconcile all site arrays
            n_new = len(results_list)
            chi_batch = np.zeros((n_new, len(stored_k)), dtype=np.float64)
            frame_indices = np.zeros(n_new, dtype=np.int32)
            site_indices = np.zeros(n_new, dtype=np.int32)
            absorber_elements = []
            successes = np.zeros(n_new, dtype=np.bool_)
            
            all_path_contributions = []
            
            for i, res in enumerate(results_list):
                f_idx = res["frame_index"]
                s_idx = res["site_index"]
                frame_indices[i] = f_idx
                site_indices[i] = s_idx
                absorber_elements.append(res["absorber_element"])
                successes[i] = res["success"]
                
                # Reconcile k
                k_res = np.asarray(res["k"], dtype=np.float64)
                chi_res = np.asarray(res["chi"], dtype=np.float64)
                if k_res.shape != stored_k.shape or not np.allclose(k_res, stored_k, rtol=1e-6, atol=1e-8):
                    chi_res = np.interp(stored_k, k_res, chi_res)
                chi_batch[i] = chi_res
                
                # Paths
                if self.store_paths and res.get("path_contributions"):
                    # Add frame_index and site_index to each path contribution for flat writing
                    for pc in res["path_contributions"]:
                        pc_copy = dict(pc)
                        pc_copy["frame_index"] = f_idx
                        pc_copy["site_index"] = s_idx
                        all_path_contributions.append(pc_copy)
            
            # 3. Overwrite existing rows or append to site_results
            if "chi" not in grp_site:
                grp_site.create_dataset("chi", data=chi_batch, maxshape=(None, len(stored_k)), chunks=(128, len(stored_k)), **_COMPRESS)
                grp_site.create_dataset("frame_index", data=frame_indices, maxshape=(None,), chunks=(1024,), **_COMPRESS)
                grp_site.create_dataset("site_index", data=site_indices, maxshape=(None,), chunks=(1024,), **_COMPRESS)
                import h5py
                dt_str = h5py.string_dtype(encoding='utf-8')
                grp_site.create_dataset("absorber_element", data=np.array(absorber_elements, dtype=dt_str), maxshape=(None,), chunks=(1024,), **_COMPRESS)
                grp_site.create_dataset("success", data=successes, maxshape=(None,), chunks=(1024,), **_COMPRESS)
            else:
                existing_frames = np.array(grp_site["frame_index"])
                existing_sites = np.array(grp_site["site_index"])
                existing_map = {(f, s): idx for idx, (f, s) in enumerate(zip(existing_frames, existing_sites))}
                
                overwrite_indices_src = []
                overwrite_indices_dst = []
                append_indices = []
                for i in range(n_new):
                    key = (frame_indices[i], site_indices[i])
                    if key in existing_map:
                        overwrite_indices_src.append(i)
                        overwrite_indices_dst.append(existing_map[key])
                    else:
                        append_indices.append(i)
                
                if overwrite_indices_src:
                    grp_site["chi"][overwrite_indices_dst] = chi_batch[overwrite_indices_src]
                    import h5py as _h5py
                    _dt_str = _h5py.string_dtype(encoding='utf-8')
                    _ae_arr = np.array([absorber_elements[i] for i in overwrite_indices_src], dtype=_dt_str)
                    grp_site["absorber_element"][overwrite_indices_dst] = _ae_arr
                    grp_site["success"][overwrite_indices_dst] = successes[overwrite_indices_src]
                
                if append_indices:
                    n_old = len(existing_frames)
                    n_append = len(append_indices)
                    n_new_total = n_old + n_append
                    
                    grp_site["chi"].resize((n_new_total, len(stored_k)))
                    grp_site["frame_index"].resize((n_new_total,))
                    grp_site["site_index"].resize((n_new_total,))
                    grp_site["absorber_element"].resize((n_new_total,))
                    grp_site["success"].resize((n_new_total,))
                    
                    grp_site["chi"][n_old:] = chi_batch[append_indices]
                    grp_site["frame_index"][n_old:] = frame_indices[append_indices]
                    grp_site["site_index"][n_old:] = site_indices[append_indices]
                    for idx_arr, orig_idx in enumerate(append_indices):
                        grp_site["absorber_element"][n_old + idx_arr] = absorber_elements[orig_idx]
                    grp_site["success"][n_old:] = successes[append_indices]

            # 4. Batch write for paths
            if self.store_paths and all_path_contributions:
                grp_paths = self._h5.require_group("path_results")
                import h5py
                dt_str = h5py.string_dtype(encoding='utf-8')
                
                if "chi" not in grp_paths:
                    pk = np.asarray(all_path_contributions[0]["k"], dtype=np.float64)
                    grp_paths.create_dataset("k_grid_paths", data=pk, **_COMPRESS)
                    
                    _pk_param = all_path_contributions[0].get("k_param")
                    if _pk_param is not None:
                        grp_paths.create_dataset("k_grid_params", data=np.asarray(_pk_param, dtype=np.float64), **_COMPRESS)
                        p_len = len(_pk_param)
                    else:
                        p_len = 0
                    
                    n_path_points = len(pk)
                    grp_paths.create_dataset("chi", shape=(0, n_path_points), maxshape=(None, n_path_points), dtype=np.float64, chunks=(1024, n_path_points), **_COMPRESS)
                    if p_len > 0:
                        grp_paths.create_dataset("amp", shape=(0, p_len), maxshape=(None, p_len), dtype=np.float64, chunks=(1024, p_len), **_COMPRESS)
                        grp_paths.create_dataset("pha", shape=(0, p_len), maxshape=(None, p_len), dtype=np.float64, chunks=(1024, p_len), **_COMPRESS)
                        grp_paths.create_dataset("lam", shape=(0, p_len), maxshape=(None, p_len), dtype=np.float64, chunks=(1024, p_len), **_COMPRESS)
                        grp_paths.create_dataset("rep", shape=(0, p_len), maxshape=(None, p_len), dtype=np.float64, chunks=(1024, p_len), **_COMPRESS)
                        
                    grp_paths.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(4096,), **_COMPRESS)
                    grp_paths.create_dataset("site_index", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(4096,), **_COMPRESS)
                    grp_paths.create_dataset("path_index", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(4096,), **_COMPRESS)
                    grp_paths.create_dataset("r_eff", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(4096,), **_COMPRESS)
                    grp_paths.create_dataset("nlegs", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(4096,), **_COMPRESS)
                    grp_paths.create_dataset("degeneracy", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(4096,), **_COMPRESS)
                    grp_paths.create_dataset("scatterer", shape=(0,), maxshape=(None,), dtype=dt_str, chunks=(4096,), **_COMPRESS)
                    grp_paths.create_dataset("cw_ratio", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(4096,), **_COMPRESS)
                
                unique_keys = set((res["frame_index"], res["site_index"]) for res in results_list)
                if "frame_index" in grp_paths and len(grp_paths["frame_index"]) > 0:
                    p_frames = np.array(grp_paths["frame_index"])
                    p_sites = np.array(grp_paths["site_index"])
                    
                    mask_keep = np.ones(len(p_frames), dtype=np.bool_)
                    for f, s in unique_keys:
                        mask_keep &= ~((p_frames == f) & (p_sites == s))
                    
                    if not np.all(mask_keep):
                        keep_indices = np.where(mask_keep)[0]
                        for name in list(grp_paths.keys()):
                            if name in ("k_grid_paths", "k_grid_params"):
                                continue
                            dset = grp_paths[name]
                            old_data = np.array(dset)[keep_indices]
                            del grp_paths[name]
                            if name == "scatterer":
                                grp_paths.create_dataset(name, data=old_data, maxshape=(None,), dtype=dt_str, chunks=(4096,), **_COMPRESS)
                            elif dset.ndim == 1:
                                grp_paths.create_dataset(name, data=old_data, maxshape=(None,), dtype=dset.dtype, chunks=(4096,), **_COMPRESS)
                            elif dset.ndim == 2:
                                grp_paths.create_dataset(name, data=old_data, maxshape=(None, dset.shape[1]), dtype=dset.dtype, chunks=(1024, dset.shape[1]), **_COMPRESS)

                n_new_paths = len(all_path_contributions)
                n_existing_paths = len(grp_paths["frame_index"])
                new_paths_size = n_existing_paths + n_new_paths
                
                grp_paths["chi"].resize((new_paths_size, len(grp_paths["k_grid_paths"])))
                if "amp" in grp_paths:
                    p_len = grp_paths["amp"].shape[1]
                    grp_paths["amp"].resize((new_paths_size, p_len))
                    grp_paths["pha"].resize((new_paths_size, p_len))
                    grp_paths["lam"].resize((new_paths_size, p_len))
                    grp_paths["rep"].resize((new_paths_size, p_len))
                
                for k_attr in ("frame_index", "site_index", "path_index", "r_eff", "nlegs", "degeneracy", "scatterer", "cw_ratio"):
                    grp_paths[k_attr].resize((new_paths_size,))
                
                stored_pk = np.array(grp_paths["k_grid_paths"])
                
                p_chi_batch = np.zeros((n_new_paths, len(stored_pk)), dtype=np.float64)
                p_frame_indices = np.zeros(n_new_paths, dtype=np.int32)
                p_site_indices = np.zeros(n_new_paths, dtype=np.int32)
                p_path_indices = np.zeros(n_new_paths, dtype=np.int32)
                p_r_effs = np.zeros(n_new_paths, dtype=np.float64)
                p_nlegs = np.zeros(n_new_paths, dtype=np.int32)
                p_degeneracies = np.zeros(n_new_paths, dtype=np.float64)
                p_scatterers = []
                p_cw_ratios = np.zeros(n_new_paths, dtype=np.float64)
                
                has_params = "amp" in grp_paths
                if has_params:
                    p_len = grp_paths["amp"].shape[1]
                    p_amps = np.zeros((n_new_paths, p_len), dtype=np.float64)
                    p_phas = np.zeros((n_new_paths, p_len), dtype=np.float64)
                    p_lams = np.zeros((n_new_paths, p_len), dtype=np.float64)
                    p_reps = np.zeros((n_new_paths, p_len), dtype=np.float64)
                
                for j, pc in enumerate(all_path_contributions):
                    pk_pc = np.asarray(pc["k"], dtype=np.float64)
                    chi_pc = np.asarray(pc["chi"], dtype=np.float64)
                    if pk_pc.shape != stored_pk.shape or not np.allclose(pk_pc, stored_pk, rtol=1e-6, atol=1e-8):
                        chi_pc = np.interp(stored_pk, pk_pc, chi_pc)
                    
                    p_chi_batch[j] = chi_pc
                    p_frame_indices[j] = pc["frame_index"]
                    p_site_indices[j] = pc["site_index"]
                    p_path_indices[j] = pc["path_index"]
                    p_r_effs[j] = pc["r_eff"]
                    p_nlegs[j] = pc["nlegs"]
                    p_degeneracies[j] = pc["degeneracy"]
                    p_scatterers.append(pc["scatterer"])
                    p_cw_ratios[j] = pc["cw_ratio"]
                    
                    if has_params:
                        p_amps[j] = pc["amp"]
                        p_phas[j] = pc["pha"]
                        p_lams[j] = pc["lam"]
                        p_reps[j] = pc["rep"]
                
                grp_paths["chi"][n_existing_paths:] = p_chi_batch
                grp_paths["frame_index"][n_existing_paths:] = p_frame_indices
                grp_paths["site_index"][n_existing_paths:] = p_site_indices
                grp_paths["path_index"][n_existing_paths:] = p_path_indices
                grp_paths["r_eff"][n_existing_paths:] = p_r_effs
                grp_paths["nlegs"][n_existing_paths:] = p_nlegs
                grp_paths["degeneracy"][n_existing_paths:] = p_degeneracies
                grp_paths["scatterer"][n_existing_paths:] = np.array(p_scatterers)
                grp_paths["cw_ratio"][n_existing_paths:] = p_cw_ratios
                
                if has_params:
                    grp_paths["amp"][n_existing_paths:] = p_amps
                    grp_paths["pha"][n_existing_paths:] = p_phas
                    grp_paths["lam"][n_existing_paths:] = p_lams
                    grp_paths["rep"][n_existing_paths:] = p_reps

    # ------------------------------------------------------------------
    # Migration helper
    # ------------------------------------------------------------------

    @classmethod
    def from_existing_output_dir(
        cls,
        output_dir: Path,
        hdf5_path: Path | None = None,
        store_paths: bool = False,
        max_paths: int | None = None,
    ) -> ExafsHDF5Store:
        """Scan an existing output tree and import all chi.dat files into HDF5.

        Creates a new HDF5 store from a ``frame_XXXX/site_XXXX`` directory tree.
        This allows migrating previously-computed outputs without re-running FEFF.

        Parameters
        ----------
        output_dir:
            Root directory of an existing pipeline output tree.
        hdf5_path:
            Destination HDF5 file.  Defaults to ``output_dir / 'results.h5'``.
        store_paths:
            Whether to also import feffNNNN.dat path files if present.
        max_paths:
            Maximum number of path files to import per site (retains shortest).
        """
        from .feff_utils import read_feff_output

        hdf5_path = hdf5_path or (output_dir / "results.h5")
        store = cls(hdf5_path, config=None, store_paths=store_paths, mode="a", max_paths=max_paths)

        imported = 0
        failed = 0
        results_list = []

        frame_dirs = sorted(output_dir.glob("frame_*"))
        for frame_dir in frame_dirs:
            if not frame_dir.is_dir():
                continue
            try:
                frame_index = int(frame_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            for site_dir in sorted(frame_dir.glob("site_*")):
                if not site_dir.is_dir():
                    continue
                try:
                    site_index = int(site_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                chi_file = site_dir / "chi.dat"
                if not chi_file.exists():
                    failed += 1
                    continue

                try:
                    k, chi = read_feff_output(site_dir)
                    path_contributions = None
                    if store_paths:
                        path_contributions = _read_path_contributions_from_dir(
                            site_dir, k_grid=k, max_paths=store.max_paths
                        )
                    results_list.append({
                        "frame_index": frame_index,
                        "site_index": site_index,
                        "k": k,
                        "chi": chi,
                        "absorber_element": "",
                        "success": True,
                        "path_contributions": path_contributions,
                    })
                    imported += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to import {site_dir}: {exc}")
                    failed += 1

        if results_list:
            logger.info(f"Writing {len(results_list)} site results (with paths) to HDF5 in batch...")
            store.write_site_results_batch(results_list)

        logger.info(f"Migration complete: {imported} sites imported, {failed} failed")
        return store

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """Return a summary of what is stored in this file."""
        meta = self._h5.get("meta", {})
        created_at = meta.attrs.get("created_at", "unknown") if meta else "unknown"
        version = meta.attrs.get("version", "unknown") if meta else "unknown"

        site_res = self._h5.get("site_results")
        if site_res is not None and "frame_index" in site_res:
            n_sites = len(site_res["frame_index"])
            n_frames = len(np.unique(site_res["frame_index"]))
        else:
            n_sites = 0
            n_frames = 0

        file_size_mb = self.path.stat().st_size / (1024**2) if self.path.exists() else 0

        return {
            "path": str(self.path),
            "version": version,
            "created_at": created_at,
            "n_frames": n_frames,
            "n_sites": n_sites,
            "n_sites_total": n_sites,
            "has_paths": bool(
                (self._h5.get("meta") or {}).attrs.get("store_paths", self.store_paths)
            ),
            "has_aggregates": "aggregates" in self._h5,
            "file_size_mb": round(file_size_mb, 2),
        }

    def read_metadata(self) -> dict:
        """Return the full metadata stored in the file.

        Returns a dict with keys:
        - ``version``, ``created_at``, ``updated_at``
        - ``feff_config``: the FeffConfig serialised as a nested dict (all
          fields, same keys as the dataclass), or ``{}`` if not stored.
        """
        meta = self._h5.get("meta")
        if meta is None:
            return {}
        result = {
            "version": meta.attrs.get("version", "unknown"),
            "created_at": meta.attrs.get("created_at", "unknown"),
            "updated_at": meta.attrs.get("updated_at", "unknown"),
        }
        raw = meta.attrs.get("feff_config")
        if raw:
            try:
                result["feff_config"] = json.loads(raw)
            except Exception:  # noqa: BLE001, S110
                result["feff_config"] = raw  # return raw string on parse failure
        else:
            result["feff_config"] = {}
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def recompute_path_chi_on_grid(path_dict: dict, k_grid: np.ndarray) -> dict:
    """Recompute χ(k) for a path contribution on a different k-grid.

    Interpolates the stored raw FEFF parameters (``amp``, ``pha``, ``lam``)
    from the native coarse grid (``k_param``) onto *k_grid*, then evaluates
    the standard EXAFS formula.  This avoids interpolating the oscillatory
    χ(k) signal directly.

    Args:
        path_dict: A path contribution dict as returned by
            :func:`_read_path_contributions_from_dir`.  Must contain
            ``amp``, ``pha``, ``lam``, ``k_param``, ``r_eff``,
            ``degeneracy``.
        k_grid: Target k-grid (Å⁻¹).

    Returns:
        A shallow copy of *path_dict* with ``k`` and ``chi`` replaced by
        the recomputed arrays evaluated on *k_grid*.
    """
    k_src = np.asarray(path_dict["k_param"], dtype=np.float64)
    amp = np.asarray(path_dict["amp"], dtype=np.float64)
    pha = np.asarray(path_dict["pha"], dtype=np.float64)
    lam = np.asarray(path_dict["lam"], dtype=np.float64)
    reff = float(path_dict["r_eff"])
    deg = float(path_dict["degeneracy"])

    amp_i = np.interp(k_grid, k_src, amp)
    pha_i = np.interp(k_grid, k_src, pha)
    lam_i = np.interp(k_grid, k_src, lam)

    with np.errstate(divide="ignore", invalid="ignore"):
        prefactor = np.where(k_grid > 0, deg * amp_i / (k_grid * reff**2), 0.0)
    damp = np.exp(-2.0 * reff / np.clip(lam_i, 1e-6, None))
    chi = prefactor * damp * np.sin(2.0 * k_grid * reff + pha_i)

    return {**path_dict, "k": k_grid, "chi": chi}


def _read_path_contributions_from_dir(
    feff_dir: Path,
    max_paths: int | None = None,
    **kwargs,
) -> list[dict]:
    """Read all feffNNNN.dat path files from a FEFF output directory.

    Uses :class:`larch.xafs.feffdat.FeffDatFile` to outsource the fragile
    ASCII parsing to larch, then computes χ(k) on FEFF's native coarse grid
    with the default path-parameter set (S₀²=1, σ²=0, ΔE₀=0).

    To evaluate path χ(k) on a different grid, pass the result through
    :func:`recompute_path_chi_on_grid`.

    Returns a list of dicts compatible with
    :meth:`ExafsHDF5Store.write_site_results_batch`'s *path_contributions*
    parameter.  Returns an empty list if no path files are found.
    """
    from larch.xafs.feffdat import FeffDatFile

    from .feff_utils import get_feff_numbered_files, parse_files_dat

    _ensure_feff_encoding_patch()

    path_files = get_feff_numbered_files(feff_dir)
    if not path_files:
        return []

    # Parse files.dat for cw_ratio (curved-wave amplitude ratio)
    files_meta: dict[str, dict] = {}
    try:
        files_meta = parse_files_dat(feff_dir)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not parse files.dat in {feff_dir}: {exc}")

    results = []
    n_failed = 0
    first_exc: Exception | None = None
    sorted_path_files = sorted(path_files)
    if max_paths is not None:
        sorted_path_files = sorted_path_files[:max_paths]
    for path_file in sorted_path_files:
        try:
            fd = FeffDatFile(str(path_file))

            reff = float(fd.reff)
            nleg = int(fd.nleg)
            deg = float(fd.degen)

            if fd.reff < 0.05 or len(fd.k) == 0:
                continue

            # Raw FEFF path parameters (always on the native coarse grid)
            _k_coarse = np.asarray(fd.k, dtype=np.float64)
            _amp = np.asarray(fd.amp, dtype=np.float64)
            _pha = np.asarray(fd.pha, dtype=np.float64)
            _lam = np.asarray(fd.lam, dtype=np.float64)
            _rep = np.asarray(fd.rep, dtype=np.float64)

            k = _k_coarse
            with np.errstate(divide="ignore", invalid="ignore"):
                prefactor = np.where(k > 0, deg * _amp / (k * reff**2), 0.0)
            damp = np.exp(-2.0 * reff / np.clip(_lam, 1e-6, None))
            chi = prefactor * damp * np.sin(2.0 * k * reff + _pha)

            # Scatterer label from geometry (geom[0] is absorber, geom[1:] scatterers)
            if len(fd.geom) >= 2:
                scatterers = [atom[0] for atom in fd.geom[1:]]
                scatterer = "-".join(scatterers)
            else:
                scatterer = "?"

            path_index = int(path_file.stem.removeprefix("feff").lstrip("0") or "0")

            file_entry = files_meta.get(path_file.name, {})
            cw_ratio = float(file_entry.get("cw_ratio", 0.0))

            results.append(
                {
                    "path_index": path_index,
                    "k": k,
                    "chi": chi,
                    "amp": _amp,
                    "pha": _pha,
                    "lam": _lam,
                    "rep": _rep,
                    "k_param": _k_coarse,
                    "r_eff": reff,
                    "nlegs": nleg,
                    "degeneracy": deg,
                    "scatterer": scatterer,
                    "cw_ratio": cw_ratio,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Skipping path file {path_file}: {exc}")
            if first_exc is None:
                first_exc = exc
            n_failed += 1

    if n_failed > 0 and not results:
        import traceback as _tb

        logger.warning(
            f"{feff_dir}: found {len(path_files)} feffNNNN.dat files but "
            f"all {n_failed} failed to parse. "
            f"First error ({type(first_exc).__name__}): {first_exc}\n"
            + _tb.format_exception_only(type(first_exc), first_exc)[0].rstrip()
        )

    return results


# ---------------------------------------------------------------------------
# Averaged-paths store (second HDF5 file)
# ---------------------------------------------------------------------------


class AveragedPathsStore:
    """Second HDF5 file that holds MD-averaged total χ and path contributions.

    Layout::

        averaged_paths.h5
        ├── meta/
        │   └── attrs: version, created_at, source_h5_path
        ├── overall_average/
        │   ├── k, chi, r, chir_mag, chir_re, chir_im
        │   └── paths/
        │       └── SS_Fe_3.22/
        │           ├── k, chi, r, chir_mag, chir_re, chir_im
        │           ├── source_frames  (int[n_unique_frames])
        │           ├── source_sites   (int[n_unique_sites])
        │           └── attrs: scatterer, nlegs, r_eff, degeneracy,
        │                     n_samples, cw_ratio, contribution_pct
        └── site_averages/
            └── site_XXXX/
                ├── k, chi, r, chir_mag, chir_re, chir_im
                └── paths/
                    └── ...
    """

    def __enter__(self) -> AveragedPathsStore:
        """Return self to support use as a context manager."""
        return self

    def __exit__(self, *_) -> None:
        """Close the HDF5 file on context manager exit."""
        self.close()

    def __init__(
        self,
        path: Path,
        source_h5_path: Path | None = None,
        mode: str = "a",
    ) -> None:
        """Open or create the averaged-paths HDF5 file."""
        import h5py

        self.path = Path(path)
        self._lock = threading.Lock()
        self._h5 = h5py.File(self.path, mode)
        if mode == "a":
            self._init_metadata(source_h5_path)

    def _init_metadata(self, source_h5_path: Path | None) -> None:
        meta = self._h5.require_group("meta")
        if "version" not in meta.attrs:
            meta.attrs["version"] = "1.0"
        now = datetime.now(timezone.utc).isoformat()
        if "created_at" not in meta.attrs:
            meta.attrs["created_at"] = now
        meta.attrs["updated_at"] = now
        if source_h5_path is not None:
            meta.attrs["source_h5_path"] = str(source_h5_path)

    def write_average(
        self,
        group_path: str,
        group: Group,
        path_contributions: dict[str, PathContribution] | None = None,
        n_total: int | None = None,
    ) -> None:
        """Write an averaged group and its path contributions."""
        with self._lock:
            grp = self._h5.require_group(group_path)
            for name in ("k", "chi", "r", "chir_mag", "chir_re", "chir_im"):
                if name in grp:
                    del grp[name]

            if n_total is not None:
                grp.attrs["n_total"] = int(n_total)
            grp.create_dataset(
                "k", data=np.asarray(group.k, dtype=np.float64), **_COMPRESS
            )
            grp.create_dataset(
                "chi", data=np.asarray(group.chi, dtype=np.float64), **_COMPRESS
            )
            if hasattr(group, "r") and group.r is not None:
                grp.create_dataset(
                    "r", data=np.asarray(group.r, dtype=np.float64), **_COMPRESS
                )
            if hasattr(group, "chir_mag") and group.chir_mag is not None:
                grp.create_dataset(
                    "chir_mag",
                    data=np.asarray(group.chir_mag, dtype=np.float64),
                    **_COMPRESS,
                )
            if hasattr(group, "chir_re") and group.chir_re is not None:
                grp.create_dataset(
                    "chir_re",
                    data=np.asarray(group.chir_re, dtype=np.float64),
                    **_COMPRESS,
                )
            if hasattr(group, "chir_im") and group.chir_im is not None:
                grp.create_dataset(
                    "chir_im",
                    data=np.asarray(group.chir_im, dtype=np.float64),
                    **_COMPRESS,
                )

            if path_contributions:
                paths_grp = grp.require_group("paths")
                for path_key, pc in path_contributions.items():
                    safe_key = path_key.replace("/", "_")
                    p_grp = paths_grp.require_group(safe_key)
                    for name in (
                        "k",
                        "chi",
                        "r",
                        "chir_mag",
                        "chir_re",
                        "chir_im",
                        "source_frames",
                        "source_sites",
                    ):
                        if name in p_grp:
                            del p_grp[name]

                    p_grp.create_dataset(
                        "k", data=np.asarray(pc.k, dtype=np.float64), **_COMPRESS
                    )
                    p_grp.create_dataset(
                        "chi",
                        data=np.asarray(pc.chi, dtype=np.float64),
                        **_COMPRESS,
                    )
                    if pc.r.size:
                        p_grp.create_dataset(
                            "r", data=np.asarray(pc.r, dtype=np.float64), **_COMPRESS
                        )
                    if pc.chir_mag.size:
                        p_grp.create_dataset(
                            "chir_mag",
                            data=np.asarray(pc.chir_mag, dtype=np.float64),
                            **_COMPRESS,
                        )
                    if pc.chir_re.size:
                        p_grp.create_dataset(
                            "chir_re",
                            data=np.asarray(pc.chir_re, dtype=np.float64),
                            **_COMPRESS,
                        )
                    if pc.chir_im.size:
                        p_grp.create_dataset(
                            "chir_im",
                            data=np.asarray(pc.chir_im, dtype=np.float64),
                            **_COMPRESS,
                        )
                    # source_frames / source_sites store *unique* frame and
                    # site indices (not one entry per sample), so they remain
                    # compact even for high-degeneracy paths.
                    if pc.source_frames.size:
                        p_grp.create_dataset(
                            "source_frames",
                            data=np.asarray(pc.source_frames, dtype=np.int64),
                            **_COMPRESS,
                        )
                    if pc.source_sites.size:
                        p_grp.create_dataset(
                            "source_sites",
                            data=np.asarray(pc.source_sites, dtype=np.int64),
                            **_COMPRESS,
                        )

                    # Store averaged raw FEFF parameters for on-the-fly recomputation
                    for _pname in ("amp", "pha", "lam", "rep"):
                        _parr = getattr(pc, _pname, np.array([]))
                        if _parr.size:
                            if _pname in p_grp:
                                del p_grp[_pname]
                            p_grp.create_dataset(
                                _pname,
                                data=np.asarray(_parr, dtype=np.float64),
                                **_COMPRESS,
                            )
                    if pc.k_param.size:
                        if "k_param" in p_grp:
                            del p_grp["k_param"]
                        p_grp.create_dataset(
                            "k_param",
                            data=np.asarray(pc.k_param, dtype=np.float64),
                            **_COMPRESS,
                        )

                    p_grp.attrs["scatterer"] = pc.scatterer
                    p_grp.attrs["nlegs"] = pc.nlegs
                    p_grp.attrs["r_eff"] = float(pc.r_eff)
                    p_grp.attrs["degeneracy"] = float(pc.degeneracy)
                    p_grp.attrs["n_samples"] = pc.n_samples
                    p_grp.attrs["cw_ratio"] = float(pc.cw_ratio)
                    if hasattr(pc, "contribution_pct"):
                        p_grp.attrs["contribution_pct"] = float(pc.contribution_pct)

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        try:
            self._h5.flush()
            self._h5.close()
        except Exception:  # noqa: BLE001, S110
            pass
