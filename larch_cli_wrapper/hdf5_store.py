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


from .feff_utils import FeffConfig

logger = logging.getLogger(__name__)

_STORE_VERSION = "1.0"
_COMPRESS = {"compression": "gzip", "compression_opts": 6}


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
    ) -> None:
        """Open or create the HDF5 file at *path*."""
        import h5py

        self.path = Path(path)
        self.store_paths = store_paths
        self.dedup_k = dedup_k
        self._lock = threading.Lock()
        self._h5 = h5py.File(self.path, mode)

        if mode == "a":
            self._init_metadata(config)

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

        # Pre-create top-level groups
        self._h5.require_group("frames")
        self._h5.require_group("aggregates")

    # ------------------------------------------------------------------
    # k-grid helpers (support both deduped and per-dataset layouts)
    # ------------------------------------------------------------------

    def _get_site_k(self, site_grp) -> np.ndarray:
        """Return the k grid for a site group, falling back to meta/k_grid_sites."""
        if "k" in site_grp:
            return np.array(site_grp["k"])
        meta = self._h5.get("meta")
        if meta is not None and "k_grid_sites" in meta:
            return np.array(meta["k_grid_sites"])
        raise KeyError(
            f"k not found in site group '{site_grp.name}' or meta/k_grid_sites"
        )

    def _get_path_k(self, path_grp) -> np.ndarray:
        """Return the k grid for a path group, falling back to meta/k_grid_paths."""
        if "k" in path_grp:
            return np.array(path_grp["k"])
        meta = self._h5.get("meta")
        if meta is not None and "k_grid_paths" in meta:
            return np.array(meta["k_grid_paths"])
        raise KeyError(
            f"k not found in path group '{path_grp.name}' or meta/k_grid_paths"
        )

    # ------------------------------------------------------------------
    # Writing – per-site results
    # ------------------------------------------------------------------

    def write_site_result(
        self,
        frame_index: int,
        site_index: int,
        k: np.ndarray,
        chi: np.ndarray,
        absorber_element: str = "",
        success: bool = True,
        path_contributions: list[dict] | None = None,
    ) -> None:
        """Write chi(k) for one site.

        Parameters
        ----------
        frame_index, site_index:
            Indices identifying the site.
        k, chi:
            Wavenumber and EXAFS signal arrays (real float64).
        absorber_element:
            Chemical symbol of the absorbing atom.
        success:
            Whether the FEFF calculation succeeded.
        path_contributions:
            List of dicts, each with keys:
            ``path_index``, ``k``, ``chi``, ``r_eff``, ``nlegs``,
            ``degeneracy``, ``scatterer``.
            Only stored when ``self.store_paths`` is True.
        """
        grp_path = f"frames/frame_{frame_index:04d}/sites/site_{site_index:04d}"
        with self._lock:
            grp = self._h5.require_group(grp_path)
            # Overwrite arrays if re-running
            for name in ("k", "chi"):
                if name in grp:
                    del grp[name]

            arr_k = np.asarray(k, dtype=np.float64)
            arr_chi = np.asarray(chi, dtype=np.float64)
            if self.dedup_k:
                meta = self._h5.require_group("meta")
                if "k_grid_sites" not in meta:
                    meta.create_dataset("k_grid_sites", data=arr_k, **_COMPRESS)
                    meta["k_grid_sites"].attrs["description"] = (
                        "Common k grid for all per-site chi arrays (Å⁻¹). "
                        "Individual site k datasets are omitted."
                    )
                else:
                    stored_k = np.array(meta["k_grid_sites"])
                    if stored_k.shape != arr_k.shape or not np.allclose(
                        stored_k, arr_k, rtol=1e-6, atol=1e-8
                    ):
                        logger.debug(
                            "k-grid mismatch for frame=%d site=%d "
                            "(%g…%g, %d pts vs %g…%g, %d pts); "
                            "interpolating chi onto stored grid.",
                            frame_index, site_index,
                            arr_k[0], arr_k[-1], len(arr_k),
                            stored_k[0], stored_k[-1], len(stored_k),
                        )
                        arr_chi = np.interp(stored_k, arr_k, arr_chi)
                        arr_k = stored_k
            else:
                grp.create_dataset("k", data=arr_k, **_COMPRESS)
            grp.create_dataset("chi", data=arr_chi, **_COMPRESS)
            grp.attrs["site_index"] = site_index
            grp.attrs["frame_index"] = frame_index
            grp.attrs["absorber_element"] = absorber_element
            grp.attrs["success"] = success

            if self.store_paths and path_contributions:
                paths_grp = grp.require_group("paths")
                for pc in path_contributions:
                    idx = pc["path_index"]
                    p_grp = paths_grp.require_group(f"path_{idx:04d}")
                    for name in ("k", "chi", "amp", "pha", "lam", "rep", "k_param"):
                        if name in p_grp:
                            del p_grp[name]
                    if self.dedup_k:
                        pk = np.asarray(pc["k"], dtype=np.float64)
                        meta = self._h5.require_group("meta")
                        if "k_grid_paths" not in meta:
                            meta.create_dataset("k_grid_paths", data=pk, **_COMPRESS)
                            meta["k_grid_paths"].attrs["description"] = (
                                "Common k grid for all per-path chi arrays (Å⁻¹). "
                                "Individual path k datasets are omitted."
                            )
                        else:
                            stored_pk = np.array(meta["k_grid_paths"])
                            if stored_pk.shape != pk.shape or not np.allclose(
                                stored_pk, pk, rtol=1e-6, atol=1e-8
                            ):
                                logger.debug(
                                    "Path k-grid mismatch for frame=%d site=%d "
                                    "path=%d (%g…%g, %d pts vs %g…%g, %d pts); "
                                    "interpolating chi onto stored grid.",
                                    frame_index, site_index, idx,
                                    pk[0], pk[-1], len(pk),
                                    stored_pk[0], stored_pk[-1], len(stored_pk),
                                )
                                pc = dict(pc)
                                pc["chi"] = np.interp(
                                    stored_pk,
                                    pk,
                                    np.asarray(pc["chi"], dtype=np.float64),
                                )
                                pk = stored_pk
                        # FEFF raw params share a common coarse grid across all paths
                        _pk_param = pc.get("k_param")
                        if _pk_param is not None and "k_grid_params" not in meta:
                            meta.create_dataset(
                                "k_grid_params",
                                data=np.asarray(_pk_param, dtype=np.float64),
                                **_COMPRESS,
                            )
                            meta["k_grid_params"].attrs["description"] = (
                                "Common native k grid for FEFF raw path parameters"
                                " (amp, pha, lam, rep) (Å⁻¹)."
                            )
                    else:
                        p_grp.create_dataset(
                            "k", data=np.asarray(pc["k"], dtype=np.float64), **_COMPRESS
                        )
                    p_grp.create_dataset(
                        "chi", data=np.asarray(pc["chi"], dtype=np.float64), **_COMPRESS
                    )
                    # Store raw FEFF parameters for on-the-fly χ recomputation
                    for _pname in ("amp", "pha", "lam", "rep", "k_param"):
                        if _pname in pc:
                            p_grp.create_dataset(
                                _pname,
                                data=np.asarray(pc[_pname], dtype=np.float64),
                                **_COMPRESS,
                            )
                    p_grp.attrs["r_eff"] = float(pc["r_eff"])
                    p_grp.attrs["nlegs"] = int(pc["nlegs"])
                    p_grp.attrs["degeneracy"] = float(pc["degeneracy"])
                    p_grp.attrs["scatterer"] = str(pc["scatterer"])
                    p_grp.attrs["cw_ratio"] = float(pc.get("cw_ratio", 0.0))


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
        frames_grp = self._h5.get("frames")
        if frames_grp is None:
            return
        for frame_name in sorted(frames_grp.keys()):
            frame_grp = frames_grp[frame_name]
            sites_grp = frame_grp.get("sites")
            if sites_grp is None:
                continue
            for site_name in sorted(sites_grp.keys()):
                s = sites_grp[site_name]
                success = bool(s.attrs.get("success", True))
                if not success:
                    continue
                yield SiteResult(
                    frame_index=int(s.attrs["frame_index"]),
                    site_index=int(s.attrs["site_index"]),
                    absorber_element=str(s.attrs.get("absorber_element", "")),
                    success=success,
                    k=self._get_site_k(s),
                    chi=np.array(s["chi"]),
                )

    def has_site_result(self, frame_index: int, site_index: int) -> bool:
        """Return True if a successful result for this site is already stored."""
        grp_path = f"frames/frame_{frame_index:04d}/sites/site_{site_index:04d}"
        with self._lock:
            if grp_path not in self._h5:
                return False
            grp = self._h5[grp_path]
            meta = self._h5.get("meta") or {}
            has_k = "k" in grp or "k_grid_sites" in meta
            return bool(grp.attrs.get("success", True)) and has_k and "chi" in grp

    def load_site_as_group(self, frame_index: int, site_index: int) -> Group:
        """Load a single site result as a Larch Group."""
        from larch import Group

        grp_path = f"frames/frame_{frame_index:04d}/sites/site_{site_index:04d}"
        s = self._h5[grp_path]
        g = Group()
        g.k = self._get_site_k(s)
        g.chi = np.array(s["chi"])
        g.frame_idx = int(s.attrs["frame_index"])
        g.site_idx = int(s.attrs["site_index"])
        g.absorber_element = str(s.attrs.get("absorber_element", ""))
        g.task_id = f"frame_{frame_index:04d}_site_{site_index:04d}"
        return g

    # ------------------------------------------------------------------
    # Reading – path contributions
    # ------------------------------------------------------------------

    def iter_path_contributions(
        self,
    ) -> Iterator[tuple[str, dict, int, int]]:
        """Iterate over all stored path contributions.

        Yields ``(path_key, info, frame_index, site_index)`` tuples where
        *path_key* is a stable string key (see
        :func:`~larch_cli_wrapper.exafs_data.make_path_key`) and *info* is a
        dict with keys: ``k``, ``chi``, ``amp``, ``pha``, ``lam``, ``rep``,
        ``r_eff``, ``nlegs``, ``degeneracy``, ``scatterer``.
        """
        from .exafs_data import make_path_key

        frames_grp = self._h5.get("frames")
        if frames_grp is None:
            return
        for frame_name in sorted(frames_grp.keys()):
            frame_grp = frames_grp[frame_name]
            sites_grp = frame_grp.get("sites")
            if sites_grp is None:
                continue
            for site_name in sorted(sites_grp.keys()):
                s = sites_grp[site_name]
                paths_grp = s.get("paths")
                if paths_grp is None:
                    continue
                frame_index = int(s.attrs["frame_index"])
                site_index = int(s.attrs["site_index"])
                for path_name in sorted(paths_grp.keys()):
                    p = paths_grp[path_name]
                    r_eff = float(p.attrs["r_eff"])
                    nlegs = int(p.attrs["nlegs"])
                    scatterer = str(p.attrs["scatterer"])
                    path_key = make_path_key(scatterer, nlegs, r_eff)
                    info: dict = {
                        "k": self._get_path_k(p),
                        "chi": np.array(p["chi"]),
                        "r_eff": r_eff,
                        "nlegs": nlegs,
                        "degeneracy": float(p.attrs["degeneracy"]),
                        "scatterer": scatterer,
                        "cw_ratio": float(p.attrs["cw_ratio"]),
                    }
                    # Yield raw FEFF parameters if stored
                    for _pname in ("amp", "pha", "lam", "rep", "k_param"):
                        if _pname in p:
                            info[_pname] = np.array(p[_pname])
                    # Fallback: k_param may be deduped in meta/k_grid_params
                    if "k_param" not in info:
                        meta = self._h5.get("meta")
                        if meta is not None and "k_grid_params" in meta:
                            info["k_param"] = np.array(meta["k_grid_params"])
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
    # Migration helper
    # ------------------------------------------------------------------

    @classmethod
    def from_existing_output_dir(
        cls,
        output_dir: Path,
        hdf5_path: Path | None = None,
        store_paths: bool = False,
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
        """
        from .feff_utils import read_feff_output

        hdf5_path = hdf5_path or (output_dir / "results.h5")
        store = cls(hdf5_path, config=None, store_paths=store_paths, mode="a")

        imported = 0
        failed = 0

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
                            site_dir, k_grid=k
                        )
                    store.write_site_result(
                        frame_index=frame_index,
                        site_index=site_index,
                        k=k,
                        chi=chi,
                        success=True,
                        path_contributions=path_contributions,
                    )
                    imported += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to import {site_dir}: {exc}")
                    failed += 1

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

        frames_grp = self._h5.get("frames", {})
        n_frames = len(frames_grp) if frames_grp else 0
        n_sites = (
            sum(len(frames_grp[f].get("sites", {})) for f in frames_grp)
            if frames_grp
            else 0
        )

        file_size_mb = self.path.stat().st_size / (1024**2) if self.path.exists() else 0

        return {
            "path": str(self.path),
            "version": version,
            "created_at": created_at,
            "n_frames": n_frames,
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


def _read_path_contributions_from_dir(feff_dir: Path) -> list[dict]:
    """Read all feffNNNN.dat path files from a FEFF output directory.

    Uses :class:`larch.xafs.feffdat.FeffDatFile` to outsource the fragile
    ASCII parsing to larch, then computes χ(k) on FEFF's native coarse grid
    with the default path-parameter set (S₀²=1, σ²=0, ΔE₀=0).

    To evaluate path χ(k) on a different grid, pass the result through
    :func:`recompute_path_chi_on_grid`.

    Returns a list of dicts compatible with
    :meth:`ExafsHDF5Store.write_site_result`'s *path_contributions*
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
    for path_file in sorted(path_files):
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
        │           ├── source_frames  (int[n_samples])
        │           ├── source_sites   (int[n_samples])
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
    ) -> None:
        """Write an averaged group and its path contributions."""
        with self._lock:
            grp = self._h5.require_group(group_path)
            for name in ("k", "chi", "r", "chir_mag", "chir_re", "chir_im"):
                if name in grp:
                    del grp[name]

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
