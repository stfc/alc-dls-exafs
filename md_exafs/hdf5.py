"""Unified HDF5 storage: batch shards and ensemble archives (ADR 0002, ADR 0003, ADR 0004).

Implements:
- BatchShardWriter / BatchShardReader for compute-node execution.
- EnsembleWriter / EnsembleReader for merged ensemble results.
- ArchiveReader: unified reader exposing .k, .chi, .r, .chir_mag, and .iter_paths().
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .feff_input import FeffConfig
from .paths import FEFF_DATA_COLS, PathResult
from .spectra import resolve_ft_params, xftf_arrays

logger = logging.getLogger("md_exafs.hdf5")

SHARD_VERSION = 1
ENSEMBLE_VERSION = 2

_COMPRESS = {"compression": "gzip", "compression_opts": 1, "shuffle": True}
_ARRAY_DTYPE = np.float32


def _as_str(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


class BatchShardWriter:
    """Writer for self-contained batch shards produced by a compute batch (ADR 0002)."""

    def __init__(
        self,
        output_path: Path | str,
        k_grid: np.ndarray,
        threshold: float = 0.0,
        config: FeffConfig | None = None,
    ) -> None:
        """Initialize BatchShardWriter."""
        self.path = Path(output_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.threshold = float(threshold)
        self.config = config

        self._f = h5py.File(self.path, "w")
        meta = self._f.create_group("meta")
        meta.attrs["format_version"] = SHARD_VERSION
        meta.attrs["archive_type"] = "batch_shard"
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["threshold"] = self.threshold
        if config is not None:
            try:
                meta.attrs["feff_config"] = json.dumps(config.feff_params)
            except Exception:
                pass

        meta.create_dataset("k_grid", data=self.k_grid)
        self._tasks_grp = self._f.create_group("tasks")
        self._task_count = 0

    def add_task_result(
        self,
        frame_idx: int,
        site_idx: int,
        absorber_element: str,
        chi: np.ndarray,
        paths: list[PathResult] | None = None,
    ) -> None:
        """Add a single snapshot's calculated χ(k) and paths to the shard."""
        task_key = f"frame_{frame_idx:04d}_site_{site_idx:04d}"
        grp = self._tasks_grp.create_group(task_key)
        grp.attrs["frame_idx"] = int(frame_idx)
        grp.attrs["site_idx"] = int(site_idx)
        grp.attrs["absorber_element"] = str(absorber_element)

        # Store chi
        chi_arr = np.asarray(chi, dtype=_ARRAY_DTYPE)
        grp.create_dataset("chi", data=chi_arr, **_COMPRESS)

        # Store raw path parameters if available (hybrid storage ADR 0003)
        if paths:
            paths_grp = grp.create_group("paths")
            n_p = len(paths)
            r_effs = np.zeros(n_p, dtype=np.float64)
            nlegs = np.zeros(n_p, dtype=np.int32)
            degens = np.zeros(n_p, dtype=np.float64)
            cw_ratios = np.zeros(n_p, dtype=np.float64)
            sig2s = np.zeros(n_p, dtype=np.float64)
            angles = np.zeros(n_p, dtype=np.float64)
            scatterers = []

            # Save coarse native k-grid once
            paths_grp.create_dataset("k_grid_params", data=paths[0].k)
            feff_stack = np.zeros(
                (n_p, len(paths[0].k), len(FEFF_DATA_COLS)), dtype=_ARRAY_DTYPE
            )

            for i, p in enumerate(paths):
                r_effs[i] = p.r_eff
                nlegs[i] = p.nlegs
                degens[i] = p.degeneracy
                cw_ratios[i] = p.cw_ratio
                sig2s[i] = p.sig2
                angles[i] = p.angle if p.angle is not None else -1.0
                scatterers.append(p.scatterer)
                feff_stack[i] = p.feff_data.astype(_ARRAY_DTYPE)

            paths_grp.create_dataset("r_eff", data=r_effs)
            paths_grp.create_dataset("nlegs", data=nlegs)
            paths_grp.create_dataset("degeneracy", data=degens)
            paths_grp.create_dataset("cw_ratio", data=cw_ratios)
            paths_grp.create_dataset("sig2", data=sig2s)
            paths_grp.create_dataset("angle", data=angles)
            paths_grp.create_dataset("scatterer", data=scatterers)
            paths_grp.create_dataset("feff_data", data=feff_stack, **_COMPRESS)

        self._task_count += 1
        self._f["meta"].attrs["n_tasks"] = self._task_count

    def close(self) -> None:
        """Flush and close the HDF5 archive."""
        if self._f:
            self._f.close()

    def __enter__(self) -> BatchShardWriter:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()


class EnsembleWriter:
    """Writer for consolidated ensemble results (ADR 0002, ADR 0003)."""

    def __init__(
        self,
        output_path: Path | str,
        k_grid: np.ndarray,
        fourier_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize EnsembleWriter."""
        self.path = Path(output_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.fourier_params = resolve_ft_params(fourier_params)

        self._f = h5py.File(self.path, "w")
        meta = self._f.create_group("meta")
        meta.attrs["format_version"] = ENSEMBLE_VERSION
        meta.attrs["archive_type"] = "ensemble"
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["fourier_params"] = json.dumps(self.fourier_params)
        meta.create_dataset("k_grid", data=self.k_grid)

        self._aggregates = self._f.create_group("aggregates")
        self._frame_avgs = self._aggregates.create_group("frame_averages")
        self._site_avgs = self._aggregates.create_group("site_averages")
        self._top_paths_grp = self._f.create_group("top_paths")

    def set_overall_average(
        self,
        k: np.ndarray,
        chi: np.ndarray,
        chi_std: np.ndarray | None = None,
        ft_results: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Store the grand ensemble average spectra and Fourier transform."""
        grp = self._aggregates.create_group("overall_average")
        grp.create_dataset("k", data=np.asarray(k, dtype=float))
        grp.create_dataset("chi", data=np.asarray(chi, dtype=float))
        if chi_std is not None:
            grp.create_dataset("chi_std", data=np.asarray(chi_std, dtype=float))

        if ft_results is None:
            ft_results = xftf_arrays(k, chi, self.fourier_params)

        grp.create_dataset("r", data=np.asarray(ft_results["r"], dtype=float))
        grp.create_dataset(
            "chir_mag", data=np.asarray(ft_results["chir_mag"], dtype=float)
        )
        grp.create_dataset(
            "chir_re", data=np.asarray(ft_results["chir_re"], dtype=float)
        )
        grp.create_dataset(
            "chir_im", data=np.asarray(ft_results["chir_im"], dtype=float)
        )

    def add_frame_average(
        self,
        frame_idx: int,
        k: np.ndarray,
        chi: np.ndarray,
        ft_results: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Store a frame average spectrum."""
        grp = self._frame_avgs.create_group(f"frame_{frame_idx:04d}")
        grp.attrs["frame_idx"] = int(frame_idx)
        grp.create_dataset("k", data=np.asarray(k, dtype=float))
        grp.create_dataset("chi", data=np.asarray(chi, dtype=float))
        if ft_results is None:
            ft_results = xftf_arrays(k, chi, self.fourier_params)
        grp.create_dataset("r", data=np.asarray(ft_results["r"], dtype=float))
        grp.create_dataset(
            "chir_mag", data=np.asarray(ft_results["chir_mag"], dtype=float)
        )

    def add_site_average(
        self,
        site_idx: int,
        k: np.ndarray,
        chi: np.ndarray,
        ft_results: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Store a site average spectrum."""
        grp = self._site_avgs.create_group(f"site_{site_idx:04d}")
        grp.attrs["site_idx"] = int(site_idx)
        grp.create_dataset("k", data=np.asarray(k, dtype=float))
        grp.create_dataset("chi", data=np.asarray(chi, dtype=float))
        if ft_results is None:
            ft_results = xftf_arrays(k, chi, self.fourier_params)
        grp.create_dataset("r", data=np.asarray(ft_results["r"], dtype=float))
        grp.create_dataset(
            "chir_mag", data=np.asarray(ft_results["chir_mag"], dtype=float)
        )

    def add_top_paths(self, top_paths: list[dict[str, Any]]) -> None:
        """Store pre-evaluated top-N composite paths (ADR 0003)."""
        for i, p in enumerate(top_paths):
            pkey = p.get("path_key", f"path_{i:04d}")
            pgrp = self._top_paths_grp.create_group(f"path_{i:04d}")
            pgrp.attrs["path_key"] = pkey
            pgrp.attrs["scatterer"] = str(p.get("scatterer", ""))
            pgrp.attrs["nlegs"] = int(p.get("nlegs", 2))
            pgrp.attrs["r_eff"] = float(p.get("r_eff", 0.0))
            pgrp.attrs["degeneracy"] = float(p.get("degeneracy", 1.0))
            pgrp.attrs["cw_ratio"] = float(p.get("cw_ratio", 0.0))
            pgrp.attrs["sig2"] = float(p.get("sig2", 0.0))

            if "chi" in p:
                pgrp.create_dataset(
                    "chi", data=np.asarray(p["chi"], dtype=_ARRAY_DTYPE)
                )
            if "chir_mag" in p:
                pgrp.create_dataset(
                    "chir_mag", data=np.asarray(p["chir_mag"], dtype=_ARRAY_DTYPE)
                )

    def close(self) -> None:
        """Close HDF5 file."""
        if self._f:
            self._f.close()

    def __enter__(self) -> EnsembleWriter:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()


class ArchiveReader:
    """Unified reader for both Batch Shards and Ensemble Archives (ADR 0004)."""

    def __init__(self, path: Path | str) -> None:
        """Initialize ArchiveReader."""
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Archive file not found: {self.path}")

    @contextmanager
    def _open(self) -> Iterator[h5py.File]:
        with h5py.File(self.path, "r") as f:
            yield f

    @property
    def is_ensemble(self) -> bool:
        """Check if archive is an ensemble archive."""
        with self._open() as f:
            if "aggregates" in f:
                return True
            version = f.get("meta", {}).attrs.get("format_version", 0)
            atype = _as_str(f.get("meta", {}).attrs.get("archive_type", ""))
            return version == ENSEMBLE_VERSION or atype == "ensemble"

    @property
    def is_shard(self) -> bool:
        """Check if archive is a batch shard."""
        with self._open() as f:
            if "tasks" in f:
                return True
            version = f.get("meta", {}).attrs.get("format_version", 0)
            atype = _as_str(f.get("meta", {}).attrs.get("archive_type", ""))
            return version == SHARD_VERSION or atype == "batch_shard"

    @property
    def k(self) -> np.ndarray:
        """Primary wavenumber grid (Å⁻¹)."""
        with self._open() as f:
            if "aggregates/overall_average/k" in f:
                return np.array(f["aggregates/overall_average/k"])
            if "meta/k_grid" in f:
                return np.array(f["meta/k_grid"])
            if "meta/k_grid_sites" in f:
                return np.array(f["meta/k_grid_sites"])
            # Fallback to first task
            if "tasks" in f:
                for t in f["tasks"].values():
                    if "k" in t:
                        return np.array(t["k"])
            raise KeyError(f"No k-grid dataset found in {self.path}")

    @property
    def chi(self) -> np.ndarray:
        """Grand ensemble average or shard-mean χ(k)."""
        with self._open() as f:
            if "aggregates/overall_average/chi" in f:
                return np.array(f["aggregates/overall_average/chi"])
            if "tasks" in f:
                chis = [np.array(t["chi"]) for t in f["tasks"].values() if "chi" in t]
                if chis:
                    return np.mean(chis, axis=0)
            raise KeyError(f"No chi spectrum found in {self.path}")

    @property
    def chi_std(self) -> np.ndarray | None:
        """Sample standard deviation of χ(k) if available."""
        with self._open() as f:
            if "aggregates/overall_average/chi_std" in f:
                return np.array(f["aggregates/overall_average/chi_std"])
            return None

    @property
    def r(self) -> np.ndarray | None:
        """Fourier transform R grid (Å)."""
        with self._open() as f:
            if "aggregates/overall_average/r" in f:
                return np.array(f["aggregates/overall_average/r"])
            return None

    @property
    def chir_mag(self) -> np.ndarray | None:
        """Magnitude of the Fourier transform |χ(R)|."""
        with self._open() as f:
            if "aggregates/overall_average/chir_mag" in f:
                return np.array(f["aggregates/overall_average/chir_mag"])
            return None

    @property
    def chir_re(self) -> np.ndarray | None:
        """Real part of the Fourier transform."""
        with self._open() as f:
            if "aggregates/overall_average/chir_re" in f:
                return np.array(f["aggregates/overall_average/chir_re"])
            return None

    @property
    def chir_im(self) -> np.ndarray | None:
        """Imaginary part of the Fourier transform."""
        with self._open() as f:
            if "aggregates/overall_average/chir_im" in f:
                return np.array(f["aggregates/overall_average/chir_im"])
            return None

    def get_site_average(self, site_idx: int) -> dict[str, np.ndarray]:
        """Return average spectra for a given site index."""
        with self._open() as f:
            key = f"aggregates/site_averages/site_{site_idx:04d}"
            if key not in f:
                raise KeyError(
                    f"Site average for site {site_idx} not found in {self.path}"
                )
            grp = f[key]
            out: dict[str, np.ndarray] = {
                "k": np.array(grp["k"]),
                "chi": np.array(grp["chi"]),
            }
            if "r" in grp and "chir_mag" in grp:
                out["r"] = np.array(grp["r"])
                out["chir_mag"] = np.array(grp["chir_mag"])
            return out

    def get_frame_average(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Return average spectra for a given frame index."""
        with self._open() as f:
            key = f"aggregates/frame_averages/frame_{frame_idx:04d}"
            if key not in f:
                raise KeyError(
                    f"Frame average for frame {frame_idx} not found in {self.path}"
                )
            grp = f[key]
            out: dict[str, np.ndarray] = {
                "k": np.array(grp["k"]),
                "chi": np.array(grp["chi"]),
            }
            if "r" in grp and "chir_mag" in grp:
                out["r"] = np.array(grp["r"])
                out["chir_mag"] = np.array(grp["chir_mag"])
            return out

    def iter_paths(self) -> Iterator[PathResult]:
        """Iterate over all stored PathResult contributions in the archive."""
        with self._open() as f:
            # 1. Check legacy PathContributionsData format (/paths)
            if "paths" in f and "feff_data" in f["paths"]:
                p_grp = f["paths"]
                k_coarse = np.array(p_grp["k_grid_params"])
                feff_data = np.array(p_grp["feff_data"])
                r_eff = np.array(p_grp["r_eff"])
                nlegs = np.array(p_grp["nlegs"])
                deg = np.array(p_grp["degeneracy"])
                cw = np.array(p_grp["cw_ratio"])
                sig2 = (
                    np.array(p_grp["sig2"]) if "sig2" in p_grp else np.zeros(len(r_eff))
                )
                scat = [_as_str(x) for x in p_grp["scatterer"]]

                # Check for merged columns
                f_idx = (
                    np.array(p_grp["frame_idx"])
                    if "frame_idx" in p_grp
                    else np.zeros(len(r_eff), dtype=int)
                )
                s_idx = (
                    np.array(p_grp["site_idx"])
                    if "site_idx" in p_grp
                    else np.zeros(len(r_eff), dtype=int)
                )
                if (
                    "format_version" in f.get("meta", {}).attrs
                    and f["meta"].attrs["format_version"] == 1
                ):
                    f_idx = np.full(
                        len(r_eff), int(f["meta"].attrs.get("frame_idx", 0))
                    )
                    s_idx = np.full(len(r_eff), int(f["meta"].attrs.get("site_idx", 0)))

                for i in range(len(r_eff)):
                    yield PathResult(
                        frame_idx=int(f_idx[i]),
                        site_idx=int(s_idx[i]),
                        r_eff=float(r_eff[i]),
                        nlegs=int(nlegs[i]),
                        degeneracy=float(deg[i]),
                        scatterer=scat[i],
                        cw_ratio=float(cw[i]),
                        k=k_coarse,
                        feff_data=feff_data[i],
                        sig2=float(sig2[i]),
                    )
                return

            # 2. Check BatchShardWriter format (/tasks)
            if "tasks" in f:
                for _task_key, tgrp in f["tasks"].items():
                    if "paths" not in tgrp:
                        continue
                    p_grp = tgrp["paths"]
                    f_idx = int(tgrp.attrs.get("frame_idx", 0))
                    s_idx = int(tgrp.attrs.get("site_idx", 0))
                    k_coarse = np.array(p_grp["k_grid_params"])
                    feff_data = np.array(p_grp["feff_data"])
                    r_eff = np.array(p_grp["r_eff"])
                    nlegs = np.array(p_grp["nlegs"])
                    deg = np.array(p_grp["degeneracy"])
                    cw = np.array(p_grp["cw_ratio"])
                    sig2 = (
                        np.array(p_grp["sig2"])
                        if "sig2" in p_grp
                        else np.zeros(len(r_eff))
                    )
                    angles = (
                        np.array(p_grp["angle"])
                        if "angle" in p_grp
                        else np.full(len(r_eff), -1.0)
                    )
                    scat = [_as_str(x) for x in p_grp["scatterer"]]

                    for i in range(len(r_eff)):
                        ang = float(angles[i]) if angles[i] >= 0 else None
                        yield PathResult(
                            frame_idx=f_idx,
                            site_idx=s_idx,
                            r_eff=float(r_eff[i]),
                            nlegs=int(nlegs[i]),
                            degeneracy=float(deg[i]),
                            scatterer=scat[i],
                            cw_ratio=float(cw[i]),
                            k=k_coarse,
                            feff_data=feff_data[i],
                            sig2=float(sig2[i]),
                            angle=ang,
                        )
                return

            # 3. Check ExafsHDF5Store format (/frames)
            if "frames" in f:
                for frame_name, f_grp in f["frames"].items():
                    frame_idx = int(frame_name.removeprefix("frame_"))
                    sites_grp = f_grp.get("sites", {})
                    for site_name, s_grp in sites_grp.items():
                        site_idx = int(site_name.removeprefix("site_"))
                        paths_grp = s_grp.get("paths", {})
                        for _path_name, p_ds in paths_grp.items():
                            k_coarse = np.array(
                                p_ds.get("k", f.get("meta/k_grid_paths", []))
                            )
                            amp = np.array(p_ds.get("amp", []))
                            pha = np.array(p_ds.get("pha", []))
                            lam = np.array(p_ds.get("lam", []))
                            rep = np.array(p_ds.get("rep", []))
                            if len(amp) == 0:
                                continue
                            feff_data = np.column_stack(
                                [
                                    np.zeros_like(amp),
                                    amp,
                                    pha,
                                    np.ones_like(amp),
                                    lam,
                                    rep,
                                ]
                            )
                            yield PathResult(
                                frame_idx=frame_idx,
                                site_idx=site_idx,
                                r_eff=float(p_ds.attrs.get("r_eff", 0.0)),
                                nlegs=int(p_ds.attrs.get("nlegs", 2)),
                                degeneracy=float(p_ds.attrs.get("degeneracy", 1.0)),
                                scatterer=_as_str(p_ds.attrs.get("scatterer", "?")),
                                cw_ratio=float(p_ds.attrs.get("cw_ratio", 0.0)),
                                k=k_coarse,
                                feff_data=feff_data,
                            )


# Reader aliases
BatchShardReader = ArchiveReader
EnsembleReader = ArchiveReader

# Re-export ExafsHDF5Store for backwards compatibility
from .hdf5_store import ExafsHDF5Store  # noqa: E402

__all__ = [
    "SHARD_VERSION",
    "ENSEMBLE_VERSION",
    "BatchShardWriter",
    "BatchShardReader",
    "EnsembleWriter",
    "EnsembleReader",
    "ArchiveReader",
    "ExafsHDF5Store",
]
