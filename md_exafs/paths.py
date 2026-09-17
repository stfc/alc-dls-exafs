"""Scattering path extraction, filtering, and EXAFS equation evaluation.

Implements exact path chi reconstruction from raw FEFF scattering factors (ADR 0003).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import ETOK

logger = logging.getLogger("md_exafs.paths")

FEFF_DATA_COLS: tuple[str, ...] = (
    "real_phc",
    "mag_feff",
    "pha_feff",
    "red_fact",
    "lam",
    "rep",
)
_COL = {name: i for i, name in enumerate(FEFF_DATA_COLS)}


@dataclass
class PathResult:
    """Represents a single FEFF scattering path contribution."""

    frame_idx: int
    site_idx: int
    r_eff: float
    nlegs: int
    degeneracy: float
    scatterer: str
    cw_ratio: float
    k: np.ndarray  # Coarse native FEFF k grid
    feff_data: np.ndarray  # Shape (M, 6): columns corresponding to FEFF_DATA_COLS
    sig2: float = 0.0
    angle: float | None = None
    path_index: int = 0


def make_path_key(
    scatterer: str,
    nlegs: int,
    r_eff: float,
    r_bin: float = 0.15,
    angle: float | None = None,
    angle_bin: float = 10.0,
) -> str:
    """Construct a canonical path grouping key.

    Args:
        scatterer: Scatterer element or hyphen-separated leg elements.
        nlegs: Number of path legs.
        r_eff: Effective path length in Å.
        r_bin: Distance bin width in Å (default 0.15).
        angle: Optional 3-body internal angle in degrees.
        angle_bin: Optional angle bin width in degrees.

    Returns:
        Canonical string key, e.g. "Cu_2_2.55" or "Cu-Cu_3_3.60_A120".
    """
    bin_r = round(round(r_eff / r_bin) * r_bin, 2)
    key = f"{scatterer}_{nlegs}_{bin_r:.2f}"
    if angle is not None and nlegs == 3 and angle > 0:
        bin_a = int(round(angle / angle_bin) * angle_bin)
        key = f"{key}_A{bin_a:03d}"
    return key


def parse_files_dat(feff_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Parse FEFF files.dat to extract per-path amplitude ratios and metadata."""
    files_dat = Path(feff_dir) / "files.dat"
    if not files_dat.exists():
        return {}

    results: dict[str, dict[str, Any]] = {}
    in_data = False

    for line in files_dat.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not in_data and "amp ratio" in stripped:
            in_data = True
            continue
        if not in_data or not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 6:
            continue
        try:
            filename = tokens[0]
            sig2 = float(tokens[1])
            cw_ratio = float(tokens[2])
            deg = float(tokens[3])
            nlegs = int(tokens[4])
            r_eff = float(tokens[5])
        except (ValueError, IndexError):
            continue
        results[filename] = {
            "sig2": sig2,
            "cw_ratio": cw_ratio,
            "deg": deg,
            "nlegs": nlegs,
            "r_eff": r_eff,
        }
    return results


def read_feff_path_dat(
    dat_path: Path | str,
    files_meta: dict[str, dict[str, Any]] | None = None,
    frame_idx: int = 0,
    site_idx: int = 0,
) -> PathResult | None:
    """Read a single feffNNNN.dat file into a PathResult object.

    Args:
        dat_path: Path to the feffNNNN.dat file.
        files_meta: Optional dictionary from parse_files_dat.
        frame_idx: Frame index to record on the PathResult.
        site_idx: Site index to record on the PathResult.

    Returns:
        PathResult or None if file cannot be read or is invalid.
    """
    dat_file = Path(dat_path)
    if not dat_file.exists():
        return None

    try:
        from larch.xafs.feffdat import FeffDatFile
    except ImportError:
        logger.error("Larch is required to read FEFF path dat files.")
        return None

    try:
        fd = FeffDatFile(str(dat_file))
        if fd.reff < 0.05 or len(fd.k) == 0:
            return None

        reff = float(fd.reff)
        nlegs = int(fd.nleg)
        deg = float(fd.degen)
        k_native = np.asarray(fd.k, dtype=float)

        feff_data = np.column_stack(
            [
                np.asarray(fd.real_phc, dtype=float),
                np.asarray(fd.mag_feff, dtype=float),
                np.asarray(fd.pha_feff, dtype=float),
                np.asarray(fd.red_fact, dtype=float),
                np.asarray(fd.lam, dtype=float),
                np.asarray(fd.rep, dtype=float),
            ]
        )

        if len(fd.geom) >= 2:
            scatterers = sorted(atom[0] for atom in fd.geom[1:])
            scatterer = "-".join(scatterers)
        else:
            scatterer = "?"

        angle: float | None = None
        if nlegs == 3 and len(fd.geom) >= 3:
            try:
                c_pos = np.array([float(x) for x in fd.geom[0][4:7]])
                n1_pos = np.array([float(x) for x in fd.geom[1][4:7]])
                n2_pos = np.array([float(x) for x in fd.geom[2][4:7]])
                d1 = float(np.linalg.norm(n1_pos - c_pos))
                d2 = float(np.linalg.norm(n2_pos - c_pos))
                apex, other = (n1_pos, n2_pos) if d1 <= d2 else (n2_pos, n1_pos)
                v1 = c_pos - apex
                v2 = other - apex
                n1_norm = np.linalg.norm(v1)
                n2_norm = np.linalg.norm(v2)
                if n1_norm > 1e-10 and n2_norm > 1e-10:
                    cos_t = np.clip(np.dot(v1, v2) / (n1_norm * n2_norm), -1.0, 1.0)
                    angle = float(np.degrees(np.arccos(cos_t)))
            except Exception:
                pass

        meta = (files_meta or {}).get(dat_file.name, {})
        cw_ratio = float(meta.get("cw_ratio", 100.0))
        sig2 = float(meta.get("sig2", 0.0))
        path_idx = int(dat_file.stem.removeprefix("feff").lstrip("0") or "0")

        return PathResult(
            frame_idx=frame_idx,
            site_idx=site_idx,
            r_eff=reff,
            nlegs=nlegs,
            degeneracy=deg,
            scatterer=scatterer,
            cw_ratio=cw_ratio,
            k=k_native,
            feff_data=feff_data,
            sig2=sig2,
            angle=angle,
            path_index=path_idx,
        )
    except Exception as exc:
        logger.debug(f"Failed to read path file {dat_file}: {exc}")
        return None


def read_paths_from_dir(
    feff_dir: Path | str,
    max_paths: int | None = None,
    threshold: float = 0.0,
    frame_idx: int = 0,
    site_idx: int = 0,
) -> list[PathResult]:
    """Read all feffNNNN.dat files from a directory, applying threshold filtering."""
    fdir = Path(feff_dir)
    files_meta = parse_files_dat(fdir)

    dat_files = sorted(fdir.glob("feff[0-9][0-9][0-9][0-9].dat"))
    results: list[PathResult] = []

    for dat_file in dat_files:
        meta = files_meta.get(dat_file.name)
        if meta and threshold > 0.0 and meta.get("cw_ratio", 0.0) < threshold:
            continue
        p = read_feff_path_dat(
            dat_file, files_meta=files_meta, frame_idx=frame_idx, site_idx=site_idx
        )
        if p is not None:
            if threshold > 0.0 and p.cw_ratio < threshold:
                continue
            results.append(p)
            if max_paths is not None and len(results) >= max_paths:
                break

    return results


def _resample(k_native: np.ndarray, values: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Interpolate a feffNNNN.dat column onto the shifted wavenumber grid."""
    min_points_for_cubic = 4
    if len(k_native) < min_points_for_cubic:
        return np.asarray(np.interp(q, k_native, values), dtype=float)
    from scipy.interpolate import UnivariateSpline

    return np.asarray(UnivariateSpline(k_native, values, s=0)(q), dtype=float)


def path_chi(
    k_native: np.ndarray,
    feff_data: np.ndarray,
    r_eff: float,
    degeneracy: float,
    k_out: np.ndarray,
    *,
    sigma2: float = 0.0,
    s02: float = 1.0,
    e0_shift: float = 0.0,
    deltar: float = 0.0,
) -> np.ndarray:
    """Evaluate one scattering path's χ(k) on k_out according to the EXAFS equation.

    Uses complex momentum p = rep + i/lambda and cubic interpolation matching Larch.

    Args:
        k_native: Coarse native k grid from FEFF (Å⁻¹).
        feff_data: (M, 6) array with columns (real_phc, mag_feff, pha_feff, red_fact, lam, rep).
        r_eff: Effective path length in Å.
        degeneracy: Path degeneracy N.
        k_out: Target output wavenumber grid in Å⁻¹.
        sigma2: Debye–Waller factor σ² in Å².
        s02: Amplitude reduction factor S₀².
        e0_shift: Threshold shift ΔE₀ in eV.
        deltar: Path length adjustment ΔR in Å.

    Returns:
        Evaluated χ(k) array on k_out.
    """
    k_out = np.asarray(k_out, dtype=float)
    feff_data = np.asarray(feff_data, dtype=float)
    if feff_data.ndim != 2 or feff_data.shape[1] != len(FEFF_DATA_COLS):
        raise ValueError(
            f"feff_data must have {len(FEFF_DATA_COLS)} columns {FEFF_DATA_COLS}, got shape {feff_data.shape}."
        )
    if r_eff <= 0:
        raise ValueError(f"r_eff must be > 0, got {r_eff}")

    energy = k_out**2 - float(e0_shift) * ETOK
    q = np.sign(energy) * np.sqrt(np.abs(energy))

    k_native = np.asarray(k_native, dtype=float)
    amp = _resample(
        k_native, feff_data[:, _COL["mag_feff"]] * feff_data[:, _COL["red_fact"]], q
    )
    pha = _resample(
        k_native, feff_data[:, _COL["real_phc"]] + feff_data[:, _COL["pha_feff"]], q
    )
    rep = _resample(k_native, feff_data[:, _COL["rep"]], q)
    lam = _resample(k_native, feff_data[:, _COL["lam"]], q)

    reff = float(r_eff)
    p_sq = (rep + 1j / lam) ** 2
    p = np.sqrt(p_sq)

    cchi = np.exp(
        -2 * reff * p.imag
        - 2 * p_sq * sigma2
        + 1j * (2 * q * reff + pha + 2 * p * (deltar - 2 * sigma2 / reff))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cchi = degeneracy * float(s02) * amp * cchi / (q * (reff + deltar) ** 2)

    chi = np.asarray(cchi.imag, dtype=float)
    chi[~np.isfinite(chi)] = 0.0

    min_points_for_extrapolation = 3
    if len(chi) >= min_points_for_extrapolation and k_out[0] == 0.0:
        chi[0] = 2 * chi[1] - chi[2]
    return chi


def total_chi(
    paths: list[PathResult],
    k_out: np.ndarray,
    *,
    sigma2: float | dict[str, float] = 0.0,
    s02: float = 1.0,
    e0_shift: float = 0.0,
) -> np.ndarray:
    """Sum χ(k) over a collection of PathResult objects."""
    k_out = np.asarray(k_out, dtype=float)
    total = np.zeros_like(k_out)
    for p in paths:
        s2 = sigma2.get(p.scatterer, 0.0) if isinstance(sigma2, dict) else float(sigma2)
        total += path_chi(
            p.k,
            p.feff_data,
            p.r_eff,
            p.degeneracy,
            k_out,
            sigma2=s2,
            s02=s02,
            e0_shift=e0_shift,
        )
    return total


def filter_paths_by_threshold(
    paths: list[PathResult],
    threshold: float,
) -> list[PathResult]:
    """Filter out paths whose cw_ratio is below the threshold percentage."""
    return [p for p in paths if p.cw_ratio >= threshold]


def group_paths_by_key(
    paths: list[PathResult],
    r_bin: float = 0.15,
) -> dict[str, list[PathResult]]:
    """Group PathResult instances by canonical path key."""
    groups: dict[str, list[PathResult]] = defaultdict(list)
    for p in paths:
        k = make_path_key(p.scatterer, p.nlegs, p.r_eff, r_bin=r_bin, angle=p.angle)
        groups[k].append(p)
    return dict(groups)


__all__ = [
    "FEFF_DATA_COLS",
    "PathResult",
    "make_path_key",
    "parse_files_dat",
    "read_feff_path_dat",
    "read_paths_from_dir",
    "path_chi",
    "total_chi",
    "filter_paths_by_threshold",
    "group_paths_by_key",
]
