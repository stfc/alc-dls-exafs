"""Import and manipulation of experimental XAS spectra (ADR 0009).

Centralizes Athena project extraction, ASCII import, and S02/E0-shift scaling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .constants import HBAR2_OVER_2M_EV_ANGSTROM2

logger = logging.getLogger("md_exafs.experimental")


def shifted_k_mask(k: np.ndarray, e0_shift: float) -> np.ndarray:
    r"""Boolean mask of k points that survive a ΔE₀ shift.

    Points where $k^2 - \Delta E_0 / (\hbar^2/2m_e) \ge 0$.
    """
    k = np.asarray(k, dtype=float)
    return (k**2 - float(e0_shift) / HBAR2_OVER_2M_EV_ANGSTROM2) >= 0.0


def scaled_chi_arrays(
    k: np.ndarray,
    chi: np.ndarray,
    s02: float = 1.0,
    e0_shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Apply comparison-only S₀² and ΔE₀ adjustments to χ(k).

    Points below the shifted threshold are dropped so the returned k-grid
    remains monotonic.
    """
    k = np.asarray(k, dtype=float)
    chi = np.asarray(chi, dtype=float)
    mask = shifted_k_mask(k, e0_shift)
    shifted_k_squared = k[mask] ** 2 - float(e0_shift) / HBAR2_OVER_2M_EV_ANGSTROM2
    return np.sqrt(shifted_k_squared), float(s02) * chi[mask]


def list_athena_groups(path: Path | str) -> list[str]:
    """Return list of spectrum group names inside an Athena project file (.prj)."""
    p = Path(path)
    if not p.exists():
        return []

    try:
        from larch.io import is_athena_project, read_athena

        if not is_athena_project(str(p)):
            return []
        project = read_athena(str(p), do_preedge=False, do_bkg=False)
        return sorted(name for name in project.__dict__ if not name.startswith("_"))
    except Exception as exc:
        logger.debug(f"Failed to inspect Athena project {path}: {exc}")
        return []


def read_experimental_spectrum(
    path: Path | str,
    group: str | None = None,
    autobk: bool = True,
    labels: list[str] | str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read an experimental spectrum using Larch and return (k, chi, metadata).

    Args:
        path: Path to experimental data file (.prj, .dat, .xdi, .chi, etc.).
        group: Optional group name if reading an Athena project file.
        autobk: Whether to run background subtraction (autobk) if only energy/mu are found.
        labels: Optional column labels for ASCII readers.

    Returns:
        tuple (k, chi, metadata_dict).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Experimental spectrum file not found: {path}")

    from larch.io import guess_filereader, is_athena_project, read_ascii, read_athena

    if isinstance(labels, str):
        label_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
    else:
        label_list = list(labels) if labels else None

    meta: dict[str, Any] = {"source_file": str(p)}

    if is_athena_project(str(p)):
        meta["reader"] = "read_athena"
        project = read_athena(str(p), match=group, do_preedge=True, do_bkg=autobk)
        groups = [
            v
            for k_name, v in project.__dict__.items()
            if not k_name.startswith("_") and hasattr(v, "__dict__")
        ]
        if not groups:
            raise ValueError(f"No spectrum groups found in Athena project {path}")
        if len(groups) > 1 and group is None:
            avail = ", ".join(
                k_name for k_name in project.__dict__ if not k_name.startswith("_")
            )
            raise ValueError(
                f"Select one Athena group before importing. Available: {avail}"
            )
        grp = groups[0]
        meta["group"] = getattr(grp, "filename", group or "unknown")
    else:
        reader_name = guess_filereader(str(p))
        meta["reader"] = reader_name
        if reader_name == "read_ascii":
            grp = read_ascii(str(p), labels=label_list)
        else:
            from larch import io

            grp = getattr(io, reader_name)(str(p))

    # Extract k and chi
    k = _first_array(grp, "k")
    chi = _first_array(grp, "chi", "chi_k")
    energy = _first_array(grp, "energy", "omega")
    mu = _first_array(grp, "mu", "xmu")

    if (
        (k is None or chi is None)
        and (energy is not None and mu is not None)
        and autobk
    ):
        from larch.xafs import autobk as larch_autobk

        larch_autobk(energy, mu, group=grp)
        k = _first_array(grp, "k")
        chi = _first_array(grp, "chi", "chi_k")

    if k is None or chi is None:
        avail_cols = ", ".join(getattr(grp, "array_labels", [])) or "none"
        raise ValueError(
            f"Could not extract k/chi from {path} (columns found: {avail_cols})"
        )

    if hasattr(grp, "e0"):
        meta["e0"] = float(grp.e0)

    return k, chi, meta


def _first_array(group: Any, *names: str) -> np.ndarray | None:
    """Return the first 1D numpy array among the specified attribute names."""
    for name in names:
        val = getattr(group, name, None)
        if val is not None:
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 1 and len(arr) > 0:
                return arr
    return None


__all__ = [
    "shifted_k_mask",
    "scaled_chi_arrays",
    "list_athena_groups",
    "read_experimental_spectrum",
]
