"""Import and manipulation of experimental XAS spectra (ADR 0009).

Centralizes Athena project extraction, ASCII import, and S02/E0-shift scaling.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .constants import HBAR2_OVER_2M_EV_ANGSTROM2

logger = logging.getLogger("md_exafs.experimental")


def shifted_k_mask(k: np.ndarray, e0_shift: float) -> np.ndarray:
    r"""Boolean mask of k points that survive a ΔE₀ shift.

    Points where $k^2 + \Delta E_0 / (\hbar^2/2m_e) \ge 0$. This matches
    ``md_exafs.paths.path_chi``, where a feature at unshifted ``k`` moves to
    ``sqrt(k^2 + ΔE₀ · ETOK)``: positive ΔE₀ shifts features to higher k.
    """
    k = np.asarray(k, dtype=float)
    return (k**2 + float(e0_shift) / HBAR2_OVER_2M_EV_ANGSTROM2) >= 0.0


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
    shifted_k_squared = k[mask] ** 2 + float(e0_shift) / HBAR2_OVER_2M_EV_ANGSTROM2
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


def _read_spectrum_file(
    p: Path, group: str | None, autobk: bool, labels: list[str] | str | None
) -> tuple[Any, dict[str, Any]]:
    from larch.io import guess_filereader, is_athena_project, read_ascii, read_athena

    label_list: list[str] | None
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
            raise ValueError(f"No spectrum groups found in Athena project {p}")
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
    return grp, meta


_DESCRIPTOR_KEYS = frozenset({"path", "group", "autobk", "labels", "kind"})


def read_experimental_spectrum(
    source: Any,
    group: str | None = None,
    autobk: bool = True,
    labels: list[str] | str | None = None,
    *,
    kind: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read an experimental spectrum and return (k, chi, metadata).

    Args:
        source: One of
            - ``Path | str``: data file (.prj, .dat, .xdi, .chi, etc.) read with Larch;
            - ``Mapping`` with ``"path"``: file descriptor
              ``{"path": ..., "labels": ..., "group": ..., "autobk": ..., "kind": ...}``
              so reader options can be forwarded through high-level functions;
            - ``Mapping`` with array data: ``{"k", "chi"}`` (or ``"chi_k"``) or
              ``{"energy", "mu"}`` (or ``"omega"``/``"xmu"``), optional ``"e0"``;
            - 2-tuple ``(x, y)``: ``(k, chi)`` by default, ``(energy, mu)`` when
              ``kind="energy_mu"``;
            - Larch ``Group`` (or any object with ``k``/``chi`` or ``energy``/``mu``
              attributes). The caller's object is never mutated.
        group: Optional group name if reading an Athena project file.
        autobk: Run ``pre_edge`` + ``autobk`` if only energy/mu are available.
        labels: Optional column labels for ASCII readers.
        kind: Interpretation of a 2-tuple source: ``None``/``"k_chi"`` or ``"energy_mu"``.

    Returns:
        tuple (k, chi, metadata_dict). ``metadata["e0"]`` is set when a finite e0 is known.
    """
    meta: dict[str, Any]
    if isinstance(source, Mapping) and "path" in source:
        unknown = sorted(str(k) for k in source if k not in _DESCRIPTOR_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown experimental descriptor key(s): {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(_DESCRIPTOR_KEYS))}"
            )
        p = Path(source["path"])
        if not p.exists():
            raise FileNotFoundError(f"Experimental spectrum file not found: {p}")
        eff_group = source.get("group", group)
        eff_autobk = bool(source.get("autobk", autobk))
        eff_labels = source.get("labels", labels)
        grp, meta = _read_spectrum_file(p, eff_group, eff_autobk, eff_labels)
        autobk = eff_autobk
        label = str(p)
    elif isinstance(source, str | Path):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Experimental spectrum file not found: {source}")
        grp, meta = _read_spectrum_file(p, group, autobk, labels)
        label = str(p)
    elif isinstance(source, Mapping):
        grp = SimpleNamespace(**{str(n): v for n, v in source.items()})
        meta, label = {"reader": "mapping"}, "mapping"
    elif isinstance(source, tuple) and len(source) == 2:  # noqa: PLR2004
        if kind in (None, "k_chi"):
            grp = SimpleNamespace(k=source[0], chi=source[1])
        elif kind == "energy_mu":
            grp = SimpleNamespace(energy=source[0], mu=source[1])
        else:
            raise ValueError(f"kind must be None, 'k_chi' or 'energy_mu', got {kind!r}")
        meta, label = {"reader": "tuple"}, "tuple"
    else:
        grp, meta, label = source, {"reader": type(source).__name__}, "group"

    k = _first_array(grp, "k")
    chi = _first_array(grp, "chi", "chi_k")
    e0_raw = getattr(grp, "e0", None)

    if k is None or chi is None:
        energy = _first_array(grp, "energy", "omega")
        mu = _first_array(grp, "mu", "xmu")
        if energy is not None and mu is not None:
            if not autobk:
                raise ValueError(
                    f"{label} contains only energy/mu; set autobk=True to extract chi(k)"
                )
            k, chi, e0_raw = _autobk_chi(energy, mu, e0_raw)

    if k is None or chi is None:
        avail_cols = ", ".join(getattr(grp, "array_labels", [])) or "none"
        raise ValueError(
            f"Could not extract k/chi from {label} (columns found: {avail_cols})"
        )
    if k.shape != chi.shape:
        raise ValueError(f"k and chi lengths differ: {k.shape[0]} vs {chi.shape[0]}")

    if e0_raw is not None and np.isfinite(float(e0_raw)):
        meta["e0"] = float(e0_raw)

    return k, chi, meta


def _autobk_chi(
    energy: np.ndarray, mu: np.ndarray, e0: Any
) -> tuple[np.ndarray | None, np.ndarray | None, Any]:
    """Run Larch pre_edge + autobk on a fresh Group; returns (k, chi, e0)."""
    from larch import Group
    from larch.xafs import autobk as larch_autobk
    from larch.xafs import pre_edge as larch_pre_edge

    if energy.shape != mu.shape:
        raise ValueError(f"energy and mu lengths differ: {len(energy)} vs {len(mu)}")
    work = Group()
    use_e0 = float(e0) if e0 is not None and np.isfinite(float(e0)) else None
    larch_pre_edge(energy, mu, group=work, e0=use_e0)
    larch_autobk(energy, mu, group=work)
    return _first_array(work, "k"), _first_array(work, "chi"), getattr(work, "e0", None)


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
