"""Headless visualization engine for 2D charts and 3D path geometry (ADR 0007).

Provides framework-agnostic Altair chart builders and WEAS-widget geometry calculators.
Consumed identically by Marimo notebooks and AiiDAlab widgets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger("md_exafs.viz")


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def group_path_results(
    paths: list[Any],
    r_bin_width: float = 0.1,
) -> list[dict[str, Any]]:
    """Group scattering paths by (scatterer, nlegs, r_bin) and average their parameters.

    Args:
        paths: Iterable of PathResult objects or dicts.
        r_bin_width: Radial distance bin width in Å (default 0.1).

    Returns:
        List of path group dictionaries sorted by effective path length.
    """
    groups: dict[tuple[str, int, float], list[Any]] = defaultdict(list)
    for p in paths:
        scat = str(_get_val(p, "scatterer", "?"))
        nlegs = int(_get_val(p, "nlegs", 2))
        r_eff = float(_get_val(p, "r_eff", 0.0))
        r_bin = round(r_eff / r_bin_width) * r_bin_width
        groups[(scat, nlegs, r_bin)].append(p)

    out = []
    for (scat, nlegs, r_bin), items in groups.items():
        first = items[0]
        k = _get_val(first, "k", None)

        feff_data_list = [
            _get_val(it, "feff_data", None)
            for it in items
            if _get_val(it, "feff_data", None) is not None
        ]
        avg_feff = np.mean(feff_data_list, axis=0) if feff_data_list else None

        degens = [float(_get_val(it, "degeneracy", 1.0)) for it in items]
        reffs = [float(_get_val(it, "r_eff", 0.0)) for it in items]
        cws = [float(_get_val(it, "cw_ratio", 0.0)) for it in items]
        sig2s = [float(_get_val(it, "sig2", 0.0)) for it in items]

        out.append(
            {
                "path_key": f"{scat}_{nlegs}_{r_bin:.2f}",
                "scatterer": scat,
                "nlegs": nlegs,
                "r_bin": r_bin,
                "r_eff": float(np.mean(reffs)),
                "degeneracy": float(np.mean(degens)),
                "cw_ratio": float(np.mean(cws)),
                "sig2": float(np.mean(sig2s)),
                "k": k,
                "feff_data": avg_feff,
                "count": len(items),
            }
        )
    return sorted(out, key=lambda x: x["r_eff"])


def build_chi_chart(
    df: Any,
    title: str = "χ(k)",
    width: int = 400,
    height: int = 250,
) -> Any:
    """Build a standard Altair line chart for χ(k)."""
    import altair as alt

    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("k:Q", title="k (Å⁻¹)"),
            y=alt.Y("chi:Q", title="χ(k)"),
            color=alt.Color("path:N", title="Path"),
            tooltip=["path:N", "k:Q", "chi:Q"],
        )
        .properties(width=width, height=height, title=title)
        .interactive()
    )


def build_chir_chart(
    df: Any,
    title: str = "|χ(R)|",
    width: int = 400,
    height: int = 250,
) -> Any:
    """Build a standard Altair line chart for Fourier transform |χ(R)|."""
    import altair as alt

    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("r:Q", title="R (Å)"),
            y=alt.Y("chir_mag:Q", title="|χ(R)|"),
            color=alt.Color("path:N", title="Path"),
            tooltip=["path:N", "r:Q", "chir_mag:Q"],
        )
        .properties(width=width, height=height, title=title)
        .interactive()
    )


def build_sigma2_chart(
    df: Any,
    title: str = "σ² vs R_eff",
    width: int = 400,
    height: int = 250,
) -> Any:
    """Build a standard Altair scatter chart for Debye-Waller σ²(R_eff)."""
    import altair as alt

    return (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X("r_eff:Q", title="R_eff (Å)"),
            y=alt.Y("sigma2:Q", title="σ² (Å²)"),
            color=alt.Color("scatterer:N", title="Scatterer"),
            tooltip=["path:N", "scatterer:N", "r_eff:Q", "sigma2:Q"],
        )
        .properties(width=width, height=height, title=title)
        .interactive()
    )


def calculate_path_vectors(
    positions: np.ndarray,
    path_atom_indices: list[int],
) -> list[dict[str, Any]]:
    """Compute 3D line/arrow segments connecting atoms in a scattering path.

    Args:
        positions: (N, 3) array of Cartesian coordinates.
        path_atom_indices: List of atom indices traversing the scattering path
            (e.g. [absorber, scatterer1, scatterer2, absorber]).

    Returns:
        List of segment dictionaries containing 'start', 'end', and 'vector'.
    """
    segments = []
    for i in range(len(path_atom_indices) - 1):
        idx_a = path_atom_indices[i]
        idx_b = path_atom_indices[i + 1]
        p_a = positions[idx_a]
        p_b = positions[idx_b]
        vec = p_b - p_a
        segments.append(
            {
                "start": p_a.tolist(),
                "end": p_b.tolist(),
                "vector": vec.tolist(),
                "length": float(np.linalg.norm(vec)),
            }
        )
    return segments


__all__ = [
    "group_path_results",
    "build_chi_chart",
    "build_chir_chart",
    "build_sigma2_chart",
    "calculate_path_vectors",
]
