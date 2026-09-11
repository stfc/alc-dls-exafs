"""Spectral averaging, Fourier transformation, and ASCII export.

Unifies Larch Fourier transform calculations and Athena ASCII export.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

FT_DEFAULTS: dict[str, Any] = {
    "kmin": 3.0,
    "kmax": 15.0,
    "kweight": 2,
    "dk": 1.0,
    "rmax": 8.0,
    "window": "kaiser",
}


def resolve_ft_params(ft_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill ft_params with FT_DEFAULTS and validate keys."""
    p = dict(ft_params or {})
    valid_keys = set(FT_DEFAULTS) | {"dk2", "with_phase", "rmax_out", "nfft", "kstep"}
    unknown = sorted(set(p) - valid_keys)
    if unknown:
        raise ValueError(
            f"Unknown Fourier-transform parameter(s): {unknown}. "
            f"Recognised keys: {sorted(valid_keys)}"
        )
    res = {**FT_DEFAULTS, **p}
    # map rmax_out to rmax if provided
    if "rmax_out" in res and "rmax" not in p:
        res["rmax"] = res["rmax_out"]
    return res


def xftf_arrays(
    k: np.ndarray,
    chi: np.ndarray,
    ft_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fourier-transform χ(k) → χ(R) using Larch.

    Args:
        k: Photoelectron wavenumber grid (Å⁻¹).
        chi: χ(k) on that grid.
        ft_params: Optional Fourier transform parameters.

    Returns:
        Dict with keys:
            'r': R grid (Å).
            'chir_mag': Magnitude |χ(R)|.
            'chir_re': Real part Re[χ(R)].
            'chir_im': Imaginary part Im[χ(R)].
            'ft_params': Fully resolved parameters used.
    """
    try:
        from larch import Group
        from larch.xafs import xftf
    except ImportError as exc:
        msg = (
            "xraylarch is required for Fourier transforms. "
            "Install with: pip install xraylarch"
        )
        raise ImportError(msg) from exc

    p = resolve_ft_params(ft_params)
    grp = Group(k=np.asarray(k, dtype=float), chi=np.asarray(chi, dtype=float))
    xftf(
        grp,
        kmin=float(p["kmin"]),
        kmax=float(p["kmax"]),
        kweight=int(p["kweight"]),
        dk=float(p["dk"]),
        window=str(p["window"]),
        rmax_out=float(p.get("rmax", p.get("rmax_out", 8.0))),
    )
    return {
        "r": np.asarray(grp.r, dtype=float),
        "chir_mag": np.asarray(np.abs(grp.chir), dtype=float),
        "chir_re": np.asarray(grp.chir.real, dtype=float),
        "chir_im": np.asarray(grp.chir.imag, dtype=float),
        "ft_params": p,
    }


def average_chi_arrays(
    k_list: list[np.ndarray],
    chi_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble average a list of χ(k) arrays onto a common k grid.

    Interpolates each spectrum onto the grid of the first member. Values
    outside each spectrum's original range are treated as NaN to prevent
    spurious dampening at high k.

    Args:
        k_list: List of k grids.
        chi_list: List of corresponding χ(k) arrays.

    Returns:
        tuple (k_ref, mean_chi, std_chi):
            - k_ref: The common wavenumber grid.
            - mean_chi: The nan-averaged χ(k).
            - std_chi: Sample standard deviation (ddof=1) across ensemble members.
    """
    if not chi_list or not k_list:
        raise ValueError("Cannot average empty list of spectra.")
    if len(chi_list) != len(k_list):
        raise ValueError("k_list and chi_list must have the same length.")

    k_ref = np.asarray(k_list[0], dtype=float)
    stack = []
    for k_item, chi_item in zip(k_list, chi_list, strict=True):
        k_arr = np.asarray(k_item, dtype=float)
        chi_arr = np.asarray(chi_item, dtype=float)
        chi_interp = np.interp(k_ref, k_arr, chi_arr, left=np.nan, right=np.nan)
        stack.append(chi_interp)

    arr = np.asarray(stack, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_chi = np.nanmean(arr, axis=0)
        if arr.shape[0] > 1:
            std_chi = np.nanstd(arr, axis=0, ddof=1)
        else:
            std_chi = np.zeros_like(mean_chi)

    return k_ref, mean_chi, std_chi


def format_chi_ascii(
    k: np.ndarray,
    chi: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format chi(k) as two-column ASCII readable by Athena and Larch.

    Args:
        k: Photoelectron wavenumber grid (Å⁻¹).
        chi: Real chi(k) values on that grid.
        metadata: Optional dictionary of metadata written as comment lines.

    Returns:
        Athena-compatible ASCII file contents as a string.
    """
    lines = [
        "# XDI/1.0 md-exafs",
        "# Column.1: k angstrom^-1",
        "# Column.2: chi",
    ]
    for key, value in (metadata or {}).items():
        lines.append(f"# {key}: {value}")
    lines.append("#---")
    lines.append("#    k           chi")
    k = np.asarray(k, dtype=float)
    chi = np.asarray(chi, dtype=float)
    lines.extend(f"{ki:12.6f} {ci:14.8g}" for ki, ci in zip(k, chi, strict=True))
    return "\n".join(lines) + "\n"


__all__ = [
    "FT_DEFAULTS",
    "resolve_ft_params",
    "xftf_arrays",
    "average_chi_arrays",
    "format_chi_ascii",
]
