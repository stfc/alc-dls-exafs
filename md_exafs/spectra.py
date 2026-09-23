"""Spectral resampling, averaging, Fourier transformation, and ASCII export.

This module owns the single canonical implementation of three operations that
were previously duplicated with differing conventions: putting a χ(k) spectrum
onto a shared k-grid (:func:`resample_chi`), averaging a set of such spectra
(:func:`average_chi_arrays`), and Fourier transforming the result
(:func:`xftf_arrays`).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger("md_exafs.spectra")

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


def resample_chi(
    k_src: np.ndarray,
    chi_src: np.ndarray,
    k_dst: np.ndarray,
) -> np.ndarray:
    """Put one χ(k) spectrum onto a different k-grid.

    Destination points inside the source range are linearly interpolated.
    Beyond it, two cases have to be told apart, because conflating them is
    how a grid-registration artifact turns into either a physics error or a
    crash:

    * **Overshoot smaller than half a source step.** The two grids describe
      the same k-range and disagree only on where the last sample sits. FEFF
      writes ``chi.dat`` with 400 or 401 rows depending on whether the grid
      starts at k=0, and a grid built with ``np.arange`` overshoots its own
      stated endpoint by a few ulp. Neither is missing data, so the endpoint
      value is carried over.
    * **Anything further out.** The source spectrum genuinely stops there —
      a run with a lower ``EXAFS`` k_max, say. Extrapolating would invent
      EXAFS oscillations, and zero-filling would damp the ensemble mean at
      exactly the k where the Debye-Waller information lives. The result is
      NaN, which :func:`average_chi_arrays` masks out and counts.

    Args:
        k_src: Source wavenumber grid, ascending (Å⁻¹).
        chi_src: χ(k) on ``k_src``.
        k_dst: Destination wavenumber grid (Å⁻¹).

    Returns:
        χ(k) on ``k_dst``, NaN where ``k_src`` offers no support.
    """
    k_s = np.asarray(k_src, dtype=float)
    chi_s = np.asarray(chi_src, dtype=float)
    k_d = np.asarray(k_dst, dtype=float)

    if k_s.size == 0:
        return np.full(k_d.shape, np.nan)
    if k_s.size == 1:
        return np.where(np.isclose(k_d, k_s[0]), chi_s[0], np.nan)

    # Half the local step at each end: the distance within which a destination
    # point is nearer to the terminal source sample than to any other.
    lo_tol = 0.5 * (k_s[1] - k_s[0])
    hi_tol = 0.5 * (k_s[-1] - k_s[-2])

    # left/right clamp to the endpoint values, then NaN out whatever lies
    # beyond the tolerance. Clamping rather than extrapolating keeps the error
    # bounded by half a step of curvature instead of amplifying the end slope.
    out = np.interp(k_d, k_s, chi_s, left=chi_s[0], right=chi_s[-1])
    out[(k_d < k_s[0] - lo_tol) | (k_d > k_s[-1] + hi_tol)] = np.nan
    return out


class ChiAverage(NamedTuple):
    """Ensemble average of a set of χ(k) spectra.

    Attributes:
        k: The common wavenumber grid (Å⁻¹).
        mean: Ensemble-mean χ(k), NaN where nothing contributed.
        std: Sample standard deviation (``ddof=1``), NaN where fewer than two
            spectra contributed — one sample gives no spread estimate, and
            reporting zero would understate the uncertainty exactly where the
            ensemble has thinned out.
        n_contributors: Number of spectra contributing at each k.
    """

    k: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_contributors: np.ndarray


def xftf_arrays(
    k: np.ndarray,
    chi: np.ndarray,
    ft_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fourier-transform χ(k) → χ(R) using Larch.

    NaN in ``chi`` marks k where the ensemble has no data. It is replaced by
    zero, which is what a finite-range Fourier transform does beyond its data
    anyway. Left in place it would not stay local: the window multiplies the
    whole array, ``0 * nan`` is ``nan``, and one uncovered point at the top of
    the grid would turn every χ(R) value into NaN.

    Zero-padding is only harmless outside the window. If uncovered k fall
    *inside* it the transform is partly built from fabricated zeros, which
    damps |χ(R)|, so that case is logged as a warning naming the k-range at
    fault. Lower ``kmax`` to the range the data actually cover.

    Args:
        k: Photoelectron wavenumber grid (Å⁻¹).
        chi: χ(k) on that grid; NaN marks k with no data.
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
        from larch.xafs import ftwindow, xftf
    except ImportError as exc:
        msg = (
            "xraylarch is required for Fourier transforms. "
            "Install with: pip install xraylarch"
        )
        raise ImportError(msg) from exc

    p = resolve_ft_params(ft_params)
    k_arr = np.asarray(k, dtype=float)
    chi_arr = np.asarray(chi, dtype=float)

    missing = np.isnan(chi_arr)
    if missing.any():
        window = ftwindow(
            k_arr,
            xmin=float(p["kmin"]),
            xmax=float(p["kmax"]),
            dx=float(p["dk"]),
            window=str(p["window"]),
        )
        in_window = missing & (np.abs(window) > 1e-7)
        if in_window.any():
            bad = k_arr[in_window]
            logger.warning(
                "Fourier transform window covers k where no spectrum has data "
                "(%.2f-%.2f A^-1 of %d points); those k are treated as zero, "
                "which damps |chi(R)|. Reduce kmax below %.2f A^-1.",
                bad.min(),
                bad.max(),
                bad.size,
                bad.min(),
            )
        chi_arr = np.nan_to_num(chi_arr, nan=0.0)

    grp = Group(k=k_arr, chi=chi_arr)
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
    *,
    k_grid: np.ndarray | None = None,
    weights: list[float] | np.ndarray | None = None,
) -> ChiAverage:
    """Ensemble average a set of χ(k) spectra onto a common k grid.

    Each spectrum is put onto the reference grid by :func:`resample_chi`, so a
    member that stops at a lower k simply stops contributing there instead of
    being zero-filled. Zero-filling would drag the mean towards zero at high
    k, which is exactly where the Debye-Waller information lives.

    Args:
        k_list: Wavenumber grid of each spectrum (Å⁻¹).
        chi_list: χ(k) of each spectrum, matching ``k_list``.
        k_grid: Reference grid to average onto. Defaults to ``k_list[0]``.
        weights: Per-spectrum weights. Defaults to equal weighting.

    Returns:
        A :class:`ChiAverage`.
    """
    if not chi_list or not k_list:
        raise ValueError("Cannot average empty list of spectra.")
    if len(chi_list) != len(k_list):
        raise ValueError("k_list and chi_list must have the same length.")
    if weights is not None and len(weights) != len(chi_list):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of "
            f"spectra ({len(chi_list)})"
        )

    k_ref = np.asarray(k_list[0] if k_grid is None else k_grid, dtype=float)
    arr = np.asarray(
        [
            resample_chi(k_item, chi_item, k_ref)
            for k_item, chi_item in zip(k_list, chi_list, strict=True)
        ],
        dtype=float,
    )

    valid = ~np.isnan(arr)
    n_contributors = np.sum(valid, axis=0, dtype=np.int32)

    w = np.ones(arr.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    # Zero the weight of every point a spectrum does not cover, so both the
    # mean and the spread are taken over the contributors alone.
    w2d = np.where(valid, w[:, None], 0.0)
    v1 = w2d.sum(axis=0)
    v2 = (w2d**2).sum(axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_chi = np.sum(np.nan_to_num(arr) * w2d, axis=0) / np.where(
            v1 > 0, v1, np.nan
        )

        # Reliability-weighted unbiased variance; reduces to the ddof=1 sample
        # estimator when the weights are equal. ddof=1 throughout the package:
        # ensemble members are samples from a distribution, not the population.
        resid2 = np.nan_to_num((arr - mean_chi) ** 2) * w2d
        denom = v1 - v2 / np.where(v1 > 0, v1, np.nan)
        std_chi = np.sqrt(resid2.sum(axis=0) / np.where(denom > 0, denom, np.nan))

    # One sample gives no spread estimate; reporting zero there would
    # understate the uncertainty where the ensemble has thinned out.
    std_chi[n_contributors < 2] = np.nan

    return ChiAverage(k_ref, mean_chi, std_chi, n_contributors)


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
    "ChiAverage",
    "average_chi_arrays",
    "format_chi_ascii",
    "resample_chi",
    "resolve_ft_params",
    "xftf_arrays",
]
