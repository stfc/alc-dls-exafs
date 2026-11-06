"""Plot-ready data series from an ensemble archive.

:func:`archive_series` reads an ``ensemble_results.h5`` archive and returns a
versioned, JSON-safe dict (lists only; NaN becomes ``None``) holding the overall
average and, on request, per-site averages and a capped sample of per-frame
spectra. The series is the contract; :func:`md_exafs.viz.series_to_vega_lite`
derives chart specifications from it.

Conventions recorded in the series ``units`` block: ``chi`` is stored
unweighted (no k**n factor). ``chi_weighted`` is k**kweight * chi, the curve
plotted in the chi(k) views, and |chi(R)| is the transform of that same
weighting with the window and range in ``ft_params``, so its unit is
A^-(kweight+1).

Fourier-transform parameters are either the archive's own (``ft_params=None``:
the stored chi(R) is used) or caller-supplied, in which case chi(R) is
recomputed from the stored unweighted chi(k) for every spectrum.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import numpy as np

from .hdf5 import ArchiveReader
from .spectra import FT_WINDOWS, resolve_ft_params, xftf_arrays

SCHEMA_VERSION = 1

# Breakdown sections ``include`` can request. The overall average is always read.
SECTIONS = ("sites", "frames")

# Fourier-transform parameters a caller may override.
FT_REQUEST_KEYS = ("kmin", "kmax", "dk", "window", "kweight", "rmax")

# k-weights offered to a UI (``ft_limits``). Requests accept 0 to 4 inclusive.
KWEIGHTS = (0, 1, 2, 3)
KWEIGHT_RANGE = (0, 4)

# Largest R_max accepted; larger values are rejected as absurd.
RMAX_MAX = 30.0


class InvalidFtParamsError(ValueError):
    """Requested Fourier-transform parameters cannot be used.

    ``problems`` holds one message per problem found.
    """

    def __init__(self, problems: list[str]) -> None:
        """Join ``problems`` into the message."""
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _floats(a: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in np.asarray(a, float)]


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return False
    return bool(np.isfinite(value))


def _resolve_ft_request(
    request: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    """Merge a caller's Fourier-transform request over the archive's defaults.

    The accepted keys are :data:`FT_REQUEST_KEYS`, deliberately narrower than
    :func:`spectra.resolve_ft_params`: ``dk2``, ``with_phase``, ``rmax_out``,
    ``nfft`` and ``kstep`` are not exposed here. Window names are strict: only
    the :data:`spectra.FT_WINDOWS` names are accepted, although larch itself
    also matches on the first three letters.

    Raises:
        InvalidFtParamsError: One message per problem: unknown keys, bad types
            or ranges, an empty window, or ``kmin >= kmax``.
    """
    problems: list[str] = []
    unknown = sorted(str(key) for key in set(request) - set(FT_REQUEST_KEYS))
    if unknown:
        problems.append(
            f"unknown Fourier-transform parameter(s) {unknown}; "
            f"allowed: {list(FT_REQUEST_KEYS)}"
        )
    merged = {**defaults, **{k: v for k, v in request.items() if k in FT_REQUEST_KEYS}}

    if not _finite_number(merged["kmin"]) or merged["kmin"] < 0:
        problems.append(f"kmin must be a finite number >= 0, got {merged['kmin']!r}")
    if not _finite_number(merged["kmax"]):
        problems.append(f"kmax must be a finite number, got {merged['kmax']!r}")
    elif _finite_number(merged["kmin"]) and merged["kmin"] >= merged["kmax"]:
        problems.append(
            f"kmin ({merged['kmin']:g}) must be below kmax ({merged['kmax']:g})"
        )
    if not _finite_number(merged["dk"]) or merged["dk"] < 0:
        problems.append(f"dk must be a finite number >= 0, got {merged['dk']!r}")
    if not _finite_number(merged["rmax"]) or not 0 < merged["rmax"] <= RMAX_MAX:
        problems.append(f"rmax must be in (0, {RMAX_MAX:g}] A, got {merged['rmax']!r}")
    if merged["window"] not in FT_WINDOWS:
        problems.append(
            f"window must be one of {list(FT_WINDOWS)}, got {merged['window']!r}"
        )
    kw = merged["kweight"]
    lo, hi = KWEIGHT_RANGE
    if (
        isinstance(kw, bool)
        or not isinstance(kw, int | np.integer)
        or not lo <= kw <= hi
    ):
        problems.append(f"kweight must be an integer from {lo} to {hi}, got {kw!r}")

    if problems:
        raise InvalidFtParamsError(problems)
    return {
        "kmin": float(merged["kmin"]),
        "kmax": float(merged["kmax"]),
        "dk": float(merged["dk"]),
        "window": str(merged["window"]),
        "kweight": int(merged["kweight"]),
        "rmax": float(merged["rmax"]),
    }


def _archive_ft_defaults(stored: dict[str, Any]) -> dict[str, Any]:
    """The archive's Fourier-transform parameters, library defaults filled in."""
    resolved = resolve_ft_params({k: stored[k] for k in FT_REQUEST_KEYS if k in stored})
    return {k: resolved[k] for k in FT_REQUEST_KEYS}


def _coverage_limits(overall: dict[str, np.ndarray]) -> dict[str, Any]:
    """First and last k with data in the overall average, and the grid top."""
    k = np.asarray(overall["k"], dtype=float)
    if "n_contributors" in overall:
        covered = np.asarray(overall["n_contributors"]) > 0
    else:
        covered = np.isfinite(np.asarray(overall["chi"], dtype=float))
    if not covered.any():
        kmin_data = kmax_data = None
    else:
        kmin_data = float(k[covered][0])
        kmax_data = float(k[covered][-1])
    return {
        "kmin_data": kmin_data,
        "kmax_data": kmax_data,
        "k_grid_max": float(k.max()) if k.size else None,
        "kweights": list(KWEIGHTS),
        "windows": list(FT_WINDOWS),
        "rmax_max": RMAX_MAX,
    }


def _ft_data_warnings(
    ft: dict[str, Any], limits: dict[str, Any]
) -> list[dict[str, Any]]:
    """Structured warnings for a window that reaches past the data."""
    if limits["kmin_data"] is None:
        return [
            {
                "code": "no_data",
                "section": "overall",
                "message": "The overall average has no k with data, so the "
                "Fourier-transform window cannot be checked against it.",
            }
        ]
    out: list[dict[str, Any]] = []
    kmin_data, kmax_data = limits["kmin_data"], limits["kmax_data"]
    if ft["kmax"] > kmax_data:
        out.append(
            {
                "code": "window_beyond_data",
                "section": "overall",
                "param": "kmax",
                "value": ft["kmax"],
                "data_limit": kmax_data,
                "message": f"kmax={ft['kmax']:g} A^-1 lies above the highest k "
                f"with data ({kmax_data:g} A^-1). The transform fills k inside "
                "the window beyond the data with zeros, which damps |chi(R)|; "
                f"lower kmax to at most {kmax_data:g} A^-1.",
            }
        )
    if ft["kmin"] < kmin_data:
        out.append(
            {
                "code": "window_beyond_data",
                "section": "overall",
                "param": "kmin",
                "value": ft["kmin"],
                "data_limit": kmin_data,
                "message": f"kmin={ft['kmin']:g} A^-1 lies below the lowest k "
                f"with data ({kmin_data:g} A^-1). The transform fills k inside "
                "the window below the data with zeros, which damps |chi(R)|; "
                f"raise kmin to at least {kmin_data:g} A^-1.",
            }
        )
    return out


def _section_ft_warnings(
    section: str, spectra: list[dict[str, np.ndarray]], ft: dict[str, Any]
) -> list[dict[str, Any]]:
    """One summary warning when the window reaches past some spectra's data."""
    affected = sum(
        any(
            w["code"] == "window_beyond_data"
            for w in _ft_data_warnings(ft, _coverage_limits(data))
        )
        for data in spectra
    )
    if not affected:
        return []
    label = {"sites": "site", "frames": "frame"}[section]
    return [
        {
            "code": "window_beyond_data",
            "section": section,
            "n_affected": affected,
            "n_shown": len(spectra),
            "message": "The Fourier-transform window reaches past the data of "
            f"{affected} of {len(spectra)} shown {label} spectra. Their k outside "
            "the data are zero-filled, which damps |chi(R)|; lower kmax or raise "
            "kmin to the range the data cover.",
        }
    ]


def _transform(data: dict[str, np.ndarray], ft: dict[str, Any]) -> dict[str, Any]:
    """Fourier transform of one stored unweighted spectrum with ``ft``.

    Coverage is reported once per series through ``ft_warnings``, not per
    spectrum, so the transform itself stays quiet.
    """
    res = xftf_arrays(
        np.asarray(data["k"], dtype=float),
        np.asarray(data["chi"], dtype=float),
        ft,
        warn_uncovered=False,
    )
    return {name: res[name] for name in ("r", "chir_mag", "chir_re", "chir_im")}


def _with_transform(
    data: dict[str, np.ndarray], ft: dict[str, Any], recompute: bool
) -> dict[str, Any]:
    """``data`` with chi(R) recomputed when asked, or when it was never stored."""
    if recompute or "chir_mag" not in data:
        return {**data, **_transform(data, ft)}
    return data


def _spectrum(data: dict[str, np.ndarray], kweight: int) -> dict[str, Any]:
    """Series entry for one averaged spectrum (any of overall, site, frame)."""
    k = np.asarray(data["k"], dtype=float)
    chi = np.asarray(data["chi"], dtype=float)
    weight = k**kweight
    out: dict[str, Any] = {
        "k": _floats(k),
        "chi": _floats(chi),
        "chi_weighted": _floats(weight * chi),
    }
    if "chi_std" in data:
        std = np.asarray(data["chi_std"], dtype=float)
        out["chi_std"] = _floats(std)
        out["chi_std_weighted"] = _floats(weight * std)
    if "n_contributors" in data:
        out["n_contributors"] = [int(v) for v in np.asarray(data["n_contributors"])]
    if "r" in data and "chir_mag" in data:
        out["r"] = _floats(data["r"])
        for name in ("chir_mag", "chir_re", "chir_im"):
            if name in data:
                out[name] = _floats(data[name])
    return out


def _even_subsample(indices: list[int], cap: int) -> list[int]:
    if len(indices) <= cap:
        return list(indices)
    if cap <= 0:
        return []
    pos = np.unique(np.round(np.linspace(0, len(indices) - 1, cap)).astype(int))
    return [indices[i] for i in pos]


def _units(kweight: int) -> dict[str, str]:
    unit_k = f"A^-{kweight}" if kweight else "dimensionless"
    return {
        "k": "A^-1",
        "r": "A",
        "chi": "dimensionless",
        "chi_weighted": unit_k,
        "chir": f"A^-{kweight + 1}",
        "chi_weighting": "none in chi (stored without a k**n factor); "
        f"chi_weighted = k**{kweight} * chi",
        "chir_weighting": f"k**{kweight} applied before the Fourier transform",
    }


def _section(
    name: str,
    ids: list[int],
    get: Callable[[int], dict[str, np.ndarray]],
    cap: int,
    include: Collection[str],
    ft: dict[str, Any],
    recompute: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Series section for sites or frames; nothing is read unless included."""
    shown = _even_subsample(ids, cap) if name in include else []
    raw = [get(i) for i in shown]
    key = f"{name.removesuffix('s')}_idx"
    spectra = [
        {key: i, **_spectrum(_with_transform(data, ft, recompute), ft["kweight"])}
        for i, data in zip(shown, raw, strict=True)
    ]
    section = {"total": len(ids), "shown_indices": shown, "spectra": spectra}
    return section, raw


def archive_series(
    path: Path | str,
    *,
    ft_params: dict[str, Any] | None = None,
    max_frames: int = 20,
    max_sites: int = 50,
    include: Collection[str] = (),
) -> dict[str, Any]:
    """Read an ensemble archive into a plot-ready, JSON-safe series.

    Args:
        path: Ensemble archive (``ensemble_results.h5``).
        ft_params: Fourier-transform parameters to use instead of the archive's.
            Any subset of ``kmin``, ``kmax``, ``dk``, ``window``, ``kweight``
            and ``rmax``; the rest default to the archive's stored values. When
            given, chi(R) is recomputed with :func:`spectra.xftf_arrays` from
            the stored unweighted chi(k) of every spectrum read. ``None`` keeps
            the stored chi(R).
        max_frames: Per-frame spectra to include, evenly subsampled in frame order.
        max_sites: Per-site spectra to include, evenly subsampled in site order.
        include: Breakdowns to read, a subset of :data:`SECTIONS`. The overall
            average is always read. A section left out keeps its ``total`` and
            comes back with no spectra, and none of its data is read.

    Returns:
        A dict with ``schema_version``, ``units``, ``ft_params`` (the resolved
        parameters used), ``ft_params_source`` (``"archive"`` or ``"request"``),
        ``ft_defaults`` (the archive's stored parameters), ``ft_limits`` (what a
        UI can offer: k with data, k-weights, windows, R_max cap),
        ``ft_warnings`` (structured), ``source``, ``overall``, ``sites``,
        ``frames`` and ``warnings``. Non-finite values are ``None``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        InvalidFtParamsError: ``ft_params`` is unusable (a ``ValueError``).
        ValueError: Not an ensemble archive, no overall average, or bad parameters.
    """
    if max_frames < 0 or max_sites < 0:
        raise ValueError("max_frames and max_sites must be non-negative")
    unknown_sections = sorted(set(include) - set(SECTIONS))
    if unknown_sections:
        raise ValueError(
            f"Unknown section(s) {unknown_sections}; expected a subset of {SECTIONS}"
        )

    reader = ArchiveReader(path)
    if not reader.is_ensemble:
        raise ValueError(f"{reader.path} is not an ensemble archive")
    try:
        overall_raw = reader.get_overall_average()
    except KeyError as exc:
        raise ValueError(f"{reader.path} has no overall average") from exc

    warnings: list[str] = []
    stored = reader.fourier_params
    if not stored:
        warnings.append(
            "The archive records no Fourier-transform parameters; library "
            "defaults are assumed for the chi(R) labels."
        )
    defaults = _archive_ft_defaults(stored)
    if ft_params is None:
        ft, source, recompute = defaults, "archive", False
    else:
        ft = _resolve_ft_request(ft_params, defaults)
        source, recompute = "request", True

    overall_limits = _coverage_limits(overall_raw)
    ft_warnings = _ft_data_warnings(ft, overall_limits)
    sites, site_raw = _section(
        "sites",
        reader.site_indices,
        reader.get_site_average,
        max_sites,
        include,
        ft,
        recompute,
    )
    frames, frame_raw = _section(
        "frames",
        reader.frame_indices,
        reader.get_frame_average,
        max_frames,
        include,
        ft,
        recompute,
    )
    ft_warnings += _section_ft_warnings("sites", site_raw, ft)
    ft_warnings += _section_ft_warnings("frames", frame_raw, ft)
    warnings.extend(w["message"] for w in ft_warnings)

    return {
        "schema_version": SCHEMA_VERSION,
        "units": _units(ft["kweight"]),
        "ft_params": ft,
        "ft_params_source": source,
        "ft_defaults": defaults,
        "ft_limits": overall_limits,
        "ft_warnings": ft_warnings,
        "source": {"n_sites": sites["total"], "n_frames": frames["total"]},
        "overall": _spectrum(
            _with_transform(overall_raw, ft, recompute), ft["kweight"]
        ),
        "sites": sites,
        "frames": frames,
        "warnings": warnings,
    }


__all__ = [
    "FT_REQUEST_KEYS",
    "InvalidFtParamsError",
    "KWEIGHTS",
    "RMAX_MAX",
    "SCHEMA_VERSION",
    "SECTIONS",
    "archive_series",
]
