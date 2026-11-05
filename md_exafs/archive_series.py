"""Plot-ready data series from an ensemble archive, and Vega-Lite views of them.

:func:`archive_series` reads an ``ensemble_results.h5`` archive and returns a
versioned, JSON-safe dict (lists only; NaN becomes ``None``) holding the overall
average, per-site averages and a capped sample of per-frame spectra. The series
is the contract; :func:`series_to_vega_lite` derives chart specifications from
it with the builders in :mod:`md_exafs.viz`.

Conventions recorded in the series ``units`` block: ``chi`` is stored
unweighted (no k**n factor); |chi(R)| is the transform of k**kweight * chi(k)
with the window and range in ``ft_params``, so its unit is A^-(kweight+1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hdf5 import ArchiveReader
from .spectra import average_chi_arrays

SCHEMA_VERSION = 1

# Largest |stored - recomputed| overall chi, as a fraction of the stored peak
# |chi|, before a warning is attached. The two are averages of the same float64
# spectra, so a healthy archive agrees to rounding; 1e-4 leaves room for the
# frame-weighting approximation on ragged k-grids and still catches a stale
# aggregate, whose error is of order the signal itself.
DEFAULT_REL_TOL = 1e-4

# Per series: 2 channels' min and max per bucket plus both endpoints must fit.
MIN_MAX_POINTS = 16

VIEWS = ("chi_k", "chi_r", "sites", "frames")


class ViewUnavailableError(ValueError):
    """The series holds no data for the requested view."""


def _keep_indices(n: int, max_points: int | None, *channels: np.ndarray) -> np.ndarray:
    """Indices to keep when thinning ``n`` points to at most ``max_points``.

    Min/max-per-bucket decimation: each of ``channels`` contributes its minimum
    and maximum in every bucket, so narrow peaks survive. Both endpoints are
    always kept; NaN is ignored when picking extremes.
    """
    if max_points is None or n <= max_points:
        return np.arange(n)
    n_buckets = max((max_points - 2) // (2 * len(channels)), 1)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    keep = {0, n - 1}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        for y in channels:
            seg = y[lo:hi]
            if np.all(np.isnan(seg)):
                continue
            keep.add(lo + int(np.nanargmin(seg)))
            keep.add(lo + int(np.nanargmax(seg)))
    return np.array(sorted(keep))


def _floats(a: np.ndarray, idx: np.ndarray) -> list[float | None]:
    return [float(v) if np.isfinite(v) else None for v in np.asarray(a, float)[idx]]


def _spectrum(data: dict[str, np.ndarray], max_points: int | None) -> dict[str, Any]:
    """Series entry for one averaged spectrum (any of overall, site, frame)."""
    k = np.asarray(data["k"], dtype=float)
    chi = np.asarray(data["chi"], dtype=float)
    ik = _keep_indices(k.size, max_points, chi)
    out: dict[str, Any] = {"k": _floats(k, ik), "chi": _floats(chi, ik)}
    if "chi_std" in data:
        out["chi_std"] = _floats(data["chi_std"], ik)
    if "n_contributors" in data:
        out["n_contributors"] = [int(v) for v in np.asarray(data["n_contributors"])[ik]]
    if "r" in data and "chir_mag" in data:
        r = np.asarray(data["r"], dtype=float)
        channels = [
            np.asarray(data[c], dtype=float)
            for c in ("chir_mag", "chir_re", "chir_im")
            if c in data
        ]
        ir = _keep_indices(r.size, max_points, *channels)
        out["r"] = _floats(r, ir)
        for name in ("chir_mag", "chir_re", "chir_im"):
            if name in data:
                out[name] = _floats(data[name], ir)
    return out


def _even_subsample(indices: list[int], cap: int) -> list[int]:
    if len(indices) <= cap:
        return list(indices)
    if cap <= 0:
        return []
    pos = np.unique(np.round(np.linspace(0, len(indices) - 1, cap)).astype(int))
    return [indices[i] for i in pos]


def _check_overall(
    reader: ArchiveReader,
    overall: dict[str, np.ndarray],
    rel_tol: float,
) -> tuple[dict[str, Any], list[str]]:
    """Compare the stored overall chi with one recomputed from frame averages.

    Frames are weighted by their spectrum count (the peak of their
    ``n_contributors``) so unequal numbers of absorbers per frame do not
    register as a mismatch; equal weights are used when counts are absent.
    """
    frame_ids = reader.frame_indices
    check: dict[str, Any] = {
        "status": "skipped",
        "rel_tol": rel_tol,
        "max_rel_deviation": None,
        "n_frames_used": 0,
    }
    if not frame_ids:
        return check, [
            "No frame averages are stored, so the stored overall average could "
            "not be checked against a recomputation."
        ]

    frames = [reader.get_frame_average(i) for i in frame_ids]
    weights: list[float] | None = None
    if all("n_contributors" in f for f in frames):
        weights = [float(np.max(f["n_contributors"])) for f in frames]
    stored_k = np.asarray(overall["k"], dtype=float)
    stored = np.asarray(overall["chi"], dtype=float)
    recomputed = average_chi_arrays(
        [f["k"] for f in frames],
        [f["chi"] for f in frames],
        k_grid=stored_k,
        weights=weights,
    ).mean

    both = np.isfinite(stored) & np.isfinite(recomputed)
    mask_differs = int(np.sum(np.isfinite(stored) != np.isfinite(recomputed)))
    check["n_frames_used"] = len(frames)
    check["weighting"] = "n_contributors" if weights is not None else "equal"
    if not both.any():
        check["status"] = "mismatch"
        return check, [
            "The stored overall average and the one recomputed from the frame "
            "averages share no finite points."
        ]

    scale = float(np.max(np.abs(stored[both])))
    deviation = float(np.max(np.abs(stored[both] - recomputed[both])))
    rel = deviation / scale if scale > 0 else deviation
    check["max_rel_deviation"] = rel
    if rel > rel_tol or mask_differs:
        check["status"] = "mismatch"
        return check, [
            "The stored overall average differs from the average recomputed from "
            f"{len(frames)} stored frame averages: max |dchi| is {rel:.2e} of the "
            f"peak |chi| (tolerance {rel_tol:.1e}), and {mask_differs} point(s) "
            "have different coverage. The stored aggregate may be stale; the "
            "stored average is plotted."
        ]
    check["status"] = "ok"
    return check, []


def archive_series(
    path: Path | str,
    *,
    max_points: int | None = None,
    max_frames: int = 20,
    max_sites: int = 50,
    rel_tol: float = DEFAULT_REL_TOL,
) -> dict[str, Any]:
    """Read an ensemble archive into a plot-ready, JSON-safe series.

    Args:
        path: Ensemble archive (``ensemble_results.h5``).
        max_points: Cap on points per plotted curve, applied by min/max-per-bucket
            decimation so peaks survive; ``None`` keeps every point. At least
            :data:`MIN_MAX_POINTS`.
        max_frames: Per-frame spectra to include, evenly subsampled in frame order.
        max_sites: Per-site spectra to include, evenly subsampled in site order.
        rel_tol: Relative tolerance of the stored-vs-recomputed overall average
            check (fraction of the peak |chi|); see :data:`DEFAULT_REL_TOL`.

    Returns:
        A dict with ``schema_version``, ``units``, ``ft_params``, ``source``,
        ``downsampling``, ``overall``, ``sites``, ``frames``, ``checks`` and
        ``warnings``. Non-finite values are ``None``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: Not an ensemble archive, no overall average, or bad parameters.
    """
    if max_points is not None and max_points < MIN_MAX_POINTS:
        raise ValueError(f"max_points must be at least {MIN_MAX_POINTS}")
    if max_frames < 0 or max_sites < 0:
        raise ValueError("max_frames and max_sites must be non-negative")
    if rel_tol <= 0:
        raise ValueError("rel_tol must be positive")

    reader = ArchiveReader(path)
    try:
        is_ensemble = reader.is_ensemble
    except AttributeError:  # an HDF5 file without a /meta group
        is_ensemble = False
    if not is_ensemble:
        raise ValueError(f"{reader.path} is not an ensemble archive")
    try:
        overall = reader.get_overall_average()
    except KeyError as exc:
        raise ValueError(f"{reader.path} has no overall average") from exc

    warnings: list[str] = []
    ft_params = reader.fourier_params
    if not ft_params:
        warnings.append("The archive records no Fourier-transform parameters.")
    kweight = ft_params.get("kweight")

    check, check_warnings = _check_overall(reader, overall, rel_tol)
    warnings.extend(check_warnings)

    site_ids = reader.site_indices
    frame_ids = reader.frame_indices
    shown_sites = _even_subsample(site_ids, max_sites)
    shown_frames = _even_subsample(frame_ids, max_frames)

    chir_unit = f"A^-{int(kweight) + 1}" if kweight is not None else "A^-(kweight+1)"
    return {
        "schema_version": SCHEMA_VERSION,
        "units": {
            "k": "A^-1",
            "r": "A",
            "chi": "dimensionless",
            "chir": chir_unit,
            "chi_weighting": "none (chi(k) is stored without a k**n factor)",
            "chir_weighting": f"k**{kweight} applied before the Fourier transform",
        },
        "ft_params": ft_params,
        "source": {
            "n_sites": len(site_ids),
            "n_frames": len(frame_ids),
        },
        "downsampling": {
            "method": "min/max per bucket",
            "max_points": max_points,
        },
        "overall": _spectrum(overall, max_points),
        "sites": {
            "total": len(site_ids),
            "shown_indices": shown_sites,
            "spectra": [
                {"site_idx": i, **_spectrum(reader.get_site_average(i), max_points)}
                for i in shown_sites
            ],
        },
        "frames": {
            "total": len(frame_ids),
            "shown_indices": shown_frames,
            "spectra": [
                {"frame_idx": i, **_spectrum(reader.get_frame_average(i), max_points)}
                for i in shown_frames
            ],
        },
        "checks": {"stored_vs_recomputed": check},
        "warnings": warnings,
    }


def _own_zoom(chart: Any, name: str) -> Any:
    """Give ``chart`` its own scale-bound zoom selection called ``name``.

    The builders add identically configured ``.interactive()`` selections, which
    Altair merges into one shared parameter when charts are concatenated. That
    would zoom a chi(k) panel and a chi(R) panel together, over unrelated axes.
    """
    import altair as alt

    chart.params = alt.Undefined
    return chart.add_params(
        alt.selection_interval(name=name, bind="scales", encodings=["x", "y"])
    )


def _frame_df(spectra: list[dict[str, Any]], key: str, label: str) -> tuple[Any, Any]:
    """Long-form (k, chi) and (r, chir_mag) frames with one ``path`` per spectrum."""
    import pandas as pd

    chi_rows, chir_rows = [], []
    for s in spectra:
        name = f"{label} {s[key]}"
        chi_rows.append(pd.DataFrame({"k": s["k"], "chi": s["chi"], "path": name}))
        if "r" in s:
            chir_rows.append(
                pd.DataFrame({"r": s["r"], "chir_mag": s["chir_mag"], "path": name})
            )
    return (
        pd.concat(chi_rows, ignore_index=True),
        pd.concat(chir_rows, ignore_index=True) if chir_rows else None,
    )


def series_to_vega_lite(
    series: dict[str, Any],
    *,
    view: str = "chi_k",
    width: int = 480,
    height: int = 260,
) -> dict[str, Any]:
    """Vega-Lite specification for one view of a series.

    Views: ``chi_k`` (overall chi(k) with a +-1 std band and a contributor-count
    strip), ``chi_r`` (|chi(R)|, Re and Im of the overall average), ``sites``
    and ``frames`` (chi(k) over |chi(R)| for the included spectra).

    The series warnings, FT parameters, units and checks travel in the spec's
    top-level ``usermeta``, so a viewer needs only the spec.

    Raises:
        ValueError: Unknown ``view``.
        ViewUnavailableError: The series holds no data for ``view``.
    """
    import altair as alt
    import pandas as pd

    from .viz import build_chi_chart, build_chir_chart

    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}; expected one of {VIEWS}")
    overall = series["overall"]
    kweight = series["ft_params"].get("kweight")
    chir_title = f"χ(R) (Å^-{int(kweight) + 1})" if kweight is not None else "χ(R)"

    if view == "chi_k":
        df = pd.DataFrame(
            {"k": overall["k"], "chi": overall["chi"], "path": "ensemble mean"}
        )
        chart: Any = build_chi_chart(
            df, title="Ensemble χ(k)", width=width, height=height
        )
        if overall.get("chi_std") is not None:
            std = pd.Series(overall["chi_std"], dtype=float)
            chi = pd.Series(overall["chi"], dtype=float)
            band_df = pd.DataFrame(
                {"k": overall["k"], "lo": chi - std, "hi": chi + std}
            )
            band = (
                alt.Chart(band_df)
                .mark_area(opacity=0.25)
                .encode(
                    x=alt.X("k:Q", title="k (Å⁻¹)"),
                    y=alt.Y("lo:Q", title="χ(k)"),
                    y2="hi:Q",
                )
            )
            chart = alt.layer(band, chart)
        if overall.get("n_contributors") is not None:
            n_df = pd.DataFrame({"k": overall["k"], "n": overall["n_contributors"]})
            strip = (
                alt.Chart(n_df)
                .mark_line(interpolate="step-after")
                .encode(
                    x=alt.X("k:Q", title="k (Å⁻¹)"),
                    y=alt.Y("n:Q", title="spectra"),
                )
                .properties(width=width, height=70)
            )
            chart = alt.vconcat(chart, strip)
    elif view == "chi_r":
        if "r" not in overall:
            raise ViewUnavailableError(
                "The series has no chi(R) for the overall average"
            )
        labels = {"chir_mag": "|χ(R)|", "chir_re": "Re χ(R)", "chir_im": "Im χ(R)"}
        parts = [
            pd.DataFrame({"r": overall["r"], "chir_mag": overall[c], "path": label})
            for c, label in labels.items()
            if c in overall
        ]
        chart = build_chir_chart(
            pd.concat(parts, ignore_index=True),
            title="Ensemble χ(R)",
            width=width,
            height=height,
        )
        chart.encoding.y = alt.Y("chir_mag:Q", title=chir_title)
    else:
        key, label = ("site_idx", "site") if view == "sites" else ("frame_idx", "frame")
        spectra = series[view]["spectra"]
        if not spectra:
            raise ViewUnavailableError(f"The series has no {label} spectra")
        chi_df, chir_df = _frame_df(spectra, key, label)
        chart = build_chi_chart(
            chi_df, title=f"χ(k) per {label}", width=width, height=height
        )
        if chir_df is not None:
            chart = alt.vconcat(
                _own_zoom(chart, "zoom_k"),
                _own_zoom(
                    build_chir_chart(
                        chir_df,
                        title=f"|χ(R)| per {label}",
                        width=width,
                        height=height,
                    ),
                    "zoom_r",
                ),
            )

    chart.usermeta = {
        "series_schema_version": series["schema_version"],
        "view": view,
        "warnings": series["warnings"],
        "ft_params": series["ft_params"],
        "units": series["units"],
        "checks": series["checks"],
        "shown": {
            "sites": series["sites"]["shown_indices"],
            "frames": series["frames"]["shown_indices"],
        },
    }
    with alt.data_transformers.disable_max_rows():
        return dict(chart.to_dict())


__all__ = [
    "DEFAULT_REL_TOL",
    "MIN_MAX_POINTS",
    "SCHEMA_VERSION",
    "VIEWS",
    "ViewUnavailableError",
    "archive_series",
    "series_to_vega_lite",
]
