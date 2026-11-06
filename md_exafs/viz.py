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


def _build_line_chart(
    df: Any,
    *,
    x_field: str,
    x_title: str,
    y_field: str,
    y_title: str,
    title: str,
    width: int,
    height: int,
) -> Any:
    """Altair line chart with one coloured line per ``path``."""
    import altair as alt

    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y(f"{y_field}:Q", title=y_title),
            color=alt.Color("path:N", title="Path"),
            tooltip=["path:N", f"{x_field}:Q", f"{y_field}:Q"],
        )
        .properties(width=width, height=height, title=title)
        .interactive()
    )


def build_chi_chart(
    df: Any,
    title: str = "χ(k)",
    width: int = 400,
    height: int = 250,
    *,
    y_field: str = "chi",
    y_title: str = "χ(k)",
) -> Any:
    """Build a standard Altair line chart for χ(k).

    ``y_field`` names the column to plot (e.g. a k-weighted χ) and ``y_title``
    its axis label.
    """
    return _build_line_chart(
        df,
        x_field="k",
        x_title="k (Å⁻¹)",
        y_field=y_field,
        y_title=y_title,
        title=title,
        width=width,
        height=height,
    )


def build_chir_chart(
    df: Any,
    title: str = "|χ(R)|",
    width: int = 400,
    height: int = 250,
    *,
    y_field: str = "chir_mag",
    y_title: str = "|χ(R)|",
) -> Any:
    """Build a standard Altair line chart for Fourier transform |χ(R)|.

    ``y_field`` names the column to plot and ``y_title`` its axis label; use a
    neutral title when the frame mixes magnitude, real and imaginary parts.
    """
    return _build_line_chart(
        df,
        x_field="r",
        x_title="R (Å)",
        y_field=y_field,
        y_title=y_title,
        title=title,
        width=width,
        height=height,
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


VIEWS = ("chi_k", "chi_r", "sites", "frames")

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class ViewUnavailableError(ValueError):
    """The series holds no data for the requested view."""


def _power(n: int) -> str:
    return str(n).translate(_SUPERSCRIPT)


def _inv_angstrom(n: int) -> str:
    """Unit label Å⁻ⁿ, or '' for a dimensionless quantity."""
    return f"Å{_power(-n)}" if n else ""


def _chi_k_label(kweight: int) -> str:
    """Axis label for the k-weighted χ(k), e.g. 'k²χ(k) (Å⁻²)'."""
    base = "χ(k)" if kweight == 0 else f"k{_power(kweight) if kweight > 1 else ''}χ(k)"
    unit = _inv_angstrom(kweight)
    return f"{base} ({unit})" if unit else base


def _weight_phrase(kweight: int) -> str:
    if kweight == 0:
        return "unweighted"
    return "k-weighted" if kweight == 1 else f"k{_power(kweight)}-weighted"


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


def _marker_rules(df: Any, values: list[float]) -> list[Any]:
    """Thin dashed vertical rules at each of ``values``.

    The rules take their data from the first row of ``df``, so they add no
    series of their own and carry no extra ``path`` label.
    """
    import altair as alt

    row = df.iloc[:1]
    return [
        alt.Chart(row)
        .mark_rule(strokeDash=[4, 3], strokeWidth=1, color="gray", opacity=0.8)
        .encode(x=alt.datum(float(v)))
        for v in values
    ]


def _spectra_frames(
    spectra: list[dict[str, Any]], key: str, label: str
) -> tuple[Any, Any]:
    """Long-form chi(k) and |chi(R)| frames, one path per spectrum.

    The |chi(R)| frame lists only spectra that carry a transform.
    """
    import pandas as pd

    chi_rows, chir_rows = [], []
    for s in spectra:
        name = f"{label} {s[key]}"
        chi_rows.append(
            pd.DataFrame({"k": s["k"], "chi_weighted": s["chi_weighted"], "path": name})
        )
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
    """Vega-Lite specification for one view of an :func:`archive_series` result.

    Views: ``chi_k`` (overall k-weighted chi(k) with a +-1 std band, dashed
    rules at kmin and kmax, and a contributor-count strip), ``chi_r`` (|chi(R)|,
    Re and Im of the overall average, with a rule at rmax), and ``sites`` and
    ``frames`` (k-weighted chi(k) over |chi(R)| for the included spectra).

    The series warnings, FT parameters and units travel in the spec's top-level
    ``usermeta``, so a viewer needs only the spec.

    Raises:
        ValueError: Unknown ``view``.
        ViewUnavailableError: The series holds no data for ``view``; ``sites``
            and ``frames`` need ``include`` in :func:`archive_series`.
    """
    import altair as alt
    import pandas as pd

    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}; expected one of {VIEWS}")
    overall = series["overall"]
    ft = series["ft_params"]
    kweight = int(ft["kweight"])
    chi_title = _chi_k_label(kweight)
    phrase = _weight_phrase(kweight)

    if view == "chi_k":
        df = pd.DataFrame(
            {
                "k": overall["k"],
                "chi_weighted": overall["chi_weighted"],
                "path": "ensemble mean",
            }
        )
        layers: list[Any] = [
            build_chi_chart(
                df,
                title=f"Ensemble χ(k), {phrase}",
                width=width,
                height=height,
                y_field="chi_weighted",
                y_title=chi_title,
            )
        ]
        if overall.get("chi_std_weighted") is not None:
            std = pd.Series(overall["chi_std_weighted"], dtype=float)
            center = pd.Series(overall["chi_weighted"], dtype=float)
            band_df = pd.DataFrame(
                {"k": overall["k"], "lo": center - std, "hi": center + std}
            )
            layers.insert(
                0,
                alt.Chart(band_df)
                .mark_area(opacity=0.25)
                .encode(
                    x=alt.X("k:Q", title="k (Å⁻¹)"),
                    y=alt.Y("lo:Q", title=chi_title),
                    y2="hi:Q",
                ),
            )
        layers.extend(_marker_rules(df, [ft["kmin"], ft["kmax"]]))
        chart = alt.layer(*layers)
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
        chir_df = pd.concat(
            [
                pd.DataFrame({"r": overall["r"], "chir": overall[c], "path": label})
                for c, label in labels.items()
                if c in overall
            ],
            ignore_index=True,
        )
        chart = alt.layer(
            build_chir_chart(
                chir_df,
                title=f"Ensemble χ(R), {phrase}",
                width=width,
                height=height,
                y_field="chir",
                y_title=f"χ(R) ({_inv_angstrom(kweight + 1)})",
            ),
            *_marker_rules(chir_df, [ft["rmax"]]),
        )
    else:
        label = view.removesuffix("s")
        spectra = series[view]["spectra"]
        if not spectra:
            raise ViewUnavailableError(
                f"The series has no {label} spectra; request them with "
                f"archive_series(include=({view!r},))"
            )
        chi_df, chir_df = _spectra_frames(spectra, f"{label}_idx", label)
        chart = build_chi_chart(
            chi_df,
            title=f"χ(k) per {label}",
            width=width,
            height=height,
            y_field="chi_weighted",
            y_title=chi_title,
        )
        if chir_df is not None:
            chir_chart = build_chir_chart(
                chir_df, title=f"|χ(R)| per {label}", width=width, height=height
            )
            chart = alt.vconcat(
                _own_zoom(chart, "zoom_k"), _own_zoom(chir_chart, "zoom_r")
            )

    chart.usermeta = {
        "series_schema_version": series["schema_version"],
        "view": view,
        "warnings": series["warnings"],
        "ft_params": ft,
        "ft_params_source": series.get("ft_params_source"),
        "ft_defaults": series.get("ft_defaults"),
        "ft_limits": series.get("ft_limits"),
        "ft_warnings": series.get("ft_warnings", []),
        "units": series["units"],
        "shown": {
            "sites": series["sites"]["shown_indices"],
            "frames": series["frames"]["shown_indices"],
        },
    }
    with alt.data_transformers.disable_max_rows():
        return dict(chart.to_dict())


__all__ = [
    "group_path_results",
    "build_chi_chart",
    "build_chir_chart",
    "build_sigma2_chart",
    "calculate_path_vectors",
    "series_to_vega_lite",
]
