#!/usr/bin/env python3
"""Compile an md-exafs analysis_summary.json into a publication Quarto HTML report.

Usage:
    render_report.py analysis_summary.json -o report/ --title "Bulk Cu MD-EXAFS"
    render_report.py analysis_summary.json --notes notes.md --no-render
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

QUARTO_TIMEOUT = 300

CUSTOM_CSS = """
body { font-feature-settings: "tnum" 1; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
  gap: 0.75rem;
  margin: 1rem 0 1.4rem 0;
}
.metric-card {
  border: 1px solid #dee2e6;
  border-top: 3px solid #1f77b4;
  border-radius: 6px;
  padding: 0.7rem 0.9rem;
  background: #fbfcfd;
}
.metric-card .label {
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #5c6773;
}
.metric-card .value {
  font-size: 1.35rem;
  font-weight: 600;
  color: #1c2430;
  margin-top: 0.15rem;
}
.metric-card .sub {
  font-size: 0.78rem;
  color: #6c757d;
  margin-top: 0.1rem;
}
table { font-size: 0.9rem; }
"""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60].strip("-") or "exafs-report"


def _fmt(val: Any, fmt: str = ".4f", suffix: str = "") -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_param(param: dict[str, Any] | None, fmt: str = ".4f") -> str:
    if not param or param.get("value") is None:
        return "—"
    v = f"{float(param['value']):{fmt}}"
    se = param.get("stderr")
    if se is not None:
        return f"{v} ± {float(se):{fmt}}"
    return v


def _path_kind_tag(path: dict[str, Any]) -> str:
    """Extract 4-leg geometry tag (e.g. ``collinear_focus``) from a path label."""
    lbl = str(path.get("label", ""))
    for tag in ("collinear_thru_abs", "collinear_focus", "rattle", "bent"):
        if lbl.endswith(f"_{tag}"):
            return tag
    return ""


def _rollup_static_paths(
    paths: list[dict[str, Any]],
    gap_tol: float = 0.25,
) -> list[dict[str, Any]]:
    """Aggregate tight static FEFF sub-paths that matched the same MD shell.

    Display-only rollup: paths sharing ``(scatterer, nlegs, path_kind,
    sigma2_md)`` with no ``r_eff`` gap above ``gap_tol`` are combined into one
    summary shell row with total degeneracy ``N`` and N-weighted mean
    ``R_eff`` and ``ΔR``.
    """
    buckets: dict[tuple[str, int, str, float], list[dict[str, Any]]] = {}
    for p in paths:
        key = (
            str(p.get("scatterer", "")),
            int(p.get("nlegs", 0)),
            _path_kind_tag(p),
            round(float(p.get("sigma2_md", 0.0)), 6),
        )
        buckets.setdefault(key, []).append(p)

    shells: list[dict[str, Any]] = []
    for (scatterer, nlegs, kind, _s2_key), members in buckets.items():
        members.sort(key=lambda item: float(item["r_eff"]))
        groups: list[list[dict[str, Any]]] = [[members[0]]]
        for m in members[1:]:
            if float(m["r_eff"]) - float(groups[-1][-1]["r_eff"]) > gap_tol:
                groups.append([m])
            else:
                groups[-1].append(m)

        for grp in groups:
            ns = [float(g["degeneracy"]) for g in grp]
            rs = [float(g["r_eff"]) for g in grp]
            drs = [float(g["deltar"]) for g in grp if g.get("deltar") is not None]
            s2s = [float(g["sigma2_md"]) for g in grp]
            n_tot = sum(ns)
            w = ns if n_tot > 0 else [1.0] * len(grp)
            w_sum = sum(w)
            r_mean = sum(wi * ri for wi, ri in zip(w, rs, strict=True)) / w_sum
            s2_mean = sum(wi * si for wi, si in zip(w, s2s, strict=True)) / w_sum
            dr_mean = (
                sum(
                    wi * float(g["deltar"])
                    for wi, g in zip(w, grp, strict=True)
                    if g.get("deltar") is not None
                )
                / w_sum
                if drs
                else None
            )
            angles = [
                (wi, float(g["angle"]))
                for wi, g in zip(w, grp, strict=True)
                if g.get("angle") is not None
            ]
            ang_mean = (
                sum(wi * a for wi, a in angles) / sum(wi for wi, _ in angles)
                if angles
                else None
            )
            label = f"{scatterer} ({nlegs}-leg)"
            if kind:
                label = f"{scatterer} ({nlegs}-leg, {kind})"
            elif nlegs == 3 and ang_mean is not None:  # noqa: PLR2004
                label = f"{scatterer} (3-leg, {ang_mean:.0f}°)"

            shells.append(
                {
                    "label": label,
                    "nlegs": nlegs,
                    "r_mean": r_mean,
                    "r_min": min(rs),
                    "r_max": max(rs),
                    "n_total": n_tot,
                    "sigma2_md": s2_mean,
                    "deltar": dr_mean,
                    "n_subpaths": len(grp),
                }
            )

    shells.sort(key=lambda s: (s["r_mean"], s["nlegs"], s["label"]))
    return shells


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No entries._"
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def _build_window_section(win: dict[str, Any], fig_rel: str | None) -> str:
    rr = win["rrange"]
    fa = win["fit_a"]
    fb = win.get("fit_b")
    gs = win["group_sites"]
    wg = win["site_group_weights"]
    att_p = win["peak_attribution"].get("paths")
    att_s = win["peak_attribution"].get("sites")

    lines: list[str] = [f"### R ∈ [{rr[0]:g}, {rr[1]:g}] Å", ""]

    # 1. Model comparison table (MD ensemble vs Static baseline vs Static fit)
    comp_rows: list[list[str]] = [
        [
            "**MD-ensemble alignment**",
            _fmt_param(fa["params"].get("s02"), ".4f"),
            _fmt_param(fa["params"].get("e0_shift"), ".2f"),
            "—",
            f"**{_fmt(fa['rfactor_r'], '.4f')}**",
            _fmt(fa["rfactor_k"], ".4f"),
            _fmt(fa["chi2_reduced"], ".1f"),
            f"{fa['nvarys']} / {_fmt(fa['n_idp'], '.1f')}",
            "—",
        ]
    ]
    if fb is not None:
        b0 = (fb.get("comparison") or {}).get("b0")
        if b0:
            b0p = b0.get("params", {})
            comp_rows.append(
                [
                    "Static baseline (MD σ², fixed)",
                    _fmt(b0p.get("s02"), ".4f"),
                    _fmt(b0p.get("e0"), ".2f"),
                    "0 (fixed)",
                    _fmt(b0.get("rfactor_r"), ".4f"),
                    _fmt(b0.get("rfactor_k"), ".4f"),
                    _fmt(b0.get("chi2_reduced"), ".1f"),
                    f"0 / {_fmt(fb['n_idp'], '.1f')}",
                    _fmt(
                        100.0 * fb["truncation_fraction"]
                        if fb.get("truncation_fraction") is not None
                        else None,
                        ".1f",
                        "%",
                    ),
                ]
            )
        comp_rows.append(
            [
                "**Static + MD σ² fit**",
                _fmt_param(fb["params"].get("s02"), ".4f"),
                _fmt_param(fb["params"].get("e0"), ".2f"),
                _fmt_param(fb["params"].get("alpha"), "+.4f"),
                f"**{_fmt(fb['rfactor_r'], '.4f')}**",
                _fmt(fb["rfactor_k"], ".4f"),
                _fmt(fb["chi2_reduced"], ".1f"),
                f"{fb['nvarys']} / {_fmt(fb['n_idp'], '.1f')}",
                _fmt(
                    100.0 * fb["truncation_fraction"]
                    if fb.get("truncation_fraction") is not None
                    else None,
                    ".1f",
                    "%",
                ),
            ]
        )

    mf = win.get("manual_fit")
    if mf is not None:
        mp = {p["name"].lower(): p for p in mf.get("params", [])}
        n_varied = sum(1 for p in mf.get("params", []) if p.get("vary"))
        amp_p = next((mp[k] for k in ("amp", "s02", "so2") if k in mp), None)
        e0_p = next((mp[k] for k in ("dele", "e0", "enot", "de0") if k in mp), None)
        dr_p = next(
            (
                p
                for p in mf.get("params", [])
                if p["name"].lower() in ("alpha", "dscale")
                or p["name"].lower().startswith("dr")
            ),
            None,
        )
        dr_cell = (
            f"{_fmt(dr_p['value'], '+.4f')} (`{dr_p['name']}`)"
            if dr_p is not None
            else "see params"
        )
        comp_rows.append(
            [
                "Manual Artemis fit (re-scored)",
                _fmt(amp_p["value"] if amp_p else None, ".4f"),
                _fmt(e0_p["value"] if e0_p else None, ".2f"),
                dr_cell,
                f"**{_fmt(mf['rfactor_r'], '.4f')}**",
                _fmt(mf["rfactor_k"], ".4f"),
                _fmt(mf.get("chi2_reduced"), ".1f"),
                f"{n_varied} / {_fmt(fa['n_idp'], '.1f')}",
                "—",
            ]
        )

    lines.append("#### Model Comparison")
    lines.append("")
    lines.append(
        _md_table(
            [
                "Model",
                "S₀²",
                "ΔE₀ (eV)",
                "α (ΔR/R) or ΔR (Å)",
                "R-factor (R)",
                "R-factor (k)",
                "χ²_red",
                "nvarys / n_idp",
                "Static trunc.",
            ],
            comp_rows,
        )
    )
    lines.append("")
    if mf is not None:
        meta = mf.get("meta") or {}
        r_logged = meta.get("r_range_logged")
        rf_logged = meta.get("r_factor_logged")
        detail_bits: list[str] = []
        if r_logged and len(r_logged) == 2:
            detail_bits.append(f"over R ∈ [{r_logged[0]:g}, {r_logged[1]:g}] Å")
        if rf_logged is not None:
            detail_bits.append(f"logged R = {rf_logged:.4f}")
        detail = f" ({', '.join(detail_bits)})" if detail_bits else ""
        lines.append(
            f"_The manual model was refined externally in Artemis{detail}. "
            "Here its χ(k) curve is re-scored on the shared experiment using "
            "this workflow's noise weighting in the window above so its "
            "R-factors are directly comparable with the MD models._"
        )
        lines.append("")
        if mf.get("params"):
            prow = [
                [
                    f"`{p['name']}`",
                    _fmt(p["value"], ".4f"),
                    _fmt(p["stderr"], ".4f"),
                    "yes" if p.get("vary") else "fixed",
                ]
                for p in mf["params"]
            ]
            lines.append("<details>")
            lines.append(
                "<summary><strong>Manual Artemis fit parameters</strong></summary>"
            )
            lines.append("")
            lines.append(_md_table(["Parameter", "Value", "Stderr", "Refined"], prow))
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # 2. Figure
    if fig_rel:
        cap = (
            "Comparison of Experiment, MD-ensemble alignment, Static baseline "
            "(MD σ²), Static + MD σ² fit, and ensemble-scaled path attribution "
            f"for R ∈ [{rr[0]:g}, {rr[1]:g}] Å."
        )
        lines.append(f"![{cap}]({fig_rel}){{width=100%}}")
        lines.append("")

    # 3. Peak attribution
    lines.append("#### χ(R) Peak Attribution (MD-Ensemble Scaled)")
    lines.append("")
    if att_p is not None and att_p.get("peaks"):
        pk_rows: list[list[str]] = []
        for pk in att_p["peaks"]:
            top_shares = ", ".join(
                f"`{s['id']}` ({s['projection']:+.2f}, Δ={s['loo_delta']:+.2f})"
                for s in pk["shares"][:4]
            )
            pk_rows.append(
                [
                    _fmt(pk["r_peak"], ".2f"),
                    f"[{pk['window'][0]:.2f}, {pk['window'][1]:.2f}]",
                    _fmt(pk["height"], ".2f"),
                    top_shares or "—",
                    _fmt(pk["unassigned"]["projection"], "+.3f"),
                ]
            )
        lines.append(
            _md_table(
                [
                    "R_peak (Å)",
                    "Window (Å)",
                    "Height",
                    "Top paths (projection, LOO Δ)",
                    "Unassigned",
                ],
                pk_rows,
            )
        )
        lines.append("")
        u_pct = _fmt(100.0 * att_p["unassigned_norm_fraction"], ".2f", "%")
        err_str = f"{att_p['max_projection_sum_error']:.1e}"
        lines.append(
            f"- Stored-paths `unassigned` norm fraction: **{u_pct}** "
            f"(max projection sum error: `{err_str}`)."
        )
        if att_s is not None:
            s_pct = _fmt(100.0 * att_s["unassigned_norm_fraction"], ".4f", "%")
            lines.append(
                f"- Per-site decomposition `unassigned` norm fraction: **{s_pct}**."
            )
        lines.append("")
    else:
        lines.append("_No peaks or stored top paths available in this window._\n")

    # 4. Site grouping & reweighting
    lines.append("#### Site Grouping & Reweighting")
    lines.append("")
    nf = gs.get("noise_floor")
    lines.append(
        f"- **Groups found:** `{gs['n_groups']}` across `{gs['n_sites']}` sites "
        f"(requested `tol={gs.get('tol')}`, sampling-noise floor "
        f"`noise_floor={_fmt(nf, '.3f')}`)."
    )
    rf_md = _fmt(wg.get("rfactor_md"), ".4f")
    rf_fit = _fmt(wg.get("rfactor_fit"), ".4f")
    e0_w = _fmt(wg.get("e0_shift"), ".2f")
    lines.append(
        f"- **Reweighting:** R-factor MD `{rf_md}` → fitted `{rf_fit}` "
        f"(`nvarys={wg.get('nvarys')}`, ΔE₀ held fixed at `{e0_w} eV`)."
    )
    lines.append("")
    grp_rows = [
        [
            g["group_id"],
            str(g["n_sites"]),
            _fmt(g["weight"], ".3f"),
            str(g["medoid_site"]),
            _fmt(g["within_group_rms"], ".3f"),
            _fmt(g["distance_to_overall"], ".3f"),
        ]
        for g in gs.get("groups", [])
    ]
    lines.append(
        _md_table(
            [
                "Group",
                "Sites",
                "MD weight",
                "Medoid",
                "Within-group RMS",
                "Dist to overall",
            ],
            grp_rows,
        )
    )
    lines.append("")

    # 5. Static + MD sigma^2 shell summary + collapsible raw sub-path breakdown
    if fb is not None and fb.get("paths"):
        shells = _rollup_static_paths(fb["paths"])
        lines.append("#### Static + MD σ² Summary by MD Shell")
        lines.append("")
        lines.append(
            "_Tight static FEFF sub-paths (`r_tol = 0.025 Å`) that share the "
            "same MD Debye–Waller shell are aggregated below for readability; "
            "expand the details block for the exact pooled sub-paths passed to "
            "Larch `feffit`._"
        )
        lines.append("")
        shell_rows = []
        for sh in shells:
            if sh["n_subpaths"] > 1:
                r_cell = (
                    f"{sh['r_mean']:.3f} ({sh['r_min']:.2f}–{sh['r_max']:.2f})"
                )
            else:
                r_cell = f"{sh['r_mean']:.3f}"
            shell_rows.append(
                [
                    sh["label"],
                    r_cell,
                    _fmt(sh["n_total"], ".1f"),
                    _fmt(sh["sigma2_md"], ".5f"),
                    _fmt(sh["deltar"], "+.4f"),
                    str(sh["n_subpaths"]),
                ]
            )
        lines.append(
            _md_table(
                [
                    "Shell / Path class",
                    "R_eff (Å)",
                    "N",
                    "σ²_MD (Å²)",
                    "ΔR (Å)",
                    "Sub-paths",
                ],
                shell_rows,
            )
        )
        lines.append("")

        n_kept = len(fb["paths"])
        n_drop = len(fb.get("dropped_paths", []))
        lines.append("<details>")
        lines.append(
            "<summary><strong>Raw pooled static sub-paths "
            f"({n_kept} kept, {n_drop} dropped)</strong></summary>"
        )
        lines.append("")
        path_rows = [
            [
                f"`{p['label']}`",
                str(p["nlegs"]),
                _fmt(p["r_eff"], ".3f"),
                _fmt(p["degeneracy"], ".1f"),
                _fmt(p["sigma2_md"], ".5f"),
                _fmt(p["deltar"], "+.4f"),
                _fmt(p["match_distance"], ".3f"),
            ]
            for p in fb["paths"]
        ]
        lines.append(
            _md_table(
                [
                    "Path",
                    "Legs",
                    "R_eff (Å)",
                    "N",
                    "σ²_MD (Å²)",
                    "ΔR (Å)",
                    "Match |dR| (Å)",
                ],
                path_rows,
            )
        )
        if fb.get("dropped_paths"):
            dropped = ", ".join(f"`{d['label']}`" for d in fb["dropped_paths"])
            lines.append(f"\nDropped paths (no MD σ² match): {dropped}\n")
        lines.append("</details>")
        lines.append("")

    # 6. Warnings callout
    all_warnings: list[tuple[str, str]] = []
    for src, wlist in (
        ("MD-ensemble alignment", fa.get("warnings", [])),
        ("Static + MD σ² fit", (fb or {}).get("warnings", [])),
        ("Site grouping", gs.get("warnings", [])),
        ("Site reweighting", wg.get("warnings", [])),
        ("Path attribution", (att_p or {}).get("warnings", [])),
    ):
        for w in wlist:
            all_warnings.append((src, w))

    if all_warnings:
        lines.append('::: {.callout-note collapse="true"}')
        lines.append(f"## Diagnostics & Warnings ({len(all_warnings)})")
        for src, msg in all_warnings:
            lines.append(f"- **{src}:** {msg}")
        lines.append(":::")
        lines.append("")

    return "\n".join(lines)


def _build_path_clustering_section(pc: dict[str, Any]) -> list[str]:
    """Explain how FEFF path samples were grouped, with the split diagnostic table."""
    lines: list[str] = ["## FEFF Path Grouping", ""]
    lines.append(
        f"Path attribution used **`{pc['mode']}`** grouping "
        f"(requested `{pc['requested']}`). `running_mean` caps each group's "
        "width at about twice the tolerance, so one broad thermal shell can "
        'be cut into several "paths". `valley` keeps a shell whole and splits '
        "only where the sample density dips between shells."
    )
    lines.append("")
    diag = pc.get("diagnostic")
    if diag is None:
        lines.append(
            "_No `results.h5` was available, so the split diagnostic was not run._"
        )
        lines.append("")
    else:
        rows = []
        for c in diag["categories"]:
            flag = "over-split" if c["n_running_mean"] > c["n_valley"] else ""
            rows.append(
                [
                    f"`{c['scatterer']}`",
                    str(c["nlegs"]),
                    _fmt(100.0 * c["share"], ".1f", "%"),
                    f"{c['r_range'][0]:.2f}–{c['r_range'][1]:.2f}",
                    str(c["n_running_mean"]),
                    str(c["n_valley"]),
                    flag,
                ]
            )
        lines.append(
            f"Split diagnostic at tolerance {diag['tol']:g} Å over "
            f"{diag['n_samples']} FEFF path samples "
            f"(categories with ≥ {100.0 * diag['min_share']:.0f}% of samples):"
        )
        lines.append("")
        lines.append(
            _md_table(
                [
                    "Scatterer",
                    "Legs",
                    "Share",
                    "R range (Å)",
                    "Groups (running mean)",
                    "Groups (valley)",
                    "",
                ],
                rows,
            )
        )
        lines.append("")
    for msg in pc.get("warnings", []):
        lines.append(f"- {msg}")
    if pc.get("warnings"):
        lines.append("")
    return lines


def build_qmd(
    summary: dict[str, Any],
    *,
    title: str,
    summary_dir: Path,
    report_dir: Path,
    notes_md: str | None = None,
) -> str:
    """Build the complete Quarto Markdown source."""
    arch = summary["archive"]
    exp = summary["experiment"]
    traj = summary["trajectory"]
    windows = summary["windows"]
    ft = arch.get("fourier_params") or (
        windows[0]["fit_a"]["ft_params"] if windows else {}
    )
    today = dt.date.today().isoformat()

    # Copy figures into report_dir/assets so Quarto embeds cleanly
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig_rels: dict[str, str | None] = {}
    for win in windows:
        src_fig = summary_dir / win["figure"]
        if src_fig.is_file():
            dst_fig = assets_dir / src_fig.name
            if src_fig.resolve() != dst_fig.resolve():
                shutil.copy2(src_fig, dst_fig)
            fig_rels[win["tag"]] = f"assets/{src_fig.name}"
        else:
            fig_rels[win["tag"]] = None

    # Headline metric cards (first window + extended if present)
    cards: list[tuple[str, str, str]] = []
    for win in windows[:2]:
        rr = win["rrange"]
        fa = win["fit_a"]
        fb = win.get("fit_b")
        s02_s = _fmt(fa["params"]["s02"]["value"], ".3f")
        e0_s = _fmt(fa["params"]["e0_shift"]["value"], "+.2f")
        cards.append(
            (
                f"MD ensemble R-factor [{rr[0]:g}–{rr[1]:g} Å]",
                _fmt(fa["rfactor_r"], ".4f"),
                f"S₀²={s02_s}, ΔE₀={e0_s} eV",
            )
        )
        if fb is not None:
            alpha_s = _fmt(fb["params"]["alpha"]["value"], "+.4f")
            trunc_s = _fmt(
                100.0 * (fb.get("truncation_fraction") or 0.0), ".1f", "%"
            )
            cards.append(
                (
                    f"Static fit R-factor [{rr[0]:g}–{rr[1]:g} Å]",
                    _fmt(fb["rfactor_r"], ".4f"),
                    f"α={alpha_s}, trunc={trunc_s}",
                )
            )
    n_sites = arch.get("n_sites", "—")
    n_frames = arch.get("n_frames", "—")
    rad = arch.get("radius", "—")
    cards.append(
        (
            "Ensemble Sampling",
            f"{arch.get('n_tasks', '—')} tasks",
            f"{n_sites} sites × {n_frames} frames · r={rad} Å",
        )
    )

    card_html = ['<div class="metric-grid">']
    for lbl, val, sub in cards:
        card_html.append(
            f'  <div class="metric-card"><div class="label">{lbl}</div>'
            f'<div class="value">{val}</div><div class="sub">{sub}</div></div>'
        )
    card_html.append("</div>")

    parts: list[str] = [
        "---",
        f'title: "{title}"',
        f'date: "{today}"',
        "format:",
        "  html:",
        "    theme: cosmo",
        "    toc: true",
        "    toc-depth: 3",
        "    embed-resources: true",
        "    smooth-scroll: true",
        "    page-layout: full",
        "---",
        "",
        f"<style>{CUSTOM_CSS}</style>",
        "",
        "## Overview",
        "",
        "\n".join(card_html),
        "",
    ]

    if notes_md:
        parts.extend(["## Physical Interpretation", "", notes_md.strip(), ""])

    pc = summary.get("path_clustering")
    if pc is not None:
        parts.extend(_build_path_clustering_section(pc))

    parts.append("## Comparison by Fit Window")
    parts.append("")
    if len(windows) > 1:
        parts.append("::: {.panel-tabset}")
        parts.append("")
    for win in windows:
        parts.append(_build_window_section(win, fig_rels[win["tag"]]))
    if len(windows) > 1:
        parts.append(":::")
        parts.append("")

    # Provenance & settings
    ft_str = ", ".join(f"`{k}={v}`" for k, v in sorted(ft.items()))
    eps_r = windows[0]["fit_a"].get("epsilon_r") if windows else None
    eps_k = windows[0]["fit_a"].get("epsilon_k") if windows else None

    arch_detail = (
        f"`{arch['path']}` (format v{arch.get('format_version')}, "
        f"{arch.get('n_top_paths')} top paths)"
    )
    exp_detail = (
        f"`{exp['path']}` ({exp['n_points']} pts, "
        f"k = {exp['k_min']:.2f}–{exp['k_max']:.2f} Å⁻¹)"
    )
    traj_detail = (
        f"`{traj['path']}` (slice `{traj['frame_index']}`, "
        f"{traj['n_frames_loaded']} frames, absorber `{traj['absorber']}`)"
        if traj.get("path")
        else "Not run"
    )
    pc_detail = (
        f"`{pc['mode']}` (requested `{pc['requested']}`), "
        f"archive `{pc['archive_used']}`"
        if pc
        else "—"
    )
    prov_rows = [
        ["Ensemble archive", arch_detail],
        ["Experimental data", exp_detail],
        ["Trajectory (static fit)", traj_detail],
        ["Fourier transform", ft_str or "Library defaults"],
        ["Noise estimate ε_R / ε_k", f"ε_R={eps_r}, ε_k={eps_k}" if eps_r else "—"],
        ["Path clustering", pc_detail],
        [
            "Manual fit",
            f"`{summary['manual_fit']['source']}`"
            if summary.get("manual_fit")
            else "Not run",
        ],
        ["Analysis runtime", f"{summary.get('elapsed_s', '—')} s"],
    ]
    parts.extend(
        [
            "## Provenance & Reproducibility",
            "",
            _md_table(["Item", "Details"], prov_rows),
            "",
            "```bash",
            summary.get("command", ""),
            "```",
            "",
        ]
    )

    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for compiling analysis_summary.json into Quarto HTML."""
    parser = argparse.ArgumentParser(
        description=(
            "Compile analysis_summary.json into a self-contained Quarto HTML report."
        )
    )
    parser.add_argument("summary_json", type=Path, help="Path to analysis_summary.json")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Report output dir"
    )
    parser.add_argument("--title", default=None, help="Report title")
    parser.add_argument("--slug", default=None, help="Output filename stem")
    parser.add_argument(
        "--notes", type=Path, default=None, help="Optional Markdown findings note"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Write .qmd only, skip quarto render"
    )
    args = parser.parse_args(argv)

    summary_path = args.summary_json.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_dir = summary_path.parent
    report_dir = (args.output_dir or (summary_dir / "report")).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    absorber = (summary.get("trajectory") or {}).get("absorber") or "EXAFS"
    title = args.title or f"{absorber} MD-EXAFS Experimental Comparison"
    slug = args.slug or _slugify(title)

    notes_md = (
        args.notes.read_text(encoding="utf-8")
        if args.notes and args.notes.is_file()
        else None
    )
    qmd_text = build_qmd(
        summary,
        title=title,
        summary_dir=summary_dir,
        report_dir=report_dir,
        notes_md=notes_md,
    )
    qmd_path = report_dir / f"{slug}.qmd"
    qmd_path.write_text(qmd_text, encoding="utf-8")
    print(f"Wrote {qmd_path}")

    if args.no_render:
        return 0

    quarto_bin = shutil.which("quarto")
    if quarto_bin is None:
        print(
            "Warning: 'quarto' CLI not found on PATH; kept .qmd file only. "
            "Install Quarto (e.g. 'brew install quarto') and run: quarto render "
            f"{qmd_path}",
            file=sys.stderr,
        )
        return 0

    proc = subprocess.run(  # noqa: S603
        [quarto_bin, "render", qmd_path.name],
        cwd=report_dir,
        capture_output=True,
        text=True,
        timeout=QUARTO_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    html_path = report_dir / f"{slug}.html"
    print(f"Rendered {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
