#!/usr/bin/env python3
"""Run the full md-exafs experimental comparison, fitting, and attribution suite.

Produces:
  - analysis_summary.json  (structured, JSON-safe results for reporting)
  - comparison_<tag>.png   (2x2 publication figure per R-range)
  - chi_k_*.vl.json / chi_r_*.vl.json (Vega-Lite specs)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ase.io import read as ase_read  # noqa: E402
from manual_fit import evaluate_manual_fit  # noqa: E402
from path_splits import (  # noqa: E402
    choose_cluster_mode,
    diagnose_path_splits,
    format_diagnosis,
)

from md_exafs import (  # noqa: E402
    ArchiveReader,
    FeffConfig,
    archive_series,
    fit_ensemble,
    fit_site_group_weights,
    fit_static,
    group_sites,
    peak_attribution,
    series_to_vega_lite,
)
from md_exafs.execution import export_ensemble  # noqa: E402
from md_exafs.experimental import read_experimental_spectrum  # noqa: E402


def _parse_rrange(spec: str) -> tuple[float, float]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"rrange must be 'rmin,rmax' (e.g. 1.0,3.0), got {spec!r}"
        )
    rmin, rmax = float(parts[0]), float(parts[1])
    if not (0.0 <= rmin < rmax):
        raise argparse.ArgumentTypeError(
            f"Invalid rrange {spec!r}: need 0 <= rmin < rmax"
        )
    return (rmin, rmax)


def _parse_sites(spec: str | None) -> list[int] | str | None:
    if spec is None or not spec.strip():
        return None
    s = spec.strip()
    if all(tok.strip().isdigit() for tok in s.split(",")):
        return [int(tok.strip()) for tok in s.split(",")]
    return s


def _rrange_tag(rrange: tuple[float, float]) -> str:
    return f"R{rrange[0]:g}_{rrange[1]:g}".replace(".", "p")


def _strip_curves(d: dict[str, Any], drop_keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in drop_keys}


def _plot_comparison(
    *,
    out_png: Path,
    rrange: tuple[float, float],
    kweight: int,
    fit_a: dict[str, Any],
    fit_b: dict[str, Any] | None,
    att_paths: dict[str, Any] | None,
    top_n_paths: int = 5,
    manual: dict[str, Any] | None = None,
) -> None:
    """Render a 2x2 publication summary figure for one R-range."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), dpi=220)
    ax_k, ax_r = axes[0]
    ax_att, ax_res = axes[1]

    exp_a = fit_a["experiment"]
    th_a = fit_a["theory"]
    k_a = np.asarray(exp_a["k"], dtype=float)
    r_a = np.asarray(exp_a["r"], dtype=float)
    exp_kw = np.asarray(exp_a["chi_weighted"], dtype=float)
    fa_kw = np.asarray(th_a["chi_weighted"], dtype=float)
    exp_rmag = np.asarray(exp_a["chir_mag"], dtype=float)
    fa_rmag = np.asarray(th_a["chir_mag"], dtype=float)

    # Panel (a): k^w chi(k)
    ax_k.plot(
        k_a, exp_kw, color="#111111", ls="--", lw=1.6, label="Experiment", zorder=5
    )
    ax_k.plot(
        k_a,
        fa_kw,
        color="#1f77b4",
        lw=1.6,
        label=f"MD ensemble (R={fit_a['rfactor']:.4f})",
        zorder=4,
    )

    # Panel (b): |chi(R)|
    ax_r.axvspan(
        rrange[0], rrange[1], color="#e8f1fa", alpha=0.7, zorder=0, label="Fit window"
    )
    rmax_valid = fit_a.get("rmax_valid")
    if rmax_valid is not None:
        ax_r.axvline(
            rmax_valid,
            color="#666666",
            ls=":",
            lw=1.1,
            label=f"rmax_valid ({rmax_valid:g} Å)",
        )
    ax_r.plot(
        r_a, exp_rmag, color="#111111", ls="--", lw=1.6, label="Experiment", zorder=5
    )
    ax_r.plot(
        r_a,
        fa_rmag,
        color="#1f77b4",
        lw=1.6,
        label=f"MD ensemble (R={fit_a['rfactor']:.4f})",
        zorder=4,
    )

    # Residuals in R-space: |chi_exp(R) - chi_model(R)| using complex Re/Im
    exp_re = np.asarray(exp_a["chir_re"], dtype=float)
    exp_im = np.asarray(exp_a["chir_im"], dtype=float)
    fa_re = np.asarray(th_a["chir_re"], dtype=float)
    fa_im = np.asarray(th_a["chir_im"], dtype=float)
    res_a = np.hypot(exp_re - fa_re, exp_im - fa_im)
    ax_res.axvspan(rrange[0], rrange[1], color="#e8f1fa", alpha=0.7, zorder=0)
    ax_res.plot(r_a, res_a, color="#1f77b4", lw=1.5, label="MD ensemble |Δχ(R)|")

    if fit_b is not None:
        th_b = fit_b["theory"]
        k_b = np.asarray(th_b["k"], dtype=float)
        r_b = np.asarray(th_b["r"], dtype=float)
        exp_b = fit_b["experiment"]
        ax_k.plot(
            k_b,
            np.asarray(th_b["chi_weighted"], dtype=float),
            color="#d95f02",
            lw=1.4,
            label=f"Static fit (MD σ²) (R={fit_b['rfactor']:.4f})",
            zorder=3,
        )
        ax_r.plot(
            r_b,
            np.asarray(th_b["chir_mag"], dtype=float),
            color="#d95f02",
            lw=1.4,
            label=f"Static fit (MD σ²) (R={fit_b['rfactor']:.4f})",
            zorder=3,
        )
        eb_re = np.asarray(exp_b["chir_re"], dtype=float)
        eb_im = np.asarray(exp_b["chir_im"], dtype=float)
        tb_re = np.asarray(th_b["chir_re"], dtype=float)
        tb_im = np.asarray(th_b["chir_im"], dtype=float)
        ax_res.plot(
            r_b,
            np.hypot(eb_re - tb_re, eb_im - tb_im),
            color="#d95f02",
            lw=1.4,
            label="Static fit (MD σ²) |Δχ(R)|",
        )

        b0 = (fit_b.get("comparison") or {}).get("b0")
        if b0 and "theory" in b0:
            th_b0 = b0["theory"]
            k_b0 = np.asarray(th_b0["k"], dtype=float)
            r_b0 = np.asarray(th_b0["r"], dtype=float)
            ax_k.plot(
                k_b0,
                np.asarray(th_b0["chi_weighted"], dtype=float),
                color="#7570b3",
                ls="-.",
                lw=1.2,
                alpha=0.85,
                label=f"Static baseline (MD σ²) (R={b0['rfactor']:.4f})",
                zorder=2,
            )
            ax_r.plot(
                r_b0,
                np.asarray(th_b0["chir_mag"], dtype=float),
                color="#7570b3",
                ls="-.",
                lw=1.2,
                alpha=0.85,
                label=f"Static baseline (MD σ²) (R={b0['rfactor']:.4f})",
                zorder=2,
            )
            b0_re = np.asarray(th_b0["chir_re"], dtype=float)
            b0_im = np.asarray(th_b0["chir_im"], dtype=float)
            ax_res.plot(
                r_b0,
                np.hypot(eb_re - b0_re, eb_im - b0_im),
                color="#7570b3",
                ls="-.",
                lw=1.2,
                alpha=0.85,
                label="Static baseline (MD σ²) |Δχ(R)|",
            )

    if manual is not None:
        th_m = manual["theory"]
        m_label = f"Manual Artemis fit (R={manual['rfactor_r']:.4f})"
        ax_k.plot(
            np.asarray(th_m["k"], dtype=float),
            np.asarray(th_m["chi_weighted"], dtype=float),
            color="#e7298a",
            ls=(0, (5, 2)),
            lw=1.3,
            label=m_label,
            zorder=3,
        )
        ax_r.plot(
            np.asarray(th_m["r"], dtype=float),
            np.asarray(th_m["chir_mag"], dtype=float),
            color="#e7298a",
            ls=(0, (5, 2)),
            lw=1.3,
            label=m_label,
            zorder=3,
        )
        em_re = np.asarray(manual["experiment"]["chir_re"], dtype=float)
        em_im = np.asarray(manual["experiment"]["chir_im"], dtype=float)
        tm_re = np.asarray(th_m["chir_re"], dtype=float)
        tm_im = np.asarray(th_m["chir_im"], dtype=float)
        ax_res.plot(
            np.asarray(th_m["r"], dtype=float),
            np.hypot(em_re - tm_re, em_im - tm_im),
            color="#e7298a",
            ls=(0, (5, 2)),
            lw=1.3,
            label="Manual Artemis fit |Δχ(R)|",
        )

    # Panel (c): Peak attribution (ensemble-aligned), including prominent shoulders
    ax_att.axvspan(rrange[0], rrange[1], color="#e8f1fa", alpha=0.7, zorder=0)
    if att_paths is not None:
        r_p = np.asarray(att_paths["r"], dtype=float)
        tot_mag = np.asarray(att_paths["total"]["chir_mag"], dtype=float)
        in_win = (r_p >= rrange[0]) & (r_p <= rrange[1])
        band_max = float(tot_mag[in_win].max()) if in_win.any() else 0.0
        ax_att.plot(
            r_p,
            tot_mag,
            color="#1f77b4",
            lw=1.8,
            label="MD ensemble total |χ(R)|",
            zorder=5,
        )
        # Collect dominant signed projection across detected peaks
        dom_proj: dict[str, float] = {}
        for pk in att_paths.get("peaks", []):
            for sh in pk.get("shares", []):
                cid = sh["id"]
                val = float(sh["projection"])
                if cid not in dom_proj or abs(val) > abs(dom_proj[cid]):
                    dom_proj[cid] = val
        comp_by_id = {c["id"]: c for c in att_paths.get("components", [])}
        # Score each component by max(|proj|, in-window peak height / band_max)
        # so strong unresolved shoulders (like 2nd-shell K in KCl) are also shown.
        comp_score: dict[str, float] = {}
        comp_peak_rel: dict[str, float] = {}
        for cid, comp in comp_by_id.items():
            cmag = np.asarray(comp["chir_mag"], dtype=float)
            rel_h = (
                float(cmag[in_win].max()) / band_max
                if band_max > 0.0 and in_win.any()
                else 0.0
            )
            comp_peak_rel[cid] = rel_h
            comp_score[cid] = max(abs(dom_proj.get(cid, 0.0)), rel_h)
        ranked_ids = sorted(comp_score, key=lambda cid: -comp_score[cid])[:top_n_paths]
        cmap = plt.get_cmap("tab10")
        for idx, cid in enumerate(ranked_ids):
            comp = comp_by_id.get(cid)
            if comp is None:
                continue
            proj = dom_proj.get(cid, 0.0)
            has_peak_proj = cid in dom_proj and abs(proj) >= 0.05
            if has_peak_proj:
                ls = "--" if proj < 0 else "-"
                lbl = f"{cid} ({proj:+.2f})"
            elif comp_peak_rel.get(cid, 0.0) >= 0.15:
                ls = "-"
                lbl = f"{cid} (shoulder)"
            else:
                ls = "-"
                lbl = f"{cid} ({proj:+.2f})" if cid in dom_proj else cid
            ax_att.plot(
                r_p,
                np.asarray(comp["chir_mag"], dtype=float),
                color=cmap((idx + 2) % 10),
                ls=ls,
                lw=1.25,
                label=lbl,
            )
        ax_att.plot(
            r_p,
            np.asarray(att_paths["unassigned"]["chir_mag"], dtype=float),
            color="#777777",
            ls=":",
            lw=1.2,
            label=f"unassigned ({att_paths['unassigned']['norm_fraction']:.1%})",
        )
        for pk in att_paths.get("peaks", []):
            ax_att.axvline(pk["r_peak"], color="#aaaaaa", ls=":", lw=0.8, zorder=1)
    else:
        ax_att.text(
            0.5,
            0.5,
            "Path attribution unavailable\n(requires format-4 archive with /top_paths)",
            ha="center",
            va="center",
            transform=ax_att.transAxes,
            color="#666666",
        )

    r_plot_max = min(float(r_a[-1]), max(rrange[1] + 1.2, 4.5))
    ax_k.set_xlabel("k (Å⁻¹)")
    ax_k.set_ylabel(f"k^{kweight} χ(k) (Å⁻{kweight})")
    ax_k.set_title(f"(a) k-space comparison · R ∈ [{rrange[0]:g}, {rrange[1]:g}] Å")
    ax_k.set_xlim(0.0, min(20.5, float(np.nanmax(k_a)) + 0.5))
    ax_k.legend(fontsize=8, loc="upper right", frameon=True)
    ax_k.grid(alpha=0.25)

    ax_r.set_xlabel("R (Å)")
    ax_r.set_ylabel(f"|χ(R)| (Å⁻{kweight + 1})")
    ax_r.set_title(f"(b) R-space magnitude · R ∈ [{rrange[0]:g}, {rrange[1]:g}] Å")
    ax_r.set_xlim(0.0, r_plot_max)
    ax_r.legend(fontsize=8, loc="upper right", frameon=True)
    ax_r.grid(alpha=0.25)

    ax_att.set_xlabel("R (Å)")
    ax_att.set_ylabel(f"|χ(R)| (Å⁻{kweight + 1})")
    ax_att.set_title("(c) MD-ensemble path attribution (solid: +, dashed: −)")
    ax_att.set_xlim(0.0, r_plot_max)
    ax_att.legend(fontsize=7.5, loc="upper right", frameon=True)
    ax_att.grid(alpha=0.25)

    ax_res.set_xlabel("R (Å)")
    ax_res.set_ylabel(f"|χ_exp(R) − χ_model(R)| (Å⁻{kweight + 1})")
    ax_res.set_title("(d) Complex R-space residuals")
    ax_res.set_xlim(0.0, r_plot_max)
    ax_res.legend(fontsize=8, loc="upper right", frameon=True)
    ax_res.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the MD-EXAFS experimental comparison suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Run md-exafs experimental comparison, alignment, static fit, "
            "attribution, and plots."
        )
    )
    parser.add_argument(
        "--archive", required=True, type=Path, help="Format-4 ensemble_results.h5"
    )
    parser.add_argument(
        "--experiment", required=True, type=Path, help="Experimental spectrum file"
    )
    parser.add_argument(
        "--experiment-kwargs",
        default=None,
        help='Optional JSON reader kwargs, e.g. \'{"labels": "chi,k"}\'',
    )
    parser.add_argument(
        "--trajectory", default=None, type=Path, help="MD trajectory for Fit B"
    )
    parser.add_argument(
        "--frame-index", default=":", help="ASE read index slice (default ':')"
    )
    parser.add_argument(
        "--config", default=None, type=Path, help="exafs_config.yaml for Fit B"
    )
    parser.add_argument(
        "--absorber", default=None, help="Absorber element symbol for Fit B"
    )
    parser.add_argument("--sites", default=None, help="Site indices or spec for Fit B")
    parser.add_argument(
        "--structure", default=None, help="Optional static structure file for Fit B"
    )
    parser.add_argument(
        "--rrange",
        action="append",
        type=_parse_rrange,
        help="R fit window 'rmin,rmax' (repeatable; default 1.0,3.0)",
    )
    parser.add_argument(
        "--kweights", default="1,2,3", help="Comma-separated kweights (default 1,2,3)"
    )
    parser.add_argument(
        "--fit-space", default="r", choices=("r", "k"), help="Fit space (default r)"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.15,
        help="group_sites relative tol (default 0.15)",
    )
    parser.add_argument(
        "--top-paths-plot", type=int, default=5, help="Top paths overlaid in panel (c)"
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("exp_comparison"), help="Output directory"
    )
    parser.add_argument(
        "--workdir", type=Path, default=None, help="Reusable FEFF workdir for Fit B"
    )
    parser.add_argument(
        "--cluster-mode",
        default="auto",
        choices=("auto", "running_mean", "valley"),
        help=(
            "FEFF path grouping for attribution and MD sigma^2. 'auto' runs "
            "the split diagnostic on results.h5 and switches to 'valley' "
            "when shells are over-split."
        ),
    )
    parser.add_argument(
        "--results-h5",
        type=Path,
        default=None,
        help="Pipeline results.h5 with path_results (default: next to --archive)",
    )
    parser.add_argument(
        "--manual-fit",
        type=Path,
        default=None,
        help="Artemis .fit export (k, chi, fit columns) of a manual fit",
    )
    parser.add_argument(
        "--manual-fit-params",
        type=Path,
        default=None,
        help="Artemis fit.yaml with the refined manual-fit parameters",
    )
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    exp_source: Any = str(args.experiment.resolve())
    exp_kwargs: dict[str, Any] = {}
    if args.experiment_kwargs:
        exp_kwargs = json.loads(args.experiment_kwargs)
        exp_source = {"path": str(args.experiment.resolve()), **exp_kwargs}

    k_exp, chi_exp, exp_meta = read_experimental_spectrum(exp_source)
    rranges = args.rrange or [(1.0, 3.0)]
    kweights = tuple(int(x.strip()) for x in args.kweights.split(",") if x.strip())

    # Path grouping: diagnose over-split shells and pick the clustering mode.
    results_h5 = (args.results_h5 or args.archive.parent / "results.h5").resolve()
    diagnosis: dict[str, Any] | None = None
    path_warnings: list[str] = []
    if results_h5.exists():
        diagnosis = diagnose_path_splits(results_h5)
        print(format_diagnosis(diagnosis), flush=True)
    if args.cluster_mode == "auto":
        if diagnosis is None:
            cluster_mode = "running_mean"
            path_warnings.append(
                f"No results.h5 at {results_h5}; cluster_mode auto fell back "
                "to running_mean."
            )
        else:
            cluster_mode = choose_cluster_mode(diagnosis)
    else:
        cluster_mode = args.cluster_mode
    if (
        diagnosis is not None
        and cluster_mode != "running_mean"
        and diagnosis["over_split"]
    ):
        names = ", ".join(
            f"{c['scatterer']}_{c['nlegs']}" for c in diagnosis["over_split"]
        )
        path_warnings.append(
            f"Over-split path populations under running_mean ({names}); using "
            f"cluster_mode={cluster_mode}, which splits only at density valleys."
        )
    print(f"Path clustering mode: {cluster_mode}", flush=True)

    archive_path = args.archive.resolve()
    if cluster_mode != "running_mean":
        if results_h5.exists():
            archive_path = (outdir / f"ensemble_{cluster_mode}.h5").resolve()
            export_ensemble(results_h5, archive_path, cluster_mode=cluster_mode)
        else:
            path_warnings.append(
                f"Cannot re-export with cluster_mode={cluster_mode}: "
                f"{results_h5} not found; reusing {archive_path}."
            )
    reader = ArchiveReader(archive_path)
    archive_info = {
        "path": str(archive_path),
        "format_version": reader.format_version,
        "radius": reader.radius,
        "n_tasks": reader.n_tasks,
        "n_sites": len(reader.site_indices),
        "n_frames": len(reader.frame_indices),
        "n_top_paths": len(reader.get_top_paths()),
        "fourier_params": reader.fourier_params,
    }
    k_overall = np.asarray(reader.get_overall_average()["k"], dtype=float)
    manual_params = (
        str(args.manual_fit_params.resolve())
        if args.manual_fit_params is not None
        else None
    )

    # Raw overlay Vega-Lite specs
    series_raw = archive_series(archive_path, experiment=exp_source)
    for view in ("chi_k", "chi_r"):
        spec = series_to_vega_lite(series_raw, view=view)
        (outdir / f"{view}_raw.vl.json").write_text(json.dumps(spec), encoding="utf-8")

    frames = None
    feff_cfg = None
    if args.trajectory is not None:
        if not args.absorber:
            parser.error(
                "--absorber is required when --trajectory is provided for Fit B"
            )
        frames = ase_read(str(args.trajectory.resolve()), index=args.frame_index)
        if not isinstance(frames, list):
            frames = [frames]
        feff_cfg = (
            FeffConfig.from_yaml(args.config)
            if args.config is not None
            else FeffConfig(radius=reader.radius or 5.0)
        )
    sites_arg = _parse_sites(args.sites)
    workdir = (
        args.workdir.resolve()
        if args.workdir is not None
        else outdir / "fit_static_work"
    )

    window_results: list[dict[str, Any]] = []
    for rrange in rranges:
        tag = _rrange_tag(rrange)
        print(
            f"=== Running analysis for R in [{rrange[0]:g}, {rrange[1]:g}] Å "
            f"({tag}) ===",
            flush=True,
        )

        fit_a = fit_ensemble(
            archive_path,
            exp_source,
            rrange=rrange,
            kweights=kweights,
            fit_space=args.fit_space,
        )
        s02_a = float(fit_a["params"]["s02"]["value"])
        e0_a = float(fit_a["params"]["e0_shift"]["value"])

        # Fit-A adjusted Vega-Lite specs
        series_a = archive_series(
            archive_path, experiment=exp_source, s02=s02_a, e0_shift=e0_a
        )
        vl_files: dict[str, str] = {}
        for view in ("chi_k", "chi_r"):
            fname = f"{view}_fit_a_{tag}.vl.json"
            (outdir / fname).write_text(
                json.dumps(series_to_vega_lite(series_a, view=view)), encoding="utf-8"
            )
            vl_files[view] = fname

        # Site grouping (noise-calibrated) & reweighting
        groups = group_sites(archive_path, rrange=rrange, tol=args.tol)
        weights = fit_site_group_weights(
            archive_path, exp_source, groups, rrange=rrange, kweights=kweights
        )

        # Peak attribution (scaled to Fit A)
        att_by_paths: dict[str, Any] | None = None
        if (reader.format_version or 0) >= 4 and reader.get_top_paths():
            att_by_paths = peak_attribution(
                archive_path,
                by="paths",
                rrange=rrange,
                s02=s02_a,
                e0_shift=e0_a,
            )
        att_by_sites = peak_attribution(
            archive_path, by="sites", rrange=rrange, s02=s02_a, e0_shift=e0_a
        )
        att_by_groups = peak_attribution(
            archive_path,
            by="site_groups",
            groups=groups,
            rrange=rrange,
            s02=s02_a,
            e0_shift=e0_a,
        )

        # Fit B (static structure + MD sigma^2) with side-by-side Fit A & B0 comparison
        fit_b: dict[str, Any] | None = None
        fit_b_elapsed: float | None = None
        if frames is not None and feff_cfg is not None:
            t_b = time.perf_counter()
            fit_b = fit_static(
                frames,
                exp_source,
                feff_cfg,
                args.absorber,
                sites=sites_arg,
                structure=args.structure,
                rrange=rrange,
                kweights=kweights,
                fit_space=args.fit_space,
                archive_path=archive_path,
                workdir=workdir,
                cluster_mode=cluster_mode,
            )
            fit_b_elapsed = time.perf_counter() - t_b

        manual: dict[str, Any] | None = None
        if args.manual_fit is not None:
            manual = evaluate_manual_fit(
                args.manual_fit.resolve(),
                ft_params=fit_a["ft_params"],
                rrange=rrange,
                kweights=kweights,
                fit_space=args.fit_space,
                series_k=k_overall,
                experiment=exp_source,
                params_file=(
                    args.manual_fit_params.resolve() if args.manual_fit_params else None
                ),
            )

        fig_name = f"comparison_{tag}.png"
        _plot_comparison(
            out_png=outdir / fig_name,
            rrange=rrange,
            kweight=int(fit_a["ft_params"]["kweight"]),
            fit_a=fit_a,
            fit_b=fit_b,
            att_paths=att_by_paths,
            top_n_paths=args.top_paths_plot,
            manual=manual,
        )

        def _summarize_att(att: dict[str, Any] | None) -> dict[str, Any] | None:
            if att is None:
                return None
            max_sum_err = 0.0
            for pk in att["peaks"]:
                tot = (
                    sum(s["projection"] for s in pk["shares"])
                    + pk["unassigned"]["projection"]
                    + pk.get("remainder", {"projection": 0.0})["projection"]
                )
                max_sum_err = max(max_sum_err, abs(tot - 1.0))
            return {
                "decomposition": att["decomposition"],
                "adjustment": att["adjustment"],
                "unassigned_norm_fraction": att["unassigned"]["norm_fraction"],
                "max_projection_sum_error": max_sum_err,
                "peaks": att["peaks"],
                "warnings": att["warnings"],
            }

        fit_b_summary: dict[str, Any] | None = None
        if fit_b is not None:
            cmp_clean = {}
            for k_cmp, v_cmp in (fit_b.get("comparison") or {}).items():
                cmp_clean[k_cmp] = _strip_curves(v_cmp, ("theory", "experiment"))
            fit_b_summary = {
                **_strip_curves(fit_b, ("experiment", "theory", "comparison")),
                "comparison": cmp_clean,
                "elapsed_s": round(fit_b_elapsed or 0.0, 2),
            }

        groups_summary = {
            **{k: v for k, v in groups.items() if k != "groups"},
            "groups": [
                {k: v for k, v in g.items() if k != "spectrum"}
                for g in groups["groups"]
            ],
        }

        window_results.append(
            {
                "tag": tag,
                "rrange": list(rrange),
                "figure": fig_name,
                "vega_lite": vl_files,
                "fit_a": _strip_curves(fit_a, ("experiment", "theory")),
                "fit_b": fit_b_summary,
                "manual_fit": (
                    None
                    if manual is None
                    else _strip_curves(manual, ("experiment", "theory"))
                ),
                "group_sites": groups_summary,
                "site_group_weights": _strip_curves(
                    weights, ("experiment", "theory_md", "theory_fit")
                ),
                "peak_attribution": {
                    "paths": _summarize_att(att_by_paths),
                    "sites": _summarize_att(att_by_sites),
                    "site_groups": _summarize_att(att_by_groups),
                },
            }
        )

    summary = {
        "schema_version": 1,
        "command": " ".join(sys.argv),
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "archive": archive_info,
        "experiment": {
            "path": str(args.experiment.resolve()),
            "kwargs": exp_kwargs,
            "n_points": int(len(k_exp)),
            "k_min": float(np.nanmin(k_exp)),
            "k_max": float(np.nanmax(k_exp)),
            "meta": {
                k: v
                for k, v in exp_meta.items()
                if isinstance(v, str | int | float | bool | type(None))
            },
        },
        "trajectory": {
            "path": str(args.trajectory.resolve()) if args.trajectory else None,
            "frame_index": args.frame_index if args.trajectory else None,
            "n_frames_loaded": len(frames) if frames is not None else 0,
            "absorber": args.absorber,
            "sites": sites_arg,
            "structure_override": args.structure,
            "config_path": str(args.config.resolve()) if args.config else None,
        },
        "path_clustering": {
            "requested": args.cluster_mode,
            "mode": cluster_mode,
            "archive_used": str(archive_path),
            "results_h5": str(results_h5) if results_h5.exists() else None,
            "diagnostic": diagnosis,
            "warnings": path_warnings,
        },
        "manual_fit": (
            None
            if args.manual_fit is None
            else {
                "source": str(args.manual_fit.resolve()),
                "params_source": manual_params,
            }
        ),
        "windows": window_results,
    }

    out_json = outdir / "analysis_summary.json"
    out_json.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Wrote {out_json} ({summary['elapsed_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
