#!/usr/bin/env python3
"""Detect over-split FEFF path populations and choose a path-clustering mode.

The width-capped ``running_mean`` clustering (the historical default) splits one
broad thermal shell into several "paths" whenever the shell is wider than about
twice the tolerance. This module compares it with ``valley`` clustering, which
splits only at density valleys between shells. Categories where ``valley``
gives fewer populations are over-split by ``running_mean``.

Reads per-sample ``path_results`` from a pipeline ``results.h5``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np

from md_exafs.debye_waller import canonical_scatterer_key
from md_exafs.paths import cluster_path_category

DEFAULT_TOL = 0.15
DEFAULT_MIN_SHARE = 0.01


def load_path_samples(results_h5: Path | str) -> dict[str, np.ndarray]:
    """Read ``r_eff``, ``nlegs``, ``scatterer``, and ``angle`` from ``path_results``."""
    with h5py.File(results_h5, "r") as f:
        if "path_results" not in f:
            raise ValueError(f"{results_h5} has no path_results group")
        pr = f["path_results"]
        raw_scat = pr["scatterer"][:]
        scatterer = np.array(
            [
                "-".join(
                    canonical_scatterer_key(
                        s.decode() if isinstance(s, bytes) else str(s)
                    )
                )
                for s in raw_scat
            ],
            dtype=object,
        )
        return {
            "r_eff": np.asarray(pr["r_eff"][:], dtype=float),
            "nlegs": np.asarray(pr["nlegs"][:], dtype=int),
            "scatterer": scatterer,
            "angle": np.asarray(pr["angle"][:], dtype=float),
        }


def _items(samples: dict[str, np.ndarray], mask: np.ndarray) -> list[dict[str, float]]:
    return [
        {"r_eff": float(r), "angle": float(a)}
        for r, a in zip(samples["r_eff"][mask], samples["angle"][mask], strict=True)
    ]


def diagnose_path_splits(
    results_h5: Path | str,
    *,
    tol: float = DEFAULT_TOL,
    min_share: float = DEFAULT_MIN_SHARE,
) -> dict[str, Any]:
    """Compare running-mean and valley clustering for every significant path category.

    Args:
        results_h5: Pipeline ``results.h5`` with ``path_results``.
        tol: Distance tolerance (Å): the running-mean width cap, and the
            valley-smoothing width.
        min_share: Ignore categories holding less than this fraction of samples.

    Returns:
        Dict with ``tol``, ``categories`` (one entry per significant category:
        ``scatterer``, ``nlegs``, ``n_samples``, ``share``, ``r_range``,
        ``n_running_mean``, ``n_valley`` and ``fragments``), and ``over_split``.
    """
    samples = load_path_samples(results_h5)
    total = len(samples["r_eff"])
    pairs = sorted(set(zip(samples["scatterer"], samples["nlegs"], strict=True)))
    categories: list[dict[str, Any]] = []
    for scatterer, nlegs in pairs:
        mask = (samples["scatterer"] == scatterer) & (samples["nlegs"] == nlegs)
        n = int(mask.sum())
        share = n / total if total else 0.0
        if share < min_share:
            continue
        items = _items(samples, mask)
        running = cluster_path_category(
            nlegs, items, r_bin=tol, cluster_mode="running_mean"
        )
        valley = cluster_path_category(nlegs, items, r_bin=tol, cluster_mode="valley")
        r_vals = samples["r_eff"][mask]
        fragments = [
            {
                "r_min": round(float(min(it["r_eff"] for it in c)), 3),
                "r_max": round(float(max(it["r_eff"] for it in c)), 3),
                "n": len(c),
            }
            for c in running
        ]
        categories.append(
            {
                "scatterer": str(scatterer),
                "nlegs": int(nlegs),
                "n_samples": n,
                "share": round(share, 4),
                "r_range": [
                    round(float(r_vals.min()), 3),
                    round(float(r_vals.max()), 3),
                ],
                "n_running_mean": len(running),
                "n_valley": len(valley),
                "fragments": fragments,
            }
        )
    over = [c for c in categories if c["n_running_mean"] > c["n_valley"]]
    return {
        "tol": float(tol),
        "min_share": float(min_share),
        "n_samples": total,
        "categories": categories,
        "over_split": over,
    }


def choose_cluster_mode(diagnosis: dict[str, Any]) -> str:
    """Return ``'valley'`` when any significant category is over-split."""
    return "valley" if diagnosis["over_split"] else "running_mean"


def format_diagnosis(diagnosis: dict[str, Any]) -> str:
    """Format a plain-text summary table of path categories."""
    header = (
        f"Path-split diagnostic (tol={diagnosis['tol']:g} Å, "
        f"{diagnosis['n_samples']} samples):"
    )
    cols = (
        f"  {'category':<14}{'nlegs':>6}{'share':>8}"
        f"{'r_range':>16}{'running':>9}{'valley':>8}"
    )
    lines = [header, cols]
    for c in diagnosis["categories"]:
        flag = "  <- over-split" if c["n_running_mean"] > c["n_valley"] else ""
        lines.append(
            f"  {c['scatterer']:<14}{c['nlegs']:>6}{c['share']:>8.1%}"
            f"{c['r_range'][0]:>8.2f}-{c['r_range'][1]:<7.2f}"
            f"{c['n_running_mean']:>9}{c['n_valley']:>8}{flag}"
        )
    return "\n".join(lines)
