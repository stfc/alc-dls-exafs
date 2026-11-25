#!/usr/bin/env python3
"""Score an experimentalist's manual (Artemis) fit against the same experiment.

The Artemis export (``*.fit``, XDI-style) carries the experiment chi(k) and the
model chi(k) in columns ``k, chi, fit``. The model's parameters and fit window
metadata live in the Artemis project file (``fit.yaml``). This module scores
the model with :func:`md_exafs.score_spectrum` so its R-factors use the exact
same residual weighting as the MD-ensemble and static fits in every R window.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from md_exafs import score_spectrum
from md_exafs.experimental import read_experimental_spectrum


def read_artemis_fit(path: Path | str) -> dict[str, np.ndarray]:
    """Read columns ``k``, ``chi``, and ``fit`` from an Artemis ``.fit`` file."""
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 3:  # noqa: PLR2004
        raise ValueError(f"{path}: expected at least columns k, chi, fit")
    return {"k": data[:, 0], "chi": data[:, 1], "fit": data[:, 2]}


def read_artemis_params(
    path: Path | str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read refined parameters and logged fit metadata from an Artemis ``fit.yaml``.

    Returns:
        ``(params, meta)`` where each param has ``name``, ``value``, ``stderr``,
        and ``vary`` (true only for refined Artemis ``guess`` parameters), and
        ``meta`` holds logged ``r_factor``, ``r_range``, ``k_range``,
        ``kweights``, ``n_idp``, ``n_varys``, and ``chi_reduced`` when present.
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return [], {}

    params: list[dict[str, Any]] = []
    for entry in doc.get("parameters", []):
        if not isinstance(entry, list) or len(entry) < 6:  # noqa: PLR2004
            continue
        name, kind, _expr, value, stderr = entry[:5]
        refined = str(kind).strip() == "guess" and str(entry[5]).strip() in (
            "1",
            "True",
            "true",
        )
        params.append(
            {
                "name": str(name),
                "value": float(value),
                "stderr": float(stderr),
                "vary": refined,
            }
        )

    meta: dict[str, Any] = {}
    for dst, src in (
        ("r_factor_logged", "r_factor"),
        ("chi_reduced_logged", "chi_reduced"),
        ("n_idp_logged", "n_idp"),
        ("n_varys_logged", "n_varys"),
    ):
        raw = doc.get(src)
        if raw is not None:
            try:
                meta[dst] = float(raw)
            except (TypeError, ValueError):
                pass

    logcache = str(doc.get("logcache") or "")
    for key, pattern in (
        ("r_range_logged", r"R-range\s*=\s*([0-9.eE+-]+)\s*-\s*([0-9.eE+-]+)"),
        ("k_range_logged", r"k-range\s*=\s*([0-9.eE+-]+)\s*-\s*([0-9.eE+-]+)"),
    ):
        m = re.search(pattern, logcache, flags=re.IGNORECASE)
        if m:
            meta[key] = [float(m.group(1)), float(m.group(2))]
    m_kw = re.search(r"k-weight\s*=\s*([0-9,\s]+)", logcache, flags=re.IGNORECASE)
    if m_kw:
        kws = [int(t) for t in re.findall(r"\d+", m_kw.group(1))]
        if kws:
            meta["kweights_logged"] = kws

    return params, meta


def evaluate_manual_fit(
    fit_file: Path | str,
    *,
    ft_params: dict[str, Any],
    rrange: tuple[float, float],
    kweights: tuple[int, ...],
    fit_space: str,
    series_k: np.ndarray,
    experiment: Any = None,
    params_file: Path | str | None = None,
) -> dict[str, Any]:
    """Score a manual Artemis fit in one R window on the shared experiment.

    Args:
        fit_file: Artemis ``.fit`` export (experiment chi(k) plus model column).
        ft_params: Fourier-transform parameters, e.g. ``fit_a["ft_params"]``.
        rrange: Fit window ``(rmin, rmax)`` in Å.
        kweights: Fitted k-weights.
        fit_space: ``"r"`` or ``"k"``.
        series_k: The k grid used for the MD curves, so the spectra line up.
        experiment: Optional shared experiment source (checked against the
            ``.fit`` file's ``chi`` column so all models are scored on identical
            data). When omitted, uses the ``.fit`` file's ``chi`` column.
        params_file: Optional Artemis ``fit.yaml`` with refined parameters.

    Returns:
        JSON-safe dict with ``rfactor``, ``rfactor_r``, ``rfactor_k``,
        ``epsilon_r``, ``epsilon_k``, ``chi_square``, ``chi2_reduced``,
        ``n_idp``, ``rrange``, ``experiment`` and ``theory`` spectra,
        ``params``, ``meta``, and ``warnings``.
    """
    cols = read_artemis_fit(fit_file)
    warnings: list[str] = []

    exp_arg: Any = (cols["k"], cols["chi"])
    if experiment is not None:
        exp_k, exp_chi, _ = read_experimental_spectrum(experiment)
        exp_arg = (exp_k, exp_chi)
        # Check parity between the shared experiment and the .fit file's chi
        k_lo = max(float(np.nanmin(exp_k)), float(np.nanmin(cols["k"])))
        k_hi = min(float(np.nanmax(exp_k)), float(np.nanmax(cols["k"])))
        if k_hi > k_lo:
            probe = np.linspace(k_lo, k_hi, 100)
            c1 = np.interp(probe, exp_k, exp_chi)
            c2 = np.interp(probe, cols["k"], cols["chi"])
            denom = max(float(np.linalg.norm(c1)), 1e-12)
            rel_diff = float(np.linalg.norm(c1 - c2)) / denom
            if rel_diff > 1e-2:
                warnings.append(
                    f"Manual fit file's chi column differs from --experiment "
                    f"(relative L2 diff {rel_diff:.2%}); scored manual model "
                    "against --experiment for apples-to-apples comparison."
                )

    scored = score_spectrum(
        exp_arg,
        (cols["k"], cols["fit"]),
        ft_params=ft_params,
        rrange=rrange,
        kweights=kweights,
        fit_space=fit_space,
        k_grid=series_k,
    )
    warnings.extend(scored["warnings"])

    params: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    if params_file is not None:
        params, meta = read_artemis_params(params_file)

    return {
        "source": str(Path(fit_file).resolve()),
        "params_source": str(Path(params_file).resolve()) if params_file else None,
        "rrange": scored["rrange"],
        "rfactor": scored["rfactor"],
        "rfactor_r": scored["rfactor_r"],
        "rfactor_k": scored["rfactor_k"],
        "epsilon_r": scored["epsilon_r"],
        "epsilon_k": scored["epsilon_k"],
        "chi_square": scored["chi_square"],
        "chi2_reduced": scored["chi2_reduced"],
        "n_idp": scored["n_idp"],
        "warnings": warnings,
        "params": params,
        "meta": meta,
        "experiment": scored["experiment"],
        "theory": scored["theory"],
    }
