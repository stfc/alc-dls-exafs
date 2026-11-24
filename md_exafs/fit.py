"""Fit an ensemble archive's chi(k) to an experimental spectrum (Fit A).

:func:`fit_ensemble` refines S0^2 and delta-E0 of the archive's overall average
chi(k); no structural parameter is free. The theory is shifted and scaled with
:func:`md_exafs.experimental.scaled_chi_arrays`, and the objective, noise
estimate and uncertainty rescaling follow ``larch.xafs.feffit`` so the result is
directly comparable with a conventional Larch fit.

The helpers (:func:`_resolve_fit_setup`, :func:`_estimate_noise_weights`,
:func:`_transform_blocks`, :func:`_compute_rfactors`) are shared with the other
fits that compare against an experiment.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from larch import Group
from larch.math import realimag
from larch.xafs.feffit import FeffitDataSet, TransformGroup
from lmfit import Minimizer, Parameters  # type: ignore[import-untyped]

from .archive_series import (
    KWEIGHT_RANGE,
    PHASE_SHIFT_ALLOWANCE_A,
    _archive_ft_defaults,
    _coverage_limits,
    _finite_number,
    _ft_data_warnings,
    _resolve_ft_request,
    _spectrum,
    _units,
    _with_transform,
)
from .experimental import read_experimental_spectrum, scaled_chi_arrays
from .hdf5 import ArchiveReader
from .spectra import resample_chi

FIT_SCHEMA_VERSION = 1

FIT_SPACES = ("r", "k", "q")
VARY_NAMES = ("s02", "e0_shift")

# Default fit window (A), capped at rmax_valid.
DEFAULT_RRANGE = (1.0, 3.0)

# Bounds on the varied parameters; e0_shift is limited so that the shifted
# k grid stays above the transform window.
S02_BOUNDS = (1e-4, 10.0)
E0_BOUNDS = (-30.0, 30.0)

# High-R window (A) in which ``estimate_noise`` measures the noise, and the
# smallest epsilon taken as a real estimate.
NOISE_RRANGE = (15.0, 30.0)
MIN_EPSILON = 1e-12


@dataclass
class _FitSetup:
    """Validated inputs of a fit against one experiment."""

    overall_raw: dict[str, np.ndarray]
    overall_k: np.ndarray
    overall_chi: np.ndarray
    exp_k: np.ndarray
    exp_chi: np.ndarray
    ft: dict[str, Any]
    rrange: tuple[float, float]
    kweights: list[int]
    fit_space: str
    vary: list[str]
    start: dict[str, float]
    rmax_valid: float | None
    warnings: list[str] = field(default_factory=list)


def _number(value: Any) -> float | None:
    """``value`` as a float, or ``None`` when missing or not finite."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _ratio(num: np.ndarray, den: np.ndarray) -> float | None:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _number(np.sum(num**2) / np.sum(den**2))


def _clean_spectrum(k: Any, chi: Any, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Finite points of a spectrum in ascending k."""
    k_arr = np.asarray(k, dtype=float)
    chi_arr = np.asarray(chi, dtype=float)
    keep = np.isfinite(k_arr) & np.isfinite(chi_arr)
    if keep.sum() < 2:  # noqa: PLR2004
        raise ValueError(f"The {what} spectrum has fewer than two finite points")
    order = np.argsort(k_arr[keep], kind="stable")
    return k_arr[keep][order], chi_arr[keep][order]


def _resolve_kweights(kweights: Sequence[int] | int) -> list[int]:
    values = [kweights] if isinstance(kweights, int | np.integer) else list(kweights)
    lo, hi = KWEIGHT_RANGE
    if not values:
        raise ValueError("kweights must not be empty")
    for kw in values:
        if (
            isinstance(kw, bool)
            or not isinstance(kw, int | np.integer)
            or not lo <= kw <= hi
        ):
            raise ValueError(f"kweights must be integers from {lo} to {hi}, got {kw!r}")
    return [int(kw) for kw in values]


def _resolve_rrange(
    rrange: tuple[float, float] | None, rmax_valid: float | None
) -> tuple[float, float]:
    if rrange is None:
        rmin, rmax = DEFAULT_RRANGE
        rmax = rmax if rmax_valid is None else min(rmax, rmax_valid)
    else:
        try:
            rmin, rmax = (float(v) for v in rrange)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"rrange must be (rmin, rmax), got {rrange!r}") from exc
    if not (math.isfinite(rmin) and math.isfinite(rmax) and 0.0 <= rmin < rmax):
        raise ValueError(f"rrange must satisfy 0 <= rmin < rmax, got {(rmin, rmax)!r}")
    return rmin, rmax


def _resolve_fit_setup(
    path: Path | str,
    experiment: Any,
    *,
    ft_params: dict[str, Any] | None,
    rrange: tuple[float, float] | None,
    kweights: Sequence[int] | int,
    fit_space: str,
    vary: Collection[str],
    s02: float,
    e0_shift: float,
) -> _FitSetup:
    """Read the archive and the experiment and validate every argument."""
    space = str(fit_space).lower()
    if space not in FIT_SPACES:
        raise ValueError(f"fit_space must be one of {FIT_SPACES}, got {fit_space!r}")
    names = [vary] if isinstance(vary, str) else list(vary)
    unknown = sorted(set(names) - set(VARY_NAMES))
    if unknown:
        raise ValueError(f"Unknown vary name(s) {unknown}; allowed: {VARY_NAMES}")
    if not _finite_number(s02) or s02 <= 0:
        raise ValueError(f"s02 must be a finite number > 0, got {s02!r}")
    if not _finite_number(e0_shift):
        raise ValueError(f"e0_shift must be a finite number, got {e0_shift!r}")
    for name, value, (lo, hi) in (
        ("s02", s02, S02_BOUNDS),
        ("e0_shift", e0_shift, E0_BOUNDS),
    ):
        if name in names and not lo <= value <= hi:
            raise ValueError(f"{name} must lie in [{lo:g}, {hi:g}], got {value!r}")
    kw_list = _resolve_kweights(kweights)

    reader = ArchiveReader(path)
    if not reader.is_ensemble:
        raise ValueError(f"{reader.path} is not an ensemble archive")
    try:
        overall_raw = reader.get_overall_average()
    except KeyError as exc:
        raise ValueError(f"{reader.path} has no overall average") from exc
    exp_k_raw, exp_chi_raw, _meta = read_experimental_spectrum(experiment)

    warnings: list[str] = []
    stored = reader.fourier_params
    if not stored:
        warnings.append(
            "The archive records no Fourier-transform parameters; library "
            "defaults are assumed."
        )
    defaults = _archive_ft_defaults(stored)
    ft = defaults if ft_params is None else _resolve_ft_request(ft_params, defaults)

    radius = reader.radius
    rmax_valid: float | None = (
        max(0.0, float(radius - PHASE_SHIFT_ALLOWANCE_A))
        if radius is not None
        else None
    )
    rmin, rmax = _resolve_rrange(rrange, rmax_valid)
    if rmax_valid is None:
        warnings.append(
            "The archive records no FEFF cluster radius, so rmax_valid cannot "
            "be determined."
        )
    elif rmax > rmax_valid:
        warnings.append(
            f"The fit window's rmax={rmax:g} A lies above rmax_valid="
            f"{rmax_valid:g} A (FEFF cluster radius {radius:g} A minus a "
            f"{PHASE_SHIFT_ALLOWANCE_A:g} A phase-shift allowance). chi(R) "
            "beyond it is not reliable because the cluster is truncated there."
        )

    overall_k, overall_chi = _clean_spectrum(
        overall_raw["k"], overall_raw["chi"], "overall"
    )
    exp_k, exp_chi = _clean_spectrum(exp_k_raw, exp_chi_raw, "experiment")
    warnings.extend(
        w["message"] for w in _ft_data_warnings(ft, _coverage_limits(overall_raw))
    )
    warnings.extend(
        w["message"].replace("overall average", "experimental spectrum")
        for w in _ft_data_warnings(ft, _coverage_limits({"k": exp_k, "chi": exp_chi}))
    )
    return _FitSetup(
        overall_raw=overall_raw,
        overall_k=overall_k,
        overall_chi=overall_chi,
        exp_k=exp_k,
        exp_chi=exp_chi,
        ft=ft,
        rrange=(rmin, rmax),
        kweights=kw_list,
        fit_space=space,
        vary=[n for n in VARY_NAMES if n in names],
        start={"s02": float(s02), "e0_shift": float(e0_shift)},
        rmax_valid=rmax_valid,
        warnings=warnings,
    )


def _make_transform(setup: _FitSetup) -> tuple[Any, np.ndarray]:
    """Larch transform and the common k grid every spectrum is put on.

    ``TransformGroup.fftf`` takes index 0 of its input as k = 0, so the grid is
    ``trans.k_`` itself (``k = 0, 0.05, ...``), never a grid starting at 0.05.
    """
    ft = setup.ft
    trans = TransformGroup(
        kmin=ft["kmin"],
        kmax=ft["kmax"],
        dk=ft["dk"],
        window=ft["window"],
        kweight=list(setup.kweights),
        rmin=setup.rrange[0],
        rmax=setup.rrange[1],
        fitspace=setup.fit_space,
    )
    trans.make_karrays()
    k_max_fit = max(
        float(np.nanmax(setup.exp_k)),
        float(np.nanmax(setup.overall_k)),
        ft["kmax"] + ft["dk"],
    )
    ikmax = min(len(trans.k_), int(1.01 + k_max_fit / trans.kstep))
    return trans, np.asarray(trans.k_[:ikmax], dtype=float)


def _onto_grid(k: np.ndarray, chi: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """``chi`` on ``k_grid``; k outside its support are zero-filled."""
    finite = np.isfinite(chi)
    return np.nan_to_num(resample_chi(k[finite], chi[finite], k_grid), nan=0.0)


def _theory_on_grid(
    setup: _FitSetup, values: dict[str, float], k_grid: np.ndarray
) -> np.ndarray:
    k_adj, chi_adj = scaled_chi_arrays(
        setup.overall_k,
        setup.overall_chi,
        s02=values["s02"],
        e0_shift=values["e0_shift"],
    )
    return _onto_grid(k_adj, chi_adj, k_grid)


def _transform_blocks(
    trans: Any,
    k_fit: np.ndarray,
    diff: np.ndarray,
    kweights: Sequence[int],
    space: str,
    eps_r: Sequence[float],
    eps_k: Sequence[float],
) -> list[np.ndarray]:
    """Residual of ``diff`` in ``space``, one block per k-weight.

    Reproduces ``larch.xafs.feffit.FeffitDataSet._residual``; each block is
    divided by that k-weight's epsilon.
    """
    irmin = int(max(0, 0.01 + trans.rmin / trans.rstep))
    irmax = int(min(trans.nfft / 2, 0.01 + trans.rmax / trans.rstep))
    iqmin = int(max(0, 0.01 + trans.kmin / trans.kstep))
    iqmax = int(min(trans.nfft / 2, 0.01 + trans.kmax / trans.kstep))
    blocks: list[np.ndarray] = []
    for i, kw in enumerate(kweights):
        if space == "k":
            blocks.append(((diff / eps_k[i]) * k_fit**kw)[iqmin:iqmax])
            continue
        chir = trans.fftf(diff, kweight=kw)
        if space == "r":
            blocks.append(realimag((chir / eps_r[i])[irmin:irmax]))
        else:
            chiq = trans.fftr(chir) / eps_r[i]
            blocks.append(realimag(chiq[iqmin:iqmax])[::2])
    return blocks


def _usable(eps: Sequence[float]) -> bool:
    return all(np.isfinite(e) and e > MIN_EPSILON for e in eps)


def _estimate_noise_weights(
    trans: Any,
    k_fit: np.ndarray,
    exp_fit: np.ndarray,
    kweights: Sequence[int],
    fit_space: str,
    warnings: list[str],
) -> tuple[list[float], list[float]]:
    """Per-k-weight ``epsilon_r`` and ``epsilon_k`` of the experiment.

    Larch's ``estimate_noise`` measures them in the high-R chi(R). When one is
    zero or not finite (a noiseless spectrum) every k-weight falls back to the
    RMS of the data in the fit space, so no k-weight outweighs the others, and
    ``warnings`` gains a note that ``chi_square`` is relative only.
    """
    ds = FeffitDataSet(data=Group(k=k_fit, chi=exp_fit), transform=trans)
    ds.estimate_noise(
        chi=exp_fit, rmin=NOISE_RRANGE[0], rmax=NOISE_RRANGE[1], all_kweights=True
    )
    n = len(kweights)
    eps_r = [float(e) for e in np.atleast_1d(ds.epsilon_r)][:n]
    eps_k = [float(e) for e in np.atleast_1d(ds.epsilon_k)][:n]
    if len(eps_r) == n and len(eps_k) == n and _usable(eps_r) and _usable(eps_k):
        return eps_r, eps_k

    ones = [1.0] * n
    space_r = "q" if fit_space == "q" else "r"
    eps_r = [
        float(np.sqrt(np.mean(b**2)))
        for b in _transform_blocks(trans, k_fit, exp_fit, kweights, space_r, ones, ones)
    ]
    eps_k = [
        float(np.sqrt(np.mean(b**2)))
        for b in _transform_blocks(trans, k_fit, exp_fit, kweights, "k", ones, ones)
    ]
    if not (_usable(eps_r) and _usable(eps_k)):
        raise ValueError("The experimental spectrum has no signal in the fit window")
    warnings.append(
        "High-R noise in the experimental spectrum was negligible, so each "
        "k-weight is normalized by the RMS of the data in the fit window; "
        "chi_square is relative only."
    )
    return eps_r, eps_k


def _compute_rfactors(
    trans: Any,
    k_fit: np.ndarray,
    exp_fit: np.ndarray,
    theory_fit: np.ndarray,
    kweights: Sequence[int],
    fit_space: str,
    eps_r: Sequence[float],
    eps_k: Sequence[float],
) -> dict[str, float | None]:
    """R-factor ``sum(resid^2) / sum(data^2)`` in the fit space, R and k."""
    out: dict[str, float | None] = {}
    for key, space in (("rfactor", fit_space), ("rfactor_r", "r"), ("rfactor_k", "k")):
        resid, data = (
            np.concatenate(
                _transform_blocks(trans, k_fit, chi, kweights, space, eps_r, eps_k)
            )
            for chi in (exp_fit - theory_fit, exp_fit)
        )
        out[key] = _ratio(resid, data)
    return out


def _correlations(names: list[str], covar: np.ndarray | None) -> dict[str, Any]:
    out: dict[str, dict[str, float]] = {}
    if covar is None:
        return out
    sd = np.sqrt(np.abs(np.diag(covar)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j or not sd[i] * sd[j] > 0:
                continue
            corr = _number(covar[i, j] / (sd[i] * sd[j]))
            if corr is not None:
                out.setdefault(a, {})[b] = corr
    return out


def _lmfit_parameters(setup: _FitSetup) -> Parameters:
    params = Parameters()
    bounds = {"s02": S02_BOUNDS, "e0_shift": E0_BOUNDS}
    for name in VARY_NAMES:
        lo, hi = bounds[name] if name in setup.vary else (-np.inf, np.inf)
        params.add(
            name, value=setup.start[name], min=lo, max=hi, vary=name in setup.vary
        )
    return params


def _optimise(
    setup: _FitSetup, resid_fn: Any, n_idp: float
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit the varied parameters and rescale statistics to ``n_idp``.

    The rescaling is that of ``larch.xafs.feffit.feffit``: ``chi_square`` and
    ``chi2_reduced`` are scaled to ``n_idp`` independent points, and the
    covariance and standard errors by
    ``redchi * nfree / (n_idp - nvarys)``.
    """
    params = _lmfit_parameters(setup)

    def fcn(p: Parameters) -> np.ndarray:
        return np.asarray(resid_fn({n: p[n].value for n in VARY_NAMES}))

    stats: dict[str, Any] = {
        "stderr": dict.fromkeys(VARY_NAMES),
        "correlations": {},
        "nvarys": 0,
        "success": True,
        "warnings": [],
    }
    if not setup.vary:
        resid = fcn(params)
        stats["chi_square"] = float(np.sum(resid**2) * n_idp / len(resid))
        stats["chi2_reduced"] = stats["chi_square"] / n_idp
        return dict(setup.start), stats

    result = Minimizer(fcn, params, scale_covar=False).leastsq()
    npts = len(result.residual)
    stats["nvarys"] = int(result.nvarys)
    stats["success"] = bool(result.success)
    stats["chi_square"] = float(result.chisqr * n_idp / npts)
    if n_idp > result.nvarys:
        stats["chi2_reduced"] = float(stats["chi_square"] / (n_idp - result.nvarys))
    else:
        stats["chi2_reduced"] = None
        stats["warnings"].append(
            f"n_idp={n_idp:.1f} does not exceed the {result.nvarys} varied "
            "parameter(s), so chi2_reduced and the uncertainties are unavailable."
        )
    covar = result.covar
    if covar is not None and n_idp > result.nvarys:
        err_scale = float(result.redchi * (result.nfree / (n_idp - result.nvarys)))
        for par in result.params.values():
            if par.stderr is not None:
                par.stderr *= np.sqrt(err_scale)
        covar = covar * err_scale
        stats["correlations"] = _correlations(list(result.var_names), covar)
        stats["stderr"] = {
            n: (_number(result.params[n].stderr) if n in setup.vary else None)
            for n in VARY_NAMES
        }
    values = {n: float(result.params[n].value) for n in VARY_NAMES}
    return values, stats


def fit_ensemble(
    path: Path | str,
    experiment: Any,
    *,
    ft_params: dict[str, Any] | None = None,
    rrange: tuple[float, float] | None = None,
    kweights: Sequence[int] | int = (1, 2, 3),
    fit_space: str = "r",
    vary: Collection[str] = ("s02", "e0_shift"),
    s02: float = 1.0,
    e0_shift: float = 0.0,
) -> dict[str, Any]:
    """Fit S0^2 and delta-E0 of an archive's overall chi(k) to an experiment.

    S0^2 and delta-E0 are applied to the simulation with
    :func:`md_exafs.experimental.scaled_chi_arrays`, never to the experiment.
    Both spectra are put on one k grid starting at 0 and compared as
    ``larch.xafs.feffit`` does: the residual is transformed in ``fit_space``
    for every k-weight and divided by that k-weight's noise estimate, so no
    k-weight dominates. ``chi_square``, ``chi2_reduced`` and the standard errors
    are rescaled to ``n_idp`` as Larch does.

    Args:
        path: Ensemble archive (``ensemble_results.h5``).
        experiment: Anything
            :func:`md_exafs.experimental.read_experimental_spectrum` accepts.
        ft_params: Fourier-transform overrides, as for
            :func:`md_exafs.archive_series.archive_series`.
        rrange: Fit window ``(rmin, rmax)`` in Angstrom. Defaults to
            ``(1.0, min(3.0, rmax_valid))``.
        kweights: One k-weight or several (each 0 to 4), fitted simultaneously.
        fit_space: ``"r"`` (Re and Im of chi(R)), ``"k"`` or ``"q"``.
        vary: Parameters to refine, a subset of ``("s02", "e0_shift")``. An
            empty collection evaluates the starting values without fitting.
        s02: Starting S0^2 (finite, positive).
        e0_shift: Starting delta-E0 in eV (finite).

    Returns:
        A JSON-safe dict with ``schema_version``, ``fit_type``, ``fit_space``,
        ``ft_params``, ``rrange``, ``kweights``, ``rmax_valid``, ``params``
        (``value``, ``stderr``, ``start`` and ``vary`` for each), ``rfactor``
        (in ``fit_space``), ``rfactor_r``, ``rfactor_k``, ``chi_square``,
        ``chi2_reduced``, ``n_idp``, ``nvarys``, ``success``,
        ``correlations``, ``warnings``, ``units``, and ``experiment`` and
        ``theory`` (the best fit) on the archive's k grid, shaped like the
        ``overall`` entry of ``archive_series``. Non-finite values are ``None``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: Not an ensemble archive, no overall average, or bad
            arguments (including unusable ``ft_params``).
    """
    setup = _resolve_fit_setup(
        path,
        experiment,
        ft_params=ft_params,
        rrange=rrange,
        kweights=kweights,
        fit_space=fit_space,
        vary=vary,
        s02=s02,
        e0_shift=e0_shift,
    )
    ft, (rmin, rmax) = setup.ft, setup.rrange
    warnings = setup.warnings
    trans, k_fit = _make_transform(setup)
    exp_fit = _onto_grid(setup.exp_k, setup.exp_chi, k_fit)
    eps_r, eps_k = _estimate_noise_weights(
        trans, k_fit, exp_fit, setup.kweights, setup.fit_space, warnings
    )

    def resid_fn(values: dict[str, float]) -> np.ndarray:
        diff = exp_fit - _theory_on_grid(setup, values, k_fit)
        return np.concatenate(
            _transform_blocks(
                trans, k_fit, diff, setup.kweights, setup.fit_space, eps_r, eps_k
            )
        )

    n_idp = float(1.0 + 2.0 * (rmax - rmin) * (ft["kmax"] - ft["kmin"]) / np.pi)
    best, stats = _optimise(setup, resid_fn, n_idp)
    warnings.extend(stats["warnings"])
    rfactors = _compute_rfactors(
        trans,
        k_fit,
        exp_fit,
        _theory_on_grid(setup, best, k_fit),
        setup.kweights,
        setup.fit_space,
        eps_r,
        eps_k,
    )

    series_k = np.asarray(setup.overall_raw["k"], dtype=float)
    k_adj, chi_adj = scaled_chi_arrays(
        setup.overall_k, setup.overall_chi, s02=best["s02"], e0_shift=best["e0_shift"]
    )
    curves = {
        "experiment": resample_chi(setup.exp_k, setup.exp_chi, series_k),
        "theory": resample_chi(k_adj, chi_adj, series_k),
    }
    return {
        "schema_version": FIT_SCHEMA_VERSION,
        "fit_type": "ensemble",
        "fit_space": setup.fit_space,
        "ft_params": ft,
        "rrange": [rmin, rmax],
        "kweights": setup.kweights,
        "rmax_valid": setup.rmax_valid,
        "params": {
            name: {
                "value": _number(best[name]),
                "stderr": stats["stderr"][name],
                "start": setup.start[name],
                "vary": name in setup.vary,
            }
            for name in VARY_NAMES
        },
        **rfactors,
        "chi_square": _number(stats["chi_square"]),
        "chi2_reduced": _number(stats["chi2_reduced"]),
        "n_idp": n_idp,
        "nvarys": stats["nvarys"],
        "success": stats["success"],
        "correlations": stats["correlations"],
        "warnings": warnings,
        "units": _units(ft["kweight"]),
        **{
            name: _spectrum(
                _with_transform({"k": series_k, "chi": chi}, ft, True), ft["kweight"]
            )
            for name, chi in curves.items()
        },
    }


from .fit_static import (  # noqa: E402
    PooledPath,
    fit_static,
    pool_paths,
    static_paths,
    to_feffit_path,
)

__all__ = [
    "FIT_SCHEMA_VERSION",
    "PooledPath",
    "fit_ensemble",
    "fit_static",
    "pool_paths",
    "static_paths",
    "to_feffit_path",
]
