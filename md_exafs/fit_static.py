"""Conventional single-structure EXAFS fit with MD-derived sigma^2 (Fit B).

:func:`fit_static` fits an experimental spectrum with ``larch.xafs.feffit`` using
ONE static structure (by default the trajectory average) whose FEFF scattering
paths are pooled over all absorbers, with the thermal ``sigma^2`` of every path
taken from the MD trajectory.

Pooling is exact rather than parameter-averaged. All paths of one cluster are
summed as complex amplitudes, each at its own length ``R_j``, and the sum is
stored as the amplitude and phase of an *effective* path at the weighted-mean
length. Averaging amplitudes, phases and lengths instead would lose the
dephasing ``<exp(2ik(R_j - Rbar))>`` between members, which a static structure
has no thermal disorder to hide.

The building blocks are :func:`static_paths` (run FEFF on the static
structure), :func:`pool_paths` (complex-sum pooling) and
:func:`to_feffit_path` (adapter to ``larch.xafs.feffdat.FeffPathGroup``).
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read as ase_read
from larch import Group
from larch.xafs.feffdat import FeffDatFile, FeffPathGroup
from larch.xafs.feffit import FeffitDataSet, TransformGroup, feffit
from lmfit import Parameters  # type: ignore[import-untyped]
from scipy.cluster.hierarchy import fcluster, linkage

from .archive_series import (
    PHASE_SHIFT_ALLOWANCE_A,
    _archive_ft_defaults,
    _coverage_limits,
    _ft_data_warnings,
    _resolve_ft_request,
    _spectrum,
    _units,
    _with_transform,
)
from .debye_waller import (
    _rattle_scatterer_key,
    calculate_grouped_msrd,
    canonical_scatterer_key,
    match_paths_within_tolerance,
    parse_site_specification,
)
from .execution import FeffTask, parse_task_outputs
from .experimental import read_experimental_spectrum
from .feff_input import FeffConfig as InputFeffConfig
from .feff_utils import FeffConfig, normalize_absorbers, run_feff_calculation
from .fit import (
    E0_BOUNDS,
    FIT_SCHEMA_VERSION,
    FIT_SPACES,
    S02_BOUNDS,
    _clean_spectrum,
    _compute_rfactors,
    _correlations,
    _estimate_noise_weights,
    _number,
    _onto_grid,
    _resolve_kweights,
    _resolve_rrange,
    _transform_blocks,
    fit_ensemble,
)
from .paths import FEFF_DATA_COLS, PathResult, make_path_key, read_paths_from_dir
from .pipeline import InputGenerator, average_structure
from .spectra import xftf_arrays

SIGMA2_MODES = ("md", "free")
DELTAR_MODES = ("isotropic", "shells")

# FEFF CRITERIA used for the static runs: lower than the pipeline default so
# weak paths are not dropped inconsistently between absorbers.
STATIC_CRITERIA = "1.0 0.5"

ALPHA_BOUNDS = (-0.2, 0.2)
DR_BOUNDS = (-0.5, 0.5)
SIGMA2_MAX = 0.05

# Warning thresholds.
PHASE_JUMP_RAD = np.pi / 2
CANCELLATION_RATIO = 0.1
DEPHASING_PHASE_RAD = 0.5
SPREAD_KMAX = 15.0
MATCH_DR_WARN_A = 0.05
CORRELATION_WARN = 0.9
TRUNCATION_WARN = 0.10
IDP_FRACTION_LIMIT = 2.0 / 3.0

_COL = {name: i for i, name in enumerate(FEFF_DATA_COLS)}
_DEFAULT_MASS = (29, 63.546)


@dataclass
class PooledPath:
    """An effective scattering path pooled over absorbers and members.

    ``amp``, ``pha``, ``rep`` and ``lam`` are the FEFF-style columns on the
    native k grid ``k``, defined so that evaluating the EXAFS equation with
    reference length ``r_eff`` and degeneracy ``degeneracy`` reproduces the
    absorber-averaged complex sum of the member paths.
    """

    label: str
    scatterer: str
    nlegs: int
    r_eff: float
    r_eff_std: float
    degeneracy: float
    count: int
    angle: float | None
    cw_ratio: float
    k: np.ndarray
    amp: np.ndarray
    pha: np.ndarray
    rep: np.ndarray
    lam: np.ndarray
    absorber: str = ""
    warnings: list[str] = field(default_factory=list)


def _complete_linkage(values: np.ndarray, tol: float) -> list[list[int]]:
    """Index groups of ``values`` whose diameter is at most ``tol``."""
    n = len(values)
    if n == 0:
        return []
    if n < 2:  # noqa: PLR2004
        return [[0]]
    labels = fcluster(
        linkage(values.reshape(-1, 1), method="complete"), t=tol, criterion="distance"
    )
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    return list(groups.values())


def _cluster_category(
    items: list[PathResult], nlegs: int, r_tol: float, angle_tol: float
) -> list[list[PathResult]]:
    """Complete-linkage clusters of one (scatterer, nlegs) category."""

    def by_r(sub: list[PathResult]) -> list[list[PathResult]]:
        r = np.array([float(p.r_eff) for p in sub])
        return [[sub[i] for i in idx] for idx in _complete_linkage(r, r_tol)]

    if nlegs != 3:  # noqa: PLR2004
        return by_r(items)
    with_angle = [p for p in items if p.angle is not None and np.isfinite(p.angle)]
    without = [p for p in items if p.angle is None or not np.isfinite(p.angle)]
    clusters: list[list[PathResult]] = []
    if with_angle:
        angles = np.array([float(p.angle) for p in with_angle])  # type: ignore[arg-type]
        for idx in _complete_linkage(angles, angle_tol):
            clusters.extend(by_r([with_angle[i] for i in idx]))
    if without:
        clusters.extend(by_r(without))
    return clusters


def _member_columns(p: PathResult, k: np.ndarray) -> np.ndarray:
    """The six ``feff_data`` columns of ``p`` on the grid ``k``."""
    data = np.asarray(p.feff_data, dtype=float)
    pk = np.asarray(p.k, dtype=float)
    if pk.shape == k.shape and np.allclose(pk, k, rtol=0.0, atol=1e-9):
        return data
    return np.column_stack([np.interp(k, pk, data[:, c]) for c in range(data.shape[1])])


def _pool_cluster(
    members: list[PathResult], n_abs: int, absorber: str, scatterer: str
) -> PooledPath:
    """Exact complex-sum pooling of one cluster (handoff section 6.1)."""
    k = np.asarray(members[0].k, dtype=float)
    n = np.array([float(p.degeneracy) for p in members])
    r = np.array([float(p.r_eff) for p in members])
    n_tot = float(n.sum())
    if not n_tot > 0:
        raise ValueError("Path cluster has no positive degeneracy")
    rbar = float(np.sum(n * r) / n_tot)
    r_std = float(np.sqrt(np.sum(n * (r - rbar) ** 2) / n_tot))

    cols = [_member_columns(p, k) for p in members]
    amp = [c[:, _COL["mag_feff"]] * c[:, _COL["red_fact"]] for c in cols]
    pha = [c[:, _COL["real_phc"]] + c[:, _COL["pha_feff"]] for c in cols]
    rep = [c[:, _COL["rep"]] for c in cols]
    lam = [c[:, _COL["lam"]] for c in cols]

    z = []
    for j in range(len(members)):
        p_j = np.sqrt((rep[j] + 1j / lam[j]) ** 2)
        z.append(
            n[j]
            * amp[j]
            * (rbar**2 / r[j] ** 2)
            * np.exp(-2.0 * r[j] * p_j.imag)
            * np.exp(1j * (2.0 * k * r[j] + pha[j]))
        )
    z_sum = np.sum(z, axis=0)
    z_pool = z_sum / n_abs
    n_pool = n_tot / n_abs

    rep_eff = sum(n[j] * rep[j] for j in range(len(members))) / n_tot
    lam_eff = sum(n[j] * lam[j] for j in range(len(members))) / n_tot
    p_eff = np.sqrt((rep_eff + 1j / lam_eff) ** 2)
    f_eff = z_pool / (n_pool * np.exp(-2.0 * rbar * p_eff.imag) * np.exp(2j * k * rbar))
    pha_eff = np.unwrap(np.angle(f_eff))
    pha_ref = sum(n[j] * pha[j] for j in range(len(members))) / n_tot
    pha_eff = pha_eff + 2.0 * np.pi * round(
        float((pha_ref[0] - pha_eff[0]) / (2.0 * np.pi))
    )

    warnings: list[str] = []
    if len(pha_eff) > 1 and float(np.max(np.abs(np.diff(pha_eff)))) > PHASE_JUMP_RAD:
        warnings.append(
            "The pooled phase has a sharp jump (> pi/2 between k points); the "
            "cubic spline of the effective path may ring there."
        )
    abs_sum = np.sum([np.abs(zj) for zj in z], axis=0)
    usable = (k > 0) & (abs_sum > 0)
    if np.any(usable):
        coherence = float(np.min(np.abs(z_sum[usable]) / abs_sum[usable]))
        if len(members) > 1 and coherence < CANCELLATION_RATIO:
            warnings.append(
                f"Members cancel deeply (min |sum Z_j| / sum |Z_j| = "
                f"{coherence:.3f}); the effective amplitude has a node."
            )
    if 2.0 * SPREAD_KMAX * r_std > DEPHASING_PHASE_RAD:
        warnings.append(
            f"R spread {r_std:.3f} A dephases members by more than "
            f"{DEPHASING_PHASE_RAD:g} rad at k={SPREAD_KMAX:g}; consider a "
            "smaller r_tol."
        )

    finite_w = [
        (float(p.angle), w)
        for p, w in zip(members, n, strict=True)
        if p.angle is not None and np.isfinite(p.angle)
    ]
    angle = (
        float(sum(a * w for a, w in finite_w) / sum(w for _, w in finite_w))
        if finite_w
        else None
    )
    return PooledPath(
        label="",
        scatterer=scatterer,
        nlegs=int(members[0].nlegs),
        r_eff=rbar,
        r_eff_std=r_std,
        degeneracy=float(n_pool),
        count=len(members),
        angle=angle,
        cw_ratio=float(np.sum(n * [float(p.cw_ratio) for p in members]) / n_tot),
        k=k,
        amp=np.abs(f_eff),
        pha=pha_eff,
        rep=np.asarray(rep_eff, dtype=float),
        lam=np.asarray(lam_eff, dtype=float),
        absorber=absorber,
        warnings=warnings,
    )


def pool_paths(
    paths: Sequence[PathResult],
    n_abs: int,
    *,
    r_tol: float = 0.025,
    angle_tol: float = 10.0,
    absorber: str = "",
) -> list[PooledPath]:
    """Pool the FEFF paths of all absorbers into effective paths.

    Paths are bucketed by canonical scatterer and number of legs, then
    clustered by complete linkage (so a cluster's diameter never exceeds the
    tolerance): on ``r_eff`` alone, or for 3-leg paths first on the
    reversal-invariant angle and then on ``r_eff``. Each cluster becomes one
    :class:`PooledPath` through an exact complex sum of its members, each with
    its own FEFF degeneracy (never multiplied for path reversal) and divided by
    ``n_abs``, the number of absorbers that ran.

    Args:
        paths: Paths of every absorber.
        n_abs: Number of absorbers the paths come from, including absorbers
            with no member in a given cluster.
        r_tol: Maximum ``r_eff`` diameter of a cluster in Angstrom.
        angle_tol: Maximum angle diameter of a 3-leg cluster in degrees.
        absorber: Absorber element recorded on every pooled path.

    Returns:
        Pooled paths sorted by ``(r_eff, scatterer, nlegs)`` with unique labels.

    Raises:
        ValueError: ``n_abs < 1`` or a non-positive tolerance.
    """
    if n_abs < 1:
        raise ValueError(f"n_abs must be >= 1, got {n_abs!r}")
    if not r_tol > 0:
        raise ValueError(f"r_tol must be > 0, got {r_tol!r}")
    if not angle_tol > 0:
        raise ValueError(f"angle_tol must be > 0, got {angle_tol!r}")

    buckets: dict[tuple[tuple[str, ...], int], list[PathResult]] = {}
    for p in paths:
        key = (canonical_scatterer_key(p.scatterer), int(p.nlegs))
        buckets.setdefault(key, []).append(p)

    pooled: list[PooledPath] = []
    for (scat_key, nlegs), items in buckets.items():
        scatterer = "-".join(scat_key)
        for members in _cluster_category(items, nlegs, r_tol, angle_tol):
            pooled.append(_pool_cluster(members, n_abs, absorber, scatterer))

    pooled.sort(key=lambda q: (q.r_eff, q.scatterer, q.nlegs))
    used: set[str] = set()
    for idx, q in enumerate(pooled):
        label = make_path_key(q.scatterer, q.nlegs, q.r_eff, r_bin=0.01, angle=q.angle)
        if label in used:
            label = f"{label}_{idx}"
        while label in used:
            label = f"{label}_"
        used.add(label)
        q.label = label
    return pooled


def _geom(pooled: PooledPath) -> list[tuple[Any, ...]]:
    """Valid non-empty ``geom`` (absorber first) for a pooled path."""

    def entry(sym: str, ipot: int, z: float) -> tuple[Any, ...]:
        if sym in atomic_numbers and atomic_numbers[sym] > 0:
            iz, mass = atomic_numbers[sym], float(atomic_masses[atomic_numbers[sym]])
        else:
            iz, mass = _DEFAULT_MASS
        return (sym or "X", int(iz), ipot, float(mass), 0.0, 0.0, float(z))

    tokens = [t for t in pooled.scatterer.split("-") if t]
    n = max(1, len(tokens))
    geom = [entry(pooled.absorber, 0, 0.0)]
    for i, sym in enumerate(tokens or ["X"]):
        geom.append(entry(sym, i + 1, pooled.r_eff * (i + 1) / n))
    return geom


def to_feffit_path(
    pooled: PooledPath,
    label: str | None = None,
    *,
    degen: float | str | None = None,
    s02: float | str = 1.0,
    e0: float | str = 0.0,
    deltar: float | str = 0.0,
    sigma2: float | str = 0.0,
) -> Any:
    """Wrap a pooled path as a ``larch.xafs.feffdat.FeffPathGroup``.

    A complete ``FeffDatFile`` is built (all 28 state fields, a valid ``geom``
    so the reduced mass is defined) and the path gets a unique non-empty
    ``filename`` and ``feffrun`` so that Larch's parameter names never collide.

    Args:
        pooled: The pooled path.
        label: Path label; defaults to ``pooled.label``.
        degen: Degeneracy value or expression; ``None`` uses ``pooled.degeneracy``.
        s02: S0^2 value or expression.
        e0: Delta-E0 value or expression (eV).
        deltar: Delta-R value or expression (Angstrom).
        sigma2: sigma^2 value or expression (Angstrom^2).

    Returns:
        A ``FeffPathGroup``.
    """
    lbl = label if label is not None else pooled.label
    if not lbl:
        lbl = f"pooled_{pooled.nlegs}_{pooled.r_eff:.4f}"
    tag = f"pooled:{lbl}"
    k = np.asarray(pooled.k, dtype=float)
    zeros = np.zeros_like(k)
    state: tuple[Any, ...] = (
        tag,  # filename
        "pooled path",  # title
        "pooled",  # version
        "K",  # shell
        pooled.absorber,  # absorber
        float(pooled.degeneracy),  # degen
        float(pooled.r_eff),  # reff
        int(pooled.nlegs),  # nleg
        1.0,  # rnorman
        0.0,  # edge
        0.0,  # gam_ch
        "pooled",  # exch
        0.0,  # vmu
        0.0,  # vfermi
        0.0,  # vint
        1.0,  # rs_int
        [],  # potentials
        _geom(pooled),  # geom
        None,  # rmass (derived from geom)
        k.tolist(),
        zeros.tolist(),  # real_phc
        np.asarray(pooled.amp, dtype=float).tolist(),  # mag_feff
        np.asarray(pooled.pha, dtype=float).tolist(),  # pha_feff
        np.ones_like(k).tolist(),  # red_fact
        np.asarray(pooled.lam, dtype=float).tolist(),
        np.asarray(pooled.rep, dtype=float).tolist(),
        np.asarray(pooled.pha, dtype=float).tolist(),  # pha
        np.asarray(pooled.amp, dtype=float).tolist(),  # amp
    )
    fd = FeffDatFile()
    fd.__setstate__(state)
    return FeffPathGroup(
        filename=tag,
        label=lbl,
        feffrun=tag,
        degen=degen,
        s02=s02,
        e0=e0,
        deltar=deltar,
        sigma2=sigma2,
        _feffdat=fd,
    )


def _resolve_indices(
    structure: Any, absorber: str | int | list[int], sites: str | Sequence[int] | None
) -> list[int]:
    """Absorber indices of ``structure``, restricted to ``sites`` if given."""
    indices = normalize_absorbers(structure, absorber)
    if sites is None:
        return indices
    if isinstance(sites, str):
        wanted = set(parse_site_specification(sites, structure.get_chemical_symbols()))
    else:
        wanted = {int(i) for i in sites}
    indices = [i for i in indices if i in wanted]
    if not indices:
        raise ValueError(f"sites={sites!r} selects none of the absorbers")
    return indices


def _run_static(
    structure: Any,
    cfg: FeffConfig | InputFeffConfig,
    indices: list[int],
    out_dir: Path,
) -> tuple[list[PathResult], int, tuple[np.ndarray, np.ndarray] | None]:
    site_dirs = {i: out_dir / f"site_{i:04d}" for i in indices}
    todo = [
        i
        for i, d in site_dirs.items()
        if not ((d / "chi.dat").exists() and (d / "feff0001.dat").exists())
    ]
    if todo:
        batch = InputGenerator(cfg).generate_single_site_inputs(  # type: ignore[arg-type]
            structure, todo, out_dir, frame_index=0
        )
        for batch_task in batch.tasks:
            run_feff_calculation(batch_task.feff_dir, verbose=False, cleanup=False)

    all_paths: list[PathResult] = []
    chis: list[tuple[np.ndarray, np.ndarray]] = []
    n_abs = 0
    for i, d in site_dirs.items():
        paths = read_paths_from_dir(d, threshold=0.0, frame_idx=0, site_idx=i)
        if not paths:
            continue
        n_abs += 1
        all_paths.extend(paths)
        task = FeffTask(frame_idx=0, site_idx=i, input_dir=d)
        k, chi, _ = parse_task_outputs(task, store_paths=False)
        if k is not None and chi is not None:
            chis.append((k, chi))
    if n_abs == 0:
        raise RuntimeError(f"FEFF produced no scattering paths in {out_dir}")
    mean_chi: tuple[np.ndarray, np.ndarray] | None = None
    if chis:
        k0 = chis[0][0]
        mean_chi = (
            k0,
            np.mean([np.interp(k0, k, chi) for k, chi in chis], axis=0),
        )
    return all_paths, n_abs, mean_chi


def static_paths(
    structure: Any,
    config: FeffConfig | InputFeffConfig,
    absorber: str | int | list[int],
    sites: str | Sequence[int] | None = None,
    workdir: Path | str | None = None,
) -> tuple[list[PathResult], int, tuple[np.ndarray, np.ndarray] | None]:
    """Run FEFF at every selected absorber of a static structure.

    ``config`` is copied; its ``criteria`` is lowered to ``"1.0 0.5"`` and a
    ``control`` that skips the potentials (a precomputed-potentials MD config)
    is reset to run everything. The ``feffNNNN.dat`` files are kept.

    Args:
        structure: Static ``ase.Atoms``.
        config: FEFF configuration.
        absorber: Element symbol, atom index or list of indices.
        sites: Optional site specification (``parse_site_specification`` string
            or indices) restricting the absorbers.
        workdir: Directory for the runs. Site directories that already hold
            ``chi.dat`` and ``feff0001.dat`` are reused. A temporary directory
            is used (and removed) when omitted.

    Returns:
        ``(paths, n_abs, mean_chi_dat)``: the paths of every absorber that ran,
        their number, and ``(k, mean chi)`` averaged over the absorbers with a
        ``chi.dat`` (``None`` if there was none).
    """
    cfg = copy.deepcopy(config)
    cfg.criteria = STATIC_CRITERIA
    if cfg.control is not None and str(cfg.control).split()[:1] == ["0"]:
        cfg.control = "1 1 1 1 1 1"
    indices = _resolve_indices(structure, absorber, sites)
    if workdir is not None:
        return _run_static(structure, cfg, indices, Path(workdir))
    with tempfile.TemporaryDirectory(prefix="md_exafs_static_") as tmp:
        return _run_static(structure, cfg, indices, Path(tmp))


def _resolve_static_structure(
    structure: Any, structures: list[Any], warnings: list[str]
) -> Any:
    if structure is None:
        if not structures:
            raise ValueError("No structures to average and no static structure given")
        warnings.append(
            "The static structure is the trajectory average. average_structure "
            "does not Kabsch-align the frames, so bond lengths of the average "
            "are slightly short (R(<x>) < <R>, about sigma_perp^2 / 2R) and a "
            "fitted delta-R can absorb that bias."
        )
        return average_structure(structures)
    if isinstance(structure, bool | int | np.integer) or (
        isinstance(structure, str) and structure.lower() in ("first", "last")
    ):
        raise ValueError(
            "A single MD frame cannot be the static structure: its frozen thermal "
            "spread would double-count the MD sigma^2. Pass None (trajectory "
            "average), an Atoms object or a structure file."
        )
    if isinstance(structure, Atoms):
        return structure
    if isinstance(structure, str | Path):
        return ase_read(str(structure))
    raise TypeError(f"Unsupported static structure {type(structure).__name__}")


def _md_sigma2(
    kept: list[PooledPath],
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]],
    match_r_tol: float,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Thermal MD sigma^2 of each pooled path, matched by R tolerance.

    Uses ``match_paths_within_tolerance`` directly so 4-leg rattle paths get a
    group too. The ``count``-weighted mean of ``sigma2_thermal_A2`` is taken
    (times 4 for a rattle path); the spread between instances is excluded as
    the pooled static path already carries it.
    """
    groups = [
        {
            "r_eff_ref": p.r_eff,
            "r_effs": [p.r_eff],
            "nlegs": p.nlegs,
            "scatterer": p.scatterer,
            "angle": p.angle,
            "absorber": p.absorber,
        }
        for p in kept
    ]
    feff_to_dw = match_paths_within_tolerance(res_2b, res_3b, groups, match_r_tol)[
        "feff_to_dw"
    ]
    out: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    for gi, p in enumerate(kept):
        entries = feff_to_dw.get(gi)
        if not entries:
            continue
        rows = [(res_2b if body == "2b" else res_3b)[di] for _d, body, di in entries]
        weights = np.array([float(r["count"]) for r in rows])
        thermal = np.array([float(r["sigma2_thermal_A2"]) for r in rows])
        value = float(np.sum(weights * thermal) / np.sum(weights))
        if p.nlegs == 4:  # noqa: PLR2004
            value *= 4.0
        dist = float(max(d for d, _b, _i in entries))
        if dist > MATCH_DR_WARN_A:
            warnings.append(
                f"{p.label}: MD group matched at |dR|={dist:.3f} A "
                f"(> {MATCH_DR_WARN_A:g} A)."
            )
        if len(entries) > 1:
            warnings.append(
                f"{p.label}: {len(entries)} MD groups match this path; their "
                "count-weighted mean sigma^2 is used."
            )
        out[gi] = {"sigma2": value, "n_groups": len(entries), "match_distance": dist}
    return out, warnings


def _deltar_spec(
    kept: list[PooledPath], mode: str, match_r_tol: float, absorber: str
) -> list[tuple[str, float, bool]]:
    """``(parameter, coefficient, per_reff)`` giving each path's delta-R."""
    iso = ("alpha", 1.0, True)
    if mode == "isotropic":
        return [iso] * len(kept)
    specs: list[tuple[str, float, bool]] = []
    ss = [i for i, p in enumerate(kept) if p.nlegs == 2]  # noqa: PLR2004
    for idx, p in enumerate(kept):
        if p.nlegs == 2:  # noqa: PLR2004
            specs.append((f"dr_{idx}", 1.0, False))
            continue
        spec = iso
        if p.nlegs == 4:  # noqa: PLR2004
            key = _rattle_scatterer_key(p.scatterer, absorber=p.absorber or None)
            near = [
                i
                for i in ss
                if canonical_scatterer_key(kept[i].scatterer) == key
                and abs(kept[i].r_eff - p.r_eff / 2.0) <= match_r_tol
            ]
            if near:
                best = min(near, key=lambda i: abs(kept[i].r_eff - p.r_eff / 2.0))
                spec = (f"dr_{best}", 2.0, False)
        specs.append(spec)
    return specs


def _expr(name: str, coef: float, per_reff: bool) -> str:
    parts = ([f"{coef:g}"] if coef != 1.0 else []) + [name]
    return "*".join(parts + (["reff"] if per_reff else []))


def _model_chi(
    kept: list[PooledPath],
    k: np.ndarray,
    *,
    s02: float,
    e0: float,
    sigma2: Sequence[float],
    deltar: Sequence[float],
) -> np.ndarray:
    """Sum of the kept paths' chi(k) at numeric parameter values."""
    total = np.zeros_like(k)
    for p, s2, dr in zip(kept, sigma2, deltar, strict=True):
        fp = to_feffit_path(p, s02=s02, e0=e0, sigma2=s2, deltar=dr)
        fp._calc_chi(k=k)
        total += np.asarray(fp.chi, dtype=float)
    return total


def _truncation(
    kept: list[PooledPath],
    mean_chi: tuple[np.ndarray, np.ndarray] | None,
    ft: dict[str, Any],
    rrange: tuple[float, float],
) -> float | None:
    """Relative chi(R) difference over the window between kept paths and chi.dat."""
    if mean_chi is None:
        return None
    k, chi = mean_chi
    zeros = [0.0] * len(kept)
    kept_chi = _model_chi(kept, k, s02=1.0, e0=0.0, sigma2=zeros, deltar=zeros)
    ref, mod = (xftf_arrays(k, c, ft, warn_uncovered=False) for c in (chi, kept_chi))
    sel = (ref["r"] >= rrange[0]) & (ref["r"] <= rrange[1])
    t_ref = (ref["chir_re"] + 1j * ref["chir_im"])[sel]
    t_mod = (mod["chir_re"] + 1j * mod["chir_im"])[sel]
    denom = float(np.linalg.norm(t_ref))
    if not denom > 0:
        return None
    return _number(float(np.linalg.norm(t_ref - t_mod)) / denom)


def _chi_square(resid: np.ndarray, n_idp: float) -> float | None:
    return _number(float(np.sum(resid**2) * n_idp / len(resid)))


def _summary(d: dict[str, Any]) -> dict[str, Any]:
    keys = ("rfactor", "rfactor_r", "rfactor_k", "chi_square", "chi2_reduced")
    return {
        "params": {
            n: v["value"] if isinstance(v, dict) else v for n, v in d["params"].items()
        },
        **{k: d.get(k) for k in keys},
        "nvarys": d.get("nvarys", 0),
    }


def fit_static(
    structures: Sequence[Any],
    experiment: Any,
    config: FeffConfig | InputFeffConfig,
    absorber: str | int | list[int],
    *,
    structure: Any = None,
    sites: str | Sequence[int] | None = None,
    sigma2: str = "md",
    deltar_mode: str = "isotropic",
    ft_params: dict[str, Any] | None = None,
    rrange: tuple[float, float] | None = None,
    kweights: Sequence[int] | int = (1, 2, 3),
    fit_space: str = "r",
    workdir: Path | str | None = None,
    msrd: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
    archive_path: Path | str | None = None,
    r_tol: float = 0.025,
    angle_tol: float = 10.0,
    match_r_tol: float = 0.15,
    force: bool = False,
) -> dict[str, Any]:
    """Fit an experiment with one static structure and MD thermal sigma^2.

    FEFF is run on the static structure at every selected absorber, the paths
    are pooled with :func:`pool_paths`, and each pooled path gets the thermal
    sigma^2 of its matching MD path group. Pooled paths without an MD match
    (including multiple-scattering paths beyond the MSRD) are dropped and
    reported, never given a guessed sigma^2. S0^2 and E0 are shared; delta-R is
    one isotropic ``alpha`` (``dR = alpha * reff``) or, in ``"shells"`` mode,
    one parameter per single-scattering shell (a 4-leg rattle path follows
    ``2 * dR`` of its shell, other multiple-scattering paths use ``alpha``).
    Degeneracies stay fixed. The fit is a ``larch.xafs.feffit`` fit with the
    same windows, k-weights and per-k-weight noise normalization as
    :func:`md_exafs.fit.fit_ensemble`.

    Args:
        structures: The MD trajectory (frames of ``ase.Atoms``), used for the
            default average structure and for the MSRD.
        experiment: Anything
            :func:`md_exafs.experimental.read_experimental_spectrum` accepts.
        config: FEFF configuration (``radius``, ``nleg``, ...).
        absorber: Element symbol, atom index or list of indices.
        structure: Static structure: ``None`` for the trajectory average, an
            ``Atoms`` object or a structure file path. Single MD frames
            (``"first"``, ``"last"`` or a frame index) are rejected.
        sites: Optional site specification restricting the absorbers.
        sigma2: ``"md"`` (fixed at the MD value) or ``"free"`` (starts there).
        deltar_mode: ``"isotropic"`` or ``"shells"``.
        ft_params: Fourier-transform overrides, as for ``archive_series``.
        rrange: Fit window; defaults to ``(1.0, min(3.0, radius - 0.5))``.
        kweights: k-weights fitted simultaneously.
        fit_space: ``"r"``, ``"k"`` or ``"q"``.
        workdir: Directory for the FEFF runs; existing site results are reused.
        msrd: Precomputed ``(res_2b, res_3b)`` from ``calculate_grouped_msrd``.
        archive_path: Optional ensemble archive. When given, Fit A, B0 (static
            paths with MD sigma^2, delta-R = 0 and S0^2/E0 from Fit A) and the
            fit are reported side by side as ``comparison``.
        r_tol: Pooling tolerance on ``r_eff`` (Angstrom).
        angle_tol: Pooling tolerance on the 3-leg angle (degrees).
        match_r_tol: Tolerance for matching MD groups to pooled paths.
        force: Run even when the varied parameters exceed 2/3 of ``n_idp``.

    Returns:
        A JSON-safe dict with ``schema_version``, ``fit_type`` (``"static"``),
        ``fit_space``, ``ft_params``, ``rrange``, ``kweights``, ``rmax_valid``,
        ``sigma2_mode``, ``deltar_mode``, ``n_absorbers``, ``params``,
        ``paths``, ``dropped_paths``, ``truncation_fraction``, ``rfactor``,
        ``rfactor_r``, ``rfactor_k``, ``chi_square``, ``chi2_reduced``,
        ``n_idp``, ``nvarys``, ``success``, ``correlations``, ``warnings``,
        ``units``, ``experiment``, ``theory`` and, with ``archive_path``,
        ``comparison`` (``fit_a``, ``b0``, ``fit_b``).

    Raises:
        ValueError: Bad arguments, a single-frame static structure, no
            matched paths, or too many free parameters without ``force``.
    """
    if sigma2 not in SIGMA2_MODES:
        raise ValueError(f"sigma2 must be one of {SIGMA2_MODES}, got {sigma2!r}")
    if deltar_mode not in DELTAR_MODES:
        raise ValueError(
            f"deltar_mode must be one of {DELTAR_MODES}, got {deltar_mode!r}"
        )
    space = str(fit_space).lower()
    if space not in FIT_SPACES:
        raise ValueError(f"fit_space must be one of {FIT_SPACES}, got {fit_space!r}")
    kw_list = _resolve_kweights(kweights)
    frames = list(structures)
    warnings: list[str] = []
    static = _resolve_static_structure(structure, frames, warnings)

    rmax_valid = max(0.0, float(config.radius) - PHASE_SHIFT_ALLOWANCE_A)
    rmin, rmax = _resolve_rrange(rrange, rmax_valid)
    if rmax > rmax_valid:
        warnings.append(
            f"The fit window's rmax={rmax:g} A lies above rmax_valid="
            f"{rmax_valid:g} A (FEFF cluster radius {config.radius:g} A minus a "
            f"{PHASE_SHIFT_ALLOWANCE_A:g} A phase-shift allowance)."
        )

    fit_a: dict[str, Any] | None = None
    if archive_path is not None:
        fit_a = fit_ensemble(
            archive_path,
            experiment,
            ft_params=ft_params,
            rrange=(rmin, rmax),
            kweights=kw_list,
            fit_space=space,
        )
        ft = fit_a["ft_params"]
    else:
        defaults = _archive_ft_defaults({})
        ft = defaults if ft_params is None else _resolve_ft_request(ft_params, defaults)

    exp_k_raw, exp_chi_raw, _meta = read_experimental_spectrum(experiment)
    exp_k, exp_chi = _clean_spectrum(exp_k_raw, exp_chi_raw, "experiment")
    warnings.extend(
        w["message"].replace("overall average", "experimental spectrum")
        for w in _ft_data_warnings(ft, _coverage_limits({"k": exp_k, "chi": exp_chi}))
    )

    trans = TransformGroup(
        kmin=ft["kmin"],
        kmax=ft["kmax"],
        dk=ft["dk"],
        window=ft["window"],
        kweight=list(kw_list),
        rmin=rmin,
        rmax=rmax,
        fitspace=space,
    )
    trans.make_karrays()
    k_top = max(float(np.nanmax(exp_k)), ft["kmax"] + ft["dk"])
    k_fit = np.asarray(
        trans.k_[: min(len(trans.k_), int(1.01 + k_top / trans.kstep))], dtype=float
    )
    exp_fit = _onto_grid(exp_k, exp_chi, k_fit)
    eps_r, eps_k = _estimate_noise_weights(
        trans, k_fit, exp_fit, kw_list, space, warnings
    )

    indices = _resolve_indices(static, absorber, sites)
    element = str(static[indices[0]].symbol)
    all_paths, n_abs, mean_chi = static_paths(static, config, absorber, sites, workdir)
    pooled = pool_paths(
        all_paths, n_abs, r_tol=r_tol, angle_tol=angle_tol, absorber=element
    )

    if msrd is None:
        if not frames:
            raise ValueError("structures are required to compute the MSRD")
        md_idx = _resolve_indices(frames[0], absorber, sites)
        msrd = calculate_grouped_msrd(
            frames,
            md_idx,
            str(frames[0][md_idx[0]].symbol),
            cutoff=config.radius,
            cutoff_3body=config.radius,
            exclude_hydrogen=config.exclude_hydrogen,
        )
    res_2b, res_3b = msrd
    matched, match_warnings = _md_sigma2(pooled, res_2b, res_3b, match_r_tol)
    warnings.extend(match_warnings)
    kept = [p for gi, p in enumerate(pooled) if gi in matched]
    md_info = [matched[gi] for gi in range(len(pooled)) if gi in matched]
    dropped: list[dict[str, Any]] = [
        {
            "label": p.label,
            "scatterer": p.scatterer,
            "nlegs": p.nlegs,
            "r_eff": p.r_eff,
            "degeneracy": p.degeneracy,
            "reason": "no matching MD path group",
        }
        for gi, p in enumerate(pooled)
        if gi not in matched
    ]
    if dropped:
        warnings.append(
            f"{len(dropped)} pooled path(s) have no MD sigma^2 match and were "
            "dropped: " + ", ".join(d["label"] for d in dropped) + "."
        )
    if not kept:
        raise ValueError("No pooled path matches an MD path group")
    for p in kept:
        warnings.extend(f"{p.label}: {w}" for w in p.warnings)

    specs = _deltar_spec(kept, deltar_mode, match_r_tol, element)
    params = Parameters()
    params.add("s02", value=1.0, min=S02_BOUNDS[0], max=S02_BOUNDS[1], vary=True)
    params.add("e0", value=0.0, min=E0_BOUNDS[0], max=E0_BOUNDS[1], vary=True)
    for name in dict.fromkeys(s[0] for s in specs):
        lo, hi = ALPHA_BOUNDS if name == "alpha" else DR_BOUNDS
        params.add(name, value=0.0, min=lo, max=hi, vary=True)
    feff_paths = []
    for i, (p, info, (dname, coef, per_reff)) in enumerate(
        zip(kept, md_info, specs, strict=True)
    ):
        if sigma2 == "free":
            top = max(SIGMA2_MAX, 3.0 * info["sigma2"])
            params.add(f"sig_{i}", value=info["sigma2"], min=0.0, max=top, vary=True)
            s2: float | str = f"sig_{i}"
        else:
            s2 = info["sigma2"]
        feff_paths.append(
            to_feffit_path(
                p, s02="s02", e0="e0", deltar=_expr(dname, coef, per_reff), sigma2=s2
            )
        )
    start = {n: float(par.value) for n, par in params.items()}
    varied = [n for n, par in params.items() if par.vary]
    n_idp = float(1.0 + 2.0 * (rmax - rmin) * (ft["kmax"] - ft["kmin"]) / np.pi)
    if len(varied) > IDP_FRACTION_LIMIT * n_idp and not force:
        raise ValueError(
            f"{len(varied)} varied parameters exceed 2/3 of n_idp={n_idp:.1f}; "
            "reduce the model (sigma2='md', deltar_mode='isotropic') or pass "
            "force=True."
        )

    dataset = FeffitDataSet(
        data=Group(k=k_fit, chi=exp_fit), paths=feff_paths, transform=trans
    )

    def fixed_noise(*args: Any, **kws: Any) -> None:
        # feffit re-estimates the noise from high-R data on every call, which
        # fails on a noiseless spectrum and would differ from Fit A.
        dataset.epsilon_r, dataset.epsilon_k = list(eps_r), list(eps_k)

    dataset.estimate_noise = fixed_noise
    result = feffit(params, dataset)
    ds = result.datasets[0]
    theory = np.asarray(ds.model.chi, dtype=float)
    rfactors = _compute_rfactors(
        trans, k_fit, exp_fit, theory, kw_list, space, eps_r, eps_k
    )
    best = {n: float(result.params[n].value) for n in params}
    stderr = {n: _number(result.params[n].stderr) for n in params}
    correlations = _correlations(list(result.var_names or []), result.covar)
    seen: set[frozenset[str]] = set()
    for a, row in correlations.items():
        for b, corr in row.items():
            if abs(corr) > CORRELATION_WARN and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                warnings.append(
                    f"Parameters {a} and {b} are correlated (r={corr:.2f})."
                )
    if not result.success:
        warnings.append("The least-squares fit did not report success.")

    path_sigma2 = [
        best[f"sig_{i}"] if sigma2 == "free" else info["sigma2"]
        for i, info in enumerate(md_info)
    ]
    path_deltar = [
        coef * best[name] * (p.r_eff if per_reff else 1.0)
        for p, (name, coef, per_reff) in zip(kept, specs, strict=True)
    ]
    truncation = _truncation(kept, mean_chi, ft, (rmin, rmax))
    if mean_chi is None:
        warnings.append("No chi.dat was available for the path-truncation check.")
    elif truncation is not None and truncation > TRUNCATION_WARN:
        warnings.append(
            f"The kept paths reproduce the static FEFF chi(R) in the fit window "
            f"only to within {100 * truncation:.0f}%: dropped or truncated paths "
            "leave out that fraction of the static signal."
        )

    out: dict[str, Any] = {
        "schema_version": FIT_SCHEMA_VERSION,
        "fit_type": "static",
        "fit_space": space,
        "ft_params": ft,
        "rrange": [rmin, rmax],
        "kweights": kw_list,
        "rmax_valid": rmax_valid,
        "sigma2_mode": sigma2,
        "deltar_mode": deltar_mode,
        "n_absorbers": n_abs,
        "params": {
            n: {
                "value": _number(best[n]),
                "stderr": stderr[n],
                "start": start[n],
                "vary": n in varied,
            }
            for n in params
        },
        "paths": [
            {
                "label": p.label,
                "scatterer": p.scatterer,
                "nlegs": p.nlegs,
                "r_eff": p.r_eff,
                "r_eff_std": p.r_eff_std,
                "degeneracy": p.degeneracy,
                "count": p.count,
                "angle": p.angle,
                "sigma2_md": info["sigma2"],
                "sigma2": _number(s2v),
                "deltar": _number(dr),
                "md_groups": info["n_groups"],
                "match_distance": info["match_distance"],
                "warnings": list(p.warnings),
            }
            for p, info, s2v, dr in zip(
                kept, md_info, path_sigma2, path_deltar, strict=True
            )
        ],
        "dropped_paths": dropped,
        "truncation_fraction": truncation,
        **rfactors,
        "chi_square": _number(result.chi_square),
        "chi2_reduced": _number(result.chi2_reduced),
        "n_idp": n_idp,
        "nvarys": int(result.nvarys),
        "success": bool(result.success),
        "correlations": correlations,
        "warnings": warnings,
        "units": _units(ft["kweight"]),
        **{
            name: _spectrum(
                _with_transform({"k": k_fit, "chi": chi}, ft, True), ft["kweight"]
            )
            for name, chi in (("experiment", exp_fit), ("theory", theory))
        },
    }

    if fit_a is not None:
        a_s02 = _number(fit_a["params"]["s02"]["value"])
        a_e0 = _number(fit_a["params"]["e0_shift"]["value"])
        b0_s02, b0_e0 = (
            (1.0 if a_s02 is None else a_s02),
            (0.0 if a_e0 is None else a_e0),
        )
        b0_theory = _model_chi(
            kept,
            k_fit,
            s02=b0_s02,
            e0=b0_e0,
            sigma2=[info["sigma2"] for info in md_info],
            deltar=[0.0] * len(kept),
        )
        b0_blocks = np.concatenate(
            _transform_blocks(
                trans, k_fit, exp_fit - b0_theory, kw_list, space, eps_r, eps_k
            )
        )
        b0_chi2 = _chi_square(b0_blocks, n_idp)
        b0 = {
            "params": {"s02": b0_s02, "e0": b0_e0},
            **_compute_rfactors(
                trans, k_fit, exp_fit, b0_theory, kw_list, space, eps_r, eps_k
            ),
            "chi_square": b0_chi2,
            "chi2_reduced": (
                _number(b0_chi2 / n_idp) if b0_chi2 is not None and n_idp > 0 else None
            ),
            "nvarys": 0,
        }
        out["comparison"] = {
            "fit_a": _summary(fit_a),
            "b0": b0,
            "fit_b": _summary(out),
        }
    return out


__all__ = [
    "PooledPath",
    "fit_static",
    "pool_paths",
    "static_paths",
    "to_feffit_path",
]
