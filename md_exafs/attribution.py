"""Origin of chi(R) peaks: attribution to paths, sites and site groups.

:func:`peak_attribution` splits the overall chi(R) of an ensemble archive into
components (stored top paths, absorber sites, or groups of sites) and reports,
for every peak, how much of it each component builds. :func:`group_sites`
clusters absorber sites by the spectrum they produce, giving the groups
``peak_attribution(by="site_groups")`` consumes.

Every spectrum is built as chi(k) on the archive's overall k grid and transformed
with :func:`md_exafs.spectra.xftf_arrays` using the same parameters. The
transform is linear, so ``unassigned = total - sum(components)`` holds exactly
and the shares of all components plus ``unassigned`` sum to one.

Per peak, ``projection`` is ``sum_W Re(conj(T) C_j) / sum_W |T|^2`` over the peak
window ``W``. It is signed: a component that cancels part of the peak has a
negative projection (for example ``+1.0`` and ``-0.3`` for a pair of
components that add up to a peak of unit size and one that removes 30 % of it).
``loo_delta`` is the drop of the peak height when the component is removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import find_peaks

from .archive_series import (
    PHASE_SHIFT_ALLOWANCE_A,
    _archive_ft_defaults,
    _coverage_limits,
    _finite_number,
    _floats,
    _ft_data_warnings,
    _resolve_ft_request,
    _spectrum,
    _units,
)
from .experimental import scaled_chi_arrays
from .hdf5 import ArchiveReader
from .spectra import resample_chi, xftf_arrays

ATTRIBUTION_SCHEMA_VERSION = 1

DECOMPOSITIONS = ("paths", "sites", "site_groups")

# Half-width (A) cap of the window a peak's shares are summed over.
PEAK_WINDOW_HALF_WIDTH_A = 0.5

# Unassigned fraction of |chi(R)| above which a path decomposition is flagged.
UNASSIGNED_WARN_FRACTION = 0.10

# Multiple of the estimated pairwise site-sampling noise floor below which
# hierarchical merges are treated as indistinguishable from sampling noise.
SITE_NOISE_MARGIN = 2.0


@dataclass
class _Context:
    reader: ArchiveReader
    overall: dict[str, np.ndarray]
    k: np.ndarray
    ft: dict[str, Any]
    rrange: tuple[float, float]
    rmax_valid: float | None
    warnings: list[str]
    r: np.ndarray
    total: np.ndarray
    in_band: np.ndarray


class _Sites(NamedTuple):
    ids: list[int]
    chi: np.ndarray  # (n_sites, n_k)
    w: np.ndarray  # per-k weight n_s / n_tot, or 1 / n_sites
    n: np.ndarray | None  # per-k contributor counts, None on legacy archives


def _ft_complex(
    k: np.ndarray, chi: np.ndarray, ft: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    res = xftf_arrays(k, chi, ft, warn_uncovered=False)
    return res["r"], res["chir_re"] + 1j * res["chir_im"]


def _on_grid(k_src: np.ndarray, values: np.ndarray, k: np.ndarray) -> np.ndarray:
    k_src = np.asarray(k_src, dtype=float)
    if k_src.shape == k.shape and np.allclose(k_src, k):
        return np.asarray(values, dtype=float)
    return resample_chi(k_src, np.asarray(values, dtype=float), k)


def _resolve_rrange(
    rrange: tuple[float, float] | None, rmax_valid: float | None
) -> tuple[float, float]:
    if rrange is None:
        rmin = 1.0
        rmax = 3.0 if rmax_valid is None else min(3.0, rmax_valid)
    else:
        try:
            rmin, rmax = (float(v) for v in rrange)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"rrange must be a (rmin, rmax) pair, got {rrange!r}"
            ) from exc
    if not (math.isfinite(rmin) and math.isfinite(rmax) and 0.0 <= rmin < rmax):
        raise ValueError(f"rrange must satisfy 0 <= rmin < rmax, got {(rmin, rmax)!r}")
    return rmin, rmax


def _resolve(
    path: Path | str,
    ft_params: dict[str, Any] | None,
    rrange: tuple[float, float] | None,
) -> _Context:
    reader = ArchiveReader(path)
    if not reader.is_ensemble:
        raise ValueError(f"{reader.path} is not an ensemble archive")
    try:
        overall = reader.get_overall_average()
    except KeyError as exc:
        raise ValueError(f"{reader.path} has no overall average") from exc

    warnings: list[str] = []
    stored = reader.fourier_params
    if not stored:
        warnings.append(
            "The archive records no Fourier-transform parameters; library "
            "defaults are assumed."
        )
    ft = _resolve_ft_request(ft_params or {}, _archive_ft_defaults(stored))

    radius = reader.radius
    rmax_valid = (
        max(0.0, float(radius - PHASE_SHIFT_ALLOWANCE_A))
        if radius is not None
        else None
    )
    if rmax_valid is None:
        warnings.append(
            "The archive records no FEFF cluster radius, so rmax_valid cannot "
            "be determined."
        )
    window = _resolve_rrange(rrange, rmax_valid)
    if rmax_valid is not None and window[1] > rmax_valid:
        warnings.append(
            f"rrange rmax={window[1]:g} A lies above rmax_valid={rmax_valid:g} A "
            "(FEFF cluster radius minus a phase-shift allowance); chi(R) there is "
            "not reliable because the cluster is truncated."
        )
    warnings.extend(
        w["message"] for w in _ft_data_warnings(ft, _coverage_limits(overall))
    )

    k = np.asarray(overall["k"], dtype=float)
    r, total = _ft_complex(k, overall["chi"], ft)
    in_band = (r >= window[0]) & (r <= window[1])
    if window[1] > float(r[-1]):
        warnings.append(
            f"rrange rmax={window[1]:g} A lies beyond the transform's R grid "
            f"({float(r[-1]):g} A); raise ft_params['rmax']."
        )
    return _Context(
        reader, overall, k, ft, window, rmax_valid, warnings, r, total, in_band
    )


def _site_arrays(ctx: _Context) -> _Sites:
    ids = ctx.reader.site_indices
    if not ids:
        raise ValueError(f"{ctx.reader.path} has no per-site averages")
    k = ctx.k
    chi = np.empty((len(ids), k.size))
    counts: list[np.ndarray] = []
    have_n = "n_contributors" in ctx.overall
    for i, s in enumerate(ids):
        data = ctx.reader.get_site_average(s)
        chi[i] = _on_grid(data["k"], data["chi"], k)
        if "n_contributors" in data:
            counts.append(np.nan_to_num(_on_grid(data["k"], data["n_contributors"], k)))
        else:
            have_n = False
    if have_n:
        n = np.asarray(counts, dtype=float)
        n_tot = np.asarray(ctx.overall["n_contributors"], dtype=float)
        safe = np.where(n_tot > 0, n_tot, 1.0)
        return _Sites(ids, chi, np.where(n_tot > 0, n / safe, 0.0), n)
    ctx.warnings.append(
        "The archive lacks per-k contributor counts (legacy format), so sites "
        "are weighted equally; the components may not sum to the overall average."
    )
    return _Sites(ids, chi, np.full(chi.shape, 1.0 / len(ids)), None)


def _site_components(sites: _Sites) -> np.ndarray:
    """Per-site contribution w_s(k) * chi_s(k) to the overall chi(k)."""
    return np.where((sites.w > 0) & np.isfinite(sites.chi), sites.w * sites.chi, 0.0)


def _scalar_weights(sites: _Sites) -> np.ndarray:
    n_sites = len(sites.ids)
    if sites.n is None:
        return np.full(n_sites, 1.0 / n_sites)
    valid = sites.w.sum(axis=0) > 0
    if not valid.any():
        return np.full(n_sites, 1.0 / n_sites)
    w = sites.w[:, valid].mean(axis=1)
    total = float(w.sum())
    return w / total if total > 0 else np.full(n_sites, 1.0 / n_sites)


def _mean_chi(sites: _Sites, idx: list[int]) -> np.ndarray:
    chi = sites.chi[idx]
    if sites.n is not None:
        n = sites.n[idx]
        ok = (n > 0) & np.isfinite(chi)
        weight = np.where(ok, n, 0.0)
    else:
        ok = np.isfinite(chi)
        weight = ok.astype(float)
    den = weight.sum(axis=0)
    num = np.where(ok, weight * chi, 0.0).sum(axis=0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _features(vec: np.ndarray, in_band: np.ndarray) -> np.ndarray:
    return np.concatenate([vec.real[in_band], vec.imag[in_band]])


def _adjust_chi(
    k: np.ndarray, chi: np.ndarray, s02: float, e0_shift: float
) -> np.ndarray:
    """Apply comparison-only S0^2 and delta-E0 scaling on the archive k grid."""
    if s02 == 1.0 and e0_shift == 0.0:
        return np.asarray(chi, dtype=float)
    ka, ca = scaled_chi_arrays(k, np.nan_to_num(chi), s02=s02, e0_shift=e0_shift)
    return resample_chi(ka, ca, k)


def _site_noise_floor(ctx: _Context, n_sites: int) -> float | None:
    """Expected pairwise RMS distance between two noise-only site averages.

    Estimated from frame-to-frame variance of the stored frame averages
    assuming independent site fluctuations within each frame:
    ``sqrt(2 * n_sites * sum_j var(frame_feat_j) / n_frames)``.
    Returns ``None`` when fewer than two frame averages are stored.
    """
    frame_ids = ctx.reader.frame_indices
    if len(frame_ids) < 2:  # noqa: PLR2004
        return None
    feats = []
    for f_idx in frame_ids:
        data = ctx.reader.get_frame_average(f_idx)
        chi_f = np.nan_to_num(_on_grid(data["k"], data["chi"], ctx.k))
        _, t_f = _ft_complex(ctx.k, chi_f, ctx.ft)
        feats.append(_features(t_f, ctx.in_band))
    mat = np.asarray(feats, dtype=float)
    tr_var = float(np.sum(np.var(mat, axis=0, ddof=1)))
    if not (math.isfinite(tr_var) and tr_var >= 0.0):
        return None
    return float(math.sqrt(2.0 * n_sites * tr_var / len(frame_ids)))


def group_sites(
    path: Path | str,
    *,
    ft_params: dict[str, Any] | None = None,
    rrange: tuple[float, float] | None = None,
    tol: float = 0.15,
    n_groups: int | None = None,
) -> dict[str, Any]:
    """Cluster absorber sites by the chi(R) spectrum each produces.

    Features are the stacked Re and Im of chi(R) of each site over ``rrange``.
    Sites are joined by average-linkage hierarchical clustering. Without
    ``n_groups`` the dendrogram is cut at ``tol`` times the RMS of the
    individual site feature norms (not the norm of the overall spectrum, which
    collapses under destructive interference between sites).

    Args:
        path: Ensemble archive.
        ft_params: Fourier-transform overrides, as in
            :func:`md_exafs.archive_series.archive_series`.
        rrange: R band (A) of the features; default ``(1, min(3, rmax_valid))``.
        tol: Relative distance threshold, a finite number >= 0.
        n_groups: Number of groups to cut to instead of ``tol``.

    Returns:
        A JSON-safe dict with ``schema_version``, ``units``, ``ft_params``,
        ``rrange``, ``rmax_valid``, ``tol`` (``None`` when ``n_groups`` is
        given), ``n_groups``, ``n_sites``, ``groups`` and ``warnings``. Each
        group has ``group_id``, ``site_indices``, ``n_sites``, ``weight`` (sums
        to 1 over groups), ``medoid_site``, ``within_group_rms``,
        ``distance_to_overall`` and ``spectrum`` (group-mean chi(k) and chi(R)).

    Raises:
        ValueError: Not an ensemble archive, no overall average or sites, or
            invalid ``rrange``, ``tol`` or ``n_groups``.
    """
    ctx = _resolve(path, ft_params, rrange)
    sites = _site_arrays(ctx)
    n_sites = len(sites.ids)

    if n_groups is not None:
        if isinstance(n_groups, bool) or not isinstance(n_groups, int | np.integer):
            raise ValueError(f"n_groups must be an integer, got {n_groups!r}")
        if not 1 <= n_groups <= n_sites:
            raise ValueError(f"n_groups must be in [1, {n_sites}], got {n_groups}")
    elif not (
        isinstance(tol, int | float | np.number) and math.isfinite(tol) and tol >= 0
    ):
        raise ValueError(f"tol must be a finite number >= 0, got {tol!r}")

    ft = ctx.ft
    x = np.array(
        [_features(_ft_complex(ctx.k, c, ft)[1], ctx.in_band) for c in sites.chi]
    )
    overall_x = _features(ctx.total, ctx.in_band)

    noise_floor = _site_noise_floor(ctx, n_sites) if n_sites >= 2 else None  # noqa: PLR2004
    if n_sites >= 2 and noise_floor is None:  # noqa: PLR2004
        ctx.warnings.append(
            "The archive stores fewer than 2 frame averages, so site grouping "
            "cannot be validated against trajectory sampling noise."
        )

    if n_sites < 2:
        labels = np.ones(n_sites, dtype=int)
    else:
        z = linkage(x, method="average")
        if n_groups is not None:
            labels = fcluster(z, t=int(n_groups), criterion="maxclust")
            if int(n_groups) > 1 and noise_floor is not None:
                min_split = float(z[n_sites - int(n_groups), 2])
                sig_thr = SITE_NOISE_MARGIN * noise_floor
                if min_split < sig_thr:
                    ctx.warnings.append(
                        f"Requested n_groups={int(n_groups)} splits sites at "
                        f"distance {min_split:.3g}, below {SITE_NOISE_MARGIN:g}x "
                        f"the sampling-noise floor ({sig_thr:.3g}); the split is "
                        "not statistically significant."
                    )
        else:
            rms_norm = float(np.sqrt(np.mean(np.sum(x**2, axis=1))))
            tol_thr = float(tol) * rms_norm
            noise_thr = (
                SITE_NOISE_MARGIN * noise_floor if noise_floor is not None else 0.0
            )
            thr = max(tol_thr, noise_thr)
            if noise_floor is not None and noise_thr > tol_thr:
                ctx.warnings.append(
                    f"tol={float(tol):g} (threshold {tol_thr:.3g}) is below "
                    f"{SITE_NOISE_MARGIN:g}x the sampling-noise floor "
                    f"({noise_thr:.3g}); effective threshold raised to merge "
                    "noise-indistinguishable sites."
                )
            labels = fcluster(z, t=thr, criterion="distance")

    weights = _scalar_weights(sites)
    groups: list[dict[str, Any]] = []
    for label in np.unique(labels):
        members = [int(i) for i in np.flatnonzero(labels == label)]
        xm = x[members]
        dist = np.linalg.norm(xm[:, None, :] - xm[None, :, :], axis=2)
        medoid = members[int(np.argmin(dist.sum(axis=1)))]
        mean_x = xm.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum((xm - mean_x) ** 2, axis=1))))
        chi_g = _mean_chi(sites, members)
        r, t_g = _ft_complex(ctx.k, chi_g, ft)
        groups.append(
            {
                "site_indices": sorted(sites.ids[i] for i in members),
                "n_sites": len(members),
                "weight": float(weights[members].sum()),
                "medoid_site": sites.ids[medoid],
                "within_group_rms": rms if len(members) > 1 else 0.0,
                "distance_to_overall": float(
                    np.linalg.norm(_features(t_g, ctx.in_band) - overall_x)
                ),
                "spectrum": _spectrum(
                    {
                        "k": ctx.k,
                        "chi": chi_g,
                        "r": r,
                        "chir_mag": np.abs(t_g),
                        "chir_re": t_g.real,
                        "chir_im": t_g.imag,
                    },
                    ft["kweight"],
                ),
            }
        )
    groups.sort(key=lambda g: (-g["weight"], g["site_indices"][0]))
    groups = [{"group_id": f"group_{i}", **g} for i, g in enumerate(groups)]

    singletons = [g["site_indices"][0] for g in groups if g["n_sites"] < 2]
    if singletons:
        ctx.warnings.append(
            f"Sites {singletons} form groups of fewer than 2 sites. Site averages "
            "carry no per-site uncertainty, so sparsely sampled sites can form "
            "spurious groups."
        )

    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "units": _units(ft["kweight"]),
        "ft_params": ft,
        "rrange": list(ctx.rrange),
        "rmax_valid": ctx.rmax_valid,
        "tol": None if n_groups is not None else float(tol),
        "noise_floor": noise_floor,
        "n_groups": len(groups),
        "n_sites": n_sites,
        "groups": groups,
        "warnings": ctx.warnings,
    }


def _path_components(ctx: _Context) -> list[dict[str, Any]]:
    version = ctx.reader.format_version
    if version is None or version < 4:
        raise ValueError(
            f"by='paths' needs archive format version >= 4 (this archive is "
            f"version {version}): older or raw pipeline archives do not store "
            "ensemble-weighted per-path chi(k). Run 'md-exafs export-ensemble' "
            "(or md_exafs.execution.export_ensemble) on results.h5, or use by='sites'."
        )
    paths = ctx.reader.get_top_paths()
    comps: list[dict[str, Any]] = []
    for p in paths:
        if "chi" not in p:
            continue
        chi = np.asarray(p["chi"], dtype=float)
        if chi.shape != ctx.k.shape:
            raise ValueError(
                f"top path {p.get('path_key')!r} has {chi.size} k points; the "
                f"overall k grid has {ctx.k.size}"
            )
        meta: dict[str, Any] = {"scatterer": str(p.get("scatterer", ""))}
        for key in ("nlegs", "count"):
            if key in p:
                meta[key] = int(p[key])
        for key in ("r_eff", "degeneracy", "cw_ratio", "sig2", "angle"):
            if key in p:
                meta[key] = float(p[key])
        key_id = str(p["path_key"])
        comps.append({"id": key_id, "label": key_id, "metadata": meta, "chi": chi})
    if not comps:
        raise ValueError(f"{ctx.reader.path} stores no top paths with chi(k)")
    return comps


def _group_site_lists(groups: Any, site_ids: list[int]) -> list[list[int]]:
    if isinstance(groups, dict):
        seq = [list(g["site_indices"]) for g in groups["groups"]]
    else:
        seq = [list(g) for g in groups]
    known = set(site_ids)
    seen: set[int] = set()
    out: list[list[int]] = []
    for g in seq:
        members = sorted(int(s) for s in g)
        if not members:
            raise ValueError("site groups must not be empty")
        missing = [s for s in members if s not in known]
        if missing:
            raise ValueError(f"site(s) {missing} are not in the archive")
        dup = [s for s in members if s in seen]
        if dup:
            raise ValueError(f"site(s) {dup} appear in more than one group")
        seen.update(members)
        out.append(members)
    if not out:
        raise ValueError("groups is empty")
    return out


def _metrics(
    t: np.ndarray, c: np.ndarray, window: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[float | None]]:
    """Projection, leave-one-out delta and R of maximum for rows of ``c``."""
    tw = t[window]
    cw = c[:, window]
    denom = float(np.sum(np.abs(tw) ** 2))
    proj = np.real(np.conj(tw)[None, :] * cw).sum(axis=1) / denom
    loo = np.abs(tw).max() - np.abs(tw[None, :] - cw).max(axis=1)
    amp = np.abs(cw)
    rw = r[window]
    r_max = [float(rw[int(np.argmax(a))]) if a.max() > 0 else None for a in amp]
    return proj, loo, r_max


def _peak_window(r: np.ndarray, minima: np.ndarray, p: int) -> np.ndarray:
    left = minima[minima < p]
    right = minima[minima > p]
    lo = int(left[-1]) if left.size else 0
    hi = int(right[0]) if right.size else len(r) - 1
    lo = max(lo, int(np.searchsorted(r, r[p] - PEAK_WINDOW_HALF_WIDTH_A, "left")))
    hi = min(hi, int(np.searchsorted(r, r[p] + PEAK_WINDOW_HALF_WIDTH_A, "right")) - 1)
    return np.arange(lo, hi + 1)


def _curves(r: np.ndarray, z: np.ndarray) -> dict[str, list[float | None]]:
    return {
        "chir_mag": _floats(np.abs(z)),
        "chir_re": _floats(z.real),
        "chir_im": _floats(z.imag),
    }


def peak_attribution(
    path: Path | str,
    *,
    by: str = "paths",
    ft_params: dict[str, Any] | None = None,
    rrange: tuple[float, float] | None = None,
    prominence: float = 0.05,
    max_components: int = 20,
    groups: Any = None,
    s02: float = 1.0,
    e0_shift: float = 0.0,
) -> dict[str, Any]:
    """Attribute each chi(R) peak of the overall average to its components.

    Args:
        path: Ensemble archive.
        by: ``"paths"`` (stored top paths, archive format >= 4),
            ``"sites"`` (absorber sites) or ``"site_groups"``.
        ft_params: Fourier-transform overrides, as in
            :func:`md_exafs.archive_series.archive_series`.
        rrange: R band (A) peaks are searched in; default
            ``(1, min(3, rmax_valid))``.
        prominence: Peak prominence as a fraction of the largest |chi(R)| in
            ``rrange``; must be > 0.
        max_components: Components kept with curves and per-peak shares; the
            rest (ranked by largest absolute projection) are summed into
            ``remainder``.
        groups: For ``by="site_groups"``, the output of :func:`group_sites` or
            a sequence of site-index sequences. ``None`` runs
            :func:`group_sites` with defaults.
        s02: Comparison-only amplitude reduction factor applied identically to
            total and all components (must be finite and > 0; default 1.0).
        e0_shift: Comparison-only threshold energy shift (eV) applied
            identically to total and all components (default 0.0).

    Returns:
        A JSON-safe dict with ``schema_version``, ``units``, ``ft_params``,
        ``rrange``, ``rmax_valid``, ``adjustment``, ``source``,
        ``decomposition``, ``r``, ``total``, ``components``, ``remainder``,
        ``unassigned``, ``peaks`` and ``warnings``. Each peak has ``r_peak``,
        ``window``, ``height``, ``shoulder_warning``, ``shares`` (``id``,
        ``projection``, ``loo_delta``, ``r_max`` per kept component),
        ``unassigned`` and, when components were truncated, ``remainder``.
        Projections of the shares, remainder and unassigned sum to 1.

    Raises:
        ValueError: Bad arguments, not an ensemble archive, or ``by="paths"`` on
            an archive older than format version 4.
    """
    if by not in DECOMPOSITIONS:
        raise ValueError(f"by must be one of {DECOMPOSITIONS}, got {by!r}")
    if not (
        isinstance(prominence, int | float)
        and math.isfinite(prominence)
        and prominence > 0
    ):
        raise ValueError(f"prominence must be a finite number > 0, got {prominence!r}")
    if isinstance(max_components, bool) or max_components < 1:
        raise ValueError(f"max_components must be >= 1, got {max_components!r}")
    if not (_finite_number(s02) and float(s02) > 0.0):
        raise ValueError(f"s02 must be a finite number > 0, got {s02!r}")
    if not _finite_number(e0_shift):
        raise ValueError(f"e0_shift must be a finite number, got {e0_shift!r}")
    s02_f, e0_f = float(s02), float(e0_shift)

    ctx = _resolve(path, ft_params, rrange)
    ft, k, r, total = ctx.ft, ctx.k, ctx.r, ctx.total
    if s02_f != 1.0 or e0_f != 0.0:
        r, total = _ft_complex(k, _adjust_chi(k, ctx.overall["chi"], s02_f, e0_f), ft)
    reader = ctx.reader

    comps: list[dict[str, Any]]
    if by == "paths":
        comps = _path_components(ctx)
    else:
        sites = _site_arrays(ctx)
        site_chi = _site_components(sites)
        if by == "sites":
            comps = [
                {
                    "id": f"site_{s}",
                    "label": f"site {s}",
                    "metadata": {"site_idx": s},
                    "chi": site_chi[i],
                }
                for i, s in enumerate(sites.ids)
            ]
        else:
            if groups is None:
                groups = group_sites(path, ft_params=ft_params, rrange=rrange)
            pos = {s: i for i, s in enumerate(sites.ids)}
            comps = []
            for idx, members in enumerate(_group_site_lists(groups, sites.ids)):
                comps.append(
                    {
                        "id": f"group_{idx}",
                        "label": f"group {idx} ({len(members)} sites)",
                        "metadata": {"site_indices": members, "n_sites": len(members)},
                        "chi": site_chi[[pos[s] for s in members]].sum(axis=0),
                    }
                )

    cmat = np.array(
        [_ft_complex(k, _adjust_chi(k, c["chi"], s02_f, e0_f), ft)[1] for c in comps]
    )
    unassigned = total - cmat.sum(axis=0)
    band = ctx.in_band
    t_norm = float(np.linalg.norm(total[band]))
    frac = float(np.linalg.norm(unassigned[band]) / t_norm) if t_norm > 0 else 0.0
    if by == "paths" and frac > UNASSIGNED_WARN_FRACTION:
        ctx.warnings.append(
            f"The stored paths leave {frac:.0%} of |chi(R)| in the window "
            "unassigned: the stored path sets are often only a subset of all "
            "paths, so this share belongs to paths that are not stored."
        )

    abs_t = np.abs(total)
    band_max = float(abs_t[band].max()) if band.any() else 0.0
    peak_idx: list[int] = []
    if band_max > 0:
        found, _ = find_peaks(abs_t, prominence=prominence * band_max)
        rmin, rmax = ctx.rrange
        peak_idx = [int(p) for p in found if rmin <= r[p] <= rmax]
    minima, _ = find_peaks(-abs_t)

    windows = [_peak_window(r, minima, p) for p in peak_idx]
    shoulder: list[bool] = []
    for p, w in zip(peak_idx, windows, strict=True):
        d2 = np.diff(abs_t[w], n=2)
        signs = np.sign(d2[d2 != 0])
        flag = bool(np.sum(signs[1:] != signs[:-1]) > 2)
        shoulder.append(flag)
        if flag:
            ctx.warnings.append(
                f"The peak at R={r[p]:.2f} A has more than two inflection points "
                "in its window; a shoulder may be absorbed into it."
            )

    per_peak = [_metrics(total, cmat, w, r) for w in windows]
    per_peak_u = [_metrics(total, unassigned[None, :], w, r) for w in windows]
    if per_peak:
        score = np.max(np.abs(np.array([m[0] for m in per_peak])), axis=0)
    else:
        score = np.linalg.norm(cmat[:, band], axis=1)
    order = [int(j) for j in np.argsort(-score, kind="stable")]
    kept, rest = order[:max_components], order[max_components:]

    remainder: dict[str, Any] | None = None
    c_rem = cmat[rest].sum(axis=0) if rest else None
    if rest and c_rem is not None:
        remainder = {
            "count": len(rest),
            "ids": [comps[j]["id"] for j in rest],
            **_curves(r, c_rem),
        }

    peaks: list[dict[str, Any]] = []
    for p, w, flag, (proj, loo, rmx), (uproj, uloo, urmx) in zip(
        peak_idx, windows, shoulder, per_peak, per_peak_u, strict=True
    ):
        entry: dict[str, Any] = {
            "r_peak": float(r[p]),
            "window": [float(r[w[0]]), float(r[w[-1]])],
            "height": float(abs_t[p]),
            "shoulder_warning": flag,
            "shares": [
                {
                    "id": comps[j]["id"],
                    "projection": float(proj[j]),
                    "loo_delta": float(loo[j]),
                    "r_max": rmx[j],
                }
                for j in sorted(kept, key=lambda idx: -abs(float(proj[idx])))
            ],
            "unassigned": {
                "projection": float(uproj[0]),
                "loo_delta": float(uloo[0]),
                "r_max": urmx[0],
            },
        }
        if c_rem is not None:
            rp, rl, rr = _metrics(total, c_rem[None, :], w, r)
            entry["remainder"] = {
                "projection": float(rp[0]),
                "loo_delta": float(rl[0]),
                "r_max": rr[0],
            }
        peaks.append(entry)

    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "units": _units(ft["kweight"]),
        "ft_params": ft,
        "rrange": list(ctx.rrange),
        "rmax_valid": ctx.rmax_valid,
        "adjustment": {"s02": s02_f, "e0_shift": e0_f},
        "source": {
            "n_sites": len(reader.site_indices),
            "n_frames": len(reader.frame_indices),
        },
        "decomposition": by,
        "r": _floats(r),
        "total": _curves(r, total),
        "components": [
            {
                "id": comps[j]["id"],
                "label": comps[j]["label"],
                "metadata": comps[j]["metadata"],
                **_curves(r, cmat[j]),
            }
            for j in kept
        ],
        "remainder": remainder,
        "unassigned": {"norm_fraction": frac, **_curves(r, unassigned)},
        "peaks": peaks,
        "warnings": ctx.warnings,
    }


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "group_sites",
    "peak_attribution",
]
