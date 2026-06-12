#!/usr/bin/env python3
"""Filter / slim an EXAFS results HDF5 file.

Copies a source HDF5 file to a new destination while excluding groups or
datasets whose paths match one or more glob-style patterns.  All metadata,
compression, and chunk settings are preserved via ``h5py.copy``.

Typical usage
-------------
Drop per-path chi(k) contributions (the usual culprit for large files) but
keep everything else (frames, site k/chi, aggregates, meta)::

    python scripts/filter_h5.py results.h5 results_slim.h5 \\
        --exclude "frames/*/sites/*/paths"

Keep only aggregates and metadata (smallest possible file)::

    python scripts/filter_h5.py results.h5 results_agg.h5 \\
        --include-only "meta" "aggregates"

HDF5 path pattern syntax
-------------------------
Patterns are matched against the *full* HDF5 path of every group/dataset
(e.g. ``frames/frame_0000/sites/site_0001/paths``).

- ``*``  matches any **single** path component  (not a separator ``/``)
- ``**`` matches **zero or more** path components (recursive wildcard)

Examples
--------
``frames/*/sites/*/paths``    – exclude all ``paths`` groups under any site
``frames/frame_0000/**``      – exclude everything under frame_0000
``aggregates/site_averages``  – exclude just the site_averages aggregate
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern matching helpers
# ---------------------------------------------------------------------------

def _glob_to_regex(pattern: str) -> re.Pattern:
    """Convert a path-glob pattern with ``*`` / ``**`` to a compiled regex.

    ``*``  → matches one path segment (no ``/``)
    ``**`` → matches any number of segments (including zero)
    """
    # Escape everything, then restore our wildcards.
    escaped = re.escape(pattern)
    # Order matters: replace escaped ** first.
    escaped = escaped.replace(r"\*\*", r"[^/]+(?:/[^/]+)*")
    escaped = escaped.replace(r"\*", r"[^/]+")
    return re.compile(r"^" + escaped + r"(/.*)?$")


def _matches_any(hdf5_path: str, patterns: list[re.Pattern]) -> bool:
    """Return True if *hdf5_path* matches any compiled pattern."""
    # Strip leading slash for normalisation.
    path = hdf5_path.lstrip("/")
    return any(p.match(path) for p in patterns)


# Matches  frames/frame_XXXX/sites/site_XXXX/paths/path_NNNN
_PATH_LEAF_RE = re.compile(
    r"^frames/frame_\d+/sites/site_\d+/paths/path_(\d+)$"
)


def _exceeds_max_paths(hdf5_path: str, max_paths: int) -> bool:
    """Return True if this HDF5 path is a per-path group beyond *max_paths*."""
    m = _PATH_LEAF_RE.match(hdf5_path.lstrip("/"))
    if m:
        return int(m.group(1)) > max_paths
    return False


# Regex patterns that identify k datasets at site and path level.
_SITE_K_RE = re.compile(r"^frames/frame_\d+/sites/site_\d+/k$")
_PATH_K_RE = re.compile(r"^frames/frame_\d+/sites/site_\d+/paths/path_\d+/k$")


class _KGrids(NamedTuple):
    """Common k grids discovered by scanning the source file."""

    site_k: np.ndarray | None  # shared k grid for all site chi arrays
    path_k: np.ndarray | None  # shared k grid for all per-path chi arrays


def _scan_k_grids(h5src, max_paths: int | None = None, rtol: float = 1e-6) -> _KGrids:
    """Scan *h5src* and return the common k grids for sites and paths.

    Returns ``None`` for a level if the k arrays are not sufficiently
    uniform (warns in that case).
    """
    site_ref: np.ndarray | None = None
    path_ref: np.ndarray | None = None
    site_ok = True
    path_ok = True
    site_count = path_count = 0

    def _visit(name, obj):
        nonlocal site_ref, path_ref, site_ok, path_ok, site_count, path_count
        import h5py
        if not isinstance(obj, h5py.Dataset):
            return
        k = obj[()]
        if _SITE_K_RE.match(name):
            site_count += 1
            if site_ref is None:
                site_ref = k
            elif site_ok and not np.allclose(site_ref, k, rtol=rtol, atol=0):
                logger.warning(
                    f"Site k arrays are not uniform (first mismatch at {name!r}).  "
                    "--dedup-k will NOT deduplicate site k arrays."
                )
                site_ok = False
        elif _PATH_K_RE.match(name):
            if max_paths is not None:
                # Skip paths that would be excluded anyway
                m = _PATH_LEAF_RE.match(name[: name.rfind("/")])
                if m and int(m.group(1)) > max_paths:
                    return
            path_count += 1
            if path_ref is None:
                path_ref = k
            elif path_ok and not np.allclose(path_ref, k, rtol=rtol, atol=0):
                logger.warning(
                    f"Path k arrays are not uniform (first mismatch at {name!r}).  "
                    "--dedup-k will NOT deduplicate path k arrays."
                )
                path_ok = False

    h5src.visititems(_visit)

    if site_count:
        logger.info(f"  Site k arrays: {site_count} found, "
                    + (f"all identical ({len(site_ref)} points, "
                       f"{site_ref[0]:.4f}–{site_ref[-1]:.4f} Å⁻¹)"
                       if site_ok and site_ref is not None else "NOT uniform – kept as-is"))
    if path_count:
        logger.info(f"  Path k arrays: {path_count} found, "
                    + (f"all identical ({len(path_ref)} points, "
                       f"{path_ref[0]:.4f}–{path_ref[-1]:.4f} Å⁻¹)"
                       if path_ok and path_ref is not None else "NOT uniform – kept as-is"))

    return _KGrids(
        site_k=site_ref if site_ok else None,
        path_k=path_ref if path_ok else None,
    )


# ---------------------------------------------------------------------------
# Recursive copy
# ---------------------------------------------------------------------------

def _copy_filtered(
    src_group,
    dst_group,
    exclude_patterns: list[re.Pattern],
    include_patterns: list[re.Pattern],
    *,
    root: str = "",
    stats: dict,
    max_paths: int | None = None,
    k_grids: _KGrids | None = None,
) -> None:
    """Recursively copy *src_group* into *dst_group*, honouring filters.

    Parameters
    ----------
    src_group, dst_group:
        Open h5py Group objects.
    exclude_patterns:
        Items whose HDF5 path matches any of these are skipped entirely.
    include_patterns:
        When non-empty, *only* items whose HDF5 path starts with one of these
        prefixes are copied; everything else is skipped.
    root:
        HDF5 path prefix accumulated during recursion (no leading slash).
    stats:
        Mutable dict accumulating ``copied`` / ``skipped`` counters.
    max_paths:
        When set, skip any ``paths/path_NNNN`` group whose index exceeds this
        value.  FEFF paths are ordered by increasing r_eff (shorter = more
        important), so keeping the first N preserves the dominant shells.
    k_grids:
        When non-None, skip ``k`` datasets at the corresponding level and
        rely on the shared grid stored in ``meta/``.
    """
    import h5py

    # Copy group attributes first.
    for attr_key, attr_val in src_group.attrs.items():
        try:
            dst_group.attrs[attr_key] = attr_val
        except Exception as exc:
            logger.warning(f"Could not copy attr '{attr_key}' on '{root}': {exc}")

    for name, item in src_group.items():
        item_path = f"{root}/{name}".lstrip("/")

        # ------------------------------------------------------------------
        # max_paths filter  (applied before pattern filters)
        # ------------------------------------------------------------------
        if max_paths is not None and _exceeds_max_paths(item_path, max_paths):
            logger.debug(f"  SKIP (max_paths={max_paths}): {item_path}")
            stats["skipped"] += 1
            continue

        # ------------------------------------------------------------------
        # dedup-k filter – skip k datasets that are stored once in meta/
        # ------------------------------------------------------------------
        if k_grids is not None and isinstance(item, h5py.Dataset):
            if k_grids.site_k is not None and _SITE_K_RE.match(item_path):
                logger.debug(f"  SKIP (dedup-k site): {item_path}")
                stats["skipped"] += 1
                continue
            if k_grids.path_k is not None and _PATH_K_RE.match(item_path):
                logger.debug(f"  SKIP (dedup-k path): {item_path}")
                stats["skipped"] += 1
                continue

        # ------------------------------------------------------------------
        # Exclusion filter
        # ------------------------------------------------------------------
        if exclude_patterns and _matches_any(item_path, exclude_patterns):
            logger.debug(f"  SKIP (excluded): {item_path}")
            stats["skipped"] += 1
            continue

        # ------------------------------------------------------------------
        # Include-only filter: skip items that do NOT match any include prefix
        # ------------------------------------------------------------------
        if include_patterns:
            path_norm = item_path.rstrip("/")
            # Keep the item if any include pattern matches it OR if the item
            # is an ancestor of an include pattern (so we can descend into it).
            matched = _matches_any(item_path, include_patterns)
            # Also keep if it is a *prefix* of any include pattern
            # (i.e. we need to descend into it to reach an included item).
            is_ancestor = any(
                re.match(r"^" + re.escape(path_norm) + r"(/|$)", p.pattern.lstrip("^").rstrip("(/.*)?$"))
                for p in include_patterns
            )
            if not matched and not is_ancestor:
                logger.debug(f"  SKIP (not included): {item_path}")
                stats["skipped"] += 1
                continue

        # ------------------------------------------------------------------
        # Copy dataset directly (preserves compression, chunks, etc.)
        # ------------------------------------------------------------------
        if isinstance(item, h5py.Dataset):
            try:
                src_group.copy(name, dst_group, name=name)
                logger.debug(f"  COPY dataset: {item_path}")
                stats["copied"] += 1
            except Exception as exc:
                logger.warning(f"Could not copy dataset '{item_path}': {exc}")
                stats["skipped"] += 1

        # ------------------------------------------------------------------
        # Recurse into sub-groups
        # ------------------------------------------------------------------
        elif isinstance(item, h5py.Group):
            child_dst = dst_group.require_group(name)
            _copy_filtered(
                item,
                child_dst,
                exclude_patterns=exclude_patterns,
                include_patterns=include_patterns,
                root=item_path,
                stats=stats,
                max_paths=max_paths,
                k_grids=k_grids,
            )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def filter_h5(
    src: Path,
    dst: Path,
    exclude: list[str] | None = None,
    include_only: list[str] | None = None,
    max_paths: int | None = None,
    dedup_k: bool = False,
    overwrite: bool = False,
) -> None:
    """Copy *src* HDF5 file to *dst*, applying filters.

    Parameters
    ----------
    src:
        Source HDF5 file (must exist).
    dst:
        Destination HDF5 file.  Must not exist unless *overwrite* is True.
    exclude:
        List of glob-style HDF5 path patterns to exclude.
    include_only:
        When given, copy *only* items matching these prefixes (and their
        ancestors).  Cannot be combined with *exclude*.
    max_paths:
        Keep only the first N FEFF paths per site (ordered by r_eff ascending).
        Paths are numbered from 1; path 1 is the nearest-neighbour shell.
        Cannot be combined with a pattern that already strips all paths.
    dedup_k:
        Deduplicate k arrays.  All per-site and per-path ``k`` datasets are
        verified to be identical and then stored once each in
        ``meta/k_grid_sites`` and ``meta/k_grid_paths``.  Individual ``k``
        datasets are dropped.  Halves storage for chi arrays without any loss
        of information.  Reconstruction: ``k = h5['meta/k_grid_sites'][()]``.
    overwrite:
        If True, an existing *dst* will be overwritten.
    """
    import h5py

    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists():
        if overwrite:
            dst.unlink()
            logger.info(f"Removed existing destination: {dst}")
        else:
            raise FileExistsError(
                f"Destination already exists: {dst}  (use --overwrite to replace)"
            )

    if exclude and include_only:
        raise ValueError("--exclude and --include-only are mutually exclusive")

    exclude_pats = [_glob_to_regex(p) for p in (exclude or [])]
    include_pats = [_glob_to_regex(p) for p in (include_only or [])]

    src_size_mb = src.stat().st_size / 1024**2
    logger.info(f"Source : {src}  ({src_size_mb:.1f} MB)")
    logger.info(f"Dest   : {dst}")

    if exclude:
        logger.info(f"Excluding patterns: {exclude}")
    if include_only:
        logger.info(f"Include-only patterns: {include_only}")
    if max_paths is not None:
        logger.info(f"Keeping at most {max_paths} FEFF path(s) per site (ordered by r_eff)")

    stats: dict[str, int] = {"copied": 0, "skipped": 0}

    with h5py.File(src, "r") as h5src, h5py.File(dst, "w") as h5dst:
        k_grids: _KGrids | None = None
        if dedup_k:
            logger.info("Scanning k arrays for deduplication...")
            k_grids = _scan_k_grids(h5src, max_paths=max_paths)

        _copy_filtered(
            h5src,
            h5dst,
            exclude_patterns=exclude_pats,
            include_patterns=include_pats,
            root="",
            stats=stats,
            max_paths=max_paths,
            k_grids=k_grids,
        )

        # Write shared k grids into meta/ after the main copy.
        if k_grids is not None:
            _COMPRESS_K = {"compression": "gzip", "compression_opts": 6}
            meta = h5dst.require_group("meta")
            if k_grids.site_k is not None:
                if "k_grid_sites" in meta:
                    del meta["k_grid_sites"]
                meta.create_dataset("k_grid_sites", data=k_grids.site_k, **_COMPRESS_K)
                meta["k_grid_sites"].attrs["description"] = (
                    "Common k grid for all per-site chi arrays (Å⁻¹).  "
                    "Individual site k datasets were removed during filtering."
                )
                logger.info(
                    f"Stored shared site k grid at meta/k_grid_sites "
                    f"({len(k_grids.site_k)} points)"
                )
            if k_grids.path_k is not None:
                if "k_grid_paths" in meta:
                    del meta["k_grid_paths"]
                meta.create_dataset("k_grid_paths", data=k_grids.path_k, **_COMPRESS_K)
                meta["k_grid_paths"].attrs["description"] = (
                    "Common k grid for all per-path chi arrays (Å⁻¹).  "
                    "Individual path k datasets were removed during filtering."
                )
                logger.info(
                    f"Stored shared path k grid at meta/k_grid_paths "
                    f"({len(k_grids.path_k)} points)"
                )

    dst_size_mb = dst.stat().st_size / 1024**2
    reduction = 100.0 * (1 - dst_size_mb / src_size_mb) if src_size_mb else 0.0
    logger.info(
        f"Done.  Copied {stats['copied']} items, skipped {stats['skipped']}."
    )
    logger.info(
        f"  {src_size_mb:.1f} MB  →  {dst_size_mb:.1f} MB"
        f"  ({reduction:.1f}% reduction)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", type=Path, help="Source HDF5 file")
    parser.add_argument("dst", type=Path, help="Destination HDF5 file")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--exclude",
        metavar="PATTERN",
        nargs="+",
        help=(
            "HDF5 path glob patterns to exclude.  "
            "Use * for a single component, ** for any depth.  "
            "Example: 'frames/*/sites/*/paths'"
        ),
    )
    group.add_argument(
        "--include-only",
        metavar="PATTERN",
        nargs="+",
        dest="include_only",
        help=(
            "Copy ONLY items matching these patterns (and their ancestors).  "
            "Example: 'meta' 'aggregates'"
        ),
    )

    parser.add_argument(
        "--strip-paths",
        action="store_true",
        dest="strip_paths",
        help=(
            "Convenience flag: exclude all per-path chi(k) contributions "
            "(equivalent to --exclude 'frames/*/sites/*/paths').  "
            "This is usually the main source of bloat."
        ),
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=None,
        metavar="N",
        dest="max_paths",
        help=(
            "Keep only the first N FEFF paths per site.  "
            "FEFF orders paths by increasing r_eff (effective path length), "
            "so path 1 is always the nearest-neighbour single-scattering shell "
            "and lower-numbered paths contribute most to the EXAFS signal.  "
            "Example: --max-paths 20 keeps the 20 shortest paths."
        ),
    )
    parser.add_argument(
        "--dedup-k",
        action="store_true",
        dest="dedup_k",
        help=(
            "Deduplicate k arrays.  All per-site (and per-path) k datasets "
            "are verified to be identical, stored once at meta/k_grid_sites "
            "(and meta/k_grid_paths), and the redundant per-dataset copies "
            "are dropped.  This typically halves the storage used by chi "
            "arrays with no information loss.  Incompatible with scripts that "
            "expect each site to carry its own k dataset — check consumers "
            "before using."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it already exists.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging (prints every copied/skipped item).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    exclude = list(args.exclude or [])
    if args.strip_paths:
        pat = "frames/*/sites/*/paths"
        if pat not in exclude:
            exclude.append(pat)
        if "path_results" not in exclude:
            exclude.append("path_results")

    filter_h5(
        src=args.src,
        dst=args.dst,
        exclude=exclude or None,
        include_only=args.include_only,
        max_paths=args.max_paths,
        dedup_k=args.dedup_k,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
