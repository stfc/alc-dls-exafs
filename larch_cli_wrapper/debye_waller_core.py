"""Core functions for Debye-Waller factor and MSRD computation from MD trajectories.

These functions are shared between the CLI (``larch-cli debye-waller``) and the
``notebooks/debye_waller.py`` Marimo notebook so that logic is not duplicated.
"""

from __future__ import annotations

import io
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.geometry import find_mic
from ase.io import read as ase_read

# ---------------------------------------------------------------------------
# Module-level logger – callers may configure this however they like.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------


def load_trajectory(
    trajectory_path: str | Path,
    skip_frames: int = 0,
) -> list[Any]:
    """Load an ASE-readable trajectory and validate it.

    Args:
        trajectory_path: Path to the trajectory file.
        skip_frames: Number of frames to skip from the start.

    Returns:
        List of ASE ``Atoms`` objects (one per frame).

    Raises:
        ValueError: If the trajectory fails validation.
        OSError: If the file cannot be read.
    """
    trajectory_path = Path(trajectory_path)
    raw = ase_read(str(trajectory_path), index=f"{skip_frames}:")
    if not isinstance(raw, list):
        raw = [raw]

    errors: list[str] = []
    warnings: list[str] = []

    if len(raw) < 2:
        errors.append(
            f"Trajectory has only {len(raw)} frame after skipping {skip_frames}. "
            "At least 2 frames are required."
        )

    n_atoms_0 = len(raw[0])
    bad_frames = [i for i, a in enumerate(raw) if len(a) != n_atoms_0]
    if bad_frames:
        errors.append(
            f"Inconsistent atom count: frame(s) {bad_frames[:5]}"
            f"{'…' if len(bad_frames) > 5 else ''} differ from frame 0 "
            f"({n_atoms_0} atoms)."
        )

    syms_0 = raw[0].get_chemical_symbols()
    bad_sym_frames = [
        i
        for i, a in enumerate(raw)
        if len(a) == n_atoms_0 and a.get_chemical_symbols() != syms_0
    ]
    if bad_sym_frames:
        errors.append(
            f"Inconsistent chemical symbols in frame(s) {bad_sym_frames[:5]}"
            f"{'…' if len(bad_sym_frames) > 5 else ''}."
        )

    if errors:
        raise ValueError("Trajectory validation failed:\n" + "\n".join(errors))

    if not raw[0].get_cell().any():
        warnings.append(
            "No periodic cell found. PBC unwrapping will be skipped "
            "(positions used as-is)."
        )

    for w in warnings:
        logger.warning(w)

    logger.info(
        "Loaded %d frames · %d atoms per frame · elements: %s",
        len(raw),
        n_atoms_0,
        ", ".join(sorted(set(syms_0))),
    )
    return raw  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Position processing
# ---------------------------------------------------------------------------


def unwrap_positions_pbc(structures: list[Any]) -> np.ndarray:
    """Unwrap atomic positions for continuous trajectories across PBC.

    Args:
        structures: List of ASE ``Atoms`` objects (one per frame).

    Returns:
        ``ndarray`` of shape ``(n_frames, n_atoms, 3)`` with unwrapped
        Cartesian positions.
    """
    logger.info("Unwrapping positions for PBC...")
    n_frames = len(structures)
    n_atoms = len(structures[0])
    unwrapped = np.zeros((n_frames, n_atoms, 3))

    unwrapped[0] = structures[0].get_positions()

    for i in range(1, n_frames):
        if i % 500 == 0:
            logger.info("  Unwrapping frame %d/%d...", i, n_frames)
        atoms = structures[i]
        cell = atoms.get_cell()
        if np.all(cell.lengths() == 0):
            unwrapped[i] = atoms.get_positions()
            continue
        cell_matrix = cell.complete()
        pbc = atoms.get_pbc()
        # Convert the previous unwrapped position to fractional coordinates
        # using the *current* frame's cell (the displacement between frames is
        # small, so the choice of cell for the reference point only affects
        # the wrap decision, not the accumulated position).
        inv_cell = np.linalg.inv(cell_matrix)
        frac_current = atoms.get_positions() @ inv_cell
        frac_previous = unwrapped[i - 1] @ inv_cell
        frac_disp = frac_current - frac_previous
        # Only wrap along periodic directions; wrapping a non-periodic
        # direction (e.g. the vacuum axis of a slab) would corrupt positions.
        frac_disp[:, pbc] -= np.round(frac_disp[:, pbc])
        unwrapped[i] = unwrapped[i - 1] + (frac_disp @ cell_matrix)

    return unwrapped


def kabsch_align(
    unwrapped_positions: np.ndarray,
    reference_idx: int = 0,
    reference_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Align all trajectory frames to a reference using the Kabsch algorithm.

    Args:
        unwrapped_positions: Array of shape ``(n_frames, n_atoms, 3)``.
        reference_idx: Frame index to use as reference (ignored if
            *reference_pos* is provided).
        reference_pos: Explicit reference positions of shape ``(n_atoms, 3)``.

    Returns:
        Aligned positions, same shape as *unwrapped_positions*.
    """
    logger.info("Aligning trajectory (Kabsch)...")
    ref_pos = (
        reference_pos
        if reference_pos is not None
        else unwrapped_positions[reference_idx]
    )
    ref_com = ref_pos.mean(axis=0)
    ref_pos_centered = ref_pos - ref_com

    aligned = np.zeros_like(unwrapped_positions)
    for i in range(len(unwrapped_positions)):
        pos = unwrapped_positions[i]
        com = pos.mean(axis=0)
        pos_c = pos - com
        H = pos_c.T @ ref_pos_centered
        U, _S, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        aligned[i] = (pos_c @ R) + ref_com
    return aligned


def process_trajectory(
    structures: list[Any],
    *,
    align: bool = True,
) -> np.ndarray:
    """Unwrap PBC positions and optionally Kabsch-align all frames.

    This is the convenience wrapper used by both the CLI and notebook.

    Args:
        structures: List of ASE ``Atoms`` objects.
        align: When ``True`` (default), a two-pass Kabsch alignment is applied
            after unwrapping.

    Returns:
        Processed positions, shape ``(n_frames, n_atoms, 3)``.
    """
    unwrapped = unwrap_positions_pbc(structures)
    if align:
        rough = kabsch_align(unwrapped)
        avg = np.mean(rough, axis=0)
        unwrapped = kabsch_align(unwrapped, reference_pos=avg)
        logger.info("Two-pass Kabsch alignment complete.")
    else:
        logger.info("Kabsch alignment skipped.")
    return unwrapped


# ---------------------------------------------------------------------------
# ADP / B-factor computation
# ---------------------------------------------------------------------------


def compute_adp_results(
    structures: list[Any],
    unwrapped: np.ndarray,
) -> dict[str, Any]:
    """Compute per-atom ADP tensors and B-factors from unwrapped positions.

    Args:
        structures: List of ASE ``Atoms`` objects (used for metadata only –
            symbols, cell, pbc).
        unwrapped: Unwrapped (and optionally aligned) positions, shape
            ``(n_frames, n_atoms, 3)``.

    Returns:
        Dictionary with keys:

        - ``"b_factors"``      – ``ndarray (n_atoms,)``
        - ``"u_tensor"``       – ``ndarray (n_atoms, 3, 3)``
        - ``"avg_positions"``  – ``ndarray (n_atoms, 3)``
        - ``"atom_names"``     – ``list[str]``
        - ``"avg_cell"``       – ``ndarray (3, 3)``
        - ``"atom_indices"``   – ``ndarray (n_atoms,)``
    """
    avg_pos = np.mean(unwrapped, axis=0)
    displacements = unwrapped - avg_pos[np.newaxis, :, :]
    u_tensor = np.einsum("fni,fnj->nij", displacements, displacements) / len(structures)
    b_factors = 8 * np.pi**2 * np.trace(u_tensor, axis1=1, axis2=2) / 3

    return {
        "b_factors": b_factors,
        "u_tensor": u_tensor,
        "avg_positions": avg_pos,
        "atom_names": structures[0].get_chemical_symbols(),
        "avg_cell": structures[0].get_cell().complete(),
        "atom_indices": np.arange(len(b_factors)),
    }


# ---------------------------------------------------------------------------
# CIF output
# ---------------------------------------------------------------------------


def save_cif_with_adp(results: dict[str, Any]) -> str:
    """Return a CIF string containing mean positions and anisotropic U tensors.

    Args:
        results: Dictionary as returned by :func:`compute_adp_results`.

    Returns:
        CIF file contents as a string.
    """
    pos = results["avg_positions"]
    names = results["atom_names"]
    u_cart = results["u_tensor"]
    cell = results["avg_cell"]
    inv_cell = np.linalg.inv(cell)
    frac_pos = pos @ inv_cell.T
    a, b, c = np.linalg.norm(cell, axis=1)

    def ang(v1: np.ndarray, v2: np.ndarray) -> float:
        return float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1,
                        1,
                    )
                )
            )
        )

    alpha = ang(cell[1], cell[2])
    beta = ang(cell[0], cell[2])
    gamma = ang(cell[0], cell[1])

    buf = io.StringIO()
    buf.write("data_MD_results\n\n")
    buf.write(f"_cell_length_a {a:.6f}\n")
    buf.write(f"_cell_length_b {b:.6f}\n")
    buf.write(f"_cell_length_c {c:.6f}\n")
    buf.write(f"_cell_angle_alpha {alpha:.4f}\n")
    buf.write(f"_cell_angle_beta  {beta:.4f}\n")
    buf.write(f"_cell_angle_gamma {gamma:.4f}\n\n")
    buf.write(
        "loop_\n"
        "_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "_atom_site_B_iso_or_equiv\n"
    )
    for i in range(len(names)):
        buf.write(
            f"{names[i]}{i + 1} {names[i]} "
            f"{frac_pos[i, 0]:.6f} {frac_pos[i, 1]:.6f} {frac_pos[i, 2]:.6f} "
            f"{results['b_factors'][i]:.4f}\n"
        )
    buf.write(
        "\nloop_\n"
        "_atom_site_aniso_label\n"
        "_atom_site_aniso_U_11\n_atom_site_aniso_U_22\n_atom_site_aniso_U_33\n"
        "_atom_site_aniso_U_23\n_atom_site_aniso_U_13\n_atom_site_aniso_U_12\n"
    )
    for i in range(len(names)):
        u = u_cart[i]
        buf.write(
            f"{names[i]}{i + 1} "
            f"{u[0, 0]:.5f} {u[1, 1]:.5f} {u[2, 2]:.5f} "
            f"{u[1, 2]:.5f} {u[0, 2]:.5f} {u[0, 1]:.5f}\n"
        )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Site specification parsing
# ---------------------------------------------------------------------------


def parse_site_specification(spec: str, symbols: list[str]) -> list[int]:
    """Parse a site specification string and return atomic indices.

    Supported formats:

    +----------+---------------------------------------+
    | Format   | Meaning                               |
    +==========+=======================================+
    | ``K``    | All K atoms                           |
    +----------+---------------------------------------+
    | ``K.1``  | First K atom (1-based within element) |
    +----------+---------------------------------------+
    | ``K.1-3``| First three K atoms                   |
    +----------+---------------------------------------+
    | ``11``   | 11th atom in full structure (1-based) |
    +----------+---------------------------------------+
    | ``11-20``| Atoms 11–20 (1-based, inclusive)      |
    +----------+---------------------------------------+

    Args:
        spec: Site specification string.
        symbols: List of chemical symbols for the structure.

    Returns:
        List of zero-based atom indices.

    Raises:
        ValueError: If the specification is invalid or no matching atoms exist.
    """
    # Pure numeric: absolute indices
    if spec.replace("-", "").replace(" ", "").isdigit():
        spec = spec.replace(" ", "")
        if "-" in spec:
            start, end = spec.split("-")
            return list(range(int(start) - 1, int(end)))
        else:
            return [int(spec) - 1]

    # Element with optional sub-index
    if "." in spec:
        element, index_part = spec.split(".", 1)
        element_indices = [i for i, sym in enumerate(symbols) if sym == element]
        if not element_indices:
            raise ValueError(f"No atoms of element '{element}' found")
        index_part = index_part.replace(" ", "")
        if "-" in index_part:
            start, end = index_part.split("-")
            start_idx = int(start) - 1
            end_idx = int(end)
            if end_idx > len(element_indices):
                raise ValueError(
                    f"'{element}' only has {len(element_indices)} atoms, "
                    f"cannot select up to {end_idx}"
                )
            return element_indices[start_idx:end_idx]
        else:
            idx = int(index_part) - 1
            if idx >= len(element_indices):
                raise ValueError(
                    f"'{element}' only has {len(element_indices)} atoms, "
                    f"cannot select index {idx + 1}"
                )
            return [element_indices[idx]]

    # Bare element symbol: all matching atoms
    matching = [i for i, sym in enumerate(symbols) if sym == spec]
    if not matching:
        raise ValueError(f"No atoms of element '{spec}' found")
    return matching


# ---------------------------------------------------------------------------
# Shared path clustering / canonicalization primitives
#
# These are used in three places: pooling MD-derived 2-body/3-body paths
# within :func:`calculate_grouped_msrd`, pooling raw per-frame FEFF paths
# into averaged path populations (``PathAggregator`` in ``exafs_data.py``),
# and matching MSRD (MD) path groups against FEFF path groups
# (:func:`match_msrd_paths_to_feff`). Keeping one implementation avoids three
# subtly different ad hoc grouping mechanisms.
# ---------------------------------------------------------------------------


def canonical_scatterer_key(scatterer: str) -> tuple[str, ...]:
    """Canonicalize a (possibly multi-atom) scatterer label for grouping.

    Sorts dash-separated element tokens so that, e.g., ``"N-C"`` and ``"C-N"``
    (which can arise purely from atom-listing order and differ frame-to-frame
    for the same physical path) always compare equal.

    Args:
        scatterer: Scatterer label, e.g. ``"O"`` or ``"N-C"``.

    Returns:
        Sorted tuple of element tokens, usable as a dict key.
    """
    return tuple(sorted(scatterer.split("-")))


def path_feature_vector(r_eff: float, angle: float | None = None) -> np.ndarray:
    """Å-equivalent feature vector for comparing/clustering scattering paths.

    For a 2-body (or already-transformed 4-leg rattle) path this is just the
    effective distance. For a 3-body path with a known angle, the angular
    mismatch is converted to an arc length via the mean leg length
    (``r_eff * radians(angle)``) so that distance and angle contribute in the
    same physical units (Å) and can share a single tolerance.

    Args:
        r_eff: Effective path length (Å).
        angle: Path angle in degrees, or ``None`` if unavailable (falls back
            to a distance-only vector, e.g. for 2-body paths or 3-body FEFF
            paths from older data without a stored angle).

    Returns:
        1-element or 2-element feature vector.
    """
    if angle is None:
        return np.array([r_eff])
    return np.array([r_eff, r_eff * np.radians(angle)])


def cluster_1d_sorted(values: np.ndarray, tol: float) -> list[np.ndarray]:
    """Greedy 1-D clustering by running-mean gap.

    Values are sorted ascending; a new cluster starts whenever the next
    value's distance from the *running mean* of the current (growing)
    cluster exceeds ``tol``. This is the sequential clustering algorithm
    historically inlined in :func:`calculate_grouped_msrd`, extracted here so
    it can be reused for FEFF-path pooling and MSRD↔FEFF matching as well.

    Args:
        values: 1-D array of scalar values to cluster.
        tol: Maximum allowed distance from a cluster's running mean.

    Returns:
        List of index arrays (into ``values``), one per cluster, in
        ascending order of the clustered values.
    """
    if len(values) == 0:
        return []
    values = np.asarray(values)
    order = np.argsort(values)
    sorted_vals = values[order]

    clusters: list[list[int]] = [[int(order[0])]]
    running_sum = [float(sorted_vals[0])]
    for i in range(1, len(sorted_vals)):
        running_mean = running_sum[-1] / len(clusters[-1])
        if abs(float(sorted_vals[i]) - running_mean) <= tol:
            clusters[-1].append(int(order[i]))
            running_sum[-1] += float(sorted_vals[i])
        else:
            clusters.append([int(order[i])])
            running_sum.append(float(sorted_vals[i]))

    return [np.array(c) for c in clusters]


def estimate_tolerance_from_elbow(
    distances: np.ndarray,
    *,
    min_tol: float = 0.02,
    max_tol: float = 1.0,
) -> dict[str, Any]:
    """Suggest a tolerance from the "elbow" of a sorted nearest-neighbor curve.

    Sorted nearest-neighbor distances typically show a shallow, noisy region
    (paths that clearly belong together) followed by a sharp rise (paths
    that clearly don't). This finds the "knee" via the point of maximum
    distance from the chord connecting the first and last sorted points —
    the same principle used to pick ``eps`` in DBSCAN, without adding a
    dependency on the ``kneed`` package.

    Args:
        distances: 1-D array of nonnegative nearest-neighbor distances
            (need not be pre-sorted).
        min_tol: Lower clamp for the suggested tolerance.
        max_tol: Upper clamp for the suggested tolerance.

    Returns:
        Dict with keys ``"tol"`` (suggested tolerance), ``"sorted_distances"``
        (ascending array, for plotting), and ``"knee_index"`` (index into
        ``sorted_distances`` of the detected knee), or an empty ``"tol":
        None`` result if fewer than 2 distances are provided.
    """
    distances = np.asarray(distances, dtype=np.float64)
    distances = distances[np.isfinite(distances)]
    if len(distances) < 2:
        return {"tol": None, "sorted_distances": distances, "knee_index": None}

    sorted_d = np.sort(distances)
    n = len(sorted_d)
    x = np.arange(n, dtype=np.float64)

    # Distance of each point from the chord (x[0], y[0]) -> (x[-1], y[-1]).
    x0, y0 = x[0], sorted_d[0]
    x1, y1 = x[-1], sorted_d[-1]
    chord = np.array([x1 - x0, y1 - y0])
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-12:
        # All distances identical -> no meaningful elbow.
        tol = float(np.clip(sorted_d[0], min_tol, max_tol))
        return {"tol": tol, "sorted_distances": sorted_d, "knee_index": 0}

    chord_unit = chord / chord_len
    pts = np.stack([x - x0, sorted_d - y0], axis=1)
    proj_len = pts @ chord_unit
    proj = np.outer(proj_len, chord_unit)
    perp_dist = np.linalg.norm(pts - proj, axis=1)

    knee_index = int(np.argmax(perp_dist))
    tol = float(np.clip(sorted_d[knee_index], min_tol, max_tol))
    return {"tol": tol, "sorted_distances": sorted_d, "knee_index": knee_index}


# ---------------------------------------------------------------------------
# Path instance model
#
# The architecture is layered:
#
#     fixed atom-index path instances   (enumerate_path_instances)
#     -> per-frame path samples         (sample_path_instances)
#     -> per-instance statistics        (compute_instance_statistics)
#     -> optional grouped MSRD stats    (group_path_instances)
#     -> optional FEFF/MD-EXAFS matching (match_msrd_paths_to_feff,
#        match_path_instances_to_feff)
#
# ``calculate_grouped_msrd`` is a convenience wrapper chaining the first four
# layers for the standalone CLI/notebook use case.
# ---------------------------------------------------------------------------


def canonical_intermediates(intermediates: Iterable[int]) -> tuple[int, ...]:
    """Canonical (orientation-independent) ordering of intermediate atoms.

    A closed scattering path traversed in reverse visits the intermediates in
    reversed order; the canonical form is the lexicographically smaller of
    the forward and reversed tuples.

    Args:
        intermediates: Ordered intermediate atom indices.

    Returns:
        Canonical tuple of intermediate atom indices.
    """
    forward = tuple(int(i) for i in intermediates)
    return min(forward, forward[::-1])


@dataclass(frozen=True)
class PathInstanceID:
    """Stable identity of a fixed atom-index path instance.

    Attributes:
        absorber: Zero-based index of the absorber atom.
        intermediates: Canonically ordered intermediate atom indices
            (see :func:`canonical_intermediates`).
    """

    absorber: int
    intermediates: tuple[int, ...]

    @property
    def nleg(self) -> int:
        """Number of legs of the closed path (intermediates + 1)."""
        return len(self.intermediates) + 1


@dataclass
class PathInstance:
    """A fixed atom-index scattering path with its orientation aliases.

    Attributes:
        path_id: Canonical identity of the instance.
        absorber: Zero-based index of the absorber atom (== path_id.absorber).
        orientations: All ordered intermediate tuples describing the same
            physical path (forward and reversed). Matching against external
            (FEFF/MD-EXAFS) ordered paths should test every orientation.
        species: Chemical symbols along the path, ``(absorber, *intermediates)``
            in canonical order.
        reference_reff: Effective path length (half total path length) in the
            reference structure, in Å.
    """

    path_id: PathInstanceID
    absorber: int
    orientations: list[tuple[int, ...]]
    species: tuple[str, ...]
    reference_reff: float

    @property
    def nleg(self) -> int:
        """Number of legs of the closed path."""
        return self.path_id.nleg


@dataclass
class PathSamples:
    """Per-frame samples of one path instance through a trajectory.

    Attributes:
        instance: The sampled path instance.
        reff: Effective path length per frame, shape ``(n_frames,)`` (Å).
            ``reff = legs.sum(axis=1) / 2`` for multi-leg paths; the plain
            absorber–scatterer distance for 2-body paths.
        legs: Leg lengths per frame, shape ``(n_frames, n_unique_legs)`` (Å).
            For 2-body paths a single leg (the bond distance). For 3-body
            paths ``[d(A, B), d(B, C), d(C, A)]`` in canonical order.
        internal_angles: Internal vertex angles per frame (degrees), shape
            ``(n_frames, n_vertices)``, ordered ``(A, B, C)`` for 3-body
            paths; ``None`` for 2-body paths.
        feff_beta: FEFF scattering angles per frame (degrees), same shape as
            ``internal_angles``. ``feff_beta = 180 - internal_angle`` at each
            vertex; the physically relevant entries for a 3-body path are the
            scatterer vertices (B and C).
    """

    instance: PathInstance
    reff: np.ndarray
    legs: np.ndarray
    internal_angles: np.ndarray | None
    feff_beta: np.ndarray | None


def _normalize_cells(structures: list[Any]) -> list[Any]:
    """Return frames sharing a single cell, for MIC path sampling.

    Constant-cell trajectories pass through unchanged. For variable-cell
    (NPT) trajectories each frame's Cartesian positions are remapped into
    the reference (frame-0) cell — fractional coordinates in the frame's
    own cell, then Cartesian in the reference cell — so a single ``cell``
    can be used with :func:`find_mic`. This measures distances in a
    "strain-removed" frame: instantaneous bond lengths are off by the
    cell strain (typically <1% for equilibrated NPT), which is small
    relative to the thermal disorder MSRD is measuring.

    Args:
        structures: List of ASE ``Atoms`` objects (one per frame).

    Returns:
        List of ``Atoms`` objects sharing the frame-0 cell (the originals
        if the cell was already constant, else cheap copies with remapped
        positions). The input objects are not mutated.
    """
    from ase import Atoms as _Atoms

    ref_cell = structures[0].get_cell().complete()
    for atoms in structures[1:]:
        if not np.allclose(atoms.get_cell().complete(), ref_cell):
            break
    else:
        return structures  # constant cell — nothing to do

    ref_cell_arr = np.asarray(structures[0].get_cell().complete())
    normalized: list[Any] = []
    for atoms in structures:
        frac = atoms.get_scaled_positions(wrap=False)
        pos = frac @ ref_cell_arr.T
        normalized.append(
            _Atoms(
                symbols=atoms.get_chemical_symbols(),
                positions=pos,
                cell=ref_cell_arr,
                pbc=atoms.get_pbc(),
            )
        )
    return normalized


def catalogue_from_sequences(
    reference_atoms: Any,
    absorber: int,
    sequences: Iterable[Iterable[int]],
) -> list[PathInstance]:
    """Build a path-instance catalogue from explicit intermediate sequences.

    This is the entry point for the MD-EXAFS pipeline, which supplies
    arbitrary FEFF-like closed paths
    ``absorber -> s1 -> s2 -> ... -> absorber`` directly.

    Args:
        reference_atoms: ASE ``Atoms`` object defining the reference geometry.
        absorber: Zero-based index of the absorber atom.
        sequences: Intermediate atom index sequences (one per path).
            Each sequence is canonicalized; both orientations are retained.

    Returns:
        List of :class:`PathInstance` objects (de-duplicated by canonical
        identity).
    """
    symbols = reference_atoms.get_chemical_symbols()
    catalogue: dict[PathInstanceID, PathInstance] = {}
    for seq in sequences:
        intermediates = tuple(int(i) for i in seq)
        if not intermediates:
            continue
        canon = canonical_intermediates(intermediates)
        path_id = PathInstanceID(absorber=int(absorber), intermediates=canon)
        if path_id in catalogue:
            continue
        reference_reff = _reference_reff(reference_atoms, absorber, canon)
        reverse = canon[::-1]
        orientations = [canon] if reverse == canon else [canon, reverse]
        catalogue[path_id] = PathInstance(
            path_id=path_id,
            absorber=int(absorber),
            orientations=orientations,
            species=(symbols[absorber], *(symbols[i] for i in canon)),
            reference_reff=reference_reff,
        )
    return list(catalogue.values())


def _reference_reff(
    reference_atoms: Any, absorber: int, intermediates: tuple[int, ...]
) -> float:
    """Effective path length (half total path length) in the reference.

    For a 2-body path (A → B → A) this is the plain absorber–scatterer
    distance; for a 3-body path it is half the triangle perimeter.
    """
    sequence = (absorber, *intermediates)
    total = 0.0
    for i, j in zip(sequence, (*sequence[1:], absorber), strict=True):
        total += float(reference_atoms.get_distance(i, j, mic=True))
    return total / 2.0


def _max_safe_mic_cutoff(cell: np.ndarray) -> float | None:
    """Return the largest sphere radius that fits inside a parallelepiped cell.

    For the minimum image convention to be unambiguous, the cutoff must be
    smaller than half the smallest perpendicular distance between two opposite
    faces of the simulation box.  This function computes that value for an
    arbitrary (non-orthogonal) cell.

    Args:
        cell: Unit cell matrix, shape ``(3, 3)`` (rows are lattice vectors).

    Returns:
        Maximum safe cutoff radius in Å, or ``None`` if the cell has zero
        volume or is not periodic (no MIC constraint).
    """
    # Ensure we have a complete 3x3 cell matrix
    cell_matrix = np.asarray(cell)
    if cell_matrix.shape != (3, 3):
        return None

    volume = float(np.abs(np.linalg.det(cell_matrix)))
    if volume <= 0.0:
        return None

    # Areas of the three faces (parallelograms)
    area_ab = float(np.linalg.norm(np.cross(cell_matrix[0], cell_matrix[1])))
    area_bc = float(np.linalg.norm(np.cross(cell_matrix[1], cell_matrix[2])))
    area_ca = float(np.linalg.norm(np.cross(cell_matrix[2], cell_matrix[0])))

    if area_ab == 0.0 or area_bc == 0.0 or area_ca == 0.0:
        return None

    # Perpendicular distances between opposite faces
    dist_ab = volume / area_ab
    dist_bc = volume / area_bc
    dist_ca = volume / area_ca

    # The largest inscribed sphere has radius half the smallest face distance
    return min(dist_ab, dist_bc, dist_ca) / 2.0


def _warn_unsafe_mic_cutoff(
    cell: Any, cutoffs: Iterable[tuple[str, float | None]]
) -> None:
    """Log a warning for cutoffs exceeding the safe MIC radius of *cell*."""
    max_safe_cutoff = _max_safe_mic_cutoff(cell.complete())
    if max_safe_cutoff is None:
        return
    for cutoff_name, cutoff_value in cutoffs:
        if (
            cutoff_value is not None
            and cutoff_value > 0
            and cutoff_value > max_safe_cutoff
        ):
            logger.warning(
                "%s=%.3f Å exceeds the maximum safe MIC cutoff "
                "for this unit cell (%.3f Å). Distances may be "
                "ambiguous because the sphere overlaps with its "
                "own periodic images. Consider using a supercell or "
                "reducing %s to <= %.3f Å.",
                cutoff_name,
                cutoff_value,
                max_safe_cutoff,
                cutoff_name,
                max_safe_cutoff,
            )


def enumerate_path_instances(
    reference_atoms: Any,
    central_indices: list[int],
    cutoff: float = 3.5,
    cutoff_3body: float | None = None,
    exclude_hydrogen: bool = True,
) -> list[PathInstance]:
    """Enumerate fixed atom-index 2-body and 3-body path instances.

    Neighbours are selected by *reference-frame* minimum-image distance for
    both 2-body and 3-body paths (one documented, consistent criterion).
    The neighbour search uses ``max(cutoff, cutoff_3body)`` so that a
    ``cutoff_3body`` larger than ``cutoff`` still sees the more distant
    neighbours.

    Args:
        reference_atoms: ASE ``Atoms`` object (typically the first frame).
        central_indices: Zero-based absorber indices; all must have the same
            chemical element.
        cutoff: Absorber–neighbour cutoff (Å) for 2-body paths.
        cutoff_3body: Maximum *effective path length* Reff (half total path
            length, Å) for 3-body paths — the FEFF-comparable definition
            (FEFF's path finder cuts on total path length = 2×Reff).
            Triangles whose reference-frame half-perimeter exceeds this are
            skipped. Completeness: a triangle with Reff ≤ X has every leg
            ≤ X (triangle inequality), so generating neighbour legs within
            ``cutoff_3body`` and filtering by Reff misses nothing. ``0`` or
            ``None`` disables 3-body paths.
        exclude_hydrogen: Exclude hydrogen from the neighbour search.

    Returns:
        List of :class:`PathInstance` objects.

    Raises:
        ValueError: If the absorbers have mixed chemical elements.
    """
    symbols = reference_atoms.get_chemical_symbols()
    if not central_indices:
        return []
    central_element = symbols[central_indices[0]]
    if {symbols[i] for i in central_indices} != {central_element}:
        raise ValueError("All central_indices must have the same element")

    cell = reference_atoms.get_cell()
    three_body = cutoff_3body is not None and cutoff_3body > 0
    search_cutoff = cutoff
    if cutoff_3body is not None and cutoff_3body > 0:
        search_cutoff = max(cutoff, cutoff_3body)
    _warn_unsafe_mic_cutoff(
        cell,
        [("cutoff", cutoff), ("cutoff_3body", cutoff_3body if three_body else None)],
    )

    if exclude_hydrogen:
        neighbor_candidates = {i for i, sym in enumerate(symbols) if sym != "H"}
    else:
        neighbor_candidates = set(range(len(symbols)))

    catalogue: dict[PathInstanceID, PathInstance] = {}

    def _add(
        absorber: int,
        intermediates: tuple[int, ...],
        max_reff: float | None = None,
    ) -> None:
        canon = canonical_intermediates(intermediates)
        path_id = PathInstanceID(absorber=absorber, intermediates=canon)
        if path_id in catalogue:
            return
        ref_reff = _reference_reff(reference_atoms, absorber, canon)
        if max_reff is not None and ref_reff > max_reff + 1e-9:
            return
        reverse = canon[::-1]
        orientations = [canon] if reverse == canon else [canon, reverse]
        catalogue[path_id] = PathInstance(
            path_id=path_id,
            absorber=absorber,
            orientations=orientations,
            species=(symbols[absorber], *(symbols[i] for i in canon)),
            reference_reff=ref_reff,
        )

    for c_idx in central_indices:
        all_indices = [
            i for i in range(len(symbols)) if i != c_idx and i in neighbor_candidates
        ]
        distances = reference_atoms.get_distances(c_idx, all_indices, mic=True)
        search_neighbors = [
            all_indices[i]
            for i in range(len(all_indices))
            if distances[i] < search_cutoff + 1e-9
        ]
        logger.info(
            "  %d neighbors within %.2f Å of atom %d",
            len(search_neighbors),
            search_cutoff,
            c_idx,
        )
        ref_dist = dict(zip(all_indices, distances, strict=True))

        # 2-body paths (absorber -> neighbour -> absorber)
        for n_idx in search_neighbors:
            if ref_dist[n_idx] < cutoff:
                _add(c_idx, (n_idx,))

        # 3-body paths (absorber -> n1 -> n2 -> absorber)
        # cutoff_3body is an Reff (half-perimeter) limit; every leg of a
        # triangle with Reff <= cutoff_3body is itself <= cutoff_3body, so
        # the neighbour-leg search below is complete and the Reff filter in
        # _add makes the semantics exact.
        if cutoff_3body:  # truthy: not None and > 0
            c3b = float(cutoff_3body)
            neighbors_3body = [n for n in search_neighbors if ref_dist[n] < c3b + 1e-9]
            for i in range(len(neighbors_3body)):
                for j in range(i + 1, len(neighbors_3body)):
                    _add(
                        c_idx,
                        (neighbors_3body[i], neighbors_3body[j]),
                        max_reff=c3b,
                    )

    return list(catalogue.values())


# ---------------------------------------------------------------------------
# Path sampling
# ---------------------------------------------------------------------------


def _angle_series(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle (degrees) between row-vector series *v1* and *v2* per frame."""
    v1_unit = v1 / np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-10)
    v2_unit = v2 / np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-10)
    cos_t = np.clip(np.sum(v1_unit * v2_unit, axis=1), -1, 1)
    return np.degrees(np.arccos(cos_t))


def sample_path_instances(
    structures: list[Any],
    catalogue: list[PathInstance],
) -> list[PathSamples]:
    """Sample every catalogue path instance through the trajectory.

    Minimum-image distances are computed from the *original* (unaligned)
    coordinates — path lengths are invariant to translation/rotation, so
    unwrapping and Kabsch alignment are unnecessary for MSRD analysis.
    Variable-cell (NPT) trajectories are supported: each frame is remapped
    into the reference (frame-0) cell before MIC distances are computed
    (see :func:`_normalize_cells`).

    Args:
        structures: List of ASE ``Atoms`` objects (one per frame).
        catalogue: Path instances from :func:`enumerate_path_instances` or
            :func:`catalogue_from_sequences`.

    Returns:
        One :class:`PathSamples` per catalogue entry, in catalogue order.
    """
    if not catalogue:
        return []
    structures = _normalize_cells(structures)
    cell = structures[0].get_cell()
    pbc = structures[0].get_pbc()

    orig_positions = np.array([atoms.get_positions() for atoms in structures])

    # Cache MIC vectors per unordered atom pair (stored in ascending-index
    # direction) so each pair is processed exactly once.
    pair_vectors: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _pair_mic(i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """(vector i->j per frame, distance per frame) from the MIC cache."""
        key = (min(i, j), max(i, j))
        if key not in pair_vectors:
            v_raw = orig_positions[:, key[1], :] - orig_positions[:, key[0], :]
            pair_vectors[key] = find_mic(v_raw, cell, pbc)
        v_mic, dists = pair_vectors[key]
        if (i, j) == key:
            return v_mic, dists
        return -v_mic, dists

    all_samples: list[PathSamples] = []
    for instance in catalogue:
        absorber = instance.absorber
        intermediates = instance.path_id.intermediates
        sequence = (absorber, *intermediates)
        leg_pairs = list(zip(sequence, (*sequence[1:], absorber), strict=True))

        leg_arrays = []
        leg_vectors = []
        for i, j in leg_pairs:
            v_mic, dists = _pair_mic(i, j)
            leg_vectors.append(v_mic)
            leg_arrays.append(dists)
        legs = np.stack(leg_arrays, axis=1)

        if len(intermediates) == 1:
            reff = legs[:, 0].copy()
            internal_angles = None
            feff_beta = None
        else:
            reff = legs.sum(axis=1) / 2.0
            # Internal vertex angles: at vertex k, the angle between the
            # vectors to the previous and next path atoms.
            n_vertices = len(sequence)
            angle_cols = []
            for k in range(n_vertices):
                v_in = -leg_vectors[k - 1]  # vertex -> previous atom
                v_out = leg_vectors[k]  # vertex -> next atom
                angle_cols.append(_angle_series(v_in, v_out))
            internal_angles = np.stack(angle_cols, axis=1)
            feff_beta = 180.0 - internal_angles

        all_samples.append(
            PathSamples(
                instance=instance,
                reff=reff,
                legs=legs,
                internal_angles=internal_angles,
                feff_beta=feff_beta,
            )
        )
    return all_samples


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _cumulants(x: np.ndarray) -> tuple[float, float]:
    """Third and fourth cumulants of a 1-D sample (Å³ and Å⁴)."""
    xc = x - np.mean(x)
    m2 = float(np.mean(xc**2))
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    return m3, m4 - 3.0 * m2**2


def _block_variance_error(x: np.ndarray) -> float:
    """Standard error of the variance via block averaging.

    Frames are split into ~sqrt(n) contiguous blocks; the scatter of the
    per-block variances estimates the uncertainty caused by time correlation.
    Returns ``nan`` when there are too few frames for meaningful blocking.
    """
    n = len(x)
    k = int(np.sqrt(n))
    if k < 2 or n < 2 * k:
        return float("nan")
    block_vars = [float(np.var(b, ddof=1)) for b in np.array_split(x, k) if len(b) > 1]
    if len(block_vars) < 2:
        return float("nan")
    return float(np.std(block_vars, ddof=1) / np.sqrt(len(block_vars)))


def _effective_n_samples(x: np.ndarray) -> float:
    """Effective sample size from the lag-1 autocorrelation."""
    n = len(x)
    if n < 3:
        return float(n)
    xc = x - np.mean(x)
    denom = float(np.dot(xc, xc))
    if denom <= 0.0:
        return float(n)
    r1 = float(np.dot(xc[:-1], xc[1:]) / denom)
    r1 = float(np.clip(r1, -0.99, 0.99))
    return float(np.clip(n * (1.0 - r1) / (1.0 + r1), 1.0, n))


def compute_instance_statistics(samples: list[PathSamples]) -> list[dict[str, Any]]:
    """Compute per-instance statistics before any grouping.

    Args:
        samples: Per-instance samples from :func:`sample_path_instances`.

    Returns:
        One dict per :class:`PathSamples` entry (same order) with keys
        ``path_id``, ``n_frames``, ``mean_reff_A``, ``sigma2_A2``
        (within-instance thermal variance), ``delta_reff_A`` (mean minus
        reference), ``third_cumulant_A3``, ``fourth_cumulant_A4``,
        ``sigma2_block_error_A2``, ``effective_n_samples``, ``mean_legs_A``
        and — for 3-body paths — ``mean_internal_angles_deg`` and
        ``mean_feff_beta_deg`` (both per-vertex tuples).
    """
    stats: list[dict[str, Any]] = []
    for s in samples:
        reff = s.reff
        third, fourth = _cumulants(reff)
        entry: dict[str, Any] = {
            "path_id": s.instance.path_id,
            "n_frames": int(len(reff)),
            "mean_reff_A": float(np.mean(reff)),
            "sigma2_A2": float(np.var(reff, ddof=1)) if len(reff) > 1 else 0.0,
            "delta_reff_A": float(np.mean(reff) - s.instance.reference_reff),
            "third_cumulant_A3": third,
            "fourth_cumulant_A4": fourth,
            "sigma2_block_error_A2": _block_variance_error(reff),
            "effective_n_samples": _effective_n_samples(reff),
            "mean_legs_A": tuple(float(v) for v in np.mean(s.legs, axis=0)),
        }
        if s.internal_angles is not None and s.feff_beta is not None:
            entry["mean_internal_angles_deg"] = tuple(
                float(v) for v in np.mean(s.internal_angles, axis=0)
            )
            entry["mean_feff_beta_deg"] = tuple(
                float(v) for v in np.mean(s.feff_beta, axis=0)
            )
        stats.append(entry)
    return stats


def _scatterer_angle_pair(instance_stat: dict[str, Any]) -> tuple[float, float]:
    """Sorted pair of internal angles at the two scatterer vertices.

    Orientation-invariant under path reversal, unlike the angle at the
    canonical first scatterer alone.
    """
    _absorber, b, c = instance_stat["mean_internal_angles_deg"]
    return (min(b, c), max(b, c))


def _anchored_angle(instance_stat: dict[str, Any]) -> float:
    """Internal angle at the scatterer with the shorter absorber leg.

    Deterministic and reversal-invariant; for triangles isosceles at the
    absorber the two scatterer angles are equal, so the tie-break is harmless.
    """
    legs = instance_stat["mean_legs_A"]  # canonical [d(A,B), d(B,C), d(C,A)]
    angles = instance_stat["mean_internal_angles_deg"]  # (A, B, C)
    return float(angles[1] if legs[0] <= legs[2] else angles[2])


def _invariant_leg_triple(instance_stat: dict[str, Any]) -> tuple[float, float, float]:
    """Leg lengths as (shorter absorber leg, opposite leg, longer absorber leg)."""
    l_ab, l_bc, l_ca = instance_stat["mean_legs_A"]
    absorber_legs = sorted([l_ab, l_ca])
    return (absorber_legs[0], l_bc, absorber_legs[1])


def _cluster_angle_pairs(
    features: list[tuple[float, float]], tol: float
) -> list[np.ndarray]:
    """Nested 1-D clustering of sorted angle pairs.

    Clusters first on the smaller angle, then on the larger angle within
    each resulting cluster, using the running-mean gap algorithm of
    :func:`cluster_1d_sorted`. (A single-pass lexicographic clustering would
    interleave pairs like ``{45, 45}`` and ``{45, 90}`` and fragment them.)

    Args:
        features: Sorted angle pairs, one per path instance.
        tol: Per-component tolerance in degrees.

    Returns:
        List of index arrays (into ``features``), one per cluster.
    """
    if not features:
        return []
    min_angles = np.array([f[0] for f in features])
    clusters: list[np.ndarray] = []
    for idx_arr in cluster_1d_sorted(min_angles, tol):
        max_angles = np.array([features[i][1] for i in idx_arr])
        for sub_idx_arr in cluster_1d_sorted(max_angles, tol):
            clusters.append(idx_arr[sub_idx_arr])
    return clusters


def group_path_instances(
    samples: list[PathSamples],
    instance_stats: list[dict[str, Any]],
    tol_dist: float = 0.1,
    tol_angle: float = 5.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool path instances into grouped 2-body and 3-body MSRD results.

    Grouping clusters instance *means* (two-stage for 3-body paths: the
    sorted pair of scatterer-vertex internal angles — reversal-invariant, so
    A→B→C→A and A→C→B→A pool into one population — then effective length).
    Each group reports both the pooled *effective* class variance (thermal +
    between-instance mean spread) and the thermal-only within-instance
    variance.

    Args:
        samples: Per-instance samples.
        instance_stats: Per-instance statistics (same order as *samples*).
        tol_dist: Distance grouping tolerance (Å).
        tol_angle: Angle grouping tolerance (degrees).

    Returns:
        Tuple ``(res_2b, res_3b)`` sorted by ``"reff"``. Dict keys include the
        legacy ``type``, ``reff``, ``sigma2``, ``count``, ``atom_indices``
        (and ``angle``/``angle_var`` for 3-body), plus ``absorber``,
        ``path_ids``, ``sigma2_effective_A2``, ``sigma2_thermal_A2``,
        ``between_instance_reff_var_A2``, ``delta_reff_A``,
        ``sigma2_block_error_A2``, ``third_cumulant_A3``,
        ``fourth_cumulant_A4``, ``effective_n_samples``; 3-body groups add
        ``internal_angle_deg``, ``feff_beta_deg``, ``scatterer_angles_deg``,
        ``vertex_angles_deg``, ``feff_beta_seq_deg`` and ``leg_lengths_A``.
        3-body geometry keys are orientation-invariant: ``internal_angle_deg``
        is the internal angle at the scatterer with the shorter absorber leg,
        ``vertex_angles_deg`` is ``(absorber vertex, smaller scatterer angle,
        larger scatterer angle)``, and ``leg_lengths_A`` is ``(shorter
        absorber leg, opposite leg, longer absorber leg)``.
    """
    paired = list(zip(samples, instance_stats, strict=True))
    pair_instances = [(s, st) for s, st in paired if s.instance.nleg == 2]
    triplet_instances = [(s, st) for s, st in paired if s.instance.nleg == 3]

    def _group_stats(
        cluster: list[tuple[PathSamples, dict[str, Any]]],
    ) -> dict[str, Any]:
        pooled = np.concatenate([s.reff for s, _st in cluster])
        weights = np.array([st["n_frames"] for _s, st in cluster], dtype=float)
        within_vars = np.array([st["sigma2_A2"] for _s, st in cluster])
        means = np.array([st["mean_reff_A"] for _s, st in cluster])
        third, fourth = _cumulants(pooled)
        sigma2_effective = float(np.var(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        return {
            "reff": float(np.mean(pooled)),
            # Effective class variance: thermal fluctuations + spread of
            # instance means. Aliased by the legacy "sigma2" key.
            "sigma2": sigma2_effective,
            "sigma2_effective_A2": sigma2_effective,
            "sigma2_thermal_A2": (
                float(np.sum(weights * within_vars) / np.sum(weights))
                if np.sum(weights) > 0
                else 0.0
            ),
            "between_instance_reff_var_A2": (
                float(np.var(means, ddof=1)) if len(means) > 1 else 0.0
            ),
            "delta_reff_A": float(
                np.mean(pooled)
                - np.mean([s.instance.reference_reff for s, _st in cluster])
            ),
            "sigma2_block_error_A2": _block_variance_error(pooled),
            "third_cumulant_A3": third,
            "fourth_cumulant_A4": fourth,
            "effective_n_samples": float(
                np.sum([st["effective_n_samples"] for _s, st in cluster])
            ),
            "count": len(cluster),
        }

    res_2b: list[dict[str, Any]] = []
    pairs_by_element: defaultdict[str, list[tuple[PathSamples, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for s, st in pair_instances:
        pairs_by_element[s.instance.species[1]].append((s, st))

    for element, entries in pairs_by_element.items():
        mean_ds = np.array([st["mean_reff_A"] for _s, st in entries])
        clusters = [
            [entries[i] for i in idx_arr]
            for idx_arr in cluster_1d_sorted(mean_ds, tol_dist)
        ]
        for cluster in clusters:
            absorber_el = cluster[0][0].instance.species[0]
            res_2b.append(
                {
                    "type": f"{absorber_el}-{element}",
                    "absorber": absorber_el,
                    "atom_indices": [
                        (s.instance.absorber, *s.instance.path_id.intermediates)
                        for s, _st in cluster
                    ],
                    "path_ids": [s.instance.path_id for s, _st in cluster],
                    **_group_stats(cluster),
                }
            )

    res_3b: list[dict[str, Any]] = []
    triplets_by_elements: defaultdict[
        tuple[str, ...], list[tuple[PathSamples, dict[str, Any]]]
    ] = defaultdict(list)
    for s, st in triplet_instances:
        triplets_by_elements[tuple(sorted(s.instance.species[1:]))].append((s, st))

    for elem_pair, entries in triplets_by_elements.items():
        # Cluster on the *sorted* pair of scatterer-vertex internal angles.
        # The angle at the canonical first scatterer alone is not
        # reversal-invariant: A->B->C->A and A->C->B->A are the same closed
        # path (FEFF counts both in one degeneracy), but for non-isosceles
        # triangles their first-scatterer angles differ, which would fragment
        # one physical population into two groups.
        angle_pairs = [_scatterer_angle_pair(st) for _s, st in entries]
        angle_clusters = [
            [entries[i] for i in idx_arr]
            for idx_arr in _cluster_angle_pairs(angle_pairs, tol_angle)
        ]
        for angle_cluster in angle_clusters:
            mean_ls = np.array([st["mean_reff_A"] for _s, st in angle_cluster])
            dist_clusters = [
                [angle_cluster[i] for i in idx_arr]
                for idx_arr in cluster_1d_sorted(mean_ls, tol_dist)
            ]
            for cluster in dist_clusters:
                absorber_el = cluster[0][0].instance.species[0]
                # Anchored (orientation-invariant) internal angle: at the
                # scatterer with the shorter absorber leg.
                cluster_internal = np.array([_anchored_angle(st) for _s, st in cluster])
                internal_angle = float(np.mean(cluster_internal))
                absorber_angle = float(
                    np.mean([st["mean_internal_angles_deg"][0] for _s, st in cluster])
                )
                scatterer_pair = tuple(
                    float(v)
                    for v in np.mean(
                        [_scatterer_angle_pair(st) for _s, st in cluster], axis=0
                    )
                )
                # Orientation-invariant ordering: (absorber vertex, smaller
                # scatterer angle, larger scatterer angle).
                vertex_angles = (absorber_angle, *scatterer_pair)
                beta_seq = tuple(180.0 - a for a in vertex_angles)
                # Orientation-invariant legs: (shorter absorber leg, opposite
                # leg, longer absorber leg).
                leg_lengths = tuple(
                    float(v)
                    for v in np.mean(
                        [_invariant_leg_triple(st) for _s, st in cluster], axis=0
                    )
                )
                res_3b.append(
                    {
                        "type": f"{absorber_el}-{elem_pair[0]}-{elem_pair[1]}",
                        "absorber": absorber_el,
                        "atom_indices": [
                            (s.instance.absorber, *s.instance.path_id.intermediates)
                            for s, _st in cluster
                        ],
                        "path_ids": [s.instance.path_id for s, _st in cluster],
                        # Internal geometric angle at the scatterer with the
                        # shorter absorber leg (NOT the FEFF beta angle).
                        "angle": internal_angle,
                        "internal_angle_deg": internal_angle,
                        "feff_beta_deg": 180.0 - internal_angle,
                        "angle_var": (
                            float(np.var(cluster_internal, ddof=1))
                            if len(cluster_internal) > 1
                            else 0.0
                        ),
                        "scatterer_angles_deg": scatterer_pair,
                        "vertex_angles_deg": vertex_angles,
                        "feff_beta_seq_deg": beta_seq,
                        "leg_lengths_A": leg_lengths,
                        **_group_stats(cluster),
                    }
                )

    return (
        sorted(res_2b, key=lambda x: x["reff"]),
        sorted(res_3b, key=lambda x: x["reff"]),
    )


# ---------------------------------------------------------------------------
# Grouped MSRD convenience wrapper
# ---------------------------------------------------------------------------


def calculate_grouped_msrd(
    structures: list[Any],
    central_indices: list[int],
    central_label: str,
    cutoff: float = 3.5,
    tol_dist: float = 0.1,
    tol_angle: float = 5.0,
    cutoff_3body: float | None = None,
    exclude_hydrogen: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate grouped MSRD for 2-body and 3-body EXAFS paths.

    Convenience wrapper chaining :func:`enumerate_path_instances`,
    :func:`sample_path_instances`, :func:`compute_instance_statistics` and
    :func:`group_path_instances`. Use those lower-level functions directly
    when per-instance samples/statistics are needed (MD-EXAFS pipeline).

    Args:
        structures: List of ASE ``Atoms`` objects.
        central_indices: Zero-based indices of the absorber atoms.
        central_label: Human-readable label for the absorber (used in logging).
        cutoff: Neighbor cutoff radius in Å.
        tol_dist: Distance grouping tolerance in Å.
        tol_angle: Angle grouping tolerance in degrees.
        cutoff_3body: Maximum *effective path length* Reff (half total path
            length, Å) for 3-body paths — comparable to FEFF's path-length
            cutoff (FEFF cuts on total path length = 2×Reff). Triangles
            whose reference-frame half-perimeter exceeds this are excluded.
            ``0`` or ``None`` disables 3-body path computation entirely.
        exclude_hydrogen: When ``True`` (default), hydrogen atoms are excluded
            from the neighbor search and will not appear as scatterers in any
            MSRD path.

    Returns:
        Tuple ``(res_2b, res_3b)`` as documented for
        :func:`group_path_instances`.
    """
    if not central_indices:
        return [], []

    logger.info(
        "Analysing MSRD paths for %s (%d sites)...",
        central_label,
        len(central_indices),
    )

    catalogue = enumerate_path_instances(
        structures[0],
        central_indices,
        cutoff=cutoff,
        cutoff_3body=cutoff_3body,
        exclude_hydrogen=exclude_hydrogen,
    )
    samples = sample_path_instances(structures, catalogue)
    instance_stats = compute_instance_statistics(samples)
    return group_path_instances(
        samples, instance_stats, tol_dist=tol_dist, tol_angle=tol_angle
    )


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------


def plot_bfactors(
    results: dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Plot per-atom B-factors coloured by element (matplotlib).

    Args:
        results: Dictionary as returned by :func:`compute_adp_results`.
        output_path: If given, the figure is saved to this path instead of
            being returned open / shown.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    import matplotlib.pyplot as plt
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors

    b_factors = results["b_factors"]
    atom_names = results["atom_names"]
    atom_indices = np.arange(len(b_factors))
    unique_elements = sorted(set(atom_names))

    def jmol_rgb(sym: str) -> tuple[float, float, float]:
        z = atomic_numbers.get(sym, 0)
        return tuple(jmol_colors[z])  # type: ignore[return-value]

    fig, ax = plt.subplots(figsize=(10, 4))
    for element in unique_elements:
        mask = np.array([n == element for n in atom_names])
        ax.scatter(
            atom_indices[mask],
            b_factors[mask],
            s=20,
            alpha=0.7,
            color=jmol_rgb(element),
            label=element,
            zorder=3,
        )
        mean_b = float(np.mean(b_factors[mask]))
        ax.axhline(
            mean_b, color=jmol_rgb(element), linestyle="--", linewidth=1, alpha=0.6
        )

    ax.set_xlabel("Atom index")
    ax.set_ylabel("B-factor (Å²)")
    ax.set_title("Debye-Waller factors per atom")
    ax.legend(title="Element", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved B-factor plot → %s", output_path)

    return fig


def plot_sigma2_vs_reff(
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Plot σ² vs Reff for 2-body and 3-body paths (matplotlib).

    Args:
        res_2b: 2-body MSRD results as returned by :func:`calculate_grouped_msrd`.
        res_3b: 3-body MSRD results as returned by :func:`calculate_grouped_msrd`.
        output_path: If given, the figure is saved to this path.

    Returns:
        The ``matplotlib.figure.Figure`` object.
    """
    import matplotlib.pyplot as plt
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors

    def jmol_rgb(sym: str) -> tuple[float, float, float]:
        z = atomic_numbers.get(sym, 0)
        return tuple(jmol_colors[z])  # type: ignore[return-value]

    fig, ax = plt.subplots(figsize=(8, 5))

    plotted_labels: set[str] = set()

    def _display_label(path_type: str) -> str:
        """Replace '-' separators with '→' for cleaner legend entries."""
        return path_type.replace("-", " → ")

    for r in res_2b:
        path_type = r["type"]
        scatterer = path_type.split("-")[1] if "-" in path_type else path_type
        color = jmol_rgb(scatterer)
        lbl = (
            _display_label(path_type)
            if path_type not in plotted_labels
            else "_nolegend_"
        )
        ax.scatter(
            r["reff"],
            r["sigma2"],
            marker="o",
            s=60,
            color=color,
            label=lbl,
            zorder=3,
        )
        plotted_labels.add(path_type)

    for r in res_3b:
        path_type = r["type"]
        parts = path_type.split("-")
        scatterer = parts[1] if len(parts) > 1 else path_type
        color = jmol_rgb(scatterer)
        lbl = (
            f"{_display_label(path_type)} (3-body)"
            if path_type not in plotted_labels
            else "_nolegend_"
        )
        ax.scatter(
            r["reff"],
            r["sigma2"],
            marker="^",
            s=60,
            color=color,
            label=lbl,
            alpha=0.8,
            zorder=3,
        )
        plotted_labels.add(path_type)

    ax.set_xlabel("Reff (Å)")
    ax.set_ylabel("σ² (Å²)")
    ax.set_title("MSRD σ² vs Reff")
    ax.legend(title="Path type", framealpha=0.8, fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved σ² vs Reff plot → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# MSRD DataFrame helper
# ---------------------------------------------------------------------------


def msrd_to_dataframe(
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]],
    n_absorbers: int,
) -> Any:
    """Convert MSRD results to a :class:`pandas.DataFrame`.

    Args:
        res_2b: 2-body path results.
        res_3b: 3-body path results.
        n_absorbers: Number of distinct absorber atoms (used for the
            multiplicity estimate).

    Returns:
        A ``pandas.DataFrame`` with columns: ``Body``, ``Path type``,
        ``Reff (Å)``, ``σ² (Å²)``, ``σ² thermal (Å²)``, ``Angle (°)``,
        ``FEFF β (°)``, ``Count``, ``MD multiplicity estimate``.

        The multiplicity column is an MD-side estimate (instances per
        absorber, with an assumed reversal factor of 2 for 3-body paths) —
        it is *not* a validated FEFF degeneracy.
    """
    import pandas as pd

    rows = [
        {
            "_row_id": i,
            "Body": "2-body",
            "Path type": r["type"],
            "Reff (Å)": r["reff"],
            "σ² (Å²)": r["sigma2"],
            "σ² thermal (Å²)": r.get("sigma2_thermal_A2", r["sigma2"]),
            "Angle (°)": float("nan"),
            "FEFF β (°)": float("nan"),
            "Count": r["count"],
            "MD multiplicity estimate": r["count"] / n_absorbers,
        }
        for i, r in enumerate(res_2b)
    ] + [
        {
            "_row_id": len(res_2b) + i,
            "Body": "3-body",
            "Path type": r["type"],
            "Reff (Å)": r["reff"],
            "σ² (Å²)": r["sigma2"],
            "σ² thermal (Å²)": r.get("sigma2_thermal_A2", r["sigma2"]),
            "Angle (°)": r.get("internal_angle_deg", r.get("angle")),
            "FEFF β (°)": r.get("feff_beta_deg", float("nan")),
            "Count": r["count"],
            # Assumed reversal multiplicity of 2 — an MD-side estimate, not
            # a FEFF degeneracy.
            "MD multiplicity estimate": 2 * r["count"] / n_absorbers,
        }
        for i, r in enumerate(res_3b)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MSRD (DW) ↔ FEFF path matching
# ---------------------------------------------------------------------------


def _mean_r_eff(path_group: dict[str, Any]) -> float:
    """Mean effective path length of a FEFF path group."""
    r_effs = path_group.get("r_effs")
    if r_effs is not None and len(r_effs) > 0:
        return float(np.mean(r_effs))
    return float(path_group.get("r_eff_ref", 0.0))


def _mean_feff_angle(path_group: dict[str, Any]) -> float | None:
    """Mean 3-body angle of a FEFF path group, or ``None`` if unavailable.

    Supports either a per-frame ``"angles"`` list (mirroring ``"r_effs"``) or
    a single averaged ``"angle"`` scalar. Returns ``None`` for 2-body/4-leg
    paths or for older data that predates angle extraction.
    """
    angles = path_group.get("angles")
    if angles is not None and len(angles) > 0:
        finite = [float(a) for a in angles if a is not None]
        if finite:
            return float(np.mean(finite))
    angle = path_group.get("angle")
    if angle is not None:
        return float(angle)
    return None


def _is_rattle_path(
    path_group: dict[str, Any],
    leg_rtol: float = 0.05,
    beta_tol_deg: float = 20.0,
) -> bool:
    """Check whether a 4-leg FEFF path is a genuine collinear rattle path.

    A rattle path is ``A → B → A → B → A``: all four legs approximately
    equal and all relevant β angles approximately 180°. Geometry is taken
    from the optional ``"rlegs"`` and ``"betas"`` keys; when neither is
    present the path is accepted (legacy behaviour for older data).

    Args:
        path_group: FEFF path group dict.
        leg_rtol: Maximum allowed relative spread of the four leg lengths.
        beta_tol_deg: Maximum allowed deviation of each β angle from 180°.

    Returns:
        ``True`` if the path is consistent with a collinear rattle path.
    """
    rlegs = path_group.get("rlegs")
    if rlegs is not None and len(rlegs) == 4:
        r = np.asarray(rlegs, dtype=float)
        mean_r = max(float(np.mean(r)), 1e-12)
        if float(np.max(r) - np.min(r)) > leg_rtol * mean_r:
            return False
    betas = path_group.get("betas")
    if betas is not None and len(betas) > 0:
        if any(abs(float(b) - 180.0) > beta_tol_deg for b in betas):
            return False
    # Legacy behaviour: accept 4-leg paths without stored geometry.
    return True


def _rattle_scatterer_key(
    scatterer: str, absorber: str | None = None
) -> tuple[str, ...] | None:
    """Canonical scatterer key for a 4-leg collinear rattle path A→B→A→B→A.

    The FEFF pipeline labels 4-leg paths with *all three* intermediate
    elements, sorted (e.g. ``"Mn-O-O"`` for an O rattle on an Mn absorber —
    the absorber is revisited mid-path, so it appears in the label). A
    genuine rattle's token multiset is therefore ``{B: 2, A: 1}`` (or
    ``{B: 3}`` when absorber and scatterer are the same element); anything
    else cannot be a rattle and must never match a 2-body DW group. A bare
    single-element label (``"O"``, legacy/test data) is taken at face value.

    Note: the token multiset cannot distinguish a true collinear rattle
    ``A→B→A→B→A`` from a bent 4-leg path visiting the same elements
    (``A→B→C→B→A`` with C the absorber species on a different site) — when
    the path carries leg geometry, :func:`_is_rattle_path` provides the
    stricter check on top of this gating.

    Args:
        scatterer: The path's scatterer label.
        absorber: Absorber element, if known — the singleton token must
            equal it (otherwise the label is not a rattle for this absorber).

    Returns:
        ``(B,)`` for a rattle-compatible label, else ``None``.
    """
    tokens = canonical_scatterer_key(scatterer)
    if len(tokens) == 1:
        return tokens
    if len(tokens) != 3:
        return None
    counts = {t: tokens.count(t) for t in set(tokens)}
    if len(counts) == 1:  # {B: 3} — absorber and scatterer are the same element
        (b,) = counts
        if absorber and absorber != b:
            return None
        return (b,)
    if len(counts) == 2:  # {B: 2, A: 1}
        double = next(t for t, n in counts.items() if n == 2)
        single = next(t for t, n in counts.items() if n == 1)
        if absorber and absorber != single:
            return None
        return (double,)
    return None


def _build_feff_trees(
    path_groups: list[dict[str, Any]],
) -> dict[tuple[str, tuple[str, ...], int], tuple[Any, list[int]]]:
    """Group FEFF paths and build a KD-tree per group for fast lookup.

    Buckets are keyed by (leg-kind, canonical scatterer, feature-ndim).

    ``leg_kind`` is one of ``"2b"`` (``nlegs == 2``), ``"4b"`` (``nlegs == 4``,
    collinear rattle), or ``"3b"`` (``nlegs == 3``). Splitting further by
    feature dimensionality lets 3-body FEFF paths with a known angle
    (2-D Å-equivalent vector) and those without one (1-D distance-only
    fallback, e.g. older data) coexist without a dimension mismatch.

    Returns:
        ``{(leg_kind, canonical_scatterer, ndim): (cKDTree, group_indices)}``
        where ``group_indices[i]`` is the original ``path_groups`` index for
        the ``i``-th point stored in the tree.
    """
    from scipy.spatial import cKDTree

    buckets: dict[tuple[str, tuple[str, ...], int], list[tuple[int, np.ndarray]]] = (
        defaultdict(list)
    )
    for gi, pg in enumerate(path_groups):
        nlegs = pg.get("nlegs")
        if nlegs not in (2, 3, 4):
            continue
        leg_kind = {2: "2b", 4: "4b", 3: "3b"}[nlegs]
        if leg_kind == "4b":
            if not _is_rattle_path(pg):
                # A 4-leg path that is not a collinear A-B-A-B-A rattle path
                # must never be matched against a 2-body DW group.
                continue
            # The pipeline label includes the revisited absorber element
            # ("Mn-O-O" for an O rattle on Mn); derive the actual rattle
            # scatterer from the token multiset.
            cat = _rattle_scatterer_key(
                str(pg.get("scatterer", "?")), absorber=pg.get("absorber")
            )
            if cat is None:
                continue
        else:
            cat = canonical_scatterer_key(str(pg.get("scatterer", "?")))
        mean_reff = _mean_r_eff(pg)
        if leg_kind == "4b":
            mean_reff = mean_reff / 2.0
        angle = _mean_feff_angle(pg) if leg_kind == "3b" else None
        vec = path_feature_vector(mean_reff, angle)
        buckets[(leg_kind, cat, vec.shape[0])].append((gi, vec))

    trees: dict[tuple[str, tuple[str, ...], int], tuple[Any, list[int]]] = {}
    for key, entries in buckets.items():
        idxs = [e[0] for e in entries]
        vecs = np.array([e[1] for e in entries])
        trees[key] = (cKDTree(vecs), idxs)
    return trees


def _query_feff_candidates(
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]],
    path_groups: list[dict[str, Any]],
    r_tol: float,
) -> tuple[
    dict[tuple[str, int], list[tuple[float, int, str, int]]],
    dict[int, list[tuple[float, str, int]]],
]:
    """Find *all* eligible FEFF paths within tolerance of every DW group.

    This is the symmetric many-to-many primitive behind
    :func:`match_msrd_paths_to_feff`: each DW group may legitimately
    correspond to *several* FEFF paths within tolerance (e.g. one MSRD σ²
    applied to several near-equivalent FEFF paths in a fit), and each FEFF
    path may be claimed by several DW groups (degeneracy). Neither direction
    is arbitrated to a single winner here — the full candidate sets are
    returned so callers can decide how to summarize them.

    Eligibility is gated by a canonicalized scatterer/element-pair key and
    compared using the Å-equivalent feature vector (distance, or
    distance + angle-arc-length for 3-body paths), exactly as in
    :func:`_match_dw_to_feff`.

    Returns:
        Tuple ``(dw_to_feff, feff_to_dw)`` where

        * ``dw_to_feff[(body, dw_idx)]`` is the list of candidate FEFF paths
          ``(distance, group_idx, feff_note, feff_nlegs)`` sorted by distance
          (empty list if none within tolerance);
        * ``feff_to_dw[group_idx]`` is the list of DW groups
          ``(distance, body, dw_idx)`` claiming that FEFF path, sorted by
          distance.
    """
    trees = _build_feff_trees(path_groups)
    dw_to_feff: dict[tuple[str, int], list[tuple[float, int, str, int]]] = {}

    def _collect(tree_entry: Any, v: np.ndarray, note: str, nlegs: int) -> list:
        tree, idxs = tree_entry
        candidates = []
        for ii in tree.query_ball_point(v, r_tol):
            d = float(np.linalg.norm(tree.data[ii] - v))
            if d <= r_tol:
                candidates.append((d, idxs[ii], note, nlegs))
        return candidates

    def _absorber_ok(m: dict[str, Any], gi: int) -> bool:
        """Gate on absorber identity when *both* sides declare one."""
        dw_abs = m.get("absorber")
        pg_abs = path_groups[gi].get("absorber")
        if dw_abs and pg_abs:
            return str(dw_abs) == str(pg_abs)
        return True

    for di, m in enumerate(res_2b):
        scatterer_key = canonical_scatterer_key(m["type"].split("-")[-1])
        candidates = []
        for leg_kind, note, feff_nlegs in (
            ("2b", "direct ss", 2),
            ("4b", "rattle (4×σ²)", 4),
        ):
            tree_entry = trees.get((leg_kind, scatterer_key, 1))
            if tree_entry is not None:
                candidates += _collect(
                    tree_entry, np.array([m["reff"]]), note, feff_nlegs
                )
        dw_to_feff[("2b", di)] = sorted(
            (c for c in candidates if _absorber_ok(m, c[1])), key=lambda c: c[0]
        )

    for di, m in enumerate(res_3b):
        elements = m["type"].split("-")[1:]  # e.g. "K-N-C" -> ["N", "C"]
        scatterer_key = canonical_scatterer_key("-".join(elements))
        # The FEFF-side angle is the internal angle at the *first-listed*
        # scatterer of the path file — an arbitrary one of the two scatterer
        # vertices (and for reversal-degenerate paths potentially either).
        # The DW group's anchored "angle" is at the scatterer with the
        # shorter absorber leg, which for mixed-element triangles is the
        # *other* vertex. Compare the FEFF angle against both DW scatterer
        # angles ("scatterer_angles_deg", sorted pair) and keep the closest.
        dw_angles: list[float] = []
        for _a in m.get("scatterer_angles_deg") or ():
            if _a is not None and float(_a) not in dw_angles:
                dw_angles.append(float(_a))
        if not dw_angles and m.get("angle") is not None:
            dw_angles = [float(m["angle"])]
        candidates = []
        tree2 = trees.get(("3b", scatterer_key, 2))
        if tree2 is not None and dw_angles:
            for _qa in dw_angles:
                candidates += _collect(
                    tree2,
                    path_feature_vector(m["reff"], _qa),
                    "direct triangular",
                    3,
                )
        # 1-D distance-only fallback bucket (FEFF paths without a stored
        # angle, e.g. older data).
        tree1 = trees.get(("3b", scatterer_key, 1))
        if tree1 is not None:
            candidates += _collect(
                tree1,
                path_feature_vector(m["reff"], None),
                "direct triangular",
                3,
            )
        # A FEFF path may be found via several DW-side angles — keep the
        # smallest distance.
        best: dict[int, tuple[float, int, str, int]] = {}
        for _cand in candidates:
            if _cand[1] not in best or _cand[0] < best[_cand[1]][0]:
                best[_cand[1]] = _cand
        dw_to_feff[("3b", di)] = sorted(
            (c for c in best.values() if _absorber_ok(m, c[1])), key=lambda c: c[0]
        )

    feff_to_dw: dict[int, list[tuple[float, str, int]]] = defaultdict(list)
    for (body, di), cands in dw_to_feff.items():
        for d, gi, _note, _nlegs in cands:
            feff_to_dw[gi].append((d, body, di))
    for gi in feff_to_dw:
        feff_to_dw[gi].sort(key=lambda c: c[0])

    return dw_to_feff, dict(feff_to_dw)


def _representative_candidate(
    cands: list[tuple[float, int, str, int]],
) -> tuple[float, int, str, int]:
    """Pick the representative match from a distance-sorted candidate list.

    For 2-body DW groups, a direct single-scattering (``nlegs == 2``) path is
    preferred over a 4-leg rattle even when the rattle is marginally nearer
    on the matching axis: the rattle assignment rests on the approximate
    ``σ²_rattle = 4σ²`` relation (and, without stored leg geometry, on the
    token-multiset gating alone), so it should only *represent* the group
    when no direct SS path is available within tolerance. The full candidate
    list is unaffected — both still count as matched for coverage.
    """
    for c in cands:
        if c[3] == 2:
            return c
    return cands[0]


def match_paths_within_tolerance(
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]] | None,
    path_groups: list[dict[str, Any]],
    r_tol: float,
) -> dict[str, Any]:
    """Symmetric many-to-many MSRD ↔ FEFF candidate matching.

    See :func:`_query_feff_candidates` for the semantics. This public wrapper
    exists so notebooks/UI code can work with the full candidate structure
    (coverage in *both* directions) without importing private helpers.

    Returns:
        Dict with keys ``"dw_to_feff"`` and ``"feff_to_dw"`` as documented
        for :func:`_query_feff_candidates`.
    """
    dw_to_feff, feff_to_dw = _query_feff_candidates(
        res_2b, res_3b or [], path_groups, r_tol
    )
    return {"dw_to_feff": dw_to_feff, "feff_to_dw": feff_to_dw}


def _match_dw_to_feff(
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]],
    path_groups: list[dict[str, Any]],
    r_tol: float,
) -> dict[tuple[str, int], tuple[int, float, str, int]]:
    """Match each MSRD (DW) path group to its nearest eligible FEFF path.

    This is a genuine (non-exclusive) nearest-neighbor match: each DW group
    independently finds its closest FEFF path within ``r_tol``, gated by a
    canonicalized scatterer/element-pair key and compared using the
    Å-equivalent feature vector (distance, or distance + angle-arc-length
    for 3-body paths). Several DW groups may legitimately match the *same*
    FEFF path — e.g. a spuriously split MD shell, or genuine
    crystallographic degeneracy — see :func:`match_msrd_paths_to_feff` for
    how this is surfaced via ``shared_with``/``contribution_pct_share``
    rather than silently arbitrated away.

    This returns only the *representative* candidate per DW group — the
    nearest, except that a direct single-scattering (2-leg) path is
    preferred over a 4-leg rattle when both are within tolerance (see
    :func:`_representative_candidate`). Use
    :func:`match_paths_within_tolerance` for the full many-to-many
    candidate sets.

    Returns:
        ``{(body, dw_idx): (group_idx, distance, feff_note, feff_nlegs)}``
        for every DW group with an eligible FEFF match within tolerance.
    """
    dw_to_feff, _feff_to_dw = _query_feff_candidates(res_2b, res_3b, path_groups, r_tol)
    matches: dict[tuple[str, int], tuple[int, float, str, int]] = {}
    for key, cands in dw_to_feff.items():
        if cands:
            d, gi, note, nlegs = _representative_candidate(cands)
            matches[key] = (gi, d, note, nlegs)
    return matches


def match_msrd_paths_to_feff(
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]] | None,
    path_groups: list[dict[str, Any]],
    r_tol: float,
    ambiguity_margin: float = 0.02,
) -> list[dict[str, Any]]:
    """Match MSRD/Debye-Waller path groups to FEFF path groups.

    Each MSRD (DW) group independently finds its nearest eligible FEFF path
    (nearest-neighbor, no exclusivity — but a direct 2-leg SS path is
    preferred over a marginally nearer 4-leg rattle for the representative,
    see :func:`_representative_candidate`). Several DW groups may share the
    same FEFF path — this is allowed and reported, not silently arbitrated:
    each row's ``contribution_pct_share`` divides that FEFF path's
    ``contribution_pct`` evenly across all DW rows whose *nearest* match is
    that path, so downstream sums don't double-count, while
    ``contribution_pct`` keeps the raw, unsplit value and ``shared_with``
    lists the other DW indices (same body) sharing this FEFF path.

    The match is many-to-many in both directions: a DW group may also have
    *several* FEFF paths within tolerance (e.g. one MSRD σ² applied to
    several near-equivalent FEFF paths in a fit). The full candidate list is
    exposed per row via ``candidate_group_idxs`` (nearest first) and
    ``n_candidates``; the single-``group_idx`` field is always the nearest
    candidate for backwards compatibility.

    Eligibility rules (DW group → FEFF path), gated by a canonicalized
    scatterer/element-pair key so e.g. an Fe path can never match an O path:

    * 2-body DW → FEFF ``nlegs == 2``: ``|reff_dw - <r_eff>| < r_tol``.
    * 2-body DW → FEFF ``nlegs == 4`` (collinear rattle):
      ``|reff_dw - <r_eff>/2| < r_tol``.
    * 3-body DW → FEFF ``nlegs == 3``, same element pair: Euclidean distance
      on the Å-equivalent feature vector ``[reff, reff·radians(angle)]``
      (falling back to distance-only if the FEFF path has no stored angle).
      Because the FEFF-side angle is measured at an arbitrary one of the two
      scatterer vertices (first-listed in the path file), it is compared
      against *both* DW scatterer angles and the smaller distance is kept.

    Args:
        res_2b: 2-body MSRD results (``type``, ``reff``, ``sigma2``,
            ``count``).
        res_3b: 3-body MSRD results (adds ``angle``); may be ``None``.
        path_groups: FEFF path groups (``nlegs``, ``scatterer``, ``r_effs``,
            ``angle``/``angles``, ``contribution_pct``).
        r_tol: Distance tolerance (Å or Å-equivalent).
        ambiguity_margin: If the runner-up candidate is within this distance
            of the nearest candidate, the row is flagged ``"ambiguous"``.

    Returns:
        One row dict per MSRD group (2-body rows first, then 3-body).
    """
    res_3b = res_3b or []
    dw_to_feff, _feff_to_dw = _query_feff_candidates(res_2b, res_3b, path_groups, r_tol)
    # Representative (nearest) match per DW group, for backwards-compatible
    # group_idx/contribution fields.
    matches: dict[tuple[str, int], tuple[int, float, str, int]] = {
        key: (lambda _c: (_c[1], _c[0], _c[2], _c[3]))(_representative_candidate(cands))
        for key, cands in dw_to_feff.items()
        if cands
    }

    # group_idx -> list of (body, dw_idx) currently matched to it, so
    # contribution_pct can be split evenly across legitimate sharers.
    sharers: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for (body, dw_i), (gi, *_rest) in matches.items():
        sharers[gi].append((body, dw_i))

    def _row(body: str, dw_i: int, m: dict[str, Any], is_2b: bool) -> dict[str, Any]:
        matched = matches.get((body, dw_i))
        group_idx = matched[0] if matched else None
        match_distance = matched[1] if matched else None
        feff_note = matched[2] if matched else "(no FEFF match)"
        feff_nlegs = matched[3] if matched else None
        cands = dw_to_feff.get((body, dw_i), [])
        reff_exafs = (
            _mean_r_eff(path_groups[group_idx]) if group_idx is not None else None
        )
        contrib = (
            float(path_groups[group_idx].get("contribution_pct", 0.0))
            if group_idx is not None
            else 0.0
        )
        n_sharing = len(sharers[group_idx]) if group_idx is not None else 0
        shared_with = (
            [dwi for (b, dwi) in sharers[group_idx] if not (b == body and dwi == dw_i)]
            if group_idx is not None
            else []
        )
        contrib_share = contrib / n_sharing if n_sharing else 0.0
        ambiguous = len(cands) > 1 and (cands[1][0] - cands[0][0]) < ambiguity_margin
        # Var(Reff) of a 4-leg rattle path A->B->A->B->A is Var(2R) = 4 Var(R),
        # so the plain 2-body sigma2 must be scaled before it is applied to
        # such a FEFF path. Exposed explicitly rather than hidden in the note.
        sigma2_feff = 4.0 * m["sigma2"] if feff_nlegs == 4 else m["sigma2"]
        scatterer = (
            m["type"].split("-")[-1] if is_2b else "-".join(m["type"].split("-")[1:])
        )
        return {
            "dw_body": body,
            "dw_idx": dw_i,
            "Body": "2-body" if is_2b else "3-body",
            "Type": m["type"],
            "Scatterer": scatterer,
            "reff_dw": m["reff"],
            "sigma2": m["sigma2"],
            "sigma2_feff": sigma2_feff,
            "count": m["count"],
            "angle": None if is_2b else m.get("angle"),
            "reff_exafs": reff_exafs,
            "nlegs_feff": feff_nlegs,
            "group_idx": group_idx,
            "feff_note": feff_note,
            # Full Å-equivalent match distance from _match_dw_to_feff (for
            # 3-body paths this includes the angle-arc-length term, so it can
            # exceed |reff_dw - reff_exafs| alone).
            "match_distance": match_distance,
            # All FEFF paths within tolerance of this DW group (nearest
            # first), not just the representative — the symmetric
            # many-to-many view. ``group_idx`` is always the first entry.
            "candidate_group_idxs": [gi for _d, gi, _n, _nl in cands],
            "n_candidates": len(cands),
            "ambiguous": ambiguous,
            "contribution_pct": contrib,
            "contribution_pct_share": contrib_share,
            "shared_with": shared_with,
        }

    rows: list[dict[str, Any]] = []
    for dw_i, m in enumerate(res_2b):
        rows.append(_row("2b", dw_i, m, True))
    for dw_i, m in enumerate(res_3b):
        rows.append(_row("3b", dw_i, m, False))

    return rows


def pool_dw_groups_by_feff_path(
    match_rows: list[dict[str, Any]],
    path_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine DW/MSRD groups that share one matched FEFF path.

    The many-to-many matcher (:func:`match_msrd_paths_to_feff`) legitimately
    assigns several DW groups to the same FEFF path — e.g. two
    crystallographically distinct triangle populations built from different
    neighbour shells that happen to share the same three leg lengths (hence
    the same Reff and one common vertex angle), which FEFF's own path finder
    pooled into a single averaged path. A static single-structure FEFF fit
    needs exactly *one* (Reff, σ²) pair per path, so the contributing DW
    groups must be combined — naively averaging their σ² values would
    ignore that they may also differ in mean Reff, understating the true
    positional disorder FEFF would need to reproduce the ensemble's
    amplitude damping.

    This applies the law of total variance for the mixture of the pooled
    groups' underlying per-instance, per-frame samples::

        Reff_combined = Σ w_k · Reff_k / Σ w_k
        σ²_combined   = Σ w_k · [σ²_k + (Reff_k − Reff_combined)²] / Σ w_k

    i.e. the weighted mean of each group's own (within-group) variance
    *plus* the weighted variance of the group means around the combined
    mean (the "between-group" spread that a naive σ² average would miss).
    Weights ``w_k`` are each DW group's raw instance count
    (``"count"``, i.e. its N_pairs): since every group being combined here
    comes from the *same* trajectory (one MSRD run), the common
    number-of-frames factor in each group's true raw sample count cancels
    out of the weighted averages, leaving the instance count as the exact
    weight (verified against a direct pooled-sample calculation). Combining
    groups from *different* trajectories/frame counts would need
    ``count × n_frames`` instead.

    4-leg rattle matches are rescaled to the FEFF path's own length/variance
    scale (``Reff_dw × 2``, using the already-scaled ``σ²_feff = 4σ²``)
    before combining, so a rattle and a direct path sharing one FEFF entry
    combine on a consistent scale.

    Args:
        match_rows: Output of :func:`match_msrd_paths_to_feff` for the same
            ``path_groups``.
        path_groups: FEFF path groups.

    Returns:
        One row per matched FEFF path, sorted by ``reff_exafs``, with keys
        ``group_idx``, ``reff_exafs``, ``nlegs_feff``, ``scatterer``,
        ``n_dw_groups``, ``reff_combined``, ``sigma2_combined``,
        ``between_group_variance_A2`` (the extra σ² contributed by Reff
        spread across the pooled DW groups — zero when ``n_dw_groups == 1``),
        ``count_total``, ``contribution_pct``, and ``members`` (the
        contributing rows' ``Type``/``reff_dw``/``angle``/``sigma2``/
        ``count``, for provenance).
    """
    by_gi: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in match_rows:
        gi = row.get("group_idx")
        if gi is not None:
            by_gi[gi].append(row)

    pooled: list[dict[str, Any]] = []
    for gi, rows in by_gi.items():
        pg = path_groups[gi]
        nlegs_feff = int(pg.get("nlegs", rows[0]["nlegs_feff"]))
        scale = 2.0 if nlegs_feff == 4 else 1.0
        reffs = np.array([r["reff_dw"] * scale for r in rows], dtype=float)
        sigma2s = np.array([r["sigma2_feff"] for r in rows], dtype=float)
        weights = np.array([float(r["count"]) for r in rows], dtype=float)
        total_w = float(np.sum(weights))
        reff_combined = float(np.sum(weights * reffs) / total_w)
        between = float(np.sum(weights * (reffs - reff_combined) ** 2) / total_w)
        sigma2_combined = float(np.sum(weights * sigma2s) / total_w) + between
        pooled.append(
            {
                "group_idx": gi,
                "reff_exafs": _mean_r_eff(pg),
                "nlegs_feff": nlegs_feff,
                "scatterer": str(pg.get("scatterer", "?")),
                "n_dw_groups": len(rows),
                "reff_combined": reff_combined,
                "sigma2_combined": sigma2_combined,
                "between_group_variance_A2": between,
                "count_total": int(sum(r["count"] for r in rows)),
                "contribution_pct": float(pg.get("contribution_pct", 0.0)),
                "members": [
                    {
                        "type": r["Type"],
                        "reff_dw": r["reff_dw"],
                        "angle": r["angle"],
                        "sigma2": r["sigma2"],
                        "count": r["count"],
                    }
                    for r in rows
                ],
            }
        )
    return sorted(pooled, key=lambda r: r["reff_exafs"])


def merge_congruent_3body_groups(
    res_3b: list[dict[str, Any]],
    leg_tol: float = 0.1,
) -> list[dict[str, Any]]:
    """Merge 3-body DW groups whose triangles are congruent, without a FEFF file.

    Two distinct sets of neighbour atoms can form triangles with the *same
    three leg lengths* — e.g. one population uses a 1st-shell scatterer +
    2nd-shell scatterer with a long hop between them; another uses the
    2nd-shell scatterer + a 3rd-shell scatterer with a short hop, giving the
    same perimeter. :func:`group_path_instances` anchors its reported angle
    at whichever scatterer has the shorter *absorber* leg, which differs
    between such populations (e.g. 55° vs 90°), so they end up as two
    separate rows despite being congruent (by the SSS theorem, congruent
    triangles have identical Reff *and* identical sets of internal angles).
    A conventional external FEFF calculation run on a static structure
    typically cannot distinguish such congruent geometries either (its own
    path/angle bookkeeping is comparably coarse — see
    :func:`match_msrd_paths_to_feff`'s "compare against both scatterer
    angles" handling, which is exactly this same congruence class), so for
    taking DW results into a static-structure FEFF calculation these
    populations should usually be combined into one (Reff, σ²) pair, with
    no FEFF file needed to discover the pairing.

    Groups are merged when they share the same element pair (``"type"``)
    and the same *sorted* triple of leg lengths
    (``"leg_lengths_A"``, invariant to which specific atom plays
    "absorber-adjacent" vs "opposite") within ``leg_tol`` (Å) of each other,
    via a Union-Find over the pairwise congruence graph. Combination uses
    the same law-of-total-variance formula as
    :func:`pool_dw_groups_by_feff_path`, weighted by each group's raw
    instance count (``"count"``) — exact when every input group comes from
    the same trajectory (frame count cancels).

    Args:
        res_3b: 3-body MSRD results, as returned by
            :func:`group_path_instances`/:func:`calculate_grouped_msrd`
            (each group must carry ``"leg_lengths_A"``).
        leg_tol: Per-leg tolerance (Å) for two sorted leg triples to be
            considered congruent. The default (0.1 Å) was validated against
            a real MD dataset: it catches every population pair
            independently confirmed (via cross-matching to an external
            FEFF calculation) to share one FEFF path, without pulling in
            genuinely different triangles.

    Returns:
        List of merged rows sorted by ``reff_combined``, each with keys
        ``type``, ``reff_combined``, ``sigma2_combined``,
        ``between_group_variance_A2`` (the extra σ² contributed by Reff
        spread across the congruent populations — zero when
        ``n_dw_groups == 1``), ``count_total``, ``n_dw_groups``, and
        ``members`` (the contributing groups' ``type``/``reff``/``angle``/
        ``sigma2``/``count``, for provenance).
    """
    parent = list(range(len(res_3b)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    by_type: defaultdict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(res_3b):
        by_type[g["type"]].append(i)

    for idxs in by_type.values():
        legs = [tuple(sorted(res_3b[i]["leg_lengths_A"])) for i in idxs]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if (
                    max(abs(x - y) for x, y in zip(legs[a], legs[b], strict=True))
                    <= leg_tol
                ):
                    _union(idxs[a], idxs[b])

    clusters: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(len(res_3b)):
        clusters[_find(i)].append(i)

    merged: list[dict[str, Any]] = []
    for idxs in clusters.values():
        members = [res_3b[i] for i in idxs]
        weights = np.array([float(m["count"]) for m in members], dtype=float)
        reffs = np.array([m["reff"] for m in members], dtype=float)
        sigma2s = np.array([m["sigma2"] for m in members], dtype=float)
        total_w = float(np.sum(weights))
        reff_combined = float(np.sum(weights * reffs) / total_w)
        between = float(np.sum(weights * (reffs - reff_combined) ** 2) / total_w)
        sigma2_combined = float(np.sum(weights * sigma2s) / total_w) + between
        merged.append(
            {
                "type": members[0]["type"],
                "reff_combined": reff_combined,
                "sigma2_combined": sigma2_combined,
                "between_group_variance_A2": between,
                "count_total": int(sum(m["count"] for m in members)),
                "n_dw_groups": len(members),
                "members": [
                    {
                        "type": m["type"],
                        "reff": m["reff"],
                        "angle": m.get("internal_angle_deg"),
                        "sigma2": m["sigma2"],
                        "count": m["count"],
                    }
                    for m in members
                ],
            }
        )
    return sorted(merged, key=lambda r: r["reff_combined"])


def extract_path_matching_diagnostics(
    res_2b: list[dict[str, Any]],
    res_3b: list[dict[str, Any]] | None = None,
    path_groups: list[dict[str, Any]] | None = None,
    r_tol: float = 0.1,
) -> dict[str, Any]:
    """Extract tidy, plot-ready records to diagnose path merging and matching.

    The returned ``records`` place every MSRD/DW group and (optionally) every
    FEFF path on a common ``reff_axis`` so that overlaps, gaps, and
    match/no-match outcomes are easy to visualise as a histogram or strip plot.

    Two kinds of diagnostics are supported:

    * **Merging** (call with ``path_groups=None``): shows where the MD-derived
      DW groups landed on the Reff axis, how many pairs merged into each
      (``count``) and their spread (``sigma``).  Adjacent groups whose
      ``reff ± sigma`` ranges overlap indicate borderline clustering.
    * **Matching** (call with ``path_groups`` supplied): additionally places
      the FEFF paths on the same axis and annotates every entity with its
      match ``status`` and the ``distance`` to its nearest partner, using the
      same symmetric many-to-many matching as
      :func:`match_msrd_paths_to_feff`. Coverage is reported in *both*
      directions: a DW group is "matched" if at least one FEFF path lies
      within tolerance, and a FEFF path is "matched" if at least one DW
      group lies within tolerance — the FEFF-side count does *not* drop
      paths merely because a closer competitor won the DW group's nearest
      match. FEFF paths covered by more than one DW group are reported via
      ``n_dw_matches`` on the FEFF record and counted in
      ``summary["n_feff_shared"]``.

    Args:
        res_2b: 2-body MSRD results.
        res_3b: 3-body MSRD results; may be ``None``.
        path_groups: FEFF path groups; ``None`` for a merging-only diagnostic.
        r_tol: Distance tolerance (Å or Å-equivalent), used for matching and
            reported in ``summary``.

    Returns:
        Dict with keys:

        * ``records``: list of per-entity dicts sharing the schema
          ``role`` (``"DW"``/``"FEFF"``), ``label``, ``element``, ``body``,
          ``reff_axis``, ``reff_raw``, ``sigma2``, ``sigma``, ``count``,
          ``contribution_pct``, ``angle``, ``nlegs``, ``matched``,
          ``status``, ``distance``, ``partner_reff_axis``, ``partner_label``,
          ``n_dw_matches`` (FEFF records only), ``idx``.
        * ``summary``: counts (``n_dw``, ``n_dw_matched``, ``n_dw_unmatched``,
          ``n_feff``, ``n_feff_matched``, ``n_feff_unmatched``,
          ``n_feff_shared``).
        * ``r_tol``: the tolerance used.
    """
    res_3b = res_3b or []
    has_feff = path_groups is not None
    pgs = path_groups or []

    dw_to_feff, feff_to_dw = _query_feff_candidates(res_2b, res_3b, pgs, r_tol)
    # Representative (nearest) match per DW group.
    matches: dict[tuple[str, int], tuple[int, float, str, int]] = {
        key: (lambda _c: (_c[1], _c[0], _c[2], _c[3]))(_representative_candidate(cands))
        for key, cands in dw_to_feff.items()
        if cands
    }

    records: list[dict[str, Any]] = []

    def _dw_record(body: str, dw_i: int, m: dict[str, Any], is_2b: bool) -> dict:
        if is_2b:
            element = m["type"].split("-")[-1]
            angle = None
            angle_std = None
        else:
            element = "-".join(m["type"].split("-")[1:])
            _angle = m.get("angle")
            angle = float(_angle) if _angle is not None else None
            _angle_var = m.get("angle_var")
            angle_std = float(_angle_var) ** 0.5 if _angle_var is not None else None
        matched = matches.get((body, dw_i)) if has_feff else None
        # The match's stored Reff is the raw FEFF path length; place it on
        # the matching axis (r_eff/2 for a 4-leg rattle) for plotting.
        partner_axis = None
        distance = None
        if matched is not None:
            gi = matched[0]
            raw_reff = _mean_r_eff(pgs[gi])
            partner_axis = raw_reff / 2.0 if matched[3] == 4 else raw_reff
            # Use the actual match distance from _match_dw_to_feff (the full
            # Å-equivalent distance, including the angle-arc-length term for
            # 3-body paths), not a Reff-axis-only recomputation — the latter
            # would understate the mismatch for a 3-body match that's close
            # in Reff but was accepted/rejected largely due to angle.
            distance = matched[1]
        sigma2 = float(m["sigma2"])
        return {
            "role": "DW",
            "label": m["type"],
            "element": element,
            "body": "2-body" if is_2b else "3-body",
            "reff_axis": float(m["reff"]),
            "reff_raw": float(m["reff"]),
            "sigma2": sigma2,
            "sigma": float(sigma2**0.5),
            "count": int(m["count"]),
            "contribution_pct": None,
            "angle": angle,
            "angle_std": angle_std,
            "nlegs": None,
            "matched": matched is not None,
            "status": (
                "unmatched"
                if not has_feff
                else ("matched" if matched is not None else "unmatched")
            ),
            "distance": distance,
            "partner_reff_axis": partner_axis,
            "partner_label": (f"path#{matched[0]}" if matched else None),
            "idx": dw_i,
        }

    for dw_i, m in enumerate(res_2b):
        records.append(_dw_record("2b", dw_i, m, True))
    for dw_i, m in enumerate(res_3b):
        records.append(_dw_record("3b", dw_i, m, False))

    n_dw = len(records)
    n_dw_matched = sum(1 for r in records if r["role"] == "DW" and r["matched"])

    n_feff_matched = 0
    n_feff_shared = 0
    if has_feff:
        for gi, pg in enumerate(pgs):
            nlegs = int(pg["nlegs"])
            raw = _mean_r_eff(pg)
            axis = raw / 2.0 if nlegs == 4 else raw
            # Symmetric coverage: every DW group within tolerance of this
            # FEFF path, not just those whose *nearest* match it is. A FEFF
            # path is "matched" if at least one DW group covers it.
            dw_matches = feff_to_dw.get(gi, [])
            partner_label = None
            partner_axis = None
            distance = None
            if dw_matches:
                # Report the closest of the (possibly several) DW matches.
                distance, body0, di0 = dw_matches[0]
                src = res_2b if body0 == "2b" else res_3b
                partner_axis = float(src[di0]["reff"])
                partner_label = src[di0]["type"]
                n_feff_matched += 1
                if len(dw_matches) > 1:
                    n_feff_shared += 1
            records.append(
                {
                    "role": "FEFF",
                    "label": f"path#{gi} {pg.get('scatterer', '?')}",
                    "element": str(pg.get("scatterer", "?")),
                    "body": f"{nlegs}-leg",
                    "reff_axis": float(axis),
                    "reff_raw": float(raw),
                    "sigma2": None,
                    "sigma": None,
                    "count": None,
                    "contribution_pct": float(pg.get("contribution_pct", 0.0)),
                    "angle": _mean_feff_angle(pg) if nlegs == 3 else None,
                    "nlegs": nlegs,
                    "matched": bool(dw_matches),
                    "status": "matched" if dw_matches else "unmatched",
                    "distance": distance,
                    "partner_reff_axis": partner_axis,
                    "partner_label": partner_label,
                    "n_dw_matches": len(dw_matches),
                    "idx": gi,
                }
            )

    n_feff = len(pgs)

    summary = {
        "n_dw": n_dw,
        "n_dw_matched": n_dw_matched,
        "n_dw_unmatched": n_dw - n_dw_matched,
        "n_feff": n_feff,
        "n_feff_matched": n_feff_matched,
        "n_feff_unmatched": n_feff - n_feff_matched,
        "n_feff_shared": n_feff_shared,
    }

    return {"records": records, "summary": summary, "r_tol": r_tol}


# ---------------------------------------------------------------------------
# External (FEFF / MD-EXAFS) ordered-path matching
# ---------------------------------------------------------------------------


@dataclass
class ExternalPath:
    """An ordered scattering path from an external source (FEFF, MD-EXAFS).

    Unlike :class:`PathInstance` (which is canonical and
    orientation-independent), this schema preserves the *ordered* geometry
    needed for exact path-identity matching.

    Attributes:
        source: Origin of the path, e.g. ``"feff"`` or ``"mdexafs"``.
        run_id: Identifier of the producing run (e.g. FEFF calculation ID).
        external_path_id: Path identifier within that run (e.g. ``"0004"``).
        nleg: Number of legs of the closed path.
        degeneracy: FEFF degeneracy, if known.
        absorber_key: Absorber species or ``ipot`` string.
        intermediate_keys: Ordered intermediate species/``ipot`` keys.
        rlegs: Ordered leg lengths (Å), if available.
        betas: Ordered FEFF β angles (degrees), if available.
        etas: Ordered FEFF η angles (degrees), if available.
        coordinates: Optional path atom coordinates.
    """

    source: str
    run_id: str
    external_path_id: str
    nleg: int
    degeneracy: int | None = None
    absorber_key: str = ""
    intermediate_keys: tuple[str, ...] = ()
    rlegs: tuple[float, ...] | None = None
    betas: tuple[float, ...] | None = None
    etas: tuple[float, ...] | None = None
    coordinates: np.ndarray | None = None


def _orientation_geometry(
    sample: PathSamples, orientation: tuple[int, ...]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None]:
    """Ordered (intermediate species, legs, betas) for one orientation.

    Legs are reordered from the canonical ``[d(A,B), d(B,C), d(C,A)]``
    storage to the orientation's traversal order; β angles are the FEFF
    scattering angles at the intermediate vertices in traversal order.
    """
    instance = sample.instance
    canon = instance.path_id.intermediates
    species_map = dict(zip(canon, instance.species[1:], strict=True))
    intermediate_keys = tuple(species_map[i] for i in orientation)

    # Canonical leg index between consecutive path atoms.
    sequence = (instance.absorber, *canon)
    canon_pairs = list(zip(sequence, (*sequence[1:], instance.absorber), strict=True))
    leg_index = {frozenset(p): k for k, p in enumerate(canon_pairs)}

    orientation_seq = (instance.absorber, *orientation)
    orientation_pairs = list(
        zip(orientation_seq, (*orientation_seq[1:], instance.absorber), strict=True)
    )
    leg_order = [leg_index[frozenset(p)] for p in orientation_pairs]
    legs = sample.legs[:, leg_order]

    betas = None
    if sample.feff_beta is not None:
        # Canonical vertex order is (A, *canon); skip the absorber vertex.
        vertex_index = {atom: k + 1 for k, atom in enumerate(canon)}
        beta_order = [vertex_index[i] for i in orientation]
        betas = sample.feff_beta[:, beta_order]
    return intermediate_keys, legs, betas


def match_path_instances_to_feff(
    samples: list[PathSamples],
    external_paths: list[ExternalPath],
    rleg_tol: float = 0.1,
    beta_tol: float = 15.0,
    reff_tol: float = 0.15,
) -> list[dict[str, Any]]:
    """Match sampled path instances to ordered external (FEFF) paths.

    This is the robust path-identity matcher for the MD-EXAFS pipeline —
    unlike the heuristic group-level :func:`match_msrd_paths_to_feff`, it
    gates on the complete ordered geometry: absorber, intermediate sequence,
    full leg-length sequence, and full β-angle sequence. Both orientations
    of every canonical instance are tested.

    Args:
        samples: Per-instance samples from :func:`sample_path_instances`.
        external_paths: Ordered external paths to match against.
        rleg_tol: Per-leg length tolerance (Å); ``None``-geometry paths skip
            the leg gate.
        beta_tol: Per-vertex β tolerance (degrees); applied only when both
            sides provide β sequences.
        reff_tol: Effective-length tolerance (Å).

    Returns:
        One row per ``(instance, orientation, external path)`` triple that
        passes all gates, sorted by cost, with keys ``path_id``,
        ``orientation``, ``external`` (the :class:`ExternalPath`), ``cost``,
        ``rleg_error_A``, ``beta_error_deg``, ``reff_error_A``.
    """
    rows: list[dict[str, Any]] = []
    for sample in samples:
        instance = sample.instance
        mean_reff = float(np.mean(sample.reff))
        for orientation in instance.orientations:
            inter_keys, legs, betas = _orientation_geometry(sample, orientation)
            mean_legs = legs.mean(axis=0)
            mean_betas = betas.mean(axis=0) if betas is not None else None
            for ext in external_paths:
                if ext.nleg != instance.nleg:
                    continue
                if ext.absorber_key and ext.absorber_key != instance.species[0]:
                    continue
                if ext.intermediate_keys and (
                    tuple(ext.intermediate_keys) != inter_keys
                ):
                    continue

                rleg_error = None
                if ext.rlegs is not None and len(ext.rlegs) == len(mean_legs):
                    rleg_error = float(
                        np.max(np.abs(np.asarray(ext.rlegs) - mean_legs))
                    )
                    if rleg_error > rleg_tol:
                        continue

                beta_error = None
                if (
                    ext.betas is not None
                    and mean_betas is not None
                    and len(ext.betas) == len(mean_betas)
                ):
                    beta_error = float(
                        np.max(np.abs(np.asarray(ext.betas) - mean_betas))
                    )
                    if beta_error > beta_tol:
                        continue

                ext_reff = (
                    float(np.sum(ext.rlegs)) / 2.0 if ext.rlegs is not None else None
                )
                reff_error = None
                if ext_reff is not None:
                    reff_error = abs(ext_reff - mean_reff)
                    if reff_error > reff_tol:
                        continue

                cost = max(
                    (rleg_error / rleg_tol) if rleg_error is not None else 0.0,
                    (beta_error / beta_tol) if beta_error is not None else 0.0,
                    (reff_error / reff_tol) if reff_error is not None else 0.0,
                )
                rows.append(
                    {
                        "path_id": instance.path_id,
                        "orientation": orientation,
                        "external": ext,
                        "cost": cost,
                        "rleg_error_A": rleg_error,
                        "beta_error_deg": beta_error,
                        "reff_error_A": reff_error,
                    }
                )
    rows.sort(key=lambda r: (r["cost"], str(r["path_id"])))
    return rows
