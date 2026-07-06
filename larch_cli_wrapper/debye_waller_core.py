"""Core functions for Debye-Waller factor and MSRD computation from MD trajectories.

These functions are shared between the CLI (``larch-cli debye-waller``) and the
``notebooks/debye_waller.py`` Marimo notebook so that logic is not duplicated.
"""

from __future__ import annotations

import io
import logging
from collections import defaultdict
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

    ref_atoms = structures[0].copy()
    ref_atoms.center()
    unwrapped[0] = ref_atoms.get_positions()

    for i in range(1, n_frames):
        if i % 500 == 0:
            logger.info("  Unwrapping frame %d/%d...", i, n_frames)
        atoms = structures[i]
        cell = atoms.get_cell()
        if np.all(cell.lengths() == 0):
            unwrapped[i] = atoms.get_positions()
            continue
        cell_matrix = cell.complete()
        inv_cell = np.linalg.inv(cell_matrix)
        frac_current = atoms.get_scaled_positions()
        frac_previous = unwrapped[i - 1] @ inv_cell.T
        frac_disp = frac_current - frac_previous
        frac_disp -= np.round(frac_disp)
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
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
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
# MSRD computation
# ---------------------------------------------------------------------------


def _trajectory_mic_arrays(
    structures: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray([atoms.get_positions() for atoms in structures], dtype=float)
    cells = np.asarray(
        [np.asarray(atoms.get_cell().complete()) for atoms in structures],
        dtype=float,
    )
    pbcs = np.asarray([atoms.get_pbc() for atoms in structures], dtype=bool)
    return positions, cells, pbcs


def _mic_search_offsets(pbc: np.ndarray) -> np.ndarray:
    axes = [
        np.array([-1.0, 0.0, 1.0]) if periodic else np.array([0.0]) for periodic in pbc
    ]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([grid.ravel() for grid in grids], axis=1)


def _mic_vectors_varying_cell(
    vectors: np.ndarray,
    cells: np.ndarray,
    pbcs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mic_vectors = np.empty_like(vectors, dtype=float)
    for pbc_pattern in np.unique(pbcs, axis=0):
        mask = np.all(pbcs == pbc_pattern, axis=1)
        if not np.any(pbc_pattern):
            mic_vectors[mask] = vectors[mask]
            continue
        these_cells = cells[mask]
        if np.any(np.abs(np.linalg.det(these_cells)) < 1e-12):
            raise ValueError("Periodic MIC requested for a frame with a singular cell.")
        inv_cells = np.linalg.inv(these_cells)
        frac = np.einsum("fi,fij->fj", vectors[mask], inv_cells)
        frac[:, pbc_pattern] -= np.round(frac[:, pbc_pattern])
        offsets = _mic_search_offsets(pbc_pattern)
        trial_frac = frac[:, None, :] + offsets[None, :, :]
        trial_cart = np.einsum("fki,fij->fkj", trial_frac, these_cells)
        norm2 = np.einsum("fki,fki->fk", trial_cart, trial_cart)
        best = np.argmin(norm2, axis=1)
        mic_vectors[mask] = trial_cart[np.arange(np.count_nonzero(mask)), best]
    distances = np.linalg.norm(mic_vectors, axis=1)
    return mic_vectors, distances


def _mic_vectors_for_pair(
    positions: np.ndarray,
    cells: np.ndarray,
    pbcs: np.ndarray,
    i_idx: int,
    j_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = positions[:, j_idx, :] - positions[:, i_idx, :]
    if not np.any(pbcs):
        distances = np.linalg.norm(vectors, axis=1)
        return vectors, distances
    same_cell = np.allclose(cells, cells[0])
    same_pbc = np.all(pbcs == pbcs[0])
    if same_cell and same_pbc:
        if np.any(pbcs[0]) and abs(float(np.linalg.det(cells[0]))) < 1e-12:
            raise ValueError("Periodic MIC requested with a singular cell.")
        mic_vectors, distances = find_mic(vectors, cells[0], pbcs[0])
        return np.asarray(mic_vectors), np.asarray(distances)
    return _mic_vectors_varying_cell(vectors, cells, pbcs)


def calculate_grouped_msrd(
    structures: list[Any],
    unwrapped_positions: np.ndarray,
    central_indices: list[int],
    central_label: str,
    cutoff: float = 3.5,
    tol_dist: float = 0.1,
    tol_angle: float = 5.0,
    cutoff_3body: float | None = None,
    exclude_hydrogen: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate grouped MSRD for 2-body and 3-body EXAFS paths."""
    if not central_indices:
        return [], []
    symbols = structures[0].get_chemical_symbols()
    central_element = symbols[central_indices[0]]
    reference_atoms = structures[0].copy()
    positions, cells, pbcs = _trajectory_mic_arrays(structures)
    if exclude_hydrogen:
        neighbor_candidates = {i for i, sym in enumerate(symbols) if sym != "H"}
        logger.info("Hydrogen excluded from neighbor search.")
    else:
        neighbor_candidates = set(range(len(symbols)))
    pair_list: list[dict[str, Any]] = []
    triplet_list: list[dict[str, Any]] = []
    logger.info(
        "Analysing MSRD paths for %s (%d sites)...",
        central_label,
        len(central_indices),
    )
    for c_idx in central_indices:
        all_indices = [
            i for i in range(len(symbols)) if i != c_idx and i in neighbor_candidates
        ]
        distances = reference_atoms.get_distances(c_idx, all_indices, mic=True)
        neighbors = [
            all_indices[i] for i in range(len(all_indices)) if distances[i] < cutoff
        ]
        logger.info(
            "  %d neighbors within %.2f Å of atom %d", len(neighbors), cutoff, c_idx
        )
        neighbor_vectors_mic: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for n_idx in neighbors:
            neighbor_vectors_mic[n_idx] = _mic_vectors_for_pair(
                positions,
                cells,
                pbcs,
                c_idx,
                n_idx,
            )
        for n_idx in neighbors:
            _v_mic, dists = neighbor_vectors_mic[n_idx]
            pair_list.append(
                {
                    "element": symbols[n_idx],
                    "dists": dists,
                    "mean_d": float(np.mean(dists)),
                    "label": f"{central_element}-{symbols[n_idx]}",
                    "c_idx": c_idx,
                    "n_idx": n_idx,
                }
            )
        if cutoff_3body == 0 or cutoff_3body is None:
            continue
        neighbors_3body = (
            [
                n
                for n in neighbors
                if np.mean(neighbor_vectors_mic[n][1]) <= cutoff_3body
            ]
            if cutoff_3body < cutoff
            else neighbors
        )
        for i in range(len(neighbors_3body)):
            for j in range(i + 1, len(neighbors_3body)):
                n1, n2 = neighbors_3body[i], neighbors_3body[j]
                v01_mic, d01 = neighbor_vectors_mic[n1]
                _v02_mic, d02 = neighbor_vectors_mic[n2]
                v12_mic, d12 = _mic_vectors_for_pair(positions, cells, pbcs, n1, n2)
                reff_series = (d01 + d12 + d02) / 2.0
                v1 = -v01_mic
                v2 = v12_mic
                v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
                v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
                v1_unit = v1 / np.maximum(v1_norm, 1e-10)
                v2_unit = v2 / np.maximum(v2_norm, 1e-10)
                cos_t = np.clip(np.sum(v1_unit * v2_unit, axis=1), -1, 1)
                angles_deg = np.degrees(np.arccos(cos_t))
                elem_pair = tuple(sorted([symbols[n1], symbols[n2]]))
                triplet_list.append(
                    {
                        "elements": elem_pair,
                        "reff_series": reff_series,
                        "mean_L": float(np.mean(reff_series)),
                        "angle": float(np.mean(angles_deg)),
                        "c_idx": c_idx,
                        "n1_idx": n1,
                        "n2_idx": n2,
                    }
                )
    pairs_by_element: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in pair_list:
        pairs_by_element[path["element"]].append(path)
    res_2b: list[dict[str, Any]] = []
    for _element, paths in pairs_by_element.items():
        paths.sort(key=lambda x: x["mean_d"])
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = [paths[0]]
        for path in paths[1:]:
            current_mean = float(np.mean([p["mean_d"] for p in current]))
            if abs(path["mean_d"] - current_mean) <= tol_dist:
                current.append(path)
            else:
                clusters.append(current)
                current = [path]
        clusters.append(current)
        for cluster in clusters:
            all_dists = np.concatenate([p["dists"] for p in cluster])
            res_2b.append(
                {
                    "type": cluster[0]["label"],
                    "reff": float(np.mean(all_dists)),
                    "sigma2": float(np.var(all_dists, ddof=1)),
                    "count": len(cluster),
                    "atom_indices": [(p["c_idx"], p["n_idx"]) for p in cluster],
                }
            )
    triplets_by_elements: defaultdict[tuple[str, ...], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for path in triplet_list:
        triplets_by_elements[path["elements"]].append(path)
    res_3b: list[dict[str, Any]] = []
    for elem_pair, paths in triplets_by_elements.items():
        paths.sort(key=lambda x: x["angle"])
        angle_clusters: list[list[dict[str, Any]]] = []
        current = [paths[0]]
        for path in paths[1:]:
            current_angle = float(np.mean([p["angle"] for p in current]))
            if abs(path["angle"] - current_angle) <= tol_angle:
                current.append(path)
            else:
                angle_clusters.append(current)
                current = [path]
        angle_clusters.append(current)
        for angle_cluster in angle_clusters:
            angle_cluster.sort(key=lambda x: x["mean_L"])
            dist_clusters: list[list[dict[str, Any]]] = []
            current = [angle_cluster[0]]
            for path in angle_cluster[1:]:
                current_reff = float(np.mean([p["mean_L"] for p in current]))
                if abs(path["mean_L"] - current_reff) <= tol_dist:
                    current.append(path)
                else:
                    dist_clusters.append(current)
                    current = [path]
            dist_clusters.append(current)
            for cluster in dist_clusters:
                all_reffs = np.concatenate([p["reff_series"] for p in cluster])
                res_3b.append(
                    {
                        "type": f"{central_element}-{elem_pair[0]}-{elem_pair[1]}",
                        "reff": float(np.mean(all_reffs)),
                        "sigma2": float(np.var(all_reffs, ddof=1)),
                        "angle": float(np.mean([p["angle"] for p in cluster])),
                        "count": len(cluster),
                        "atom_indices": [
                            (p["c_idx"], p["n1_idx"], p["n2_idx"]) for p in cluster
                        ],
                    }
                )
    return (
        sorted(res_2b, key=lambda x: x["reff"]),
        sorted(res_3b, key=lambda x: x["reff"]),
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
        n_absorbers: Number of distinct absorber atoms (used for degeneracy).

    Returns:
        A ``pandas.DataFrame`` with columns: ``Body``, ``Path type``,
        ``Reff (Å)``, ``σ² (Å²)``, ``Angle (°)``, ``Count``, ``Degeneracy``.
    """
    import pandas as pd

    rows = [
        {
            "_row_id": i,
            "Body": "2-body",
            "Path type": r["type"],
            "Reff (Å)": r["reff"],
            "σ² (Å²)": r["sigma2"],
            "Angle (°)": float("nan"),
            "Count": r["count"],
            "Degeneracy": r["count"] / n_absorbers,
        }
        for i, r in enumerate(res_2b)
    ] + [
        {
            "_row_id": len(res_2b) + i,
            "Body": "3-body",
            "Path type": r["type"],
            "Reff (Å)": r["reff"],
            "σ² (Å²)": r["sigma2"],
            "Angle (°)": r["angle"],
            "Count": r["count"],
            "Degeneracy": 2 * r["count"] / n_absorbers,
        }
        for i, r in enumerate(res_3b)
    ]
    return pd.DataFrame(rows)
