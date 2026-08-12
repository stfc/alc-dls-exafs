"""Marimo notebook for MD Debye-Waller & FEFF Analysis.

Computes Debye-Waller factors and MSRD for EXAFS/FEFF from MD trajectories.
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==6.0.0",
#     "ase==3.27.0",
#     "marimo>=0.19.11",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "weas-widget==0.1.26",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="MD Debye-Waller & FEFF Analysis")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # MD Debye-Waller & FEFF Analysis

    This notebook computes **Debye-Waller factors** (B-factors / ADP tensors) and
    **Mean Square Relative Displacements (MSRD)** for EXAFS/FEFF analysis from
    molecular dynamics trajectories.

    ### Workflow
    1. Select an MD run and configure parameters
    2. Unwrap PBC positions and optionally Kabsch-align frames
    3. Compute per-atom U tensors and B-factors → export CIF with ADP
    4. (Optional) Select an absorber site to compute 2-body and 3-body MSRD paths
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**⚠️ Active development notice** — This notebook is under active "
            "development and has not yet been validated for production use. "
            "Results should **not** be used in publications without independent "
            "verification."
            "\n\n"
            "Please report any issues or feedback on the GitHub repository: "
            "https://github.com/stfc/alc-dls-exafs/issues"
        ),
        kind="warn",
    )
    return


@app.cell
def _():
    import logging
    import tempfile
    from pathlib import Path

    import altair as alt
    import numpy as np
    import pandas as pd
    from ase import Atoms
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors
    from ase.geometry import find_mic
    from ase.io import read as ase_read
    from weas_widget.atoms_viewer import AtomsViewer
    from weas_widget.base_widget import BaseWidget
    from weas_widget.utils import ASEAdapter

    # I disabled the controls in the GUi, because the style is not loaded
    # properly inside Marimo notebook
    guiConfig = {"controls": {"enabled": False}}

    # Notebook-level logger (writes to console/terminal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("dw_notebook")
    return (
        ASEAdapter,
        Atoms,
        AtomsViewer,
        BaseWidget,
        Path,
        alt,
        ase_read,
        atomic_numbers,
        find_mic,
        guiConfig,
        jmol_colors,
        logger,
        np,
        pd,
        tempfile,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1 · Trajectory & Parameters
    """)
    return


@app.cell
def _(mo):
    trajectory_file = mo.ui.file(
        label="Upload trajectory file",
        filetypes=[
            ".xyz",
            ".traj",
            ".extxyz",
            ".vasp",
            ".POSCAR",
            ".cif",
            ".lammps",
            ".dump",
            ".nc",
            ".h5",
        ],
        multiple=False,
    )
    trajectory_file
    return (trajectory_file,)


@app.cell
def _(Path, mo, trajectory_file):
    skip_frames = mo.ui.number(
        value=0,
        start=0,
        stop=100_000,
        step=1,
        label="Frames to skip at start",
    )
    no_align = mo.ui.switch(label="Skip Kabsch alignment", value=False)
    _default_prefix = (
        Path(trajectory_file.value[0].name).stem if trajectory_file.value else "output"
    )
    output_prefix = mo.ui.text(value=_default_prefix, label="Output file prefix")

    mo.hstack(
        [
            mo.vstack([skip_frames, output_prefix]),
            mo.vstack([no_align]),
        ],
        gap=2,
    )
    return no_align, output_prefix, skip_frames


@app.cell
def _(mo):
    mo.md("""
    ## 2 · Core Functions
    """)
    return


@app.cell
def _(ASEAdapter, AtomsViewer, BaseWidget, guiConfig):
    def view_atoms(atoms, model_style=1):
        v = AtomsViewer(BaseWidget(guiConfig=guiConfig))
        v.atoms = ASEAdapter.to_weas(atoms)
        v.model_style = model_style
        return v._widget

    return (view_atoms,)


@app.cell
def _(find_mic, logger, np):
    import io
    from collections import defaultdict

    try:
        from larch_cli_wrapper.debye_waller_core import (
            _max_safe_mic_cutoff,
            calculate_grouped_msrd,
            compute_adp_results,
            kabsch_align,
            parse_site_specification,
            save_cif_with_adp,
            unwrap_positions_pbc,
        )
    except ImportError:
        # ---------------------------------------------------------------------------
        # Fallback: functions inlined from larch_cli_wrapper.debye_waller_core
        # so that this notebook works in WASM / sandbox mode without the local
        # package on sys.path.
        # ---------------------------------------------------------------------------

        def _max_safe_mic_cutoff(cell):
            """Return the largest sphere radius that fits inside a parallelepiped cell."""
            cell_matrix = np.asarray(cell)
            if cell_matrix.shape != (3, 3):
                return None
            volume = float(np.abs(np.linalg.det(cell_matrix)))
            if volume <= 0.0:
                return None
            area_ab = float(np.linalg.norm(np.cross(cell_matrix[0], cell_matrix[1])))
            area_bc = float(np.linalg.norm(np.cross(cell_matrix[1], cell_matrix[2])))
            area_ca = float(np.linalg.norm(np.cross(cell_matrix[2], cell_matrix[0])))
            if area_ab == 0.0 or area_bc == 0.0 or area_ca == 0.0:
                return None
            return (
                min(
                    volume / area_ab,
                    volume / area_bc,
                    volume / area_ca,
                )
                / 2.0
            )

        def unwrap_positions_pbc(structures):
            """Unwrap atomic positions for continuous trajectories across PBC."""
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
                inv_cell = np.linalg.inv(cell_matrix)
                frac_current = atoms.get_positions() @ inv_cell
                frac_previous = unwrapped[i - 1] @ inv_cell
                frac_disp = frac_current - frac_previous
                frac_disp[:, pbc] -= np.round(frac_disp[:, pbc])
                unwrapped[i] = unwrapped[i - 1] + (frac_disp @ cell_matrix)

            return unwrapped

        def kabsch_align(unwrapped_positions, reference_idx=0, reference_pos=None):
            """Align all trajectory frames to a reference using the Kabsch algorithm."""
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

        def compute_adp_results(structures, unwrapped):
            """Compute per-atom ADP tensors and B-factors from unwrapped positions."""
            avg_pos = np.mean(unwrapped, axis=0)
            displacements = unwrapped - avg_pos[np.newaxis, :, :]
            u_tensor = np.einsum("fni,fnj->nij", displacements, displacements) / len(
                structures
            )
            b_factors = 8 * np.pi**2 * np.trace(u_tensor, axis1=1, axis2=2) / 3

            return {
                "b_factors": b_factors,
                "u_tensor": u_tensor,
                "avg_positions": avg_pos,
                "atom_names": structures[0].get_chemical_symbols(),
                "avg_cell": structures[0].get_cell().complete(),
                "atom_indices": np.arange(len(b_factors)),
            }

        def save_cif_with_adp(results):
            """Return a CIF string with mean positions and anisotropic U tensors."""
            pos = results["avg_positions"]
            names = results["atom_names"]
            u_cart = results["u_tensor"]
            cell = results["avg_cell"]
            inv_cell = np.linalg.inv(cell)
            frac_pos = pos @ inv_cell.T
            a, b, c = np.linalg.norm(cell, axis=1)

            def ang(v1, v2):
                return float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(v1, v2)
                                / (np.linalg.norm(v1) * np.linalg.norm(v2)),
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

        def parse_site_specification(spec, symbols):
            """Parse a site specification string and return atomic indices."""
            if spec.replace("-", "").replace(" ", "").isdigit():
                spec = spec.replace(" ", "")
                if "-" in spec:
                    start, end = spec.split("-")
                    return list(range(int(start) - 1, int(end)))
                else:
                    return [int(spec) - 1]

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

            matching = [i for i, sym in enumerate(symbols) if sym == spec]
            if not matching:
                raise ValueError(f"No atoms of element '{spec}' found")
            return matching

        def _max_safe_mic_cutoff(cell):
            """Return the largest sphere radius that fits inside a parallelepiped cell."""
            cell_matrix = np.asarray(cell)
            if cell_matrix.shape != (3, 3):
                return None
            volume = float(np.abs(np.linalg.det(cell_matrix)))
            if volume <= 0.0:
                return None
            area_ab = float(np.linalg.norm(np.cross(cell_matrix[0], cell_matrix[1])))
            area_bc = float(np.linalg.norm(np.cross(cell_matrix[1], cell_matrix[2])))
            area_ca = float(np.linalg.norm(np.cross(cell_matrix[2], cell_matrix[0])))
            if area_ab == 0.0 or area_bc == 0.0 or area_ca == 0.0:
                return None
            return (
                min(
                    volume / area_ab,
                    volume / area_bc,
                    volume / area_ca,
                )
                / 2.0
            )

        def calculate_grouped_msrd(
            structures,
            unwrapped_positions,
            central_indices,
            central_label,
            cutoff=3.5,
            tol_dist=0.1,
            tol_angle=5.0,
            cutoff_3body=None,
            exclude_hydrogen=True,
        ):
            """Calculate grouped MSRD for 2-body and 3-body EXAFS paths."""
            if not central_indices:
                return [], []

            orig_positions = np.array([atoms.get_positions() for atoms in structures])

            symbols = structures[0].get_chemical_symbols()
            central_element = symbols[central_indices[0]]
            reference_atoms = structures[0].copy()
            cell = structures[0].get_cell()
            pbc = structures[0].get_pbc()

            max_safe_cutoff = _max_safe_mic_cutoff(cell.complete())
            if max_safe_cutoff is not None:
                for cutoff_name, cutoff_value in [
                    ("cutoff", cutoff),
                    ("cutoff_3body", cutoff_3body),
                ]:
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

            if exclude_hydrogen:
                neighbor_candidates = {i for i, sym in enumerate(symbols) if sym != "H"}
                logger.info("Hydrogen excluded from neighbor search.")
            else:
                neighbor_candidates = set(range(len(symbols)))

            pair_list = []
            triplet_list = []

            logger.info(
                "Analysing MSRD paths for %s (%d sites)...",
                central_label,
                len(central_indices),
            )

            for c_idx in central_indices:
                all_indices = [
                    i
                    for i in range(len(symbols))
                    if i != c_idx and i in neighbor_candidates
                ]
                distances = reference_atoms.get_distances(c_idx, all_indices, mic=True)
                neighbors = [
                    all_indices[i]
                    for i in range(len(all_indices))
                    if distances[i] < cutoff
                ]
                logger.info(
                    "  %d neighbors within %.2f Å of atom %d",
                    len(neighbors),
                    cutoff,
                    c_idx,
                )

                neighbor_vectors_mic = {}
                for n_idx in neighbors:
                    v_raw = orig_positions[:, n_idx, :] - orig_positions[:, c_idx, :]
                    v_mic, dists = find_mic(v_raw, cell, pbc)
                    neighbor_vectors_mic[n_idx] = (v_mic, dists)

                for n_idx in neighbors:
                    v_mic, dists = neighbor_vectors_mic[n_idx]
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
                        v02_mic, d02 = neighbor_vectors_mic[n2]
                        v12_raw = orig_positions[:, n2, :] - orig_positions[:, n1, :]
                        v12_mic, d12 = find_mic(v12_raw, cell, pbc)
                        d20 = d02
                        L = d01 + d12 + d20

                        v1 = -v01_mic
                        v2 = v12_mic
                        v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
                        v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
                        v1_unit = v1 / np.maximum(v1_norm, 1e-10)
                        v2_unit = v2 / np.maximum(v2_norm, 1e-10)
                        cos_t = np.clip(np.sum(v1_unit * v2_unit, axis=1), -1, 1)
                        angles_deg = np.degrees(np.arccos(cos_t))

                        reff_series = L / 2.0
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

            pairs_by_element = defaultdict(list)
            for path in pair_list:
                pairs_by_element[path["element"]].append(path)

            res_2b = []
            for _element, paths in pairs_by_element.items():
                paths.sort(key=lambda x: x["mean_d"])
                clusters = []
                current = [paths[0]]
                for path in paths[1:]:
                    if (
                        abs(
                            path["mean_d"]
                            - float(np.mean([p["mean_d"] for p in current]))
                        )
                        <= tol_dist
                    ):
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

            triplets_by_elements = defaultdict(list)
            for path in triplet_list:
                triplets_by_elements[path["elements"]].append(path)

            res_3b = []
            for elem_pair, paths in triplets_by_elements.items():
                paths.sort(key=lambda x: x["angle"])
                angle_clusters = []
                current = [paths[0]]
                for path in paths[1:]:
                    if (
                        abs(
                            path["angle"]
                            - float(np.mean([p["angle"] for p in current]))
                        )
                        <= tol_angle
                    ):
                        current.append(path)
                    else:
                        angle_clusters.append(current)
                        current = [path]
                angle_clusters.append(current)

                for angle_cluster in angle_clusters:
                    angle_cluster.sort(key=lambda x: x["mean_L"])
                    dist_clusters = []
                    current = [angle_cluster[0]]
                    for path in angle_cluster[1:]:
                        if (
                            abs(
                                path["mean_L"]
                                - float(np.mean([p["mean_L"] for p in current]))
                            )
                            <= tol_dist
                        ):
                            current.append(path)
                        else:
                            dist_clusters.append(current)
                            current = [path]
                    dist_clusters.append(current)
                    for cluster in dist_clusters:
                        all_reffs = np.concatenate([p["reff_series"] for p in cluster])
                        res_3b.append(
                            {
                                "type": (
                                    f"{central_element}-{elem_pair[0]}-{elem_pair[1]}"
                                ),
                                "reff": float(np.mean(all_reffs)),
                                "sigma2": float(np.var(all_reffs, ddof=1)),
                                "angle": float(np.mean([p["angle"] for p in cluster])),
                                "count": len(cluster),
                                "atom_indices": [
                                    (p["c_idx"], p["n1_idx"], p["n2_idx"])
                                    for p in cluster
                                ],
                            }
                        )

            return (
                sorted(res_2b, key=lambda x: x["reff"]),
                sorted(res_3b, key=lambda x: x["reff"]),
            )

    return (
        _max_safe_mic_cutoff,
        calculate_grouped_msrd,
        compute_adp_results,
        kabsch_align,
        parse_site_specification,
        save_cif_with_adp,
        unwrap_positions_pbc,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Load Trajectory
    """)
    return


@app.cell
def _(Path, ase_read, mo, skip_frames, tempfile, trajectory_file):
    structures = None
    _load_status = mo.callout(
        mo.md("⬆️  Upload a trajectory file above to begin."),
        kind="info",
    )

    if trajectory_file.value:
        _tf = trajectory_file.value[0]
        _suffix = Path(_tf.name).suffix or ".xyz"
        _tmp = tempfile.NamedTemporaryFile(suffix=_suffix, delete=False)
        _tmp.write(_tf.contents)
        _tmp.close()
        try:
            _raw = ase_read(_tmp.name, index=f"{skip_frames.value}:")
            if not isinstance(_raw, list):
                _raw = [_raw]

            # ── Validation ────────────────────────────────────────────────
            _errors = []
            _warnings = []

            # 1. Must have more than one frame
            if len(_raw) < 2:
                _errors.append(
                    f"Trajectory has only **{len(_raw)} frame** after skipping "
                    f"{skip_frames.value}. At least 2 frames are required."
                )

            # 2. Consistent atom count across all frames
            _n_atoms_0 = len(_raw[0])
            _bad_frames = [i for i, _a in enumerate(_raw) if len(_a) != _n_atoms_0]
            if _bad_frames:
                _errors.append(
                    f"Inconsistent atom count: frame(s) {_bad_frames[:5]}"
                    f"{'…' if len(_bad_frames) > 5 else ''} "
                    f"differ from frame 0 ({_n_atoms_0} atoms)."
                )

            # 3. Consistent chemical symbols across all frames
            _syms_0 = _raw[0].get_chemical_symbols()
            _bad_sym_frames = [
                i
                for i, _a in enumerate(_raw)
                if len(_a) == _n_atoms_0 and _a.get_chemical_symbols() != _syms_0
            ]
            if _bad_sym_frames:
                _errors.append(
                    f"Inconsistent chemical symbols in frame(s) "
                    f"{_bad_sym_frames[:5]}"
                    f"{'…' if len(_bad_sym_frames) > 5 else ''}."
                )

            # 4. Warn if no periodic cell
            if not _raw[0].get_cell().any():
                _warnings.append(
                    "No periodic cell found. PBC unwrapping will be skipped "
                    "(positions used as-is)."
                )

            if _errors:
                _load_status = mo.callout(
                    mo.md(
                        "❌ **Trajectory validation failed:**\n\n"
                        + "\n\n".join(f"- {e}" for e in _errors)
                    ),
                    kind="danger",
                )
            else:
                structures = _raw
                elements_str = ", ".join(sorted(set(_syms_0)))
                _msg = (
                    f"✅ Loaded **{len(structures)} frames** · "
                    f"**{_n_atoms_0} atoms** per frame · "
                    f"Elements: `{elements_str}`"
                )
                if _warnings:
                    _msg += "\n\n" + "\n\n".join(f"⚠️ {w}" for w in _warnings)
                _load_status = mo.callout(
                    mo.md(_msg),
                    kind="success" if not _warnings else "warn",
                )
        except (OSError, ValueError) as _e:
            _load_status = mo.callout(
                mo.md(f"❌ Failed to load trajectory: `{_e}`"),
                kind="danger",
            )
        finally:
            Path(_tmp.name).unlink(missing_ok=True)

    _load_status
    return (structures,)


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Unwrap & Align
    """)
    return


@app.cell
def _(kabsch_align, logger, mo, no_align, structures, unwrap_positions_pbc):
    unwrapped = None
    _status = mo.md("")

    if structures is not None:
        with mo.status.spinner("Unwrapping PBC positions..."):
            unwrapped = unwrap_positions_pbc(structures)

        if not no_align.value:
            with mo.status.spinner("Kabsch alignment – pass 1…"):
                _rough = kabsch_align(unwrapped)
                _avg = __import__("numpy").mean(_rough, axis=0)
            with mo.status.spinner("Kabsch alignment – pass 2…"):
                unwrapped = kabsch_align(unwrapped, reference_pos=_avg)
            logger.info("Two-pass Kabsch alignment complete.")
        else:
            logger.info("Kabsch alignment skipped.")

        _status = mo.callout(
            mo.md(
                f"✅ Trajectory processed · "
                f"Shape: `{unwrapped.shape}` (frames × atoms × xyz)"
            ),
            kind="success",
        )

    _status
    return (unwrapped,)


@app.cell
def _(mo, structures, view_atoms):
    if structures is not None:
        mo.output.append(view_atoms(structures))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 · B-factors & ADP Tensor
    """)
    return


@app.cell
def _(
    compute_adp_results,
    mo,
    np,
    output_prefix,
    save_cif_with_adp,
    structures,
    unwrapped,
):
    results = None
    _b_status = mo.md("")

    if unwrapped is not None and structures is not None:
        results = compute_adp_results(structures, unwrapped)
        _b_factors = results["b_factors"]

        # ── Per-element summary table ──────────────────────────────────────
        _names = results["atom_names"]
        _unique = sorted(set(_names))
        _rows = []
        for _el in _unique:
            _mask = np.array([n == _el for n in _names])
            _bvals = _b_factors[_mask]
            _rows.append(
                {
                    "Element": _el,
                    "N atoms": int(_mask.sum()),
                    "Mean B (Å²)": f"{np.mean(_bvals):.3f}",
                    "Std B (Å²)": f"{np.std(_bvals):.3f}",
                    "Min B (Å²)": f"{np.min(_bvals):.3f}",
                    "Max B (Å²)": f"{np.max(_bvals):.3f}",
                }
            )
        _overall = f"{np.mean(_b_factors):.3f}"

        # ── CIF download button ───────────────────────────────────────────
        _cif_content = save_cif_with_adp(results)
        _cif_filename = f"{output_prefix.value}_with_adp.cif"
        _download_btn = mo.download(
            data=_cif_content.encode(),
            filename=_cif_filename,
            label="⬇ Download CIF with ADP",
        )

        _b_status = mo.vstack(
            [
                mo.md(f"**Overall mean B-factor:** {_overall} Å²"),
                mo.ui.table(_rows, label="Debye-Waller factors by element"),
                _download_btn,
            ]
        )

    _b_status
    return (results,)


@app.cell
def _(Atoms, structures):
    """Build the Atoms object shown for path visualisation.

    The *time-averaged* crystal structure is meaningless here: atoms drift
    across cell boundaries and exchange sites (this trajectory stores
    unwrapped coordinates, with ~70% of atoms outside the stored cell), so the
    arithmetic mean of per-atom Cartesian positions lands between lattice
    sites and collapses to a smeared blob. MSRD paths are enumerated in the
    frame-0 geometry (``enumerate_path_instances(structures[0], ...)``), so
    the path viewer uses a **wrapped frame-0 snapshot** — the reference frame
    the atom indices refer to. B-factors / ADPs / CIF output still come from
    the aligned unwrapped trajectory via ``results``.
    """
    avg_atoms = None
    if structures is not None:
        avg_atoms = structures[0].copy()
        avg_atoms.wrap()  # pull periodic images back into the unit cell
    return (avg_atoms,)


@app.cell
def _(mo):
    mo.md("""
    ## 6 · B-factor Plot
    """)
    return


@app.cell
def _(alt, atomic_numbers, jmol_colors, mo, np, pd, results):
    _plot = mo.md("_Run steps above first._")

    if results is not None:
        _b = results["b_factors"]
        _names = results["atom_names"]
        _df_b = pd.DataFrame(
            {
                "Atom index": np.arange(len(_b)),
                "Element": _names,
                "B-factor (Å²)": _b,
            }
        )

        # JMOL element colours
        def _jmol_hex(sym):
            z = atomic_numbers.get(sym, 0)
            r, g, b = jmol_colors[z]
            return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

        _unique_els = sorted(set(_names))
        _el_color_scale = alt.Scale(
            domain=_unique_els,
            range=[_jmol_hex(el) for el in _unique_els],
        )

        _hover_b = alt.selection_point(on="mouseover", nearest=True, empty=False)

        _pts = (
            alt.Chart(_df_b)
            .mark_point(filled=True)
            .encode(
                x=alt.X("Atom index:Q", title="Atom index"),
                y=alt.Y("B-factor (Å²):Q", title="B-factor (Å²)"),
                color=alt.Color("Element:N", title="Element", scale=_el_color_scale),
                size=alt.condition(_hover_b, alt.value(120), alt.value(25)),
                opacity=alt.condition(_hover_b, alt.value(1.0), alt.value(0.6)),
                tooltip=[
                    alt.Tooltip("Atom index:Q", title="Atom index"),
                    alt.Tooltip("Element:N", title="Element"),
                    alt.Tooltip("B-factor (Å²):Q", format=".4f", title="B (Å²)"),
                ],
            )
            .add_params(_hover_b)
        )

        # Horizontal mean line per element
        _mean_lines = (
            alt.Chart(_df_b)
            .mark_rule(strokeDash=[4, 4], strokeWidth=1, opacity=0.6)
            .encode(
                y=alt.Y("mean(B-factor (Å²)):Q"),
                color=alt.Color("Element:N", scale=_el_color_scale),
                tooltip=[
                    alt.Tooltip("Element:N", title="Element"),
                    alt.Tooltip(
                        "mean(B-factor (Å²)):Q", format=".4f", title="Mean B (Å²)"
                    ),
                ],
            )
        )

        _plot = (
            alt.layer(_mean_lines, _pts)
            .properties(
                height=300, width="container", title="Debye-Waller factors per atom"
            )
            .interactive()
        )

    _plot
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 · MSRD Path Analysis

    Specify an **absorber site** to calculate mean square relative displacements
    for EXAFS/FEFF.

    | Format | Meaning |
    |--------|----------|
    | `K` | All K atoms |
    | `K.1` | First K atom (1-based within element) |
    | `K.1-3` | First three K atoms |
    | `11` | 11th atom in structure (1-based, any element) |
    | `11-20` | Atoms 11–20 (1-based, inclusive) |
    """)
    return


@app.cell
def _(mo):
    element_spec = mo.ui.text(
        placeholder="e.g.  K  or  Cu.1  or  11-20",
        label="Absorber site specification",
    )
    cutoff = mo.ui.number(
        value=3.5,
        start=0.5,
        stop=20.0,
        step=0.1,
        label="Neighbor cutoff (Å)",
    )
    tol_dist = mo.ui.number(
        value=0.1,
        start=0.01,
        stop=2.0,
        step=0.01,
        label="Distance grouping tolerance (Å)",
    )
    tol_angle = mo.ui.number(
        value=5.0,
        start=0.1,
        stop=45.0,
        step=0.5,
        label="Angle grouping tolerance (°)",
    )
    cutoff_3body = mo.ui.number(
        value=0.0,
        start=0.0,
        stop=20.0,
        step=0.1,
        label="3-body neighbor cutoff (Å) (0 = skip)",
    )
    exclude_hydrogen_cb = mo.ui.checkbox(
        label="Exclude hydrogen from neighbor search",
        value=True,
    )
    run_msrd = mo.ui.run_button(label="▶  Run MSRD Analysis")

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([element_spec, cutoff, tol_dist]),
                    mo.vstack([cutoff_3body, tol_angle, exclude_hydrogen_cb]),
                ],
                gap=2,
            ),
            run_msrd,
        ]
    )
    return (
        cutoff,
        cutoff_3body,
        element_spec,
        exclude_hydrogen_cb,
        run_msrd,
        tol_angle,
        tol_dist,
    )


@app.cell
def _(
    _max_safe_mic_cutoff,
    calculate_grouped_msrd,
    cutoff,
    cutoff_3body,
    element_spec,
    exclude_hydrogen_cb,
    mo,
    output_prefix,
    parse_site_specification,
    pd,
    results,
    run_msrd,
    structures,
    tol_angle,
    tol_dist,
    unwrapped,
):
    _msrd_ui = mo.md("_Configure absorber site and click **Run MSRD Analysis**._")
    msrd_df = None
    msrd_path_indices = None

    if (
        run_msrd.value
        and structures is not None
        and unwrapped is not None
        and results is not None
    ):
        _spec = element_spec.value.strip()
        if not _spec:
            _msrd_ui = mo.callout(
                mo.md("⚠️ Please enter a site specification."), kind="warn"
            )
        else:
            _symbols = structures[0].get_chemical_symbols()
            try:
                _indices = parse_site_specification(_spec, _symbols)
            except ValueError as _err:
                _msrd_ui = mo.callout(mo.md(f"❌ {_err}"), kind="danger")
                _indices = None

            if _indices is not None:
                _c3b = cutoff_3body.value if cutoff_3body.value != 0 else 0

                # Check whether the chosen cutoffs exceed the safe MIC radius
                _warnings = []
                _max_safe = _max_safe_mic_cutoff(structures[0].get_cell().complete())
                if _max_safe is not None:
                    if cutoff.value > _max_safe:
                        _warnings.append(
                            f"**cutoff** = {cutoff.value:.3f} Å exceeds the "
                            f"maximum safe MIC cutoff **{_max_safe:.3f} Å**. "
                            f"Distances may be ambiguous; reduce cutoff to ≤ {_max_safe:.3f} Å "
                            f"or use a supercell."
                        )
                    if _c3b > 0 and _c3b > _max_safe:
                        _warnings.append(
                            f"**cutoff_3body** = {_c3b:.3f} Å exceeds the "
                            f"maximum safe MIC cutoff **{_max_safe:.3f} Å**. "
                            f"Angles/path lengths may be ambiguous; reduce cutoff_3body to "
                            f"≤ {_max_safe:.3f} Å or use a supercell."
                        )

                with mo.status.spinner("Computing MSRD paths…"):
                    _res2b, _res3b = calculate_grouped_msrd(
                        structures,
                        _indices,
                        _spec,
                        cutoff=cutoff.value,
                        tol_dist=tol_dist.value,
                        tol_angle=tol_angle.value,
                        cutoff_3body=_c3b,
                        exclude_hydrogen=exclude_hydrogen_cb.value,
                    )

                # ── Build display tables ─────────────────────────────────
                _rows2b = [
                    {
                        "Path type": r["type"],
                        "Reff (Å)": f"{r['reff']:.4f}",
                        "σ² (Å²)": f"{r['sigma2']:.6f}",
                        "Count": r["count"],
                        "Degeneracy": f"{r['count'] / len(_indices):.1f}",
                    }
                    for r in _res2b
                ]
                _rows3b = [
                    {
                        "Path type": r["type"],
                        "Reff (Å)": f"{r['reff']:.4f}",
                        "σ² (Å²)": f"{r['sigma2']:.6f}",
                        "Angle (°)": f"{r['angle']:.1f}",
                        "Count": r["count"],
                        "Degeneracy": f"{2 * r['count'] / len(_indices):.1f}",
                    }
                    for r in _res3b
                ]

                # ── Build combined DataFrame ──────────────────────────────
                _df_rows = [
                    {
                        "_row_id": i,
                        "Body": "2-body",
                        "Path type": r["type"],
                        "Reff (Å)": r["reff"],
                        "σ² (Å²)": r["sigma2"],
                        "Angle (°)": float("nan"),
                        "Count": r["count"],
                        "Degeneracy": r["count"] / len(_indices),
                    }
                    for i, r in enumerate(_res2b)
                ] + [
                    {
                        "_row_id": len(_res2b) + i,
                        "Body": "3-body",
                        "Path type": r["type"],
                        "Reff (Å)": r["reff"],
                        "σ² (Å²)": r["sigma2"],
                        "Angle (°)": r["angle"],
                        "Count": r["count"],
                        "Degeneracy": 2 * r["count"] / len(_indices),
                    }
                    for i, r in enumerate(_res3b)
                ]
                msrd_df = pd.DataFrame(_df_rows)
                msrd_path_indices = [r["atom_indices"] for r in _res2b] + [
                    r["atom_indices"] for r in _res3b
                ]
                _msrd_ui = mo.vstack(
                    [
                        mo.md(
                            f"**Site:** `{_spec}` · "
                            f"**{len(_indices)} absorber(s)** · "
                            f"**{len(_res2b)} two-body** and "
                            f"**{len(_res3b)} three-body** paths found"
                        ),
                        mo.callout(
                            mo.md("\n\n".join(_warnings)),
                            kind="warn",
                        )
                        if _warnings
                        else mo.md(""),
                        mo.md("### 2-Body Paths"),
                        mo.ui.table(_rows2b)
                        if _rows2b
                        else mo.md("_No 2-body paths found._"),
                        mo.md("### 3-Body Paths"),
                        mo.ui.table(_rows3b)
                        if _rows3b
                        else mo.md("_No 3-body paths (cutoff = 0 or no triangles)._"),
                        # Button to download the dataframe as CSV
                        mo.download(
                            data=msrd_df.to_csv(index=False).encode(),
                            # line too long:
                            filename=f"{output_prefix.value}_msrd_paths_"
                            f"{_spec.replace(' ', '_')}.csv",
                            label="⬇ Download MSRD paths as CSV",
                        ),
                    ]
                )

    _msrd_ui
    return msrd_df, msrd_path_indices


@app.cell
def _(alt, atomic_numbers, jmol_colors, mo, msrd_df):
    msrd_chart = None

    if msrd_df is not None:
        # JMOL colour per path type, keyed on the first scatterer element
        # e.g. "Fe-N" → N colour;  "Fe-N-C" → N colour
        def _jmol_hex(sym):
            z = atomic_numbers.get(sym, 0)
            r, g, b = jmol_colors[z]
            return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

        _pt_domain = list(msrd_df["Path type"].unique())
        _pt_range = [_jmol_hex(pt.split("-")[1]) for pt in _pt_domain]
        _pt_color_scale = alt.Scale(domain=_pt_domain, range=_pt_range)

        _base = alt.Chart(msrd_df)

        _points = _base.mark_point(filled=True).encode(
            x=alt.X("Reff (Å):Q", title="Reff (Å)"),
            y=alt.Y("σ² (Å²):Q", title="σ² (Å²)"),
            color=alt.Color("Path type:N", title="Path type", scale=_pt_color_scale),
            shape=alt.Shape(
                "Body:N",
                title="Body",
                scale=alt.Scale(
                    domain=["2-body", "3-body"], range=["circle", "triangle-up"]
                ),
            ),
            size=alt.value(60),
            opacity=alt.value(0.75),
            tooltip=[
                alt.Tooltip("Body:N", title="Body"),
                alt.Tooltip("Path type:N", title="Path type"),
                alt.Tooltip("Reff (Å):Q", format=".4f", title="Reff (Å)"),
                alt.Tooltip("σ² (Å²):Q", format=".6f", title="σ² (Å²)"),
                alt.Tooltip("Angle (°):Q", format=".1f", title="Angle (°)"),
                alt.Tooltip("Degeneracy:Q", format=".2f", title="Degeneracy"),
                alt.Tooltip("Count:Q", title="Count"),
            ],
        )

        _chart = alt.layer(_points).properties(height=340, width="container")
        msrd_chart = mo.ui.altair_chart(_chart, label="Click a point to inspect it")

    msrd_chart if msrd_chart is not None else mo.md(
        "_Run MSRD analysis above to see the path plot._"
    )
    return (msrd_chart,)


@app.cell
def _(mo, path_index_slider, path_view, selected_path_info, show_all_paths):
    mo.vstack(
        [
            mo.hstack([show_all_paths, path_index_slider], justify="start")
            if selected_path_info is not None
            else mo.md("_Select a path first._"),
            path_view,
        ]
    )
    return


@app.cell
def _(avg_atoms, find_mic, mo, msrd_chart, msrd_path_indices, np, results):
    _sel = msrd_chart.value if msrd_chart is not None else None
    selected_path_info = None

    if _sel is not None and len(_sel) > 0:
        _r = _sel.iloc[0]
        _body = _r["Body"]
        _row_id = int(_r["_row_id"])
        _path_pairs = msrd_path_indices[
            _row_id
        ]  # list of (c_idx, n_idx) or (c_idx, n1, n2)

        # ── Build atom index table ────────────────────────────────────────
        _syms = results["atom_names"]
        _pos = results["avg_positions"]
        _cell = results["avg_cell"]
        _pbc = avg_atoms.get_pbc() if avg_atoms is not None else [True, True, True]

        _atom_rows = []
        _all_vectors = []
        _path_instances = []  # one entry per equivalent path pair
        for _pair in _path_pairs:
            _c = _pair[0]
            _neighbors = list(_pair[1:])
            _row = {"Absorber idx": _c, "Absorber element": _syms[_c]}
            _instance_vecs = []

            if _body == "3-body" and len(_neighbors) == 2:
                # Chain MIC vectors so the three legs form a closed triangle.
                # leg c→n1
                _n1, _n2 = _neighbors
                _v1_raw = _pos[_n1] - _pos[_c]
                _v1_mic, _ = find_mic([_v1_raw], _cell, _pbc)
                _v1 = _v1_mic[0]
                _d1 = float(np.linalg.norm(_v1))
                _orig_c = _pos[_c]
                # leg n1→n2 (origin chains from c + v1)
                _v12_raw = _pos[_n2] - _pos[_n1]
                _v12_mic, _ = find_mic([_v12_raw], _cell, _pbc)
                _v12 = _v12_mic[0]
                _d12 = float(np.linalg.norm(_v12))
                _orig_n1 = _orig_c + _v1
                # leg n2→c closes the triangle exactly
                _v_ret = -(_v1 + _v12)
                _d_ret = float(np.linalg.norm(_v_ret))
                _orig_n2 = _orig_n1 + _v12

                _row["n1 idx"] = _n1
                _row["n1 element"] = _syms[_n1]
                _row["n1 dist (Å)"] = f"{_d1:.4f}"
                _row["n1 vector (Å)"] = f"({_v1[0]:.3f}, {_v1[1]:.3f}, {_v1[2]:.3f})"
                _row["n2 idx"] = _n2
                _row["n2 element"] = _syms[_n2]
                _row["n2 dist (Å)"] = f"{_d_ret:.4f}"
                _row["n1→n2 dist (Å)"] = f"{_d12:.4f}"
                _row["n1→n2 vector (Å)"] = (
                    f"({_v12[0]:.3f}, {_v12[1]:.3f}, {_v12[2]:.3f})"
                )

                _e1 = {
                    "from_idx": _c,
                    "to_idx": _n1,
                    "leg": "n1",
                    "vector": _v1,
                    "dist": _d1,
                    "origin": _orig_c,
                }
                _e12e = {
                    "from_idx": _n1,
                    "to_idx": _n2,
                    "leg": "n1\u2192n2",
                    "vector": _v12,
                    "dist": _d12,
                    "origin": _orig_n1,
                }
                _e2 = {
                    "from_idx": _n2,
                    "to_idx": _c,
                    "leg": "n2",
                    "vector": _v_ret,
                    "dist": _d_ret,
                    "origin": _orig_n2,
                }
                for _e in (_e1, _e12e, _e2):
                    _all_vectors.append(_e)
                    _instance_vecs.append(_e)
            else:
                for _k, _n in enumerate(_neighbors):
                    _v_raw = _pos[_n] - _pos[_c]
                    _v_mic, _ = find_mic([_v_raw], _cell, _pbc)
                    _v = _v_mic[0]
                    _d = float(np.linalg.norm(_v))
                    _label = "n1" if _k == 0 else "n2"
                    _row[f"{_label} idx"] = _n
                    _row[f"{_label} element"] = _syms[_n]
                    _row[f"{_label} dist (Å)"] = f"{_d:.4f}"
                    _row[f"{_label} vector (Å)"] = (
                        f"({_v[0]:.3f}, {_v[1]:.3f}, {_v[2]:.3f})"
                    )
                    _entry = {
                        "from_idx": _c,
                        "to_idx": _n,
                        "leg": _label,
                        "vector": _v,
                        "dist": _d,
                    }
                    _all_vectors.append(_entry)
                    _instance_vecs.append(_entry)

            _atom_rows.append(_row)
            _path_instances.append(_instance_vecs)

        selected_path_info = {
            "body": _body,
            "path_type": _r["Path type"],
            "reff": _r["Reff (Å)"],
            "sigma2": _r["σ² (Å²)"],
            "path_pairs": _path_pairs,
            "path_instances": _path_instances,
            "vectors": _all_vectors,
            "avg_atoms": avg_atoms,
        }

        _angle_str = f" · Angle: {_r['Angle (°)']:.1f}°" if _body == "3-body" else ""
        _summary = mo.vstack(
            [
                mo.md(
                    f"### Selected path: **{_r['Path type']}** ({_body}){_angle_str}\n"
                    f"**Reff** = {_r['Reff (Å)']:.4f} Å · "
                    f"**σ²** = {_r['σ² (Å²)']:.6f} Å² · "
                    f"**Degeneracy** = {_r['Degeneracy']:.2f} · "
                    f"**Count** = {int(_r['Count'])}"
                ),
                mo.md("#### Atom indices and bond vectors (average structure)"),
                mo.ui.table(_atom_rows),
            ]
        )
    else:
        _summary = mo.callout(
            mo.md("Click a point in the chart above to see its details here."),
            kind="info",
        )

    _summary
    return (selected_path_info,)


@app.cell
def _(mo, selected_path_info):
    _n = (
        len(selected_path_info["path_instances"])
        if selected_path_info is not None
        else 1
    )
    show_all_paths = mo.ui.checkbox(label="Show all equivalent paths", value=False)
    path_index_slider = mo.ui.slider(
        start=0,
        stop=max(_n - 1, 0),
        step=1,
        value=0,
        label=f"Path instance (0\u2013{max(_n - 1, 0)})",
        show_value=True,
    )
    return path_index_slider, show_all_paths


@app.cell
def _(
    ASEAdapter,
    AtomsViewer,
    BaseWidget,
    guiConfig,
    mo,
    np,
    path_index_slider,
    selected_path_info,
    show_all_paths,
):
    path_view = mo.md("_Select a path above to visualise it._")
    if selected_path_info is not None:
        _avg_atoms = selected_path_info["avg_atoms"]
        _instances = selected_path_info["path_instances"]
        _n_total = len(_instances)
        _pos = _avg_atoms.get_positions()

        # Pick which instances to display
        if show_all_paths.value:
            _active_instances = list(range(_n_total))
            _label = f"All {_n_total} equivalent paths"
        else:
            _idx = path_index_slider.value
            _active_instances = [_idx]
            _label = f"Instance {_idx + 1} of {_n_total}"

        # Leg colours
        _leg_colors = {
            "n1": "#e05c2e",
            "n2": "#2e94e0",
            "n1\u2192n2": "#2eac50",
        }

        _vf_groups: dict = {}
        _highlight = set()
        for _i in _active_instances:
            for _v in _instances[_i]:
                _leg = _v["leg"]
                _color = _leg_colors.get(_leg, "#aaaaaa")
                # Unique key per (leg, instance)
                # so all-paths mode doesn't collapse entries
                _key = f"leg_{_leg}_i{_i}"
                if _key not in _vf_groups:
                    _vf_groups[_key] = {
                        "origins": [],
                        "vectors": [],
                        "color": _color,
                        "radius": 0.12,
                    }
                _vf_groups[_key]["origins"].append(
                    _v["origin"].tolist()
                    if "origin" in _v
                    else _pos[_v["from_idx"]].tolist()
                )
                _vf_groups[_key]["vectors"].append(np.array(_v["vector"]).tolist())
                _highlight.add(_v["from_idx"])
                _highlight.add(_v["to_idx"])

        _viewer = AtomsViewer(BaseWidget(guiConfig=guiConfig))
        _viewer.atoms = ASEAdapter.to_weas(_avg_atoms)
        _viewer.model_style = 1
        # boundary
        _viewer.boundary = [[-0.15, 1.15], [-0.15, 1.15], [-0.15, 1.15]]
        _viewer.highlight = list(_highlight)
        _viewer.vf.settings = _vf_groups
        _viewer.vf.show = True
        path_view = _viewer._widget
    return (path_view,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### References
    - Rehr & Albers, *Rev. Mod. Phys.* **72**, 621 (2000) – EXAFS path definitions
    - Kabsch, *Acta Cryst. A* **32**, 922 (1976) – Optimal rotation algorithm
    - Debye-Waller factor: B = 8π²⟨u²⟩
    """)
    return


if __name__ == "__main__":
    app.run()
