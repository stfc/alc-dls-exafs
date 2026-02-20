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

__generated_with = "0.19.11"
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
    1. Upload a trajectory file and configure parameters
    2. Unwrap PBC positions and optionally Kabsch-align frames
    3. Compute per-atom U tensors and B-factors → export CIF with ADP
    4. (Optional) Select an absorber site to compute 2-body and 3-body MSRD paths
    """)
    return


@app.cell
def _():
    import logging
    from collections import defaultdict

    import numpy as np
    import pandas as pd
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
        AtomsViewer,
        BaseWidget,
        defaultdict,
        guiConfig,
        logger,
        np,
        pd,
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
def _(mo):
    skip_frames = mo.ui.number(
        value=0,
        start=0,
        stop=100_000,
        step=1,
        label="Frames to skip at start",
    )
    no_align = mo.ui.switch(label="Skip Kabsch alignment", value=False)
    output_prefix = mo.ui.text(value="output", label="Output file prefix")

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


@app.function
def parse_site_specification(spec, symbols):
    """Parse site specification string and return list of atomic indices.

    Supported formats
    -----------------
    'K'       – all K atoms
    'K.1'     – first K atom (1-based within element)
    'K.1-3'   – first three K atoms
    '11'      – 11th atom in full structure (1-based)
    '11-20'   – atoms 11–20 (1-based, inclusive)
    """
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
    else:
        matching = [i for i, sym in enumerate(symbols) if sym == spec]
        if not matching:
            raise ValueError(f"No atoms of element '{spec}' found")
        return matching


@app.cell
def _(logger, np):
    def unwrap_positions_pbc(structures):
        """Unwrap atomic positions.

        This helps to produce continuous trajectories across
        periodic boundary conditions.

        Returns:
        -------
        unwrapped : ndarray, shape (n_frames, n_atoms, 3)
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
                logger.info(f"  Unwrapping frame {i}/{n_frames}...")
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

    return (unwrap_positions_pbc,)


@app.cell
def _(logger, np):
    def kabsch_align(unwrapped_positions, reference_idx=0, reference_pos=None):
        """Align all trajectory frames to a reference using the Kabsch algorithm.

        Returns:
        -------
        aligned : ndarray, shape (n_frames, n_atoms, 3)
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

    return (kabsch_align,)


@app.cell
def _(defaultdict, logger, np):
    def calculate_grouped_msrd(
        structures,
        unwrapped_positions,
        central_indices,
        central_label,
        cutoff=3.5,
        tol_dist=0.1,
        tol_angle=5.0,
        cutoff_3body=None,
    ):
        """Calculate grouped MSRD for 2-body and 3-body EXAFS paths.

        Returns:
        -------
        res_2b : list[dict]   – sorted by reff
        res_3b : list[dict]   – sorted by reff
        """
        from ase.geometry import find_mic

        if not central_indices:
            return [], []

        symbols = structures[0].get_chemical_symbols()
        central_element = symbols[central_indices[0]]
        reference_atoms = structures[0].copy()
        cell = structures[0].get_cell()
        pbc = structures[0].get_pbc()

        pair_list = []
        triplet_list = []

        logger.info(
            f"Analysing MSRD paths for {central_label} "
            f"({len(central_indices)} sites)..."
        )

        for c_idx in central_indices:
            all_indices = [i for i in range(len(symbols)) if i != c_idx]
            distances = reference_atoms.get_distances(c_idx, all_indices, mic=True)
            neighbors = [
                all_indices[i] for i in range(len(all_indices)) if distances[i] < cutoff
            ]
            logger.info(
                f"  {len(neighbors)} neighbors within {cutoff} Å of atom {c_idx}"
            )

            # Pre-compute MIC vectors for all neighbors
            neighbor_vectors_mic = {}
            for n_idx in neighbors:
                v_raw = (
                    unwrapped_positions[:, n_idx, :] - unwrapped_positions[:, c_idx, :]
                )
                v_mic, dists = find_mic(v_raw, cell, pbc)
                neighbor_vectors_mic[n_idx] = (v_mic, dists)

            # ── 2-body paths ──────────────────────────────────────────────
            for n_idx in neighbors:
                v_mic, dists = neighbor_vectors_mic[n_idx]
                pair_list.append(
                    {
                        "element": symbols[n_idx],
                        "dists": dists,
                        "mean_d": np.mean(dists),
                        "label": f"{central_element}-{symbols[n_idx]}",
                    }
                )

            # ── 3-body paths ──────────────────────────────────────────────
            if cutoff_3body == 0:
                continue
            cutoff_for_3body = cutoff_3body if cutoff_3body is not None else cutoff
            if cutoff_for_3body < cutoff:
                neighbors_3body = [
                    n
                    for n in neighbors
                    if np.mean(neighbor_vectors_mic[n][1]) <= cutoff_for_3body
                ]
            else:
                neighbors_3body = neighbors

            for i in range(len(neighbors_3body)):
                for j in range(i + 1, len(neighbors_3body)):
                    n1, n2 = neighbors_3body[i], neighbors_3body[j]
                    v01_mic, d01 = neighbor_vectors_mic[n1]
                    v02_mic, d02 = neighbor_vectors_mic[n2]
                    v12_raw = (
                        unwrapped_positions[:, n2, :] - unwrapped_positions[:, n1, :]
                    )
                    v12_mic, d12 = find_mic(v12_raw, cell, pbc)
                    d20 = d02
                    L = d01 + d12 + d20

                    # Angle at n1 (intermediate atom)
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
                            "mean_L": np.mean(reff_series),
                            "angle": np.mean(angles_deg),
                        }
                    )

        # ── Cluster 2-body paths ──────────────────────────────────────────
        pairs_by_element = defaultdict(list)
        for path in pair_list:
            pairs_by_element[path["element"]].append(path)

        res_2b = []
        for _element, paths in pairs_by_element.items():
            paths.sort(key=lambda x: x["mean_d"])
            clusters, current = [], [paths[0]]
            for path in paths[1:]:
                if (
                    abs(path["mean_d"] - np.mean([p["mean_d"] for p in current]))
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
                        "reff": np.mean(all_dists),
                        "sigma2": np.var(all_dists, ddof=1),
                        "count": len(cluster),
                    }
                )

        # ── Cluster 3-body paths ──────────────────────────────────────────
        triplets_by_elements = defaultdict(list)
        for path in triplet_list:
            triplets_by_elements[path["elements"]].append(path)

        res_3b = []
        for elem_pair, paths in triplets_by_elements.items():
            paths.sort(key=lambda x: x["angle"])
            angle_clusters, current = [], [paths[0]]
            for path in paths[1:]:
                if (
                    abs(path["angle"] - np.mean([p["angle"] for p in current]))
                    <= tol_angle
                ):
                    current.append(path)
                else:
                    angle_clusters.append(current)
                    current = [path]
            angle_clusters.append(current)

            for angle_cluster in angle_clusters:
                angle_cluster.sort(key=lambda x: x["mean_L"])
                dist_clusters, current = [], [angle_cluster[0]]
                for path in angle_cluster[1:]:
                    if (
                        abs(path["mean_L"] - np.mean([p["mean_L"] for p in current]))
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
                            "type": f"{central_element}-{elem_pair[0]}-{elem_pair[1]}",
                            "reff": np.mean(all_reffs),
                            "sigma2": np.var(all_reffs, ddof=1),
                            "angle": np.mean([p["angle"] for p in cluster]),
                            "count": len(cluster),
                        }
                    )

        return (
            sorted(res_2b, key=lambda x: x["reff"]),
            sorted(res_3b, key=lambda x: x["reff"]),
        )

    return (calculate_grouped_msrd,)


@app.cell
def _(np):
    def save_cif_with_adp(results):
        """Return a CIF string containing mean positions and anisotropic U tensors."""
        import io

        pos = results["avg_positions"]
        names = results["atom_names"]
        u_cart = results["u_tensor"]
        cell = results["avg_cell"]
        inv_cell = np.linalg.inv(cell)
        frac_pos = pos @ inv_cell.T
        a, b, c = np.linalg.norm(cell, axis=1)

        def ang(v1, v2):
            return np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1,
                        1,
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

    return (save_cif_with_adp,)


@app.cell
def _(mo):
    mo.md("""
    ## 3 · Load Trajectory
    """)
    return


@app.cell
def _(mo, skip_frames, trajectory_file):
    import os
    import pathlib
    import tempfile

    structures = None
    _load_status = mo.callout(
        mo.md("⬆️  Upload a trajectory file above to begin."),
        kind="info",
    )

    if trajectory_file.value:
        _tf = trajectory_file.value[0]
        _suffix = pathlib.Path(_tf.name).suffix or ".xyz"
        _tmp = tempfile.NamedTemporaryFile(suffix=_suffix, delete=False)
        _tmp.write(_tf.contents)
        _tmp.close()
        try:
            from ase.io import read as _ase_read

            structures = _ase_read(_tmp.name, index=f"{skip_frames.value}:")
            if not isinstance(structures, list):
                structures = [structures]
            elements_str = ", ".join(sorted(set(structures[0].get_chemical_symbols())))
            _load_status = mo.callout(
                mo.md(
                    f"✅ Loaded **{len(structures)} frames** · "
                    f"**{len(structures[0])} atoms** per frame · "
                    f"Elements: `{elements_str}`"
                ),
                kind="success",
            )
        except (OSError, ValueError) as _e:
            _load_status = mo.callout(
                mo.md(f"❌ Failed to load trajectory: `{_e}`"),
                kind="danger",
            )
        finally:
            os.unlink(_tmp.name)

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

        mo.callout(
            mo.md(
                f"✅ Trajectory processed · "
                f"Shape: `{unwrapped.shape}` (frames × atoms × xyz)"
            ),
            kind="success",
        )
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
def _(mo, np, output_prefix, save_cif_with_adp, structures, unwrapped):
    results = None
    _b_status = mo.md("")

    if unwrapped is not None and structures is not None:
        _avg_pos = np.mean(unwrapped, axis=0)
        _displacements = unwrapped - _avg_pos[np.newaxis, :, :]
        _u_tensor = np.einsum("fni,fnj->nij", _displacements, _displacements) / len(
            structures
        )
        _b_factors = 8 * np.pi**2 * np.trace(_u_tensor, axis1=1, axis2=2) / 3

        results = {
            "b_factors": _b_factors,
            "u_tensor": _u_tensor,
            "avg_positions": _avg_pos,
            "atom_names": structures[0].get_chemical_symbols(),
            "avg_cell": structures[0].get_cell().complete(),
            "atom_indices": np.arange(len(_b_factors)),
        }

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
def _(mo):
    mo.md("""
    ## 6 · B-factor Plot
    """)
    return


@app.cell
def _(alt, mo, np, pd, results):
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

        _hover_b = alt.selection_point(on="mouseover", nearest=True, empty=False)

        _pts = (
            alt.Chart(_df_b)
            .mark_point(filled=True)
            .encode(
                x=alt.X("Atom index:Q", title="Atom index"),
                y=alt.Y("B-factor (Å²):Q", title="B-factor (Å²)"),
                color=alt.Color("Element:N", title="Element"),
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
                color=alt.Color("Element:N"),
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
    mo.md("""
    ## 7 · MSRD Path Analysis

    Specify an **absorber site** to calculate mean square relative displacements
    for EXAFS/FEFF.

    | Format | Meaning |
    |--------|---------|
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
        label="3-body cutoff (Å) — set 0 to skip 3-body paths",
    )
    run_msrd = mo.ui.run_button(label="▶  Run MSRD Analysis")

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([element_spec, cutoff, tol_dist]),
                    mo.vstack([cutoff_3body, tol_angle]),
                ],
                gap=2,
            ),
            run_msrd,
        ]
    )
    return cutoff, cutoff_3body, element_spec, run_msrd, tol_angle, tol_dist


@app.cell
def _(
    calculate_grouped_msrd,
    cutoff,
    cutoff_3body,
    element_spec,
    mo,
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

                with mo.status.spinner("Computing MSRD paths…"):
                    _res2b, _res3b = calculate_grouped_msrd(
                        structures,
                        unwrapped,
                        _indices,
                        _spec,
                        cutoff=cutoff.value,
                        tol_dist=tol_dist.value,
                        tol_angle=tol_angle.value,
                        cutoff_3body=_c3b,
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
                        "Body": "2-body",
                        "Path type": r["type"],
                        "Reff (Å)": r["reff"],
                        "σ² (Å²)": r["sigma2"],
                        "Angle (°)": float("nan"),
                        "Count": r["count"],
                        "Degeneracy": r["count"] / len(_indices),
                    }
                    for r in _res2b
                ] + [
                    {
                        "Body": "3-body",
                        "Path type": r["type"],
                        "Reff (Å)": r["reff"],
                        "σ² (Å²)": r["sigma2"],
                        "Angle (°)": r["angle"],
                        "Count": r["count"],
                        "Degeneracy": 2 * r["count"] / len(_indices),
                    }
                    for r in _res3b
                ]
                msrd_df = pd.DataFrame(_df_rows)
                _msrd_ui = mo.vstack(
                    [
                        mo.md(
                            f"**Site:** `{_spec}` · "
                            f"**{len(_indices)} absorber(s)** · "
                            f"**{len(_res2b)} two-body** and "
                            f"**{len(_res3b)} three-body** paths found"
                        ),
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
                            filename=f"msrd_paths_{_spec.replace(' ', '_')}.csv",
                            label="⬇ Download MSRD paths as CSV",
                        ),
                    ]
                )

    _msrd_ui
    return (msrd_df,)


@app.cell
def _(alt, mo, msrd_df):
    _plot = mo.md("_Run MSRD analysis above to see the path plot._")

    if msrd_df is not None:
        _hover = alt.selection_point(
            on="mouseover",
            nearest=True,
            empty=False,
        )

        _base = alt.Chart(msrd_df)

        _points = (
            _base.mark_point(filled=True)
            .encode(
                x=alt.X("Reff (Å):Q", title="Reff (Å)"),
                y=alt.Y("σ² (Å²):Q", title="σ² (Å²)"),
                color=alt.Color("Path type:N", title="Path type"),
                shape=alt.Shape("Body:N", title="Body"),
                size=alt.condition(_hover, alt.value(180), alt.value(60)),
                opacity=alt.condition(_hover, alt.value(1.0), alt.value(0.55)),
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
            .add_params(_hover)
        )

        # Vertical rule that snaps to the hovered point's x position
        _rule = (
            _base.mark_rule(color="gray", strokeDash=[4, 4], strokeWidth=1)
            .encode(x=alt.X("Reff (Å):Q"))
            .transform_filter(_hover)
        )

        # Text label above hovered point showing σ²
        _label = (
            _base.mark_text(align="left", dx=8, dy=-8, fontSize=11)
            .encode(
                x=alt.X("Reff (Å):Q"),
                y=alt.Y("σ² (Å²):Q"),
                text=alt.Text("σ² (Å²):Q", format=".5f"),
            )
            .transform_filter(_hover)
        )

        _chart = (
            alt.layer(_points, _rule, _label)
            .properties(height=340, width="container")
            .interactive()
        )
        _plot = _chart

    _plot
    return


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _(mo):
    mo.md("""
    ---
    ### References
    - Rehr & Albers, *Rev. Mod. Phys.* **72**, 621 (2000) – EXAFS path definitions
    - Kabsch, *Acta Cryst. A* **32**, 922 (1976) – Optimal rotation algorithm
    - Debye-Waller factor: B = 8π²⟨u²⟩
    """)
    return


if __name__ == "__main__":
    app.run()
