"""EXAFS paths: chi contributions · Debye-Waller σ² · 3-D structure.

Three complementary views per scattering path:
  • k-space and R-space chi contribution  (EXAFS pipeline HDF5)
  • σ² / Reff Debye-Waller scatter plot   (MD trajectory)
  • 3-D averaged structure with path arrows

Path matching notes
-------------------
FEFF paths (from averaged paths HDF5):
    scatterer : element symbol of the scattering atom
    nlegs     : number of path legs
                  2 → single-scattering (A→B→A);  r_eff = bond distance
                  4 → rattle / collinear MS (A→B→A→B→A);  r_eff ≈ 2×r_bond
                  3 → triangular path;  r_eff = half total path length
    r_eff_ref : effective path length = half total path length (Å)

DW/MSRD (from calculate_grouped_msrd):
    type  : "Absorber-Scatterer"  (2-body only)
    reff  : mean bond distance (Å)   ← equal to r_eff for nlegs=2 only
    σ²    : variance of bond distances = σ²_EXAFS for nlegs=2

Only nlegs=2 FEFF paths are matched to DW σ² directly.
For nlegs=4 collinear rattle paths: σ²_rattle ≈ 4 × σ²_single (approximate).
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=6.0.0",
#     "ase>=3.27.0",
#     "h5py",
#     "marimo>=0.21.1",
#     "numpy>=2.0",
#     "pandas>=2.0",
#     "xraylarch>=0.9",
#     "weas-widget>=0.1.26",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="EXAFS Paths — chi + DW + Structure")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import logging
    from collections import defaultdict

    import altair as alt
    import h5py
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

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("paths_combined")
    guiConfig = {"controls": {"enabled": False}}
    return (
        ASEAdapter,
        Atoms,
        AtomsViewer,
        BaseWidget,
        alt,
        ase_read,
        atomic_numbers,
        defaultdict,
        find_mic,
        guiConfig,
        h5py,
        jmol_colors,
        logger,
        np,
        pd,
    )


@app.cell
def _(np):
    from larch import Group
    from larch.xafs import xftf

    # k / R chart display limits — not FT window (those are UI sliders below)
    KMIN = 2.0
    KMAX = 14.0
    RMIN = 0.0

    def perform_FT(
        k,
        chi,
        kmin=3.0,
        kmax=12.0,
        kweight=2,
        dk=1.0,
        k_window="hanning",
        rmax=10.0,
    ):
        g = Group(k=k, chi=chi)
        xftf(
            g,
            kmin=kmin,
            kmax=kmax,
            dk=dk,
            kweight=kweight,
            window=k_window,
            rmax_out=rmax,
        )
        return g

    def compute_chi_from_params(k_grid, amp, pha, lam, rep, reff, degen, k_param):
        """Recompute χ(k) from averaged FEFF raw parameters on an arbitrary grid.

        Uses the full complex-momentum EXAFS formula (matching larch's
        FeffPathGroup._calc_chi with default path parameters) by linearly
        interpolating amp/pha/lam/rep from the native coarse FEFF grid.
        """
        amp_i = np.interp(k_grid, k_param, amp)
        pha_i = np.interp(k_grid, k_param, pha)
        lam_i = np.interp(k_grid, k_param, lam)
        rep_i = np.interp(k_grid, k_param, rep)

        q = k_grid
        # complex momentum squared: (rep + i/λ)²
        pp = (rep_i + 1j / np.clip(lam_i, 1e-6, None)) ** 2
        p = np.sqrt(pp)

        # Full EXAFS equation with S0²=1, σ²=0, ΔE₀=0, ΔR=0
        cchi = np.exp(-2 * reff * p.imag + 1j * (2 * q * reff + pha_i))
        cchi = degen * amp_i * cchi / (q * reff**2)
        return np.asarray(cchi.imag, dtype=np.float64)

    return KMAX, KMIN, RMIN, compute_chi_from_params, perform_FT


@app.cell
def _(defaultdict, find_mic, logger, np):
    def unwrap_positions_pbc(structures):
        n_frames = len(structures)
        n_atoms = len(structures[0])
        unwrapped = np.zeros((n_frames, n_atoms, 3))
        ref_atoms = structures[0].copy()
        ref_atoms.center()
        unwrapped[0] = ref_atoms.get_positions()
        for i in range(1, n_frames):
            atoms = structures[i]
            cell = atoms.get_cell()
            if np.all(cell.lengths() == 0):
                unwrapped[i] = atoms.get_positions()
                continue
            cell_matrix = cell.complete()
            inv_cell = np.linalg.inv(cell_matrix)
            frac_disp = atoms.get_scaled_positions() - unwrapped[i - 1] @ inv_cell.T
            frac_disp -= np.round(frac_disp)
            unwrapped[i] = unwrapped[i - 1] + frac_disp @ cell_matrix
        return unwrapped

    def kabsch_align(unwrapped_positions, reference_pos=None):
        ref_pos = reference_pos if reference_pos is not None else unwrapped_positions[0]
        ref_com = ref_pos.mean(axis=0)
        ref_centered = ref_pos - ref_com
        aligned = np.zeros_like(unwrapped_positions)
        for i in range(len(unwrapped_positions)):
            pos = unwrapped_positions[i]
            pos_c = pos - pos.mean(axis=0)
            U, _S, Vt = np.linalg.svd(pos_c.T @ ref_centered)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            aligned[i] = pos_c @ R + ref_com
        return aligned

    def compute_adp_results(structures, unwrapped):
        avg_pos = np.mean(unwrapped, axis=0)
        disp = unwrapped - avg_pos[np.newaxis]
        u_tensor = np.einsum("fni,fnj->nij", disp, disp) / len(structures)
        b_factors = 8 * np.pi**2 * np.trace(u_tensor, axis1=1, axis2=2) / 3
        return {
            "b_factors": b_factors,
            "u_tensor": u_tensor,
            "avg_positions": avg_pos,
            "atom_names": structures[0].get_chemical_symbols(),
            "avg_cell": structures[0].get_cell().complete(),
        }

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
        """Calculate grouped MSRD for 2-body and 3-body EXAFS paths.

        Returns (res_2b, res_3b) each sorted by reff.
        Matches the full implementation in debye_waller.py.
        """
        if not central_indices:
            return [], []

        symbols = structures[0].get_chemical_symbols()
        central_element = symbols[central_indices[0]]
        reference_atoms = structures[0].copy()
        cell = structures[0].get_cell()
        pbc = structures[0].get_pbc()

        if exclude_hydrogen:
            neighbor_candidates = {i for i, sym in enumerate(symbols) if sym != "H"}
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
                all_indices[i] for i in range(len(all_indices)) if distances[i] < cutoff
            ]
            logger.info(
                "  %d neighbors within %.2f Å of atom %d", len(neighbors), cutoff, c_idx
            )

            # Cache MIC vectors — reused for 3-body
            neighbor_vectors_mic = {}
            for n_idx in neighbors:
                v_raw = (
                    unwrapped_positions[:, n_idx, :] - unwrapped_positions[:, c_idx, :]
                )
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
            for _i in range(len(neighbors_3body)):
                for _j in range(_i + 1, len(neighbors_3body)):
                    n1, n2 = neighbors_3body[_i], neighbors_3body[_j]
                    v01_mic, d01 = neighbor_vectors_mic[n1]
                    v02_mic, d02 = neighbor_vectors_mic[n2]
                    v12_raw = (
                        unwrapped_positions[:, n2, :] - unwrapped_positions[:, n1, :]
                    )
                    v12_mic, d12 = find_mic(v12_raw, cell, pbc)
                    L = d01 + d12 + d02
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
                            "reff_series": L / 2.0,
                            "mean_L": float(np.mean(L / 2.0)),
                            "angle": float(np.mean(angles_deg)),
                            "c_idx": c_idx,
                            "n1_idx": n1,
                            "n2_idx": n2,
                        }
                    )

        # ── 2-body grouping: by element, then cluster by distance ─────────
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
                    abs(path["mean_d"] - float(np.mean([p["mean_d"] for p in current])))
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

        # ── 3-body grouping: by element pair, angle shell, distance shell ─
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
                    abs(path["angle"] - float(np.mean([p["angle"] for p in current])))
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

    return (
        calculate_grouped_msrd,
        compute_adp_results,
        kabsch_align,
        unwrap_positions_pbc,
    )


@app.cell
def _(mo):
    hdf5_file = mo.ui.file_browser(
        initial_path=".",
        filetypes=[".h5", ".hdf5"],
        label="Averaged paths HDF5 file",
        multiple=False,
    )
    traj_file = mo.ui.file_browser(
        initial_path=".",
        filetypes=[".extxyz", ".xyz"],
        label="MD trajectory file for DW/MSRD (optional — leave unselected to skip)",
        multiple=False,
    )
    demeter_file = mo.ui.file_browser(
        initial_path=".",
        filetypes=[".dat", ".txt"],
        label="Demeter paths.dat file (optional — for path pre-selection)",
        multiple=False,
    )
    mo.vstack([hdf5_file, traj_file, demeter_file])
    return demeter_file, hdf5_file, traj_file


@app.cell
def _(demeter_file, pd):
    """Parse an optional Demeter / Athena paths.dat file."""
    import re as _re

    def parse_demeter_paths(filepath):
        """Return a DataFrame of FEFF paths from a Demeter paths.dat file.

        Columns: path_num, degen, reff, legs, rank, path_type, path_str,
                 atoms (space-separated non-absorber atoms), scatterer (element).
        """
        rows = []
        with open(filepath) as fh:
            for line in fh:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                m = _re.match(r"^\s*(\d{4})\s+([\d.]+)\s+([\d.]+)\s+---\s+", line)
                if not m:
                    continue
                path_num = int(m.group(1))
                degen = float(m.group(2))
                reff = float(m.group(3))
                rest = line[m.end() :]
                last_at = rest.rfind("@")
                if last_at < 0:
                    continue
                path_str = rest[: last_at + 1].strip()
                after = rest[last_at + 1 :].split()
                if len(after) < 3:
                    continue
                rank = float(after[1])
                legs = int(after[2])
                path_type = " ".join(after[3:])
                atoms_raw = [t for t in path_str.split() if t != "@"]
                scatterer = _re.sub(r"[\d.]+", "", atoms_raw[0]) if atoms_raw else "?"
                rows.append(
                    {
                        "path_num": path_num,
                        "degen": degen,
                        "reff": reff,
                        "legs": legs,
                        "rank": rank,
                        "path_type": path_type,
                        "path_str": path_str,
                        "atoms": " ".join(atoms_raw),
                        "scatterer": scatterer,
                    }
                )
        return pd.DataFrame(rows) if rows else None

    demeter_df = None
    if demeter_file.value:
        try:
            demeter_df = parse_demeter_paths(demeter_file.path(0))
        except Exception:
            pass  # silently skip unreadable files
    return (demeter_df,)


@app.cell
def _(mo):
    ft_kmin = mo.ui.slider(
        start=0.0,
        stop=6.0,
        value=3.0,
        step=0.1,
        label="k_min (Å⁻¹)",
        show_value=True,
    )
    ft_kmax = mo.ui.slider(
        start=5.0,
        stop=20.0,
        value=12.0,
        step=0.5,
        label="k_max (Å⁻¹)",
        show_value=True,
    )
    ft_kweight = mo.ui.dropdown(
        options=[1, 2, 3],
        value=2,
        label="k-weight",
    )
    ft_dk = mo.ui.slider(
        start=0.0,
        stop=4.0,
        value=1.0,
        step=0.1,
        label="dk (Å⁻¹)",
        show_value=True,
    )
    ft_window = mo.ui.dropdown(
        options=["hanning", "kaiser", "parzen", "welch", "sine"],
        value="hanning",
        label="Window",
    )
    ft_rmax = mo.ui.slider(
        start=2.0,
        stop=20.0,
        value=10.0,
        step=0.5,
        label="R_max (Å)",
        show_value=True,
    )
    residual_mode_box = mo.ui.switch(
        value=False,
        label="Residual = total − selected paths (off: total − all stored paths)",
    )
    show_paths_box = mo.ui.switch(
        value=True, label="Show individual path contributions"
    )
    mo.vstack(
        [
            mo.md("### Fourier Transform"),
            mo.hstack([ft_kmin, ft_kmax, ft_dk], gap=2),
            mo.hstack([ft_kweight, ft_window, ft_rmax], gap=2),
            mo.hstack([residual_mode_box, show_paths_box], gap=4),
        ]
    )
    return (
        ft_dk,
        ft_kmax,
        ft_kmin,
        ft_kweight,
        ft_rmax,
        ft_window,
        residual_mode_box,
        show_paths_box,
    )


@app.cell
def _(mo):
    equil_box = mo.ui.number(
        value=0, start=0, stop=10_000, step=1, label="Equilibration frames to skip"
    )
    n_paths_box = mo.ui.number(
        value=100, start=1, step=1, label="Max paths per site per frame"
    )
    r_tol_box = mo.ui.number(
        value=0.25, start=0.01, stop=2.0, step=0.01, label="r_eff tolerance Å"
    )
    skip_frames_box = mo.ui.number(
        value=0, start=0, stop=100_000, step=1, label="Traj skip frames (DW)"
    )
    cutoff_box = mo.ui.number(
        value=8.0, start=0.5, stop=20.0, step=0.1, label="Neighbor cutoff Å (DW)"
    )
    dw_tol_box = mo.ui.number(
        value=0.1, start=0.02, stop=1.0, step=0.01, label="DW shell grouping tol Å"
    )
    tol_angle_box = mo.ui.number(
        value=5.0, start=0.1, stop=45.0, step=0.5, label="Angle grouping tol (°)"
    )
    cutoff_3body_box = mo.ui.number(
        value=6.0, start=0.0, stop=20.0, step=0.1, label="3-body cutoff Å (0=skip)"
    )
    site_indices_box = mo.ui.text(
        value="",
        label="Site indices override (comma-separated, e.g. 125,126; blank=auto)",
        full_width=True,
    )
    k_step_box = mo.ui.number(
        value=0.05,
        start=0.01,
        stop=0.5,
        step=0.01,
        label="k-grid step Å⁻¹ (path recomputation)",
    )
    recompute_paths_box = mo.ui.checkbox(
        value=False,
        label="Recompute path χ from raw FEFF params",
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([equil_box, n_paths_box, r_tol_box]),
                    mo.vstack([skip_frames_box, cutoff_box, dw_tol_box]),
                    mo.vstack([tol_angle_box, cutoff_3body_box]),
                ],
                gap=2,
            ),
            site_indices_box,
            mo.hstack([k_step_box, recompute_paths_box], gap=2),
        ]
    )
    return (
        cutoff_3body_box,
        cutoff_box,
        dw_tol_box,
        k_step_box,
        r_tol_box,
        recompute_paths_box,
        site_indices_box,
        skip_frames_box,
        tol_angle_box,
    )


@app.cell
def _(mo):
    btn_load = mo.ui.run_button(label="▶ Load & compute")
    btn_load
    return (btn_load,)


@app.cell
def _(btn_load, h5py, hdf5_file, mo, np):
    merged_path_data = None
    total_chi_hdf5 = None
    kref_hdf5 = None
    absorber_el_h5 = None

    mo.stop(
        not btn_load.value,
        mo.callout(
            mo.md("Select files above then click **▶ Load & compute** to start."),
            kind="info",
        ),
    )
    mo.stop(
        not hdf5_file.value,
        mo.callout(mo.md("Select an averaged-paths HDF5 file above."), kind="warn"),
    )

    _hdf5_avg = hdf5_file.path(0)

    with mo.status.spinner("Loading averaged paths from HDF5..."):
        with h5py.File(_hdf5_avg, "r") as fh:
            meta = fh.get("meta", {})
            source_h5_path = str(meta.attrs.get("source_h5_path", ""))

            oa = fh["overall_average"]
            kref_hdf5 = np.array(oa["k"])
            total_chi_hdf5 = np.array(oa["chi"])

            # Count total successful spectra.
            # Priority: (1) n_total attr on overall_average (written by pipeline
            # v1.1+), (2) max(n_samples) across paths — the most prevalent path
            # appears in every frame, so max == n_total_overall.
            total_spectra = int(oa.attrs.get("n_total", 0))
            n_sites = 0

            site_avg_grp = fh.get("site_averages", {})
            n_sites = len(site_avg_grp)

            path_groups = []
            paths_grp = oa.get("paths")
            if paths_grp:
                for path_key in sorted(paths_grp.keys()):
                    p = paths_grp[path_key]
                    attrs = dict(p.attrs)
                    n_samples = int(attrs.get("n_samples", 0))
                    contribution_pct = float(attrs.get("contribution_pct", 0.0))

                    _k_path = np.array(p["k"])
                    _chi_path = np.array(p["chi"])
                    if len(_k_path) != len(kref_hdf5) or not np.allclose(
                        _k_path, kref_hdf5
                    ):
                        _chi_path = np.interp(
                            kref_hdf5, _k_path, _chi_path, left=0.0, right=0.0
                        )

                    # Read averaged raw FEFF parameters for on-the-fly recomputation
                    _raw_params = {}
                    for _pname in ("amp", "pha", "lam", "rep"):
                        if _pname in p:
                            _raw_params[_pname] = np.array(p[_pname])
                    if "k_param" in p:
                        _raw_params["k_param"] = np.array(p["k_param"])

                    pg = {
                        "path_key": path_key,
                        "scatterer": str(attrs.get("scatterer", "?")),
                        "nlegs": int(attrs.get("nlegs", 2)),
                        "r_eff_ref": float(attrs.get("r_eff", 0.0)),
                        "r_effs": [float(attrs.get("r_eff", 0.0))],
                        "chi_list": [_chi_path],
                        "chi_weights": [n_samples],
                        "frame_set": {path_key},
                        "kref": kref_hdf5,
                        "sparse": False,
                        "n_samples": n_samples,
                        "contribution_pct": contribution_pct,
                        "degeneracy": float(attrs.get("degeneracy", 0.0)),
                        "cw_ratio": float(attrs.get("cw_ratio", 0.0)),
                        **_raw_params,
                    }
                    path_groups.append(pg)

            # Resolve total_spectra now that n_samples values are available.
            if total_spectra == 0 and path_groups:
                # Fallback: the most prevalent path appears in every frame,
                # so max(n_samples) == n_total_overall.
                total_spectra = max(pg["n_samples"] for pg in path_groups)
            if total_spectra == 0:
                total_spectra = 1

            # Compute residual (missing path contributions dropped by FEFF criteria).
            # Each path's chi is averaged over the frames where it exists, so we
            # weight by n_samples / total_spectra to get its true contribution to
            # the overall average spectrum.
            _chi_sum_paths = np.zeros_like(total_chi_hdf5)
            for _pg in path_groups:
                _w = _pg.get("chi_weights", [1])
                _n = _w[0] if _w else 1
                _chi_sum_paths += _pg["chi_list"][0] * (_n / total_spectra)
            _chi_residual = total_chi_hdf5 - _chi_sum_paths

            merged_path_data = {
                "merged_sites": {
                    "path_groups": path_groups,
                    "kref": kref_hdf5,
                    "total_frames": total_spectra,
                    "chi_residual": _chi_residual,
                }
            }

            # Read absorber element from source HDF5
            if source_h5_path:
                try:
                    with h5py.File(source_h5_path, "r") as src:
                        frames_grp = src.get("frames", {})
                        for _fn in sorted(frames_grp.keys())[:1]:
                            _sites_grp = frames_grp[_fn].get("sites", {})
                            for _sn in sorted(_sites_grp.keys())[:1]:
                                _raw_el = _sites_grp[_sn].attrs.get(
                                    "absorber_element", b""
                                )
                                absorber_el_h5 = (
                                    _raw_el.decode("utf-8", errors="ignore")
                                    if isinstance(_raw_el, bytes)
                                    else str(_raw_el)
                                ).strip()
                except Exception:
                    pass

            # ── Extract FEFF site indices from averaged HDF5 ─────────────────
            # These are the actual atom indices that FEFF was run on.
            # We use them to restrict the DW calculation to the same sites.
            feff_site_indices: list[int] = []
            if site_avg_grp:
                # site_averages/site_XXXX → XXXX is the atom index
                for _sk in site_avg_grp.keys():
                    try:
                        feff_site_indices.append(int(_sk.split("_")[-1]))
                    except (IndexError, ValueError):
                        pass
            elif paths_grp:
                # Fallback: union of source_sites across all paths
                _all_source_sites: set[int] = set()
                for _pk in paths_grp.keys():
                    _sp = paths_grp[_pk]
                    if "source_sites" in _sp:
                        _all_source_sites.update(int(x) for x in _sp["source_sites"][:])
                feff_site_indices = sorted(_all_source_sites)

    _n_groups = len(merged_path_data["merged_sites"]["path_groups"])
    _site_idx_str = (
        ", ".join(str(s) for s in feff_site_indices[:5])
        + ("…" if len(feff_site_indices) > 5 else "")
        if feff_site_indices
        else "(unknown)"
    )
    mo.callout(
        mo.md(
            f"Loaded **{_n_groups} averaged path groups** · "
            f"**{n_sites} sites** · **{total_spectra} spectra** · "
            f"Absorber: **{absorber_el_h5 or '(unknown)'}** · "
            f"FEFF sites: **{_site_idx_str}**"
        ),
        kind="success",
    )
    return (
        absorber_el_h5,
        feff_site_indices,
        kref_hdf5,
        merged_path_data,
        total_chi_hdf5,
    )


@app.cell
def _(
    ft_dk,
    ft_kmax,
    ft_kmin,
    ft_kweight,
    ft_rmax,
    ft_window,
    kref_hdf5,
    mo,
    perform_FT,
    total_chi_hdf5,
):
    g_total_hdf5 = None
    mo.stop(kref_hdf5 is None)
    g_total_hdf5 = perform_FT(
        kref_hdf5,
        total_chi_hdf5,
        kmin=ft_kmin.value,
        kmax=ft_kmax.value,
        kweight=ft_kweight.value,
        dk=ft_dk.value,
        k_window=ft_window.value,
        rmax=ft_rmax.value,
    )
    return (g_total_hdf5,)


@app.cell
def _(
    compute_chi_from_params,
    ft_dk,
    ft_kmax,
    ft_kmin,
    ft_kweight,
    ft_rmax,
    ft_window,
    g_total_hdf5,
    k_step_box,
    kref_hdf5,
    merged_path_data,
    np,
    perform_FT,
    recompute_paths_box,
    total_chi_hdf5,
):
    display_merged_data = merged_path_data
    display_kref = kref_hdf5
    display_total_chi = total_chi_hdf5
    display_g_total = g_total_hdf5

    if (
        recompute_paths_box.value
        and merged_path_data is not None
        and "amp" in merged_path_data["merged_sites"]["path_groups"][0]
    ):
        _k_step = float(k_step_box.value)
        _k_new = np.arange(0.05, 20.0 + _k_step / 2, _k_step)

        _new_pgs = []
        for _pg in merged_path_data["merged_sites"]["path_groups"]:
            if all(p in _pg for p in ("amp", "pha", "lam", "rep", "k_param")):
                _chi_r = compute_chi_from_params(
                    _k_new,
                    _pg["amp"],
                    _pg["pha"],
                    _pg["lam"],
                    _pg["rep"],
                    _pg["r_eff_ref"],
                    _pg["degeneracy"],
                    _pg["k_param"],
                )
                _pg_new = dict(_pg)
                _pg_new["chi_list"] = [_chi_r]
                _pg_new["kref"] = _k_new
                _new_pgs.append(_pg_new)
            else:
                _new_pgs.append(_pg)

        _chi_sum = np.zeros_like(_k_new)
        _total_frames = merged_path_data["merged_sites"]["total_frames"]
        for _pg in _new_pgs:
            _w = _pg.get("chi_weights", [1])
            _n = _w[0] if _w else 1
            _chi_sum += _pg["chi_list"][0] * (_n / _total_frames)
        _chi_residual = np.interp(_k_new, kref_hdf5, total_chi_hdf5) - _chi_sum

        display_merged_data = {
            "merged_sites": {
                "path_groups": _new_pgs,
                "kref": _k_new,
                "total_frames": _total_frames,
                "chi_residual": _chi_residual,
            }
        }
        display_kref = _k_new
        display_total_chi = np.interp(_k_new, kref_hdf5, total_chi_hdf5)
        display_g_total = perform_FT(
            display_kref,
            display_total_chi,
            kmin=ft_kmin.value,
            kmax=ft_kmax.value,
            dk=ft_dk.value,
            kweight=ft_kweight.value,
            k_window=ft_window.value,
            rmax=ft_rmax.value,
        )
    return (
        display_g_total,
        display_kref,
        display_merged_data,
        display_total_chi,
    )


@app.cell
def _(
    Atoms,
    absorber_el_h5,
    ase_read,
    btn_load,
    calculate_grouped_msrd,
    compute_adp_results,
    cutoff_3body_box,
    cutoff_box,
    dw_tol_box,
    feff_site_indices: list[int],
    kabsch_align,
    mo,
    np,
    site_indices_box,
    skip_frames_box,
    tol_angle_box,
    traj_file,
    unwrap_positions_pbc,
):
    avg_atoms = None
    msrd_2b = None
    msrd_3b = None
    dw_results = None

    mo.stop(not btn_load.value)
    mo.stop(not traj_file.value)  # silently skip — trajectory is optional
    _traj = traj_file.path(0)

    with mo.status.spinner("Loading trajectory..."):
        _raw = ase_read(_traj, index=f"{skip_frames_box.value}:")
        if not isinstance(_raw, list):
            _raw = [_raw]

    with mo.status.spinner("Unwrapping PBC..."):
        _unwrapped = unwrap_positions_pbc(_raw)

    with mo.status.spinner("Kabsch alignment (2-pass)..."):
        _rough = kabsch_align(_unwrapped)
        _unwrapped = kabsch_align(_unwrapped, reference_pos=np.mean(_rough, axis=0))

    dw_results = compute_adp_results(_raw, _unwrapped)
    avg_atoms = Atoms(
        symbols=dw_results["atom_names"],
        positions=dw_results["avg_positions"],
        cell=dw_results["avg_cell"],
        pbc=_raw[0].get_pbc(),
    )

    # Harmonise absorber selection with FEFF paths.
    # Priority: (1) manual override, (2) auto-extracted from HDF5,
    # (3) fall back to all sites matching the absorber element.
    _symbols = dw_results["atom_names"]
    _absorber_el = absorber_el_h5 or list(dict.fromkeys(_symbols))[0]

    _manual_override = site_indices_box.value.strip()
    if _manual_override:
        try:
            _central_indices = [
                int(x.strip()) for x in _manual_override.split(",") if x.strip()
            ]
        except ValueError:
            _central_indices = []
            mo.callout(
                mo.md(
                    "Invalid site indices override — must be comma-separated integers."
                ),
                kind="alert",
            )
    elif feff_site_indices:
        _central_indices = [i for i in feff_site_indices if i < len(_symbols)]
    else:
        _central_indices = [i for i, s in enumerate(_symbols) if s == _absorber_el]

    if not _central_indices:
        mo.callout(
            mo.md(
                f"No valid central indices for absorber **{_absorber_el}**. "
                "Check site-indices override or trajectory/HDF5 mismatch."
            ),
            kind="alert",
        )
        mo.stop()

    _site_source = (
        "manual"
        if _manual_override
        else ("HDF5" if feff_site_indices else "element match")
    )

    with mo.status.spinner(f"Computing MSRD paths for absorber={_absorber_el}..."):
        msrd_2b, msrd_3b = calculate_grouped_msrd(
            _raw,
            _unwrapped,
            _central_indices,
            _absorber_el,
            cutoff=cutoff_box.value,
            tol_dist=dw_tol_box.value,
            tol_angle=tol_angle_box.value,
            cutoff_3body=cutoff_3body_box.value or None,
            exclude_hydrogen=True,
        )

    mo.callout(
        mo.md(
            f"Trajectory: **{len(_raw)} frames** · Absorber: **{_absorber_el}** "
            f"({len(_central_indices)} sites, source: {_site_source}) · "
            f"**{len(msrd_2b)} 2-body** + **{len(msrd_3b)} 3-body** MSRD groups"
        ),
        kind="success",
    )
    return avg_atoms, dw_results, msrd_2b, msrd_3b


@app.cell
def _():
    # mo.stop(msrd_2b is None)
    # _rows_2b = [
    #     {"Body": "2-body", "Type": m["type"], "Reff_DW (Å)": round(m["reff"], 4),
    #      "σ² (Å²)": round(m["sigma2"], 6), "N_pairs": m["count"], "Angle (°)": ""}
    #     for m in msrd_2b
    # ]
    # _rows_3b = [
    #     {"Body": "3-body", "Type": m["type"], "Reff_DW (Å)": round(m["reff"], 4),
    #      "σ² (Å²)": round(m["sigma2"], 6), "N_pairs": m["count"],
    #      "Angle (°)": round(m["angle"], 1)}
    #     for m in (msrd_3b or [])
    # ]
    # _dw_df = pd.DataFrame(_rows_2b + _rows_3b)
    # mo.ui.table(_dw_df, label="DW/MSRD shell groups (sorted by Reff)")
    return


@app.cell
def _(display_merged_data, mo, msrd_2b, msrd_3b, np, pd, r_tol_box):
    mo.stop(display_merged_data is None or msrd_2b is None)

    _pgs = display_merged_data["merged_sites"]["path_groups"]
    _tol = r_tol_box.value
    _rows = []

    # ── 2-body DW groups (master) ─────────────────────────────────────────
    for _dw_i, _m in enumerate(msrd_2b):
        _scatterer = _m["type"].split("-")[-1]
        _reff_dw = _m["reff"]
        _group_idx = None
        _reff_exafs = None
        _feff_nlegs = None
        _feff_note = "(no FEFF match)"

        # direct single-scattering match (nlegs=2) — pick closest, not first
        _ss_matches = [
            (_gi, float(np.mean(_pg["r_effs"])))
            for _gi, _pg in enumerate(_pgs)
            if (
                _pg["nlegs"] == 2
                and _pg["scatterer"] == _scatterer
                and abs(float(np.mean(_pg["r_effs"])) - _reff_dw) < _tol
            )
        ]
        if _ss_matches:
            _group_idx, _reff_exafs = min(
                _ss_matches, key=lambda x: abs(x[1] - _reff_dw)
            )
            _feff_nlegs = 2
            _feff_note = "direct ss"

        # collinear rattle (nlegs=4): r_eff_rattle ≈ 2 × r_bond
        if _group_idx is None:
            _rattle_matches = [
                (_gi, float(np.mean(_pg["r_effs"])))
                for _gi, _pg in enumerate(_pgs)
                if (
                    _pg["nlegs"] == 4
                    and _pg["scatterer"] == _scatterer
                    and abs(float(np.mean(_pg["r_effs"])) / 2.0 - _reff_dw) < _tol
                )
            ]
            if _rattle_matches:
                _group_idx, _reff_exafs = min(
                    _rattle_matches, key=lambda x: abs(x[1] / 2.0 - _reff_dw)
                )
                _feff_nlegs = 4
                _feff_note = "rattle (4×σ²)"

        _contrib = (
            _pgs[_group_idx]["contribution_pct"] if _group_idx is not None else 0.0
        )
        _rows.append(
            {
                "dw_body": "2b",
                "dw_idx": _dw_i,
                "Body": "2-body",
                "Type": _m["type"],
                "Scatterer": _scatterer,
                "Reff_DW (Å)": round(_reff_dw, 4),
                "σ² (Å²)": round(_m["sigma2"], 6),
                "N_pairs": _m["count"],
                "Angle (°)": None,
                "Reff_EXAFS (Å)": round(_reff_exafs, 4)
                if _reff_exafs is not None
                else None,
                "Nlegs_FEFF": _feff_nlegs,
                "group_idx": _group_idx,
                "FEFF match": _feff_note,
                "contribution_pct": _contrib,
            }
        )

    # ── 3-body DW groups (master) ─────────────────────────────────────────
    for _dw_i, _m in enumerate(msrd_3b or []):
        _parts = _m["type"].split("-")  # e.g. "K-N-C"
        _scatterer = "-".join(_parts[1:])  # "N-C"
        _reff_dw = _m["reff"]
        _group_idx = None
        _reff_exafs = None
        _feff_note = "(no FEFF match)"

        _tri_matches = [
            (_gi, float(np.mean(_pg["r_effs"])))
            for _gi, _pg in enumerate(_pgs)
            if (
                _pg["nlegs"] == 3
                and abs(float(np.mean(_pg["r_effs"])) - _reff_dw) < _tol
            )
        ]
        if _tri_matches:
            _group_idx, _reff_exafs = min(
                _tri_matches, key=lambda x: abs(x[1] - _reff_dw)
            )
            _feff_note = "direct triangular"

        _contrib = (
            _pgs[_group_idx]["contribution_pct"] if _group_idx is not None else 0.0
        )
        _rows.append(
            {
                "dw_body": "3b",
                "dw_idx": _dw_i,
                "Body": "3-body",
                "Type": _m["type"],
                "Scatterer": _scatterer,
                "Reff_DW (Å)": round(_reff_dw, 4),
                "σ² (Å²)": round(_m["sigma2"], 6),
                "N_pairs": _m["count"],
                "Angle (°)": round(_m["angle"], 1),
                "Reff_EXAFS (Å)": round(_reff_exafs, 4)
                if _reff_exafs is not None
                else None,
                "Nlegs_FEFF": 3 if _group_idx is not None else None,
                "group_idx": _group_idx,
                "FEFF match": _feff_note,
                "contribution_pct": _contrib,
            }
        )

    path_table_df = pd.DataFrame(_rows)
    # path_table_df
    return (path_table_df,)


@app.cell
def _(contrib_range, path_table_df):
    filtered_path_table_df = path_table_df[
        (path_table_df["contribution_pct"] >= contrib_range.value[0])
        & (path_table_df["contribution_pct"] <= contrib_range.value[1])
    ]
    return (filtered_path_table_df,)


@app.cell
def _(demeter_df, mo):
    """Demeter path selector panel — only shown when a paths.dat file is loaded."""
    demeter_selector = None
    demeter_apply_toggle = None
    _panel = None

    if demeter_df is not None:
        demeter_apply_toggle = mo.ui.switch(
            value=True,
            label="Restrict selection table to Demeter paths",
        )
        _display_cols_dem = [
            "path_num",
            "degen",
            "reff",
            "legs",
            "rank",
            "path_type",
            "atoms",
        ]
        demeter_selector = mo.ui.table(
            demeter_df[_display_cols_dem].reset_index(drop=True),
            selection="multi",
            label="Demeter paths \u2014 select rows to restrict (leave empty = use all)",
        )
        _panel = mo.vstack(
            [
                mo.md("### Demeter path filter"),
                mo.md(
                    f"Loaded **{len(demeter_df)} paths** from Demeter file. "
                    "Select rows to restrict the path table below, or leave empty to use all."
                ),
                demeter_apply_toggle,
                demeter_selector,
            ]
        )
    _panel
    return demeter_apply_toggle, demeter_selector


@app.cell
def _(
    demeter_apply_toggle,
    demeter_df,
    demeter_selector,
    filtered_path_table_df,
    pd,
    r_tol_box,
):
    """Apply Demeter filter on top of the contribution-range filter."""
    display_path_table_df = filtered_path_table_df

    if (
        demeter_df is None
        or demeter_apply_toggle is None
        or not demeter_apply_toggle.value
    ):
        pass  # pass-through: no Demeter filter active
    else:
        # Use selected Demeter paths; fall back to all if nothing selected
        sel_dem = (
            demeter_selector.value
            if demeter_selector is not None and len(demeter_selector.value) > 0
            else demeter_df
        )
        _tol = r_tol_box.value

        def _dem_match(row):
            reff_exafs = row["Reff_EXAFS (Å)"]
            nlegs = row["Nlegs_FEFF"]
            if pd.isna(reff_exafs) or pd.isna(nlegs):
                return False
            for _, dem in sel_dem.iterrows():
                if abs(reff_exafs - dem["reff"]) <= _tol and int(nlegs) == dem["legs"]:
                    return True
            return False

        display_path_table_df = filtered_path_table_df[
            filtered_path_table_df.apply(_dem_match, axis=1)
        ]
    return (display_path_table_df,)


@app.cell
def _(display_path_table_df, mo):
    path_selector = None
    mo.stop(display_path_table_df is None)

    _display_cols = [
        "Body",
        "Type",
        "Reff_DW (Å)",
        "σ² (Å²)",
        "Angle (°)",
        "N_pairs",
        "Reff_EXAFS (Å)",
        "Nlegs_FEFF",
        "FEFF match",
        "contribution_pct",
    ]
    path_selector = mo.ui.table(
        display_path_table_df[_display_cols],
        selection="multi",
        label="Select one or more path groups",
    )
    path_selector
    return (path_selector,)


@app.cell
def _(merged_path_data, mo):
    mo.stop(merged_path_data is None)
    contrib_range = mo.ui.range_slider(
        start=0.0,
        stop=100.0,
        value=[0.0, 100.0],
        step=0.5,
        label="Contribution filter (% of total χ)",
        show_value=True,
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("### Path contribution filter"),
            mo.md(
                "Only paths whose `contribution_pct` falls within this range "
                "appear in the selection table."
            ),
            contrib_range,
        ]
    )
    return (contrib_range,)


@app.cell
def _(display_merged_data, dw_scatter, mo, path_selector, path_table_df, pd):
    selected_pgs = []
    selected_rows = []
    mo.stop(display_merged_data is None)

    _pgs = display_merged_data["merged_sites"]["path_groups"]

    # Union selections from the table and the scatter chart
    _sel_indices = set()
    if path_selector is not None and len(path_selector.value) > 0:
        _sel_indices.update(path_selector.value.index.tolist())
    if dw_scatter is not None and len(dw_scatter.value) > 0:
        _sel_indices.update(dw_scatter.value.index.tolist())

    for _row_idx in sorted(_sel_indices):
        _row = path_table_df.iloc[_row_idx]
        selected_rows.append(_row)
        _gi = _row["group_idx"]
        selected_pgs.append(_pgs[int(_gi)] if pd.notna(_gi) else None)
    return selected_pgs, selected_rows


@app.cell
def _(
    KMAX,
    KMIN,
    RMIN,
    alt,
    atomic_numbers,
    display_g_total,
    display_kref,
    display_merged_data,
    display_total_chi,
    ft_dk,
    ft_kmax,
    ft_kmin,
    ft_kweight,
    ft_rmax,
    ft_window,
    jmol_colors,
    mo,
    np,
    pd,
    perform_FT,
    residual_mode_box,
    selected_pgs,
    selected_rows,
    show_paths_box,
):
    mo.stop(not selected_pgs)

    _ft_kw = {
        "kmin": ft_kmin.value,
        "kmax": ft_kmax.value,
        "kweight": ft_kweight.value,
        "dk": ft_dk.value,
        "k_window": ft_window.value,
        "rmax": ft_rmax.value,
    }
    _kref = display_merged_data["merged_sites"]["kref"]
    _total_sites = display_merged_data["merged_sites"]["total_frames"]
    _k_mask = (_kref >= KMIN) & (_kref <= KMAX)

    def _jmol_hex(sym):
        _z = atomic_numbers.get(sym.split("-")[0], 0)
        _r, _g, _b = jmol_colors[_z]
        return f"#{int(_r * 255):02x}{int(_g * 255):02x}{int(_b * 255):02x}"

    # dash patterns for differentiating same-element paths
    _DASHES = [[1, 0], [8, 3], [4, 3], [2, 3], [6, 2, 2, 2]]

    # ── Build per-path labels, chi arrays, and Jmol colors ───────────────
    # DW groups without a matching FEFF path (selected_pgs entry is None) are skipped.
    # Multiple DW rows can map to the same FEFF path group (group_idx).  For the
    # chi sum we count each unique FEFF group once; the per-panel plot still shows
    # one curve per DW row so users can inspect individual DW entries.
    _path_labels = []
    _path_chis = []
    _path_colors = []
    _path_dashes = []
    _color_count: dict = {}  # jmol_hex → count (for dash cycling)
    _seen_group_indices: set[int] = set()  # FEFF groups already added to sum
    _chi_sum_sel = np.zeros_like(_kref)
    for _pg, _row in zip(selected_pgs, selected_rows, strict=False):
        if _pg is None:
            continue  # DW group with no FEFF match — skip chi
        _angle_str = (
            f"  θ={_row['Angle (°)']:.1f}°"
            if _row["Body"] == "3-body" and _row["Angle (°)"] is not None
            else ""
        )
        _lbl = (
            f"{_row['Type']}  R={_row['Reff_DW (Å)']:.2f}Å  "
            f"σ²={_row['σ² (Å²)']:.5f}{_angle_str}"
        )
        _path_labels.append(_lbl)
        _col = _jmol_hex(_row["Scatterer"])
        _cnt = _color_count.get(_col, 0)
        _path_colors.append(_col)
        _path_dashes.append(_DASHES[_cnt % len(_DASHES)])
        _color_count[_col] = _cnt + 1
        _w = _pg.get("chi_weights", [])
        if _w:
            _chi_p = np.dot(np.array(_w), np.array(_pg["chi_list"])) / _total_sites
        else:
            _chi_p = np.mean(_pg["chi_list"], axis=0) * (
                len(_pg["frame_set"]) / _total_sites
            )
        _path_chis.append(_chi_p)
        # Only add to the sum once per unique FEFF path group.
        _gi_val = _row.get("group_idx")
        if pd.notna(_gi_val):
            _gi_int = int(_gi_val)
            if _gi_int not in _seen_group_indices:
                _seen_group_indices.add(_gi_int)
                _chi_sum_sel += _chi_p

    mo.stop(
        not _path_labels,
        mo.callout(
            mo.md("No selected DW groups have matching FEFF paths."), kind="info"
        ),
    )

    # ── Color / dash scales (Jmol, dash-differentiated per element) ───────
    _n = len(_path_labels)
    _SUM_COL = "#cc3300"
    _HDF5_COL = "#111111"
    _RESID_COL = "#888888"

    _COMBINED = "── Combined"
    _panel_order = [_COMBINED] + _path_labels
    _trace_domain = _path_labels + [
        "Sum of selected",
        "HDF5 total",
        "FEFF residual",
        "Selection residual",
    ]
    _color_scale = alt.Scale(
        domain=_trace_domain,
        range=_path_colors + [_SUM_COL, _HDF5_COL, _RESID_COL, _RESID_COL],
    )
    _dash_scale = alt.Scale(
        domain=_trace_domain, range=_path_dashes + [[6, 3], [1, 0], [3, 3], [2, 2]]
    )
    _sw_scale = alt.Scale(domain=_trace_domain, range=[1.8] * _n + [2.5, 2.0, 1.5, 1.5])

    # ── k-space long-form table ───────────────────────────────────────────
    _rows_k = []
    for _lbl, _chi_p in zip(_path_labels, _path_chis, strict=False):
        for _k_val, _y_val in zip(
            _kref[_k_mask], (_chi_p * _kref**2)[_k_mask], strict=False
        ):
            _rows_k.append(
                {"k": _k_val, "k²χ(k)": _y_val, "panel": _lbl, "trace": _lbl}
            )

    _g_sum = perform_FT(_kref, _chi_sum_sel, **_ft_kw)
    # Sum of selected: native path k-grid
    for _k_val, _y_s in zip(
        _kref[_k_mask], (_chi_sum_sel * _kref**2)[_k_mask], strict=False
    ):
        _rows_k.append(
            {
                "k": _k_val,
                "k²χ(k)": _y_s,
                "panel": _COMBINED,
                "trace": "Sum of selected",
            }
        )
    # HDF5 total: its own native k-grid, already k²-weighted correctly
    _k_mask_h5 = (display_kref >= KMIN) & (display_kref <= KMAX)
    for _k_val, _y_h in zip(
        display_kref[_k_mask_h5],
        (display_total_chi * display_kref**2)[_k_mask_h5],
        strict=False,
    ):
        _rows_k.append(
            {"k": _k_val, "k²χ(k)": _y_h, "panel": _COMBINED, "trace": "HDF5 total"}
        )

    # Residual k-space in Combined panel
    _use_sel_residual = residual_mode_box.value
    _resid_label = "Selection residual" if _use_sel_residual else "FEFF residual"
    if _use_sel_residual:
        # total − sum of selected paths; interpolate path sum onto display_kref grid
        _chi_residual_k = display_total_chi - np.interp(
            display_kref, _kref, _chi_sum_sel
        )
        _resid_kref = display_kref
    else:
        _chi_residual_k = display_merged_data["merged_sites"].get("chi_residual")
        _resid_kref = display_kref
    if _chi_residual_k is not None:
        _k_mask_res_k = (_resid_kref >= KMIN) & (_resid_kref <= KMAX)
        for _k_val, _y_r in zip(
            _resid_kref[_k_mask_res_k],
            (_chi_residual_k * _resid_kref**2)[_k_mask_res_k],
            strict=False,
        ):
            _rows_k.append(
                {"k": _k_val, "k²χ(k)": _y_r, "panel": _COMBINED, "trace": _resid_label}
            )

    # ── R-space long-form table ───────────────────────────────────────────
    _rows_r = []
    for _i, (_lbl, _chi_p, _pg) in enumerate(
        zip(_path_labels, _path_chis, selected_pgs, strict=False)
    ):
        _g_p = perform_FT(_kref, _chi_p, **_ft_kw)
        _r_mask = (_g_p.r >= RMIN) & (_g_p.r <= ft_rmax.value)
        for _r_val, _mag in zip(_g_p.r[_r_mask], _g_p.chir_mag[_r_mask], strict=False):
            _rows_r.append({"R": _r_val, "|χ(R)|": _mag, "panel": _lbl, "trace": _lbl})

    _r_mask_g = (display_g_total.r >= RMIN) & (display_g_total.r <= ft_rmax.value)
    _r_mask_s = (_g_sum.r >= RMIN) & (_g_sum.r <= ft_rmax.value)
    # Each trace uses its own R-grid as x
    for _r_val, _mag_s in zip(
        _g_sum.r[_r_mask_s], _g_sum.chir_mag[_r_mask_s], strict=False
    ):
        _rows_r.append(
            {
                "R": _r_val,
                "|χ(R)|": _mag_s,
                "panel": _COMBINED,
                "trace": "Sum of selected",
            }
        )
    for _r_val, _mag_h in zip(
        display_g_total.r[_r_mask_g], display_g_total.chir_mag[_r_mask_g], strict=False
    ):
        _rows_r.append(
            {"R": _r_val, "|χ(R)|": _mag_h, "panel": _COMBINED, "trace": "HDF5 total"}
        )

    # Residual R-space in Combined panel
    if _chi_residual_k is not None:
        _g_res = perform_FT(_resid_kref, _chi_residual_k, **_ft_kw)
        _r_mask_res = (_g_res.r >= RMIN) & (_g_res.r <= ft_rmax.value)
        for _r_val, _mag_r in zip(
            _g_res.r[_r_mask_res], _g_res.chir_mag[_r_mask_res], strict=False
        ):
            _rows_r.append(
                {
                    "R": _r_val,
                    "|χ(R)|": _mag_r,
                    "panel": _COMBINED,
                    "trace": _resid_label,
                }
            )

    _df_k = pd.DataFrame(_rows_k)
    _df_r = pd.DataFrame(_rows_r)

    if not show_paths_box.value:
        _df_k = _df_k[_df_k["panel"] == _COMBINED]
        _df_r = _df_r[_df_r["panel"] == _COMBINED]
        _panel_order = [_COMBINED]

    # ── Chart builder ─────────────────────────────────────────────────────
    _row_enc = alt.Row(
        "panel:N",
        sort=_panel_order,
        header=alt.Header(
            labelFontSize=10, labelLimit=450, labelOrient="left", title=None
        ),
    )

    # param-bound interval controls scale domain only — no mark greying.
    _zoom = alt.param(bind="scales", select={"type": "interval"})

    def _stacked(df, x_col, y_col, x_title, y_title):
        return (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X(f"{x_col}:Q", title=x_title),
                y=alt.Y(f"{y_col}:Q", title=y_title),
                color=alt.Color(
                    "trace:N",
                    scale=_color_scale,
                    legend=alt.Legend(title="Trace", orient="right"),
                ),
                strokeDash=alt.StrokeDash("trace:N", scale=_dash_scale, legend=None),
                strokeWidth=alt.StrokeWidth("trace:N", scale=_sw_scale, legend=None),
                row=_row_enc,
                tooltip=[
                    alt.Tooltip(f"{x_col}:Q", format=".3f"),
                    alt.Tooltip(f"{y_col}:Q", format=".5f"),
                    alt.Tooltip("trace:N"),
                ],
            )
            .properties(width=500, height=120)
            .add_params(_zoom)
            .resolve_scale(y="independent")
        )

    _chart_k = _stacked(_df_k, "k", "k²χ(k)", "k (Å⁻¹)", "k²χ(k)")
    _chart_r = _stacked(_df_r, "R", "|χ(R)|", "R (Å)", "|χ(R)|")

    stacked_k_chart = mo.ui.altair_chart(_chart_k)
    stacked_r_chart = mo.ui.altair_chart(_chart_r)
    return stacked_k_chart, stacked_r_chart


@app.cell
def _(mo):
    dw_show_all = mo.ui.switch(
        value=True,
        label="DW plot: show all paths (off = filtered paths only)",
    )
    return (dw_show_all,)


@app.cell
def _(
    alt,
    atomic_numbers,
    dw_show_all,
    filtered_path_table_df,
    jmol_colors,
    mo,
    path_table_df,
    pd,
):
    dw_scatter = None
    mo.stop(path_table_df is None)

    _df_dw = (path_table_df if dw_show_all.value else filtered_path_table_df).copy()
    mo.stop(
        len(_df_dw) == 0,
        mo.callout(
            mo.md("No DW groups found — load a trajectory file above."), kind="info"
        ),
    )

    _df_dw["label"] = (
        _df_dw["Type"] + "  R_DW=" + _df_dw["Reff_DW (Å)"].round(3).astype(str) + "Å"
    )
    _df_dw["FEFF match filled"] = _df_dw["FEFF match"].fillna("")
    _df_dw["Angle str"] = _df_dw["Angle (°)"].apply(
        lambda x: f"{x:.1f}°" if x is not None and not pd.isna(x) else ""
    )

    def _jmol_hex(sym):
        _z = atomic_numbers.get(sym.split("-")[0], 0)
        _r, _g, _b = jmol_colors[_z]
        return f"#{int(_r * 255):02x}{int(_g * 255):02x}{int(_b * 255):02x}"

    _scat_els = sorted(_df_dw["Scatterer"].unique())
    _jmol_scale = alt.Scale(
        domain=_scat_els,
        range=[_jmol_hex(s) for s in _scat_els],
    )

    _pts = (
        alt.Chart(_df_dw)
        .mark_point(filled=True)
        .encode(
            x=alt.X("Reff_DW (Å):Q", title="Reff_DW (Å)"),
            y=alt.Y("σ² (Å²):Q", title="σ² (Å²)"),
            color=alt.Color("Scatterer:N", scale=_jmol_scale),
            size=alt.Size(
                "contribution_pct:Q",
                title="Contrib. %",
                scale=alt.Scale(domain=[0, 30], range=[30, 300], clamp=True),
                legend=alt.Legend(orient="bottom", titleLimit=100),
            ),
            shape=alt.Shape(
                "Body:N",
                scale=alt.Scale(
                    domain=["2-body", "3-body"], range=["circle", "triangle-up"]
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Path"),
                alt.Tooltip("Reff_DW (Å):Q", format=".4f"),
                alt.Tooltip("σ² (Å²):Q", format=".6f"),
                alt.Tooltip("Angle str:N", title="Angle"),
                alt.Tooltip("FEFF match filled:N", title="FEFF match"),
                alt.Tooltip("contribution_pct:Q", format=".2f", title="Contrib. %"),
            ],
        )
        .properties(
            height=320, width="container", title="σ² vs Reff_DW — click to select"
        )
        .interactive()
    )

    dw_scatter = mo.ui.altair_chart(_pts)
    return (dw_scatter,)


@app.cell
def _(dw_scatter, dw_show_all, mo, viewer):
    _dw_panel = mo.vstack([dw_show_all, dw_scatter]) if dw_scatter is not None else None
    _display = (
        mo.hstack([viewer, _dw_panel], widths=[30, 70])
        if _dw_panel is not None
        else viewer
    )
    _display
    return


@app.cell
def _(mo, selected_rows):
    show_all_instances_cb = mo.ui.checkbox(
        label="Show all equivalent site instances",
        value=True,
    )
    _n = len(selected_rows) if selected_rows else 0
    mo.vstack([mo.md(f"_{_n} path group(s) selected._"), show_all_instances_cb])
    return (show_all_instances_cb,)


@app.cell
def _(mo, selected_rows, stacked_k_chart, stacked_r_chart):
    mo.stop(not selected_rows or stacked_k_chart is None)
    mo.hstack([stacked_k_chart, stacked_r_chart], widths="equal")
    return


@app.cell
def _(
    ASEAdapter,
    AtomsViewer,
    BaseWidget,
    atomic_numbers,
    avg_atoms,
    dw_results,
    find_mic,
    guiConfig,
    jmol_colors,
    mo,
    msrd_2b,
    msrd_3b,
    selected_rows,
    show_all_instances_cb,
):
    viewer = mo.callout(
        mo.md("Load a trajectory to enable 3-D visualisation."), kind="info"
    )
    mo.stop(avg_atoms is None)

    _pos = dw_results["avg_positions"]
    _cell = avg_atoms.get_cell()
    _pbc = avg_atoms.get_pbc()

    def _jmol_hex3d(sym):
        _z = atomic_numbers.get(sym.split("-")[0], 0)
        _r, _g, _b = jmol_colors[_z]
        return f"#{int(_r * 255):02x}{int(_g * 255):02x}{int(_b * 255):02x}"

    _vf_groups = {}
    _highlight = set()

    if selected_rows:
        # One color per selected path, keyed on scatterer element.
        # If two paths share the same scatterer element, lighten the second by blending with white.
        _color_use: dict = {}  # scatterer_sym → count
        _arrow_colors = []
        for _row3d in selected_rows:
            _scat_sym = _row3d["Scatterer"].split("-")[0]
            _base = _jmol_hex3d(_scat_sym)
            _cnt = _color_use.get(_scat_sym, 0)
            if _cnt == 0:
                _arrow_colors.append(_base)
            else:
                # blend toward white by 30% per repeat
                _bv = int(_base[1:3], 16), int(_base[3:5], 16), int(_base[5:7], 16)
                _f = min(0.30 * _cnt, 0.70)
                _rv = tuple(int(_c + (_f * (255 - _c))) for _c in _bv)
                _arrow_colors.append("#{:02x}{:02x}{:02x}".format(*_rv))
            _color_use[_scat_sym] = _cnt + 1

        for _pi, (_row3d, _color3d) in enumerate(
            zip(selected_rows, _arrow_colors, strict=False)
        ):
            _body = _row3d["dw_body"]
            _dw_i = int(_row3d["dw_idx"])
            _m = msrd_2b[_dw_i] if _body == "2b" else (msrd_3b or [])[_dw_i]
            _all_indices = _m["atom_indices"]
            _active = _all_indices if show_all_instances_cb.value else _all_indices[:1]

            if _body == "2b":
                # 2-body: draw go (absorber→scatterer) and return arrows
                _orig_go, _vec_go, _orig_ret, _vec_ret = [], [], [], []
                for _pair in _active:
                    _abs_i, _scat_i = _pair
                    _v_raw = _pos[_scat_i] - _pos[_abs_i]
                    _v_mic, _ = find_mic([_v_raw], _cell, _pbc)
                    _v = _v_mic[0]
                    _orig_go.append(_pos[_abs_i].tolist())
                    _vec_go.append(_v.tolist())
                    _orig_ret.append((_pos[_abs_i] + _v).tolist())
                    _vec_ret.append((-_v).tolist())
                    _highlight.add(_abs_i)
                    _highlight.add(_scat_i)
                _vf_groups[f"go_{_pi}"] = {
                    "origins": _orig_go,
                    "vectors": _vec_go,
                    "color": _color3d,
                    "radius": 0.12,
                }
                _vf_groups[f"ret_{_pi}"] = {
                    "origins": _orig_ret,
                    "vectors": _vec_ret,
                    "color": _color3d,
                    "radius": 0.07,
                }

            else:
                # 3-body: draw triangular path c→n1→n2→c using chained MIC origins
                _leg1_origs, _leg1_vecs = [], []
                _leg2_origs, _leg2_vecs = [], []
                _leg3_origs, _leg3_vecs = [], []
                for _triple in _active:
                    _c, _n1, _n2 = _triple
                    # leg c→n1
                    _v1_mic, _ = find_mic([_pos[_n1] - _pos[_c]], _cell, _pbc)
                    _v1 = _v1_mic[0]
                    _orig_c = _pos[_c]
                    # leg n1→n2 (chained origin so the triangle closes properly)
                    _v12_mic, _ = find_mic([_pos[_n2] - _pos[_n1]], _cell, _pbc)
                    _v12 = _v12_mic[0]
                    _orig_n1 = _orig_c + _v1
                    # leg n2→c closes the triangle
                    _orig_n2 = _orig_n1 + _v12
                    _v_ret = -(_v1 + _v12)
                    _leg1_origs.append(_orig_c.tolist())
                    _leg1_vecs.append(_v1.tolist())
                    _leg2_origs.append(_orig_n1.tolist())
                    _leg2_vecs.append(_v12.tolist())
                    _leg3_origs.append(_orig_n2.tolist())
                    _leg3_vecs.append(_v_ret.tolist())
                    _highlight.update([_c, _n1, _n2])
                _vf_groups[f"leg1_{_pi}"] = {
                    "origins": _leg1_origs,
                    "vectors": _leg1_vecs,
                    "color": _color3d,
                    "radius": 0.12,
                }
                _vf_groups[f"leg2_{_pi}"] = {
                    "origins": _leg2_origs,
                    "vectors": _leg2_vecs,
                    "color": _color3d,
                    "radius": 0.09,
                }
                _vf_groups[f"leg3_{_pi}"] = {
                    "origins": _leg3_origs,
                    "vectors": _leg3_vecs,
                    "color": _color3d,
                    "radius": 0.07,
                }

    _viewer = AtomsViewer(BaseWidget(guiConfig=guiConfig))
    _viewer.atoms = ASEAdapter.to_weas(avg_atoms)
    _viewer.model_style = 0
    _viewer.atom_scales = [0.333] * len(avg_atoms)
    _viewer.boundary = [[-0.2, 1.2], [-0.2, 1.2], [-0.2, 1.2]]
    _viewer.show_bonded_atoms = True
    if _highlight:
        _viewer.highlight = list(_highlight)
    if _vf_groups:
        _viewer.vf.settings = _vf_groups
        _viewer.vf.show = True
    viewer = _viewer._widget
    return (viewer,)


@app.cell
def _(mo, path_table_df):
    mo.stop(path_table_df is None)
    mo.ui.table(path_table_df, label="All path groups")
    return


if __name__ == "__main__":
    app.run()
