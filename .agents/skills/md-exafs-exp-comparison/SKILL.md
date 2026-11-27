---
name: md-exafs-exp-comparison
description: Run an end-to-end MD-EXAFS workflow comparing an MD trajectory against experimental EXAFS data — pipeline/format-4 archive export, MD-ensemble alignment, static + MD σ² FEFFIT (with unrefined static baseline), χ(R) peak attribution, noise-calibrated site grouping/reweighting, and a self-contained Quarto HTML report. Use when running md-exafs against experiment, fitting MD ensembles or static structures to χ(k)/μ(E), attributing EXAFS peaks, or generating an EXAFS comparison report.
---

# MD-EXAFS Experimental Comparison Workflow

End-to-end workflow from an MD trajectory and experimental spectrum to a verified, self-contained Quarto HTML report. Every script lives alongside this skill (`scripts/run_comparison.py`, `scripts/render_report.py`).

## Steps

### 1. Preflight & sampling budget
1. Confirm the active environment has `md-exafs`, `feff8l`, and `quarto` on `PATH` (when running from a repo worktree, prepend `<repo>/.venv/bin` to `PATH`).
2. Inspect the three inputs in the run directory:
   - **Trajectory:** count frames and absorber atoms (`ase.io.read` or line count). Estimate FEFF task count `n_sites * n_frames`. When `n_sites * n_frames > 2000` on a local machine, confirm subsampling (sites, frame slice, worker count) with the user before launching FEFF.
   - **Experiment:** read the first 10 lines to verify format (`.prj`, `.xdi`, `.dat`, `.chi`, headerless ASCII) and column order. For headerless or reversed columns (e.g. `(chi, k)`), use a file-descriptor dict `{"path": "<file>", "labels": "chi,k"}` (or `--experimental-kwargs '{"labels": "chi,k"}'` on the CLI) rather than mutating the original file.
   - **Config (`exafs_config.yaml`):** check `radius` (`rmax_valid = radius - 0.5` Å), `kmin`, `kmax`, `kweight`, `dk`, `window`, and `n_workers`.
3. **Completion criterion:** environment binaries verified, exact `(sites, frame_index, workers, experiment_kwargs, rranges)` decided, and `rmax_valid` confirmed to cover every requested R-range.

### 2. Produce a format-4 ensemble archive
Choose the branch matching the workspace state:
- **Existing `ensemble_results.h5` (format_version >= 4 with `/top_paths`):** reuse directly.
- **Existing pipeline `results.h5`:** convert without re-running FEFF:
  ```bash
  md-exafs export-ensemble <results.h5> -o <outdir>/ensemble_results.h5
  ```
- **Fresh run from trajectory:** run `md-exafs pipeline` with format-4 export and path retention enabled (set `OMP_NUM_THREADS=1` to avoid oversubscribing cores):
  ```bash
  OMP_NUM_THREADS=1 md-exafs pipeline <traj> <absorber_spec> \
    --config <config.yaml> --output <outdir> \
    --ase-kwargs '{"index": "<slice>"}' \
    --parallel --workers <N> --reuse-potentials --potential-link-mode symlink \
    --hdf5 --keep-paths --ensemble --cleanup --clean-scratch \
    --plot-include all,paths \
    --experimental <exp_file> [--experimental-kwargs '<json>']
  ```
- **Completion criterion:** `ArchiveReader(archive)` reports `format_version == 4`, finite `radius`, expected `n_tasks`, and non-empty `get_top_paths()`.

### 3. Run fitting, attribution, and figure generation
Run the bundled analysis driver for every requested R-range (typically default first-shell `1.0,3.0` plus an extended multi-shell window within `rmax_valid`):
```bash
python <skill-dir>/scripts/run_comparison.py \
  --archive <ensemble_results.h5> \
  --trajectory <traj> \
  --frame-index "<slice>" \
  --config <config.yaml> \
  --absorber <El> \
  --sites "<site_spec_or_indices>" \
  --experiment <exp_file> \
  [--experiment-kwargs '<json>'] \
  --rrange 1.0,3.0 --rrange 1.0,4.8 \
  --cluster-mode auto \
  [--manual-fit <experimentalist>/<name>.fit --manual-fit-params <experimentalist>/fit.yaml] \
  --outdir <analysis_dir> \
  --workdir <fit_static_work>
```
This script runs **MD-ensemble alignment** (`fit_ensemble`), **Static baseline (MD σ²)** + **Static + MD σ² fit** (`fit_static` with `archive_path=`, reusing `workdir` across R-ranges), noise-calibrated `group_sites`, `peak_attribution` (`paths`, `sites`, `site_groups`, scaled to the MD-ensemble $S_0^2$/$\Delta E_0$), **Site-population reweighting** (`fit_site_group_weights`), Vega-Lite specs, and multi-panel comparison PNGs (Experiment, MD ensemble, Static baseline, Static fit, and ensemble-scaled top paths).

**Path grouping (`--cluster-mode`).** The default `running_mean` caps each FEFF path group's width at about twice the tolerance, so one broad thermal shell can be split into several "paths" (for example, the first-shell Cl at 3.1–3.3 Å). `auto` (default) runs `scripts/path_splits.py` on `results.h5` (`path_results`), compares `running_mean` with `valley` clustering for every category holding at least 1% of samples, and switches to `valley` when any category is over-split. `valley` splits only at density valleys between shells. When it is selected, the script re-exports `ensemble_valley.h5` from `results.h5`, so FEFF is not rerun. The MD σ² groups in `fit_static` take the same `cluster_mode`. Read the "FEFF Path Grouping" section of the report to confirm the choice.

**Manual fit (`--manual-fit`).** When the experimentalist's Artemis project is available, pass its `.fit` export (columns `k, chi, fit`) and `fit.yaml`. `scripts/manual_fit.py` scores the manual model with the same residual weighting as the MD fits, in every R-window, and adds it to the comparison figure and table.

- **Completion criterion:** `<analysis_dir>/analysis_summary.json` written with `success: true` for every R-range, every peak's projections + unassigned summing to `1.0 ± 1e-5`, `path_clustering.mode` recorded with the diagnostic, and all comparison PNGs rendered and visually inspected via image read.

### 4. Interpret warnings and compile the Quarto report
1. Inspect every entry in `warnings` across MD-ensemble alignment, Static + MD σ² fit, `group_sites`, and `peak_attribution`. Consult [`references/interpretation.md`](references/interpretation.md) when interpreting high `chi2_reduced`, `group_sites` noise-floor merges, `truncation_fraction`, `unassigned`, or 4-leg path classifications.
2. Optionally write a brief Markdown findings note (`notes.md`) capturing domain-specific physical interpretation (e.g. coordination shells, lattice mismatch, disorder vs static dephasing).
3. Compile the self-contained HTML report:
   ```bash
   python <skill-dir>/scripts/render_report.py <analysis_dir>/analysis_summary.json \
     -o <analysis_dir>/report \
     --title "<System> MD-EXAFS Experimental Comparison" \
     [--notes <analysis_dir>/notes.md]
   ```
   Pass `--no-render` only if `quarto` is absent (writes the `.qmd` cleanly for later compilation).
- **Completion criterion:** `<analysis_dir>/report/<slug>.html` exists (non-empty), Quarto exited 0, and every figure referenced in the `.qmd` exists on disk.

### 5. Deliver the summary
Report the compiled HTML report path, the headline quantitative comparison (MD ensemble vs Static baseline vs Static fit R-factors, $S_0^2$, $\Delta E_0$, $\alpha$, key attributed shells, site-group outcome), and any non-trivial warnings.
