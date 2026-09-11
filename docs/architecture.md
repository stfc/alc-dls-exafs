# MD-EXAFS ecosystem architecture

This document outlines the architecture, data structures, and storage layout of the MD-EXAFS ecosystem.

---

## 1. Multi-tiered topology

The ecosystem is partitioned into three tiers to separate numerical physics, workstation automation, and high-performance computing concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MD-EXAFS Core (`md_exafs`)                   │
│  • Trajectory Analysis & Selection      • Debye–Waller Engine (MSRD/ADP)│
│  • FEFF Card & Input Generation         • Potentials Manager            │
│  • Path Decomposition & Math            • Spectral Averaging & Fourier  │
│  • Experimental Alignment Math          • Unified HDF5 Shards/Ensemble  │
│  • BatchExecutor (Shard-and-Stream)     • Headless Visualisation Engine │
└───────────────────────┬─────────────────────────┬───────────────────────┘
                        │                         │
        ┌───────────────┴──────────────┐          │
        ▼                              ▼          ▼
┌──────────────────────┐ ┌────────────────────────┐ ┌─────────────────────┐
│  Tier 1: Zero-Ops    │ │  Tier 2: Workstation   │ │ Tier 3: Facility    │
│  Interactive Browser │ │  CLI (`md-exafs`)      │ │ HPC & Provenance    │
│  • Marimo (Local)    │ │  • Single-node multi-  │ │ • `aiida-feff`      │
│  • Static WASM app   │ │    threading / Slurm   │ │   (HPC daemon jobs) │
│    (Pyodide, zero-   │ │  • Streaming chunks    │ │ • `aiidalab-exafs`  │
│     install screen)  │ │  • Single-shard K=1    │ │   (Jupyter web GUI) │
└──────────────────────┘ └────────────────────────┘ └─────────────────────┘
```

Tier 1 provides zero-install browser exploration through Marimo and WebAssembly (Pyodide). Tier 2 provides command-line automation for local workstations and single-node cluster jobs via `md-exafs`. Tier 3 wraps the core scientific routines inside `aiida-feff` and `aiidalab-exafs` to manage daemon scheduling, multi-node cluster execution, and graph provenance.

---

## 2. Shard-and-stream batch execution

Thermal molecular dynamics trajectories typically contain hundreds of snapshots and multiple absorbing centres. Evaluating every configuration independently generates thousands of small text files (`xmu.dat`, `chi.dat`, `feffNNNN.dat`), which can quickly exhaust inode limits and overwhelm cluster scratch file systems.

To avoid unbounded disk growth, `md_exafs.execution.BatchExecutor` streams calculations through a bounded loop:

1. **Partitioning**: Tasks are divided into sequential chunks of size `chunk_size` (typically 256).
2. **Parallel execution**: Within each chunk, calculations execute concurrently across available worker processes.
3. **Serialised shard writes**: Upon completing each task, the resulting $\chi(k)$ spectrum and path scattering parameters are appended to a self-contained HDF5 file (`batch_shard.h5`).
4. **Immediate scratch eviction**: Scratch directories for the chunk are removed immediately after being written to the shard. That is, scratch disk usage is bounded to at most `chunk_size` calculations regardless of trajectory length.
5. **Shard aggregation**: For multi-node runs, the $K$ batch shards produced across the cluster are collected and consolidated into `ensemble_results.h5` via `merge_shards()`.

---

## 3. Hybrid path decomposition storage

Storing fully evaluated $\chi(k)$ and Fourier transform $|\tilde{\chi}(R)|$ arrays for every scattering path across every trajectory frame inflates archive sizes by more than an order of magnitude. Conversely, discarding path-level scattering factors leaves downstream tools unable to adjust physical parameters after the calculation finishes.

We resolve this tension through a hybrid storage scheme:

- **Per-frame paths (batch shards and ensemble archives)**:
  Each surviving scattering path is stored using its raw 6-column FEFF scattering factors: (i) central-atom phase shift `real_phc`, (ii) backscattering magnitude `mag_feff`, (iii) backscattering phase shift `pha_feff`, (iv) reduction factor `red_fact`, (v) mean free path `lam`, and (vi) real momentum `rep`. These are saved on FEFF's native coarse $k$-grid alongside geometric attributes ($R_\text{eff}$, leg count $N_\text{legs}$, degeneracy $N$, scatterer label, amplitude ratio `cw_ratio`, and $\sigma^2$). Because the raw parameters are preserved, one can recompute $\chi(k)$ with modified values of $S_0^2$, $\sigma^2$, or $\Delta E_0$ without re-running FEFF.

- **Ensemble-averaged paths (ensemble archive only)**:
  For the top-$N$ composite paths (defaulting to 25), the archive additionally stores pre-evaluated $\chi(k)$ and $|\tilde{\chi}(R)|$ arrays on the standard fine grids. This enables instant rendering in visualisation widgets and CLI plots without reconstruction latency.

---

## 4. Potential precomputation and linking

Self-consistent field (SCF) potential calculation accounts for roughly 80% of compute time in standard FEFF runs. Because atomic coordinates in thermal trajectories oscillate around an equilibrium structure, precomputing potentials once per absorbing site on a representative structure (`CONTROL 1 1 1 0 0 0`) and reusing them for all subsequent path calculations (`CONTROL 0 0 0 1 1 1`) reduces computational cost by roughly four-fifths. In practice, the resulting spectral discrepancy is negligible for well-equilibrated systems.

`md_exafs.potentials.PotentialsManager` defines the canonical manifest of FEFF8L potential files (`phase.pad`, `pot.pad`, and accompanying JSON tables) and distributes them to worker task directories using symbolic links.

---

## 5. Generalised absorber selection

Molecular dynamics simulations often feature variable ordering, reactive events, or permutation of atom indices between frames. Assuming a static topology based on frame 0 risks selecting incorrect atomic coordinates.

`md_exafs.selection` resolves absorbers on a per-frame basis:
1. `resolve_frame_absorbers` evaluates the selection string against the chemical symbol list of each frame, confirming that all chosen sites belong to the same element.
2. `resolve_trajectory_absorbers` tracks absorbing indices independently across every sampled frame while verifying that the absorbing species remains invariant throughout the ensemble.

---

## 6. HDF5 archive schema

### Batch shard (`batch_shard.h5`)
```
/meta/
    format_version  = 1
    archive_type    = "batch_shard"
    created_at      = ISO-8601 string
    threshold       = float
    k_grid          = float64[N]
/tasks/
    frame_0000_site_0000/
        attrs: frame_idx, site_idx, absorber_element
        chi         = float32[N] (compressed)
        paths/ (optional)
            k_grid_params = float64[M]
            feff_data     = float32[P, M, 6] (compressed)
            r_eff         = float64[P]
            nlegs         = int32[P]
            degeneracy    = float64[P]
            cw_ratio      = float64[P]
            sig2          = float64[P]
            scatterer     = str[P]
```

### Ensemble archive (`ensemble_results.h5`)
```
/meta/
    format_version  = 2
    archive_type    = "ensemble"
    fourier_params  = JSON string
    k_grid          = float64[N]
/aggregates/
    overall_average/
        k, chi, chi_std, r, chir_mag, chir_re, chir_im
    frame_averages/
        frame_0000/ ... frame_NNNN/
    site_averages/
        site_0000/ ... site_MMMM/
/top_paths/
    path_0000/ ... path_0024/
        attrs: path_key, scatterer, nlegs, r_eff, degeneracy, cw_ratio, sig2
        chi         = float32[N]
        chir_mag    = float32[K]
```

---

## 7. Relation to EDACA and prior work

The computational approach of ensemble-averaging full multiple-scattering calculations over molecular dynamics snapshots was pioneered by the authors of **EDACA**. By performing explicit configuration-by-configuration FEFF calculations, EDACA demonstrated that classical MD ensembles capture asymmetric thermal disorder and anharmonic bond-length distributions more reliably than harmonic single-structure Debye–Waller approximations.

During the development of `md_exafs`, EDACA served as an invaluable reference oracle. Comparing intermediate scattering matrices, per-path phase shifts, and ensemble averages directly against EDACA on benchmark systems provided the numerical baseline that enabled us to debug and validate our pipeline.

`md_exafs` extends this methodology to accommodate facility-scale workloads through several architectural developments:
1. **Streaming execution and compressed sharding**: Replacing loose ASCII files with chunked execution and compressed HDF5 shards (`batch_shard.h5`), strictly bounding peak scratch disk consumption.
2. **Hybrid path decomposition**: Storing native 6-column FEFF scattering factors alongside pre-evaluated dominant paths, allowing researchers to re-evaluate spectra under varying $S_0^2$, $\sigma^2$, or $\Delta E_0$ without re-running FEFF.
3. **Potential precomputation and linking**: Computing SCF potentials once on an equilibrium configuration and distributing them via symbolic links, reducing compute time by approximately four-fifths.
4. **Trajectory-based statistical disorder**: Calculating MSRD, cumulants ($C_3, C_4$), autocorrelation sample corrections, and ADP CIF models with explicit support for non-orthogonal periodic cells.
5. **Multi-tiered ecosystem**: Unifying standalone Python scripts, workstation CLI commands (`md-exafs`), AiiDA provenance-tracked HPC workflows (`aiida-feff`), and interactive browser visualisations (`aiidalab-exafs`).
