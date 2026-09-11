# MD-EXAFS (`md-exafs`)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

`md-exafs` provides the core scientific engine and command-line interface for calculating ensemble-averaged Extended X-ray Absorption Fine Structure (EXAFS) spectra from molecular dynamics trajectories.

The package sits at the base of the MD-EXAFS software ecosystem:
- **MD-EXAFS Core and CLI (`md-exafs`)**: Trajectory analysis, Debye–Waller extraction, FEFF card and input generation, path reconstruction, and streaming batch execution.
- **[AiiDA-FEFF](https://github.com/stfc/aiida-feff)**: AiiDA plugin for provenance tracking, multi-node HPC daemon scheduling and batch orchestration.
- **[AiiDAlab-EXAFS](https://github.com/stfc/aiidalab-exafs)**: Interactive browser interface inside Jupyter and AiiDAlab.

---

## Capabilities

The package implements the physical and numerical routines required to bridge thermal configurations and experimental absorption spectra:

- **Exact EXAFS reconstruction**: Evaluates the EXAFS equation for individual scattering paths using the complex electron momentum $p = \mathrm{rep} + i/\lambda$, matching Larch and FEFF conventions.
- **Shard-and-stream batch execution**: Runs FEFF in bounded chunks across available CPU cores, streams calculated spectra and per-path parameters directly into HDF5 shards (`batch_shard.h5`), and prunes scratch directories immediately to keep disk usage within fixed bounds.
- **Hybrid path storage**: Stores raw 6-column FEFF scattering parameters on native grids for per-frame paths to permit post-hoc adjustment of $S_0^2$, $\sigma^2$, or $\Delta E_0$, while pre-evaluating fine-grid $\chi(k)$ and $|\tilde{\chi}(R)|$ for the dominant composite ensemble paths.
- **Debye–Waller and ADP extraction**: Extracts path variances ($\sigma^2$), third and fourth cumulants ($C_3, C_4$), autocorrelation sample corrections ($N_\text{eff}$), and atomic displacement parameter (ADP) CIF models directly from trajectories, with support for non-orthogonal periodic cells.
- **Per-frame absorber resolution**: Supports element labels, explicit index lists, and element-relative indexing (`"Cu:0,2"`), validating chemical species consistency across static and permuted topologies.
- **Experimental alignment**: Imports beamline ASCII columns and Athena projects (`.prj`), applying $\Delta E_0$ threshold shifts and $S_0^2$ scaling on common wavenumber grids.
- **Headless visualisation**: Generates standard Altair spectral charts and 3D scattering vectors for WEAS-widget without framework dependencies.

---

## Installation

### From source

```bash
git clone https://github.com/stfc/alc-dls-exafs.git
cd alc-dls-exafs

# Core engine (zero CLI or plotting dependencies)
pip install -e .

# With command-line interface (typer, rich, matplotlib)
pip install -e ".[cli]"

# Full installation including interactive notebooks (marimo, weas-widget, altair)
pip install -e ".[cli,notebooks]"
```

Using `uv`:
```bash
uv pip install -e ".[cli,notebooks]"
```

---

## Command-line interface

The CLI is available as `md-exafs`. The legacy alias `larch-cli` remains available for transition.

### 1. Full pipeline execution
The `pipeline` command coordinates input generation, parallel FEFF execution, HDF5 archiving, and ensemble spectral averaging in a single run.

```bash
# Single structure (K-edge, first absorber site)
md-exafs pipeline structure.cif Cu

# Trajectory run: process all Cu sites across all frames into HDF5
md-exafs pipeline trajectory.xyz Cu --all-sites --all-frames --hdf5 --keep-paths

# Precompute potentials on the representative structure to accelerate trajectory runs
md-exafs pipeline trajectory.xyz Cu --all-sites --all-frames --hdf5 --reuse-potentials --parallel --workers 8

# Sub-sample trajectory frames (e.g. every 5th frame)
md-exafs pipeline trajectory.xyz Cu --ase-kwargs '{"index": "::5"}'
```

### 2. Spectral and path analysis
The `analyze` command evaluates and plots simulated spectra and path contributions from an existing HDF5 archive (`results.h5` or `ensemble_results.h5`):

```bash
# Plot ensemble average and path decomposition
md-exafs analyze results.h5 --plot-include average,paths --show

# Prune negligible paths (e.g. keep paths with at least 5% of peak amplitude)
md-exafs analyze results.h5 --plot-include average,paths --min-cw-ratio 5.0 --max-paths 20

# Compare against experimental data with an energy shift and amplitude factor
md-exafs analyze results.h5 --experimental measurement.dat --e0-shift 2.5 --s02 0.85
```

### 3. Debye–Waller extraction
Extract path variances ($\sigma^2$), cumulants ($C_3, C_4$), and ADP tensors directly from molecular dynamics configurations:

```bash
# Calculate MSRD up to 5.0 Å from Cu absorbing sites
md-exafs debye-waller trajectory.xyz --prefix cu_dw --site-spec Cu --cutoff 5.0

# Export an average CIF structure with anisotropic thermal ellipsoids
md-exafs debye-waller trajectory.xyz --prefix cu_dw --adp-cif
```

### 4. Modular workflow steps

```bash
# Step 1: Generate FEFF inputs only
md-exafs generate structure.cif Cu --output feff_inputs/ --radius 6.0 --edge K

# Step 2: Run FEFF across generated input directories
md-exafs run-feff feff_inputs/ --parallel --workers 4

# Step 3: Analyse results
md-exafs analyze feff_inputs/ --output plots/ --plot-include sites --show
```

---

## Python API quickstart

The `md_exafs` package can be imported directly in Python scripts and computational notebooks.

### Building FEFF inputs
```python
from ase.build import bulk
from md_exafs.feff_input import FeffConfig, build_feff_inp

atoms = bulk("Cu", "fcc", a=3.61)
config = FeffConfig.from_preset("quick")
config.radius = 5.5

feff_inp = build_feff_inp(atoms, config=config, absorber_idx=0)
```

### Evaluating individual scattering paths
```python
import numpy as np
from md_exafs.paths import path_chi

# Calculate chi(k) from raw 6-column FEFF scattering factors
k_out = np.linspace(2.0, 16.0, 141)
chi = path_chi(
    k_native=k_coarse,
    feff_data=feff_data_6cols,
    r_eff=2.55,
    degeneracy=12.0,
    k_out=k_out,
    sigma2=0.005,
    s02=0.9,
    e0_shift=1.5,
)
```

### Batch execution and shard aggregation
```python
from pathlib import Path
from md_exafs.execution import BatchExecutor, FeffTask, merge_shards
from md_exafs.hdf5 import ArchiveReader

tasks = [
    FeffTask(frame_idx=0, site_idx=0, input_dir=Path("run_0000"), absorber_element="Cu"),
    FeffTask(frame_idx=0, site_idx=1, input_dir=Path("run_0001"), absorber_element="Cu"),
]

# Run chunked execution into a self-contained shard
executor = BatchExecutor(tasks, output_h5="batch_shard_0.h5", chunk_size=128, n_workers=4)
executor.run()

# Merge multiple shards into an ensemble archive
ensemble_file = merge_shards(
    shard_paths=["batch_shard_0.h5", "batch_shard_1.h5"],
    ensemble_path="ensemble_results.h5",
)

# Inspect results through the unified archive reader
reader = ArchiveReader(ensemble_file)
print(f"Grand average chi points: {len(reader.chi)}")
```

### In-memory Debye–Waller screening
```python
from ase.io import read
from md_exafs.debye_waller import calculate_grouped_msrd

trajectory = read("trajectory.xyz", index=":")
msrd_results = calculate_grouped_msrd(trajectory, site_spec="Cu", max_reff=5.0)

for path in msrd_results["paths"][:5]:
    print(f"{path['label']}: R_eff = {path['r_eff']:.3f} Å, σ² = {path['sigma2']:.5f} Å²")
```

---

## Documentation

- [System architecture](docs/architecture.md): Description of the shard-and-stream and hybrid storage models.
- [CLI reference](docs/cli.md): Command documentation and parameter descriptions.
- [Python API reference](docs/python_api.md): Function and class signatures for library modules.
- [Ecosystem integration](docs/ecosystem.md): Details on coupling with `AiiDA-FEFF` and `AiiDAlab-EXAFS`.

---

## Citation

If you use `md-exafs` in your research, please cite:

```bibtex
@software{md_exafs,
  author       = {Kane Shenton and Joshua Elliott and Alin M. Elena},
  title        = {MD-EXAFS: Ensemble-Averaged EXAFS Spectroscopy from Molecular Dynamics Trajectories},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/stfc/alc-dls-exafs}}
}
```

See [`CITATION.cff`](CITATION.cff) for full citation metadata.

---

## License

This project is licensed under the BSD-3-Clause License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built on top of the excellent [Larch](https://xraypy.github.io/xraylarch/) project
- FEFF calculations powered by the [FEFF Project](https://feff.phys.washington.edu/). Specifically, the Open Source version of FEFF8 (FEFF8L) is used by default.
- Structure handling via [ASE](https://wiki.fysik.dtu.dk/ase/) and [pymatgen](https://pymatgen.org/)
- Trajectory-based ensemble analysis inspired by EDACA, which pioneered the software methodology and served as an invaluable reference during pipeline validation.
