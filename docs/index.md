# MD-EXAFS Documentation

`md-exafs` is the foundational Python engine and command-line interface for ensemble-averaged Extended X-ray Absorption Fine Structure (EXAFS) spectroscopy from molecular dynamics trajectories.

Part of the **MD-EXAFS Ecosystem**:
- **MD-EXAFS Core & CLI (`md-exafs`)**: Trajectory processing, Debye–Waller extraction, FEFF card/input generation, and streaming batch execution.
- **[AiiDA-FEFF](https://github.com/stfc/aiida-feff)**: HPC scheduling, batch orchestration, and graph provenance tracking.
- **[AiiDAlab-EXAFS](https://github.com/stfc/aiidalab-exafs)**: Interactive Jupyter web dashboard.

---

## Interactive browser applications

The ecosystem includes zero-install WebAssembly applications running entirely within your browser via Pyodide. No server or local installation is required.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Debye–Waller disorder screening
:link: interactive/debye_waller
:link-type: doc

<span class="badge-wasm">WASM App</span>

Extract path variances ($\sigma^2$), third/fourth cumulants, and atomic displacement parameters directly from trajectory files in your browser.
:::

:::{grid-item-card} Nanoparticle cluster builder
:link: interactive/nanoparticle_builder
:link-type: doc

<span class="badge-wasm">WASM App</span>

Generate mono- and bimetallic core-shell nanoparticles (e.g. Au@Ag) with custom shapes, radii, and lattice parameters.
:::

::::

---

## Quickstart

::::{tab-set}

:::{tab-item} Command-line interface
Run the full simulation pipeline from the command line:

```bash
# Process all Cu sites across all trajectory frames into HDF5
md-exafs pipeline trajectory.xyz Cu --all-sites --all-frames --hdf5 --reuse-potentials --parallel

# Plot the ensemble average and per-path contributions
md-exafs analyze pipeline_output/results.h5 --plot-include average,paths --show
```
See the [CLI Reference](cli.md) for complete options.
:::

:::{tab-item} Python library
Import core domain logic directly without CLI or database overhead:

```python
from ase.build import bulk
from md_exafs.feff_input import FeffConfig, build_feff_inp
from md_exafs.paths import path_chi

# Generate inputs
atoms = bulk("Cu", "fcc", a=3.61)
feff_inp = build_feff_inp(atoms, config=FeffConfig.from_preset("quick"))

# Evaluate fine-structure oscillations
chi = path_chi(k_native, feff_data, r_eff=2.55, degeneracy=12.0, k_out=k_grid)
```
See the [Python API Reference](python_api.md).
:::

:::{tab-item} High-performance computing
For large multi-node campaigns with full data provenance, submit through `AiiDA-FEFF`:

```python
from aiida.plugins import WorkflowFactory
EnsembleWorkChain = WorkflowFactory("feff.ensemble")
# Dispatches parallel batch shards across HPC cluster nodes
```
See [Ecosystem Integration](ecosystem.md).
:::

::::

---

## Table of contents

```{toctree}
:maxdepth: 2
:caption: Interactive applications

interactive/index
interactive/debye_waller
interactive/nanoparticle_builder
```

```{toctree}
:maxdepth: 2
:caption: Core documentation

architecture
cli
python_api
ecosystem
```
