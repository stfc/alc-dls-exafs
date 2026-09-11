# MD-EXAFS ecosystem integration

This document outlines how `md-exafs` connects with **AiiDA-FEFF** and **AiiDAlab-EXAFS**.

---

## 1. Ecosystem topology

```
                  ┌───────────────────────────────┐
                  │       MD-EXAFS Core           │
                  │        (`md_exafs`)           │
                  │  (Pure Python, zero-AiiDA)    │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│           AiiDA-FEFF            │       │         AiiDAlab-EXAFS          │
│        (`aiida-feff`)           │◄──────┤        (`aiidalab-feff`)        │
│ • Slurm/PBS HPC Daemon Jobs     │       │ • Jupyter / AiiDAlab Web App    │
│ • ExafsArchiveData Node         │       │ • In-Memory DW Pre-screening    │
│ • Remote Batch Sharding         │       │ • Interactive Paths Explorer    │
│ • Graph Provenance Tracking     │       │ • Workflow Submission Wizard    │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

The division of responsibilities is strict:
1. `md_exafs` contains all physics, statistical estimators, card generation rules, and HDF5 serialisation routines. It has no knowledge of AiiDA or web frameworks.
2. `aiida-feff` handles HPC scheduler submission, remote data staging, daemon monitoring, and relational graph provenance. This is the actual aiida plugin.
3. `aiidalab-exafs` provides interactive user interface widgets within Jupyter and AiiDAlab.

---

## 2. Shared data architecture: ExafsArchiveData

`ExafsArchiveData` (implemented in `aiida_feff.data.archive`) wraps either a batch shard (`batch_shard.h5`) or an ensemble archive (`ensemble_results.h5`). It delegates array and path queries directly to `md_exafs.hdf5.ArchiveReader`:

```python
# Accessing properties on an ExafsArchiveData node
archive_node: ExafsArchiveData = workchain.outputs.archive

# Primary spectra (delegates to md_exafs.hdf5.ArchiveReader)
k = archive_node.k
chi = archive_node.chi
r = archive_node.r
chir_mag = archive_node.chir_mag

# Querying path contributions
paths = archive_node.iter_paths()

# Legacy converter for external tools expecting XasData
xas_node = archive_node.to_xas_data()
```

---

## 3. Remote cluster execution

When `EnsembleExafsWorkChain` executes calculations on a remote cluster:

1. **Remote execution**: The batch runner script `_run_batch.py` runs on the cluster node, delegating directly to `md_exafs.execution.BatchExecutor`.
2. **Local sharding**: FEFF processes run concurrently on the allocated cores. Outputs are streamed into a single `batch_shard.h5` archive on node scratch.
3. **Network transfer reduction**: Rather than retrieving thousands of individual `xmu.dat`, `chi.dat`, and `feffNNNN.dat` files across SFTP, AiiDA retrieves only `batch_shard.h5`, reducing the file transfer count by more than 95%.
4. **Ensemble aggregation**: The workflow invokes the calcfunction `merge_exafs_shards(*shards)`. This calls `md_exafs.execution.merge_shards`, which computes ensemble averages, calculates Fourier transforms, and emits a single consolidated `ExafsArchiveData` node.

---

## 4. Parameter delegation

`FeffParameters` in `AiiDA-FEFF` is implemented as a thin `Dict` adapter. Rather than maintaining separate validation and card serialisation rules, it delegates directly to `md_exafs.FeffConfig`:

```python
from aiida_feff.data.parameters import FeffParameters

params = FeffParameters(dict={
    "edge": "K",
    "radius": 5.5,
    "s02": 0.9,
    "scf": "4.0 0 30 0.2 1",
})

# Under the hood, delegates validation and tag normalisation to md_exafs
feff_config = params.to_feff_config()
user_tags = params.to_pymatgen_user_tags()
```

---

## 5. In-memory trajectory screening in AiiDAlab

Debye–Waller and MSRD analysis involves only minimum-image geometric distances and coordinate moments; it requires no electronic structure calculations and negligible compute time. Storing intermediate MSRD results as provenance-tracked AiiDA nodes adds database overhead without scientific benefit.

In `aiidalab-exafs`, trajectory screening is provided by `aiidalab_feff.dw_widget.DebyeWallerScreeningWidget`. When a trajectory is uploaded, the widget invokes `md_exafs.debye_waller` in-memory. It computes $\sigma^2(R_\text{eff})$ and isotropic thermal $B$-factors which can be used to help constrain FEFF fits or just to inform the user about the expected thermal disorder in the system.

---

## 6. Shared visualisation engine

Both the Marimo notebook (`notebooks/paths_explorer.py`) and the AiiDAlab widget (`aiidalab_feff.widgets.paths_explorer`) consume `md_exafs.viz`:
- `group_path_results()` ensures consistent path clustering and distance binning across both interfaces.
- `build_chi_chart()` and `build_chir_chart()` construct identical Altair spectral charts.
- `calculate_path_vectors()` supplies 3D polyline vectors to WEAS-widget.
