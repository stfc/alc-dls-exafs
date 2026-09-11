# Command-line interface reference (`md-exafs`)

The `md-exafs` command-line executable automates FEFF calculations, Debye–Waller extraction, and spectrum analysis on workstations and cluster nodes.

The tool is invoked as `md-exafs`. The legacy command `larch-cli` is also retained.

---

## Command overview

```
Usage: md-exafs [OPTIONS] COMMAND [ARGS]...

Commands:
  pipeline        Run the complete EXAFS processing pipeline.
  analyze         Analyze EXAFS data, plot spectra, and inspect paths.
  debye-waller    Compute MSRD, higher cumulants, and ADPs from trajectory.
  generate        Generate FEFF input files only.
  run-feff        Execute FEFF calculations across input directories.
  config-example  Create an example YAML configuration file.
  cache           Inspect or clear calculation cache.
  info            Show environment and dependency information.
```

---

## 1. pipeline

Coordinates input preparation, parallel FEFF calculations, HDF5 archiving, and spectral analysis in a single step.

```bash
md-exafs pipeline [OPTIONS] STRUCTURE ABSORBER
```

### Arguments
- `STRUCTURE` (Path, required): Path to atomic configuration or trajectory file (CIF, XYZ, POSCAR, extxyz).
- `ABSORBER` (str, required): Absorber specification:
  - Element symbol (e.g. `"Cu"`): processes the first site by default, or all sites if `--all-sites` is provided.
  - Comma-separated index list (e.g. `"0,1,2"`): processes explicit atomic sites. All selected atoms must belong to the same species.
  - Element-relative indices (e.g. `"Cu:0,2"`): selects the first and third copper sites in the cell.

### Key options
| Option | Description |
| :--- | :--- |
| `-o, --output PATH` | Base output directory (default: `pipeline_output`). |
| `-c, --config PATH` | Custom YAML configuration file. |
| `-p, --preset PRESET` | Base configuration preset: `quick`, `nscf`, or `publication`. |
| `--all-sites / --no-all-sites` | Process all sites of the selected element. |
| `--all-frames / --no-all-frames` | Evaluate every frame in the input trajectory. |
| `--ase-kwargs JSON` | JSON string passed to `ase.io.read()` (e.g. `'{"index": "::10"}'`). |
| `--hdf5 / --no-hdf5` | Store results in a compressed HDF5 archive (default: enabled). |
| `--hdf5-file PATH` | Custom destination for the HDF5 archive. |
| `--keep-paths / --no-keep-paths` | Retain individual `feffNNNN.dat` files for path analysis. |
| `--reuse-potentials` | Compute SCF potentials once on a representative structure and reuse across all snapshots. |
| `--potentials-structure PATH` | User-supplied structure for initial potential precomputation. |
| `--parallel / --sequential` | Enable multi-process calculation across cores. |
| `--workers INT` | Number of worker processes to allocate. |
| `--stream-chunk-size INT` | Chunk size for streaming batch execution (default: 256). |
| `-e, --experimental PATH` | Experimental spectrum file (`.prj`, `.dat`, `.chi`) to overlay. |
| `--e0-shift FLOAT` | Energy shift $\Delta E_0$ (eV) for experimental comparison. |
| `--s02 FLOAT` | Amplitude reduction factor $S_0^2$ for experimental comparison. |
| `--show / --no-show` | Display matplotlib plots interactively on completion. |

---

## 2. analyze

Extracts, averages, and plots spectra and scattering paths from an existing HDF5 archive (`results.h5`, `ensemble_results.h5`) or directory tree.

```bash
md-exafs analyze [OPTIONS] [INPUTS]...
```

### Arguments
- `INPUTS` (Path, optional): One or more HDF5 results archives or directories containing `chi.dat`.

### Key options
| Option | Description |
| :--- | :--- |
| `-o, --output PATH` | Directory for exported figures (default: `analysis`). |
| `--plot-include TEXT` | Components to plot: `average`, `frames`, `sites`, `paths`, or `all`. |
| `--max-paths INT` | Maximum number of individual scattering paths shown in the paths panel. |
| `--min-cw-ratio FLOAT` | Relative amplitude threshold percentage (e.g. `5.0` retains paths with at least 5% peak amplitude). |
| `-e, --experimental PATH` | Overlay experimental spectrum file (`.prj`, `.dat`, `.chi`). |
| `--e0-shift FLOAT` | Energy shift $\Delta E_0$ (eV) applied to experimental data. |
| `--s02 FLOAT` | Amplitude factor $S_0^2$ applied to experimental data. |
| `--kmin FLOAT` | Minimum $k$ for Fourier transform in Å⁻¹ (default: 2.0). |
| `--kmax FLOAT` | Maximum $k$ for Fourier transform in Å⁻¹ (default: 12.0). |
| `--kweight INT` | Wavenumber weighting exponent $k^w$ (1, 2, or 3). |
| `--dk FLOAT` | Fourier transform window taper width in Å⁻¹ (default: 1.0). |
| `--window TEXT` | Window function: `kaiser`, `hanning`, `parzen`, `welch`, `gaussian`, or `sine`. |
| `--show / --no-show` | Display plots interactively. |

---

## 3. debye-waller

Extracts Mean Squared Relative Displacements (MSRD, $\sigma^2$), higher cumulants ($C_3, C_4$), autocorrelation sample corrections, and atomic displacement parameters from molecular dynamics trajectories.

```bash
md-exafs debye-waller [OPTIONS] TRAJECTORY
```

### Arguments
- `TRAJECTORY` (Path, required): Path to MD trajectory file.

### Key options
| Option | Description |
| :--- | :--- |
| `--prefix TEXT` | Prefix for generated CSV and summary output files. |
| `--site-spec TEXT` | Absorbing site specification (e.g. `"Cu"` or `"0,1,2"`). |
| `--cutoff FLOAT` | Radial cutoff distance $R_\text{max}$ in Å (default: 6.0). |
| `--adp-cif` | Export an average crystallographic CIF file containing anisotropic ADP tensors. |
| `--n-blocks INT` | Block count for block-variance statistical error estimation. |

---

## 4. generate

Constructs `feff.inp` input files without invoking the FEFF executable.

```bash
md-exafs generate [OPTIONS] STRUCTURE ABSORBER
```

### Key options
- `-o, --output PATH`: Target directory for generated input files.
- `-p, --preset PRESET`: Preset configuration (`quick`, `nscf`, or `publication`).
- `--radius FLOAT`: Cluster and scattering path cutoff radius in Å.
- `--edge TEXT`: Target absorption edge (`K`, `L1`, `L2`, `L3`, etc.).
- `--exclude-hydrogen / --keep-hydrogen`: Strip hydrogen atoms prior to input generation.

---

## 5. run-feff

Executes FEFF across directories containing previously prepared `feff.inp` files.

```bash
md-exafs run-feff [OPTIONS] [DIRECTORIES]...
```

### Key options
- `--parallel / --sequential`: Multi-process execution.
- `-w, --workers INT`: Worker process count.
- `--feff-bin TEXT`: FEFF binary name or absolute path (default: `feff`).
- `--cleanup / --no-cleanup`: Remove intermediate scratch files upon completion.

---

## 6. config-example

Generates documented YAML configuration templates for custom workflows.

```bash
md-exafs config-example [OPTIONS]
```

### Key options
- `-o, --output PATH`: File path for the generated template (default: `exafs_config.yaml`).
- `-p, --preset PRESET`: Base preset to scaffold (`quick`, `publication`, or `nscf`).
