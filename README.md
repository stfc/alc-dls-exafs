# EXAFS Processing Pipeline

A comprehensive CLI and interactive toolkit for Extended X-ray Absorption Fine Structure (EXAFS) processing using Larch and FEFF.

## Features

- 🌐 **Interactive Marimo App**: Web-based interface for EXAFS processing
- 🖥️ **Command Line Interface**: Streamlined CLI for batch processing
- ⇶ **Multiple Processing Modes**: Single structure, trajectory/ensemble processing
- 🔧 **FEFF Integration**: Automated FEFF input generation and calculation
- 📈 **Plotting**: Publication-ready plots with matplotlib and plotly
- 📊 **Parallel Processing**: Multi-core support for large datasets
- 💾 **Smart Caching**: Intelligent caching to avoid redundant calculations

## Quick Start

### Getting the code

You can get the code by cloning the repository or downloading it as a ZIP file.

#### Clone the repository (recommended)

```bash
git clone https://github.com/stfc/alc-dls-exafs.git
cd alc-dls-exafs
```

#### Download ZIP file

```bash
curl -LO https://github.com/stfc/alc-dls-exafs/archive/refs/heads/main.zip
unzip main.zip
cd alc-dls-exafs-main
```

You can also get it by going to GitHub: https://github.com/stfc/alc-dls-exafs, clicking on the green "Code" button, and then selecting "Download ZIP".

### Installation

#### Linux/macOS

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/stfc/alc-dls-exafs.git
cd alc-dls-exafs

# Install with pip (recommended). Run this from within the project directory
pip install -e .

# Or with all optional dependencies
pip install -e ".[full]"
```

#### Windows

```powershell
# Clone the repository
git clone https://github.com/stfc/alc-dls-exafs.git
cd alc-dls-exafs

# Install with pip (recommended). Run this from within the project directory
pip install -e .

# Or with all optional dependencies
pip install -e ".[full]"
```

Note that if you don't have `git` available, you can download the package directly from GitHub (https://github.com/stfc/alc-dls-exafs) and then follow the above steps. Alternatively, you can install the package directly from GitHub like this:

```bash
# Install with pip directly from git archive
pip install https://github.com/stfc/alc-dls-exafs/archive/refs/heads/main.zip

```

#### Alternative: Using uv (faster)

```bash
# Install uv if not already installed
pip install uv

# Install the project
uv pip install -e .
```

### Running the Interactive App

Launch the interactive Marimo application for a web-based EXAFS processing experience:

```bash
marimo run exafs_pipeline.py
```

This will open a web interface in your browser where you can:
- Upload structure files (CIF, XYZ, POSCAR, etc.)
- Configure FEFF parameters interactively
- Process single structures or trajectories
- Visualize results with interactive plots
- Export data and figures

If you want to edit the notebook, you can do so instead using:

```bash
marimo edit exafs_pipeline.py
```

### Command Line Usage

The CLI provides a streamlined interface for batch processing:

#### Process a Single Structure

```bash
# Basic EXAFS processing (defaults to K-edge)
larch-cli process structure.cif Fe

```

This will generate the FEFF input file for your structure, assuming e.g. K-edge and taking the first Fe site in the structure. It will then run FEFF8L and process the output (generate a plot).

#### Process a Trajectory

```bash
# Process MD trajectory
larch-cli process trajectory.xyz Fe \
  --trajectory \
  --output results/trajectory/ \
  --parallel \
  --workers 4
```

This will generate FEFF input files for each structure in a trajectory, and run FEFF8L on each one. It will run up to 4 workers in parallel (each one dealing with a separate structure). It will then average the XAFS for all structures and plot the results.

#### Generate FEFF Input Only

```bash
# Generate FEFF input files without running calculation
larch-cli generate structure.cif Fe --output feff_input/
```

#### Run FEFF Calculation

```bash
# Run FEFF in existing directory
larch-cli run-feff feff_input/
```

### Configuration Files

Use YAML configuration files for complex setups:

```yaml
# config.yaml
spectrum_type: "EXAFS"
edge: "K"
radius: 8.0
kmin: 2.0
kmax: 14.0
kweight: 2
method: "auto"
parallel: true
n_workers: 4

user_tag_settings:
  S02: "1.0"
  CONTROL: "1 1 1 1 1 1"
  NLEG: "6"
```

You can create a `config.yaml` file in your working directory to customize the pipeline settings. You can then use this by setting the `--config` option when running the CLI commands. For example:

```bash
larch-cli process structure.cif Fe --config config.yaml
```

## Project Structure

```
├── src/larch_cli_wrapper/      # Core package
│   ├── cli.py                  # Command line interface
│   ├── wrapper.py              # Main processing wrapper
│   ├── feff_utils.py           # FEFF utilities
│   └── cache_utils.py          # Caching system
├── exafs_pipeline.py           # Interactive Marimo app
└── tests/                      # Test suite
```

## Dependencies

### Core Requirements
- **Python** ≥ 3.10
- **xraylarch** ≥ 0.9.47 - EXAFS analysis library
- **typer** ≥ 0.12.0 - CLI framework
- **rich** - Terminal formatting
- **matplotlib** ≥ 3.5 - Plotting
- **marimo** ≥ 0.14.16 - Interactive notebooks
- **ase** ≥ 3.22.1 - Atomic structure handling

### Optional Dependencies
- **pymatgen** ≥ 2022.7 - Alternative FEFF input generator

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/your-username/alc-dls-exafs.git
cd alc-dls-exafs
```

2. **Create a virtual environment:**
```bash
# Using conda (recommended)
conda create -n exafs-dev python=3.12 --channel conda-forge
conda activate exafs-dev

# Alternative: Using micromamba
micromamba create -n exafs-dev python=3.12 --channel conda-forge
micromamba activate exafs-dev
```


3. **Install in development mode:**
```bash
uv pip install -e ".[dev]"

# Or without uv
pip install -e ".[dev]"
```

### Development Workflow

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make changes and add tests:**
```bash
# Run tests
pytest tests/

# Run linting and formatting
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/
```

3. **Commit and push:**
```bash
git add .
git commit -m "Add your feature description"
git push origin feature/your-feature-name
```

4. **Create a Pull Request** on GitHub

### Code Style

- Follow [PEP 8](https://pep8.org/) for Python code style
- Use [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting
- Add type hints where appropriate
- Write docstrings for all public functions and classes
- Include tests for new functionality

### Testing

Run the test suite:
```bash
# All tests
pytest

# Specific test file
pytest tests/test_wrapper.py

# With coverage
pytest --cov=larch_cli_wrapper
```

### Documentation

TODO

<!-- Update documentation when adding features:
- Update relevant files in `docs/`
- Update this README if needed
- Add docstrings to new functions/classes -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: TODO
- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/stfc/alc-dls-exafs/issues)
- **Discussions**: Join discussions on the project's GitHub page

## Citation

TODO


## Acknowledgments

- Built on top of the excellent [Larch](https://xraypy.github.io/xraylarch/) project
- FEFF calculations powered by the [FEFF Project](https://feff.phys.washington.edu/). Specifically, the Open Source version of FEFF8 (FEFF8L) is used by default.
- Structure handling via [ASE](https://wiki.fysik.dtu.dk/ase/) and [pymatgen](https://pymatgen.org/)
