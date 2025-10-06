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
# Install with pip (recommended). Run this from within the project directory
pip install .
```

#### Windows

```powershell
# Install with pip (recommended). Run this from within the project directory
pip install .
```

Note that if you don't have `git` available, you can download the package directly from GitHub (https://github.com/stfc/alc-dls-exafs) and then follow the above steps. Alternatively, you can install the package directly from GitHub like this:

```bash
# Install with pip directly from git archive
pip install https://github.com/stfc/alc-dls-exafs/archive/refs/heads/main.zip

```

### Running the Interactive App

Launch the interactive Marimo application for a web-based EXAFS processing experience:

```bash
marimo run notebooks/exafs_pipeline.py
```

This will open a web interface in your browser where you can:
- Upload structure files (CIF, XYZ, POSCAR, etc.)
- Configure FEFF parameters interactively
- Process single structures or trajectories
- Visualize results with interactive plots
- Export data and figures

If you want to edit the notebook, you can do so instead using:

```bash
marimo edit notebooks/exafs_pipeline.py
```

### Command Line Usage

The CLI provides a streamlined interface for batch processing:

#### Available Commands

```bash
# Show system information and check dependencies
larch-cli info

# Create example configuration file
larch-cli config-example --output my_config.yaml --preset publication

# Generate FEFF input files only
larch-cli generate structure.cif Fe --output feff_inputs/

# Run FEFF calculations in directories
larch-cli run-feff feff_inputs/ --parallel --workers 4

# Analyze existing FEFF outputs and create plots
larch-cli analyze outputs/ --plot-mode frames --show

# Run complete pipeline (generate + run + analyze)
larch-cli pipeline structure.cif Fe --output results/

# Manage cache
larch-cli cache info
larch-cli cache clear
```

#### Detailed Examples

##### Complete Pipeline Processing

```bash
# Basic EXAFS processing (defaults to K-edge, first site)
larch-cli pipeline structure.cif Fe

# Process all Fe sites in structure with custom settings
larch-cli pipeline structure.cif Fe --all-sites --kmax 15 --radius 8.0

# Process trajectory with parallel execution
larch-cli pipeline trajectory.xyz Fe --all-frames --parallel --workers 4

# Publication-quality processing with custom output
larch-cli pipeline structure.cif Fe --preset publication --output results/ --style publication
```

##### Step-by-Step Processing

```bash
# Step 1: Generate FEFF input files
larch-cli generate structure.cif Fe --output feff_inputs/ --radius 6.0 --edge K

# Step 2: Run FEFF calculations
larch-cli run-feff feff_inputs/ --parallel --workers 2

# Step 3: Analyze results and create plots
larch-cli analyze feff_inputs/ --output plots/ --plot-mode sites --show
```

##### Advanced Options

```bash
# Process specific sites by index
larch-cli pipeline structure.cif "0,1,2" --output multi_site/

# Use custom configuration file
larch-cli pipeline structure.cif Fe --config my_config.yaml

# Force recalculation and keep intermediate files
larch-cli pipeline structure.cif Fe --force --no-cleanup

# Different plot modes and styles
larch-cli analyze outputs/ --plot-mode overall --style presentation
larch-cli analyze outputs/ --plot-mode frames --with-phase
larch-cli analyze outputs/ --plot-mode sites --kweight 3
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
larch-cli pipeline structure.cif Fe --config config.yaml
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
- **pymatgen** ≥ 2025.1.24 -  FEFF input generator

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

This project is licensed under the BSD-3 License. See the [LICENSE](LICENSE) file for details.

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
