# EXAFS Processing Pipeline

A comprehensive CLI and interactive toolkit for Extended X-ray Absorption Fine Structure (EXAFS) processing using Larch and FEFF.

## Features

- 🚀 **Interactive Marimo App**: Web-based interface for EXAFS processing
- 🖥️ **Command Line Interface**: Streamlined CLI for batch processing
- 📊 **Multiple Processing Modes**: Single structure, trajectory, and ensemble processing
- 🔧 **FEFF Integration**: Automated FEFF input generation and calculation
- 📈 **Advanced Plotting**: Publication-ready plots with matplotlib and plotly
- ⚡ **Parallel Processing**: Multi-core support for large datasets
- 💾 **Smart Caching**: Intelligent caching to avoid redundant calculations

## Quick Start

### Installation

#### Linux/macOS

```bash
# Clone the repository
git clone https://github.com/stfc/alc-dls-exafs.git
cd alc-dls-exafs

# Install with pip (recommended)
pip install -e .

# Or with all optional dependencies
pip install -e .[full]
```

#### Windows

```powershell
# Clone the repository
git clone https://github.com/stfc/alc-dls-exafs.git
cd alc-dls-exafs

# Install with pip
pip install -e .

# Or with all optional dependencies
pip install -e .[full]
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

### Command Line Usage

The CLI provides a streamlined interface for batch processing:

#### Process a Single Structure

```bash
# Basic EXAFS processing
larch-cli process structure.cif Fe --output results/

# With custom parameters
larch-cli process structure.cif Fe \
  --output results/ \
  --edge K \
  --kmin 3.0 \
  --kmax 12.0 \
  --kweight 3
```

#### Process a Trajectory

```bash
# Process MD trajectory
larch-cli process trajectory.xyz Fe \
  --trajectory \
  --output results/trajectory/ \
  --parallel \
  --n-workers 4
```

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
method: "template"
parallel: true
n_workers: 4

user_tag_settings:
  S02: "1.0"
  CONTROL: "1 1 1 1 1 1"
  NLEG: "6"
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
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n exafs-dev python=3.11
conda activate exafs-dev
```

3. **Install in development mode:**
```bash
pip install -e .[dev]
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

# Run linting
flake8 src/
black src/ tests/

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
- Use [Black](https://black.readthedocs.io/) for code formatting
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

Update documentation when adding features:
- Update relevant files in `docs/`
- Update this README if needed
- Add docstrings to new functions/classes

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
- FEFF calculations powered by the [FEFF Project](https://feff.phys.washington.edu/)
- Structure handling via [ASE](https://wiki.fysik.dtu.dk/ase/) and [pymatgen](https://pymatgen.org/)
