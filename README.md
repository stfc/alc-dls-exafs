# ALS/DLS-Spectroscopy EXAFS Project

A modern, robust CLI wrapper for larch EXAFS processing with support for single structures and MD trajectories.

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
# Install the package in editable mode
pip install -e .

# Or install with development dependencies
pip install -e .
pip install -r requirements-dev.txt

# Or use the Makefile for complete setup
make dev-setup
```

### Package Installation (Recommended)

Once installed, the CLI is available system-wide:

```bash
# Install and use the CLI command
pip install -e .
larch-cli structure structure.cif Fe --output-dir results

# Or use the module interface
python -m larch_cli_wrapper.cli structure structure.cif Fe --output-dir results
```

### Optional Dependencies

The package supports several optional feature sets:

```bash
# Advanced features (pymatgen for enhanced FEFF parameters)
pip install -e .[advanced]

# Performance optimization (numba for fast trajectory processing)  
pip install -e .[performance]

# All optional features
pip install -e .[full]

# Development tools (testing, linting, etc.)
pip install -e .[dev]
```

## Larch CLI Wrapper

This project provides a powerful, user-friendly CLI wrapper around larch for EXAFS processing. The wrapper supports both single structure analysis and time-averaged MD trajectory processing with cross-platform compatibility and performance optimization.

### Key Features

- **🏗️ Modular Python API**: Use `LarchWrapper` class for programmatic access
- **⚡ Streamlined CLI**: Two main commands (`structure`, `trajectory`) plus utilities  
- **🔄 Complete Pipeline**: Run FEFF generation, calculation, and post-processing in one command
- **🎯 Flexible Processing**: Process individual structures or entire MD trajectories
- **📊 Rich Output**: Colored terminal output with progress indicators and diagnostics
- **📈 Customizable Plots**: Generate publication-quality PDF and SVG plots
- **🚀 Performance Optimized**: Optional numba acceleration for trajectory processing
- **⚡ Parallel Processing**: Multi-core support for trajectory processing with automatic worker management
- **🌍 Cross-Platform**: Windows, macOS, and Linux compatibility

### Quick Start

#### CLI Usage

Process a single structure (complete pipeline):
```bash
larch-cli structure examples/structure.cif Fe --output-dir results
```

Process MD trajectory (time-averaged EXAFS):
```bash
larch-cli trajectory trajectory.xyz Au --interval 10 --output-dir traj_results
```

Generate FEFF input only:
```bash
larch-cli structure structure.cif Fe --input-only --output-dir feff_input
```
#### Python API Usage

```python
from pathlib import Path
from larch_cli_wrapper.wrapper import LarchWrapper

# Initialize wrapper
wrapper = LarchWrapper()

# Process single structure - complete pipeline
exafs_group, plot_paths = wrapper.run_full_pipeline(
    structure_path=Path('structure.cif'),
    absorber='Fe',
    output_dir=Path('feff_output')
)

# Process MD trajectory with time-averaging
trajectory_group, plot_paths = wrapper.process_trajectory(
    trajectory_path=Path('trajectory.xyz'),
    absorber='Au',
    output_dir=Path('traj_output'),
    sample_interval=5  # Process every 5th frame
)

# Process MD trajectory with parallel processing (faster)
trajectory_group, plot_paths = wrapper.process_trajectory(
    trajectory_path=Path('trajectory.xyz'),
    absorber='Au',
    output_dir=Path('traj_output'),
    sample_interval=5,
    parallel=True,      # Enable parallel processing
    n_workers=4         # Use 4 parallel workers
)

# Step-by-step processing for more control
input_file = wrapper.generate_feff_input(Path('structure.cif'), 'Fe', Path('feff'))
success = wrapper.run_feff(Path('feff'))
exafs_group = wrapper.process_feff_output(Path('feff'))
plot_paths = wrapper.plot_fourier_transform(exafs_group, show_plot=True)
```

### CLI Commands

The CLI provides 5 focused commands:

#### Main Processing Commands

##### `structure` - Single Structure Processing
Process a single structure from any supported format (CIF, XYZ, VASP, PDB, etc.).

```bash
larch-cli structure STRUCTURE_FILE ABSORBER [OPTIONS]
```

**Key Features:**
- Supports all ASE-compatible structure formats
- Can extract specific frames from trajectory files
- Full pipeline or input-only mode
- Customizable EXAFS parameters

**Example:**
```bash
larch-cli structure Hematite.cif Fe --output-dir results --kweight 3 --show
```

##### `trajectory` - MD Trajectory Processing  
Process MD trajectories for time-averaged EXAFS analysis.

```bash
larch-cli trajectory TRAJECTORY_FILE ABSORBER [OPTIONS]
```

**Key Features:**
- Handles large trajectory files efficiently
- **Parallel processing** for improved performance on multi-core systems
- Optional numba acceleration for performance
- Configurable frame intervals
- Time-averaged spectral analysis

**Example:**
```bash
# Sequential processing (traditional)
larch-cli trajectory md_run.xyz Au --interval 10 --output-dir traj_analysis

# Parallel processing (faster on multi-core systems)
larch-cli trajectory md_run.xyz Au --interval 10 --output-dir traj_analysis --parallel --workers 4
```

#### Utility Commands

##### `run-feff` - Execute FEFF Calculation
Run FEFF calculation on existing input files.

```bash
larch-cli run-feff FEFF_DIR [OPTIONS]
```

##### `process` - Post-Process FEFF Output
Process existing FEFF output and generate plots.

```bash
larch-cli process FEFF_DIR [OPTIONS]
```

##### `version` - System Diagnostics
Show comprehensive system information and diagnostics.

```bash
larch-cli version
```

**Displays:**
- Package version and dependencies
- FEFF executable paths and versions
- Python environment details
- Optional dependency status

### Common Options

All main commands support these options:

- `--output-dir, -o`: Output directory (default varies by command)
- `--input-only`: Generate FEFF input only, skip calculation
- `--method, -m`: Method (auto, larixite, pymatgen)  
- `--spectrum, -s`: Spectrum type (EXAFS, XANES, etc.)
- `--edge`: Absorption edge (K, L1, L2, L3, etc.)
- `--radius, -r`: Cluster radius in Angstroms
- `--kweight, -k`: k-weighting for Fourier transform  
- `--window, -w`: Window function (hanning, kaiser, etc.)
- `--kmin/kmax`: k-range for Fourier transform
- `--show/--no-show`: Display plots interactively

### Advanced Usage Examples

#### Custom Fourier Transform Parameters
```python
from larch_cli_wrapper.wrapper import LarchWrapper

wrapper = LarchWrapper()
exafs_group = wrapper.process_feff_output(
    feff_dir,
    kweight=3,          # Higher k-weighting
    window='kaiser',    # Different window function  
    kmin=3.0,          # Custom k-range
    kmax=12.0
)
```

#### Trajectory Processing with Performance Optimization
```python
# Install performance extras first: pip install -e .[performance]
trajectory_group, plots = wrapper.process_trajectory(
    trajectory_path,
    absorber='Au',
    output_dir=Path('trajectory_output'),
    frame_interval=10,           # Every 10th frame
    use_numba_optimization=True  # Fast averaging (requires numba)
)
```

#### Custom Plot Generation
```python
plot_paths = wrapper.plot_fourier_transform(
    exafs_group,
    output_dir=Path('custom_plots'),
    filename_base='my_spectrum',
    kweight=2,
    show_plot=False,
    save_formats=['pdf', 'svg', 'png']  # Multiple formats
)
```

#### Advanced FEFF Input Generation
```python
# Using pymatgen for enhanced FEFF parameters (requires advanced extras)
input_file = wrapper.generate_feff_input(
    structure_path,
    absorber='Fe',
    output_dir=Path('advanced_feff'),
    method='pymatgen',    # Enhanced parameter sets
    spectrum_type='XANES',
    edge='L3',
    radius=12.0
)
```

#### Error Handling and Diagnostics
```python
import logging
from larch_cli_wrapper.wrapper import LarchWrapper

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

wrapper = LarchWrapper()

try:
    # Check system capabilities
    if wrapper.check_feff_executable():
        result = wrapper.run_full_pipeline(structure_path, absorber, output_dir)
    else:
        print("FEFF executable not found!")
        
except FileNotFoundError as e:
    print(f"File not found: {e}")
except RuntimeError as e:
    print(f"FEFF calculation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Migration from Legacy Scripts

If migrating from older EXAFS processing scripts:

**Before (manual processing):**
```python
from larch.xafs.feffrunner import feff8l
from larch.xafs import xftf
# ... extensive manual setup and processing
```

**After (using wrapper):**
```python
from larch_cli_wrapper.wrapper import LarchWrapper

wrapper = LarchWrapper()
exafs_group, plots = wrapper.run_full_pipeline(structure_path, absorber, output_dir)
```

See migration examples in `examples/` directory.

## Project Structure

```
├── src/larch_cli_wrapper/        # Main package
│   ├── __init__.py
│   ├── wrapper.py               # Core LarchWrapper class
│   └── cli.py                   # CLI interface using typer
├── examples/                    # Example workflows and data  
│   ├── example_wrapper_usage.py # API usage examples
│   ├── example_pymatgen_usage.py # Advanced pymatgen examples
│   └── simplified_example.py   # Basic usage patterns
├── docs/                       # Comprehensive documentation
│   ├── quick_start.md          # Getting started guide
│   ├── project_structure.md    # Detailed project organization
│   └── pymatgen_integration.md # Advanced features guide
├── tests/                      # Comprehensive test suite
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Core dependencies
└── requirements-dev.txt        # Development dependencies
```

## Dependencies

### Core Dependencies
- **xraylarch** (≥0.9.47): EXAFS processing engine and FEFF interface
- **typer** (≥0.12.0): Modern CLI framework with rich features
- **rich**: Beautiful terminal output with progress indicators  
- **matplotlib** (≥3.5): Publication-quality plotting
- **ase** (≥3.22.1): Atomic structure handling and file I/O
- **larixite** (≥0.1.0): Optimized CIF to FEFF conversion
- **numpy** (≥1.21): Numerical computing foundation

### Optional Dependencies
- **pymatgen** (≥2022.7): Advanced spectrum types and enhanced FEFF parameters
- **numba** (≥0.56): High-performance trajectory processing acceleration

## Development

This project uses modern Python packaging with `pyproject.toml` and includes comprehensive development tooling.

### Development Setup
```bash
# Complete development setup
make dev-setup

# Or manually:
pip install -e .
pip install -r requirements-dev.txt
pre-commit install
```

### Code Quality Tools
- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing
- **pre-commit**: Git hooks for code quality

### Common Commands
```bash
make help              # Show all available commands
make test              # Run tests
make test-coverage     # Run tests with coverage
make lint              # Run linting
make format            # Format code
make type-check        # Type checking
make build             # Build package
make clean             # Clean build artifacts
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=larch_wrapper --cov=cli --cov-report=html

# Run specific tests
pytest tests/test_larch_wrapper.py
```

### Project Structure
See `docs/project_structure.md` for detailed information about the project organization.
