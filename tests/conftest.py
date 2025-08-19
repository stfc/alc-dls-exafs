"""Test configuration and fixtures for CLI test suite."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def tmp_structure_file(tmp_path: Path) -> Path:
    """Create a temporary structure file for testing."""
    structure_file = tmp_path / "test_structure.cif"
    structure_file.write_text("""
# CIF file for testing
_chemical_name_common 'Test Structure'
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M 'P m -3 m'

loop_
_atom_site_label
_atom_site_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0.0 0.0 0.0
O1 O 0.5 0.0 0.0
O2 O 0.0 0.5 0.0
O3 O 0.0 0.0 0.5
""")
    return structure_file


@pytest.fixture
def tmp_trajectory_file(tmp_path: Path) -> Path:
    """Create a temporary trajectory file for testing."""
    trajectory_file = tmp_path / "test_trajectory.xyz"
    trajectory_content = ""
    
    # Create a simple 3-frame trajectory
    for frame in range(3):
        trajectory_content += f"""4
Frame {frame}
Fe 0.0 0.0 {frame * 0.1}
O 1.0 0.0 0.0
O 0.0 1.0 0.0
O 0.0 0.0 1.0

"""
    
    trajectory_file.write_text(trajectory_content)
    return trajectory_file


@pytest.fixture
def tmp_config_file(tmp_path: Path) -> Path:
    """Create a temporary configuration file for testing."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text("""
# Test configuration file
spectrum_type: EXAFS
edge: K
radius: 8.0
method: larixite
kmin: 2.0
kmax: 14.0
kweight: 2
window: hanning
dk: 1.0
parallel: false
force_recalculate: false
user_tag_settings:
  S02: "0.8"
  SCF: "5.0 0 30 0.1 1"
""")
    return config_file


@pytest.fixture
def tmp_feff_directory(tmp_path: Path) -> Path:
    """Create a temporary FEFF directory with input and output files."""
    feff_dir = tmp_path / "feff"
    feff_dir.mkdir()
    
    # Create feff.inp
    feff_inp = feff_dir / "feff.inp"
    feff_inp.write_text("""
TITLE Test FEFF calculation
CONTROL 1 1 1 1 1 1
PRINT 1 0 0 0 0 3
RMAX 8.0

POTENTIALS
0 26 Fe
1 8 O

ATOMS
0.00000   0.00000   0.00000   0   Fe1
1.00000   0.00000   0.00000   1   O1
0.00000   1.00000   0.00000   1   O2
0.00000   0.00000   1.00000   1   O3
END
""")
    
    # Create chi.dat (FEFF output)
    chi_dat = feff_dir / "chi.dat"
    chi_dat.write_text("""
# k chi(k) |chi(k)| phase(k) @#
  1.000   0.123   0.123   1.234
  2.000   0.234   0.234   2.345
  3.000   0.345   0.345   3.456
  4.000   0.456   0.456   4.567
  5.000   0.567   0.567   5.678
""")
    
    # Create feff.log
    feff_log = feff_dir / "feff.log"
    feff_log.write_text("""
FEFF 9.6.4 test run
This is a test log file
Calculation completed successfully
""")
    
    return feff_dir


@pytest.fixture
def tmp_trajectory_output(tmp_path: Path) -> Path:
    """Create a temporary trajectory output directory with frame subdirectories."""
    traj_dir = tmp_path / "trajectory_output"
    traj_dir.mkdir()
    
    # Create frame directories
    for i in range(5):
        frame_dir = traj_dir / f"frame_{i:04d}"
        frame_dir.mkdir()
        
        # Create chi.dat in each frame
        chi_file = frame_dir / "chi.dat"
        chi_file.write_text(f"""
# Frame {i} chi data
  1.000   {0.1 + i * 0.01}   {0.1 + i * 0.01}   1.234
  2.000   {0.2 + i * 0.01}   {0.2 + i * 0.01}   2.345
  3.000   {0.3 + i * 0.01}   {0.3 + i * 0.01}   3.456
""")
    
    return traj_dir


@pytest.fixture
def invalid_files(tmp_path: Path) -> dict:
    """Create various invalid files for testing error handling."""
    files = {}
    
    # Empty file
    files['empty'] = tmp_path / "empty.cif"
    files['empty'].write_text("")
    
    # Binary file
    files['binary'] = tmp_path / "binary.dat"
    files['binary'].write_bytes(b"\x00\x01\x02\x03\x04\x05")
    
    # Text file with invalid content
    files['invalid'] = tmp_path / "invalid.cif"
    files['invalid'].write_text("This is not a valid CIF file")
    
    # Very large text file
    files['large'] = tmp_path / "large.txt"
    files['large'].write_text("x" * 10000)  # 10KB of 'x'
    
    return files


# Test markers for different test categories
pytest_markers = {
    "unit": "Unit tests - fast, isolated tests",
    "integration": "Integration tests - test component interaction",
    "performance": "Performance tests - may take longer to run",
    "stress": "Stress tests - test system limits",
    "slow": "Slow tests - tests that take significant time"
}


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker, description in pytest_markers.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


class TestConstants:
    """Constants used across test files."""
    
    # Valid edge types
    VALID_EDGES = ["K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"]
    
    # Valid methods
    VALID_METHODS = ["auto", "larixite", "pymatgen"]
    
    # Valid plot styles
    VALID_PLOT_STYLES = ["publication", "presentation", "quick"]
    
    # Valid absorber symbols
    VALID_ABSORBERS = ["Fe", "Cu", "Zn", "Ni", "Co", "Mn", "Cr"]
    
    # Invalid values for testing
    INVALID_EDGES = ["X", "K1", "L4", "M6", "invalid", ""]
    INVALID_METHODS = ["invalid_method", "feff", "quantum", ""]
    INVALID_PLOT_STYLES = ["invalid_style", "custom", "matplotlib", ""]


class MockHelpers:
    """Helper methods for creating consistent mocks across tests."""
    
    @staticmethod
    def create_successful_wrapper_mock():
        """Create a mock LarchWrapper that simulates successful operations."""
        from unittest.mock import Mock
        from larch_cli_wrapper.wrapper import ProcessingResult
        
        mock_wrapper = Mock()
        mock_wrapper.generate_feff_input.return_value = Path("/tmp/output")
        mock_wrapper.run_feff.return_value = True
        mock_wrapper.process_feff_output.return_value = Mock()
        mock_wrapper.plot_results.return_value = {"pdf": Path("/tmp/plot.pdf")}
        mock_wrapper.process.return_value = ProcessingResult(
            exafs_group=Mock(),
            plot_paths={"pdf": Path("/tmp/plot.pdf")},
            processing_mode="single_frame"
        )
        mock_wrapper.print_diagnostics.return_value = None
        mock_wrapper.get_cache_info.return_value = {
            "enabled": True,
            "cache_dir": "/tmp/cache",
            "files": 5,
            "size_mb": 12.5
        }
        mock_wrapper.clear_cache.return_value = None
        
        # Context manager support
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        
        return mock_wrapper
    
    @staticmethod
    def create_failing_wrapper_mock(exception=None):
        """Create a mock LarchWrapper that simulates failures."""
        from unittest.mock import Mock
        
        if exception is None:
            exception = RuntimeError("Mock operation failed")
        
        mock_wrapper = Mock()
        mock_wrapper.generate_feff_input.side_effect = exception
        mock_wrapper.run_feff.return_value = False
        mock_wrapper.process_feff_output.side_effect = exception
        mock_wrapper.process.side_effect = exception
        
        # Context manager support
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        
        return mock_wrapper
