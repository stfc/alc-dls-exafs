"""Comprehensive tests for the feff_utils module."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from ase import Atoms

from larch_cli_wrapper.feff_utils import (
    EdgeType,
    FeffConfig,
    SpectrumType,
    generate_feff_input,
    read_feff_output,
    run_feff_calculation,
)


class TestFeffConfig:
    """Test FeffConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FeffConfig()
        assert config.edge == "K"
        assert config.radius == 8.0
        assert config.spectrum_type == "EXAFS"
        assert config.method == "auto"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FeffConfig(
            edge="L3", radius=6.0, spectrum_type="EXAFS", method="larixite"
        )
        assert config.edge == "L3"
        assert config.radius == 6.0
        assert config.spectrum_type == "EXAFS"
        assert config.method == "larixite"

    def test_invalid_radius(self):
        """Test invalid radius values."""
        with pytest.raises(ValueError):
            FeffConfig(radius=-1.0)


class TestEnums:
    """Test enum definitions."""

    def test_edge_type_enum(self):
        """Test EdgeType enum."""
        assert EdgeType.K == "K"
        assert EdgeType.L1 == "L1"
        assert EdgeType.L2 == "L2"
        assert EdgeType.L3 == "L3"

    def test_spectrum_type_enum(self):
        """Test SpectrumType enum."""
        assert SpectrumType.EXAFS == "EXAFS"


class TestRunFeffCalculation:
    """Test FEFF calculation execution."""

    @patch("sys.stdout")
    @patch("sys.stderr")
    def test_run_calculation_success(self, mock_stderr, mock_stdout):
        """Test successful FEFF calculation."""
        # Create a real temporary directory for this test
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            feff_dir = Path(temp_dir)

            # Create the required input file
            input_file = feff_dir / "feff.inp"
            input_file.write_text("TITLE Test FEFF calculation")

            # Create the chi.dat file that the function expects to check success
            chi_file = feff_dir / "chi.dat"
            chi_file.write_text("# Test chi.dat\n0.0 1.0 0.5\n1.0 2.0 1.0\n")

            # Mock the imported feff8l function inside the function scope
            with patch.object(
                __import__("larch.xafs.feffrunner", fromlist=["feff8l"]), "feff8l"
            ) as mock_feff8l:
                mock_feff8l.return_value = True

                result = run_feff_calculation(feff_dir, verbose=False)
                assert result is True
                mock_feff8l.assert_called_once()

    def test_run_calculation_missing_input(self):
        """Test calculation with missing input file."""
        feff_dir = Path("/test/feff_dir")
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="FEFF input file"):
                run_feff_calculation(feff_dir, verbose=False)


class TestReadFeffOutput:
    """Test FEFF output reading functions."""

    def test_read_output_file_not_found(self):
        """Test reading output when file doesn't exist."""
        feff_dir = Path("/test/feff_dir")
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="FEFF output"):
                read_feff_output(feff_dir)

    @patch("larch.io.read_ascii")
    def test_read_output_success_larch(self, mock_read_ascii):
        """Test successful output reading with larch."""
        feff_dir = Path("/test/feff_dir")
        mock_data = Mock()
        mock_data.chi = np.array([0.1, 0.2, 0.3])
        mock_data.k = np.array([1, 2, 3])
        mock_read_ascii.return_value = mock_data

        with patch("pathlib.Path.exists", return_value=True):
            chi, k = read_feff_output(feff_dir)

        assert np.array_equal(chi, [0.1, 0.2, 0.3])
        assert np.array_equal(k, [1, 2, 3])


class TestFeffInput:
    """Test FEFF input generation functions."""

    @patch("larch_cli_wrapper.feff_utils.generate_larixite_input")
    def test_generate_feff_input_larixite(self, mock_larixite):
        """Test FEFF input generation with larixite."""
        atoms = Atoms("Fe2O", positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]])
        config = FeffConfig()
        mock_larixite.return_value = Path("/test/output")

        output_dir = Path("/test/output")
        result = generate_feff_input(atoms, "Fe", output_dir, config)

        mock_larixite.assert_called_once_with(atoms, "Fe", output_dir, config)
        assert result == Path("/test/output")

    def test_generate_feff_input_invalid_absorber(self):
        """Test FEFF input generation with invalid absorber."""
        atoms = Atoms("Fe2O", positions=[[0, 0, 0], [1.8, 0, 0], [0, 1.8, 0]])
        config = FeffConfig()

        with pytest.raises(ValueError, match="Absorber element"):
            generate_feff_input(atoms, "Cu", Path("/test/output"), config)


if __name__ == "__main__":
    pytest.main([__file__])
