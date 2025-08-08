"""Comprehensive tests for the LarchWrapper class."""

from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from larch_cli_wrapper.wrapper import PYMATGEN_AVAILABLE, LarchWrapper


# Fixtures
@pytest.fixture
def wrapper():
    """Fixture for LarchWrapper instance with suppressed logging."""
    return LarchWrapper(verbose=False)


@pytest.fixture
def mock_structure_file(tmp_path):
    """Create a mock CIF file."""
    cif_file = tmp_path / "test.cif"
    cif_file.write_text(
        """data_test
_cell_length_a 5.0
_cell_length_b 5.0  
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe 0.0 0.0 0.0
"""
    )
    return cif_file


@pytest.fixture
def mock_xyz_file(tmp_path):
    """Create a mock XYZ file."""
    xyz_file = tmp_path / "test.xyz"
    xyz_file.write_text(
        """1
Frame 1
Fe 0.0 0.0 0.0
"""
    )
    return xyz_file


class TestLarchWrapperBasic:
    """Test basic functionality and initialization."""

    def test_init_default(self):
        """Test default wrapper initialization."""
        wrapper = LarchWrapper(verbose=False)
        assert wrapper.last_exafs_group is None
        assert wrapper._temp_files == []
        assert isinstance(wrapper.config, dict)

    def test_init_with_verbose(self):
        """Test wrapper initialization with verbose logging."""
        wrapper = LarchWrapper(verbose=True)
        assert wrapper.logger.level == 20  # INFO level

    def test_context_manager(self):
        """Test context manager functionality."""
        with LarchWrapper(verbose=False) as wrapper:
            assert isinstance(wrapper, LarchWrapper)
            wrapper._temp_files.append(Path("dummy"))
        # Files should be cleaned up after context exit

    def test_cleanup_temp_files(self, wrapper, tmp_path):
        """Test temporary file cleanup."""
        temp_file1 = tmp_path / "temp1.txt"
        temp_file2 = tmp_path / "temp2.txt"
        temp_file1.write_text("temp")
        temp_file2.write_text("temp")

        wrapper._temp_files.extend([temp_file1, temp_file2])

        wrapper.cleanup_temp_files()
        assert not temp_file1.exists()
        assert not temp_file2.exists()
        assert len(wrapper._temp_files) == 0

    def test_cleanup_nonexistent_files(self, wrapper):
        """Test cleanup with files that don't exist."""
        wrapper._temp_files.append(Path("/nonexistent/file.txt"))
        wrapper.cleanup_temp_files()  # Should not raise exception
        assert len(wrapper._temp_files) == 0


class TestLarchWrapperValidation:
    """Test parameter validation methods."""

    def test_validate_parameters_valid(self, wrapper):
        """Test parameter validation with valid inputs."""
        wrapper._validate_parameters("EXAFS", "K")
        wrapper._validate_parameters("XANES", "L3")
        # Should not raise exceptions

    def test_validate_parameters_invalid_spectrum(self, wrapper):
        """Test parameter validation with invalid spectrum type."""
        with pytest.raises(ValueError, match="Invalid spectrum type"):
            wrapper._validate_parameters("INVALID", "K")

    def test_validate_parameters_invalid_edge(self, wrapper):
        """Test parameter validation with invalid edge."""
        with pytest.raises(ValueError, match="Unsupported edge"):
            wrapper._validate_parameters("EXAFS", "INVALID")

    def test_validate_absorber_symbol(self, wrapper):
        """Test absorber validation with element symbols."""
        wrapper._validate_absorber("Fe")
        wrapper._validate_absorber("Cu")
        # Should not raise exceptions

    def test_validate_absorber_index(self, wrapper):
        """Test absorber validation with numeric indices."""
        wrapper._validate_absorber("0")
        wrapper._validate_absorber("5")
        # Should not raise exceptions

    def test_validate_structure_file_exists(self, wrapper, mock_structure_file):
        """Test structure file validation with existing file."""
        wrapper._validate_structure_file(mock_structure_file)
        # Should not raise exception

    def test_validate_structure_file_missing(self, wrapper):
        """Test structure file validation with missing file."""
        with pytest.raises(FileNotFoundError):
            wrapper._validate_structure_file(Path("/nonexistent/file.cif"))

    def test_validate_dependencies_available(self, wrapper):
        """Test dependency validation when available."""
        wrapper._validate_dependencies("larixite", "EXAFS")
        # Should not raise exception

    def test_validate_dependencies_missing_pymatgen(self, wrapper):
        """Test dependency validation when pymatgen missing."""
        with patch("larch_cli_wrapper.wrapper.PYMATGEN_AVAILABLE", False):
            with pytest.raises(ImportError, match="pymatgen is required"):
                wrapper._validate_dependencies("pymatgen", "XANES")


class TestLarchWrapperConfiguration:
    """Test configuration and preset methods."""

    def test_use_preset_quick(self, wrapper):
        """Test quick preset configuration."""
        wrapper.use_preset("quick")
        assert wrapper.config["radius"] == 8.0
        assert wrapper.config["spectrum_type"] == "EXAFS"

    def test_use_preset_publication(self, wrapper):
        """Test publication preset configuration."""
        wrapper.use_preset("publication")
        assert wrapper.config["radius"] == 12.0

    def test_use_preset_invalid(self, wrapper):
        """Test invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            wrapper.use_preset("invalid_preset")

    def test_get_capabilities(self, wrapper):
        """Test capabilities reporting."""
        caps = wrapper.get_capabilities()
        assert "numba_available" in caps
        assert "pymatgen_available" in caps
        assert "available_presets" in caps
        assert isinstance(caps["available_presets"], list)


class TestLarchWrapperDiagnostics:
    """Test diagnostic and system information methods."""

    @patch("larch_cli_wrapper.wrapper.find_exe")
    def test_get_feff_version(self, mock_find_exe, wrapper):
        """Test FEFF version detection."""
        mock_find_exe.return_value = "/path/to/feff"
        version = wrapper.get_feff_version()
        assert isinstance(version, str)

    def test_get_available_feff_executables(self, wrapper):
        """Test FEFF executable detection."""
        with patch("larch_cli_wrapper.wrapper.find_exe") as mock_find:
            mock_find.side_effect = ["/path/feff8", None, "/path/feff9"]
            execs = wrapper.get_available_feff_executables()
            assert isinstance(execs, dict)

    def test_run_diagnostics(self, wrapper):
        """Test comprehensive diagnostics."""
        with (
            patch.object(wrapper, "get_feff_version", return_value="Test FEFF"),
            patch.object(
                wrapper,
                "get_available_feff_executables",
                return_value={"feff8l": "/path/feff8l"},
            ),
        ):
            diagnostics = wrapper.run_diagnostics()

            assert "system_info" in diagnostics
            assert "platform" in diagnostics["system_info"]
            assert "dependencies" in diagnostics
            assert "feff_info" in diagnostics
            assert "capabilities" in diagnostics
            assert "warnings" in diagnostics
            assert "recommendations" in diagnostics

    def test_print_diagnostics(self, wrapper, capsys):
        """Test diagnostics printing."""
        with patch.object(
            wrapper,
            "run_diagnostics",
            return_value={
                "system_info": {"python_version": "3.8.0", "platform": "linux"},
                "dependencies": {"pymatgen_available": False},
                "feff_info": {
                    "version": "Test FEFF",
                    "executables": {"feff8l": "/path/feff8l"},
                },
                "presets": {"exafs": {}},
                "warnings": [],
                "recommendations": ["Install pymatgen"],
            },
        ):
            wrapper.print_diagnostics()
            captured = capsys.readouterr()
            assert "DIAGNOSTICS" in captured.out


class TestLarchWrapperFeffInput:
    """Test FEFF input generation methods."""

    @patch("larch_cli_wrapper.wrapper.cif2feffinp")
    def test_generate_feff_input_creates_directory(
        self, mock_cif2feff, wrapper, mock_structure_file, tmp_path
    ):
        """Test that generate_feff_input creates output directory."""
        mock_cif2feff.return_value = "FEFF INPUT CONTENT"

        output_dir = tmp_path / "output"

        result = wrapper.generate_feff_input(mock_structure_file, "Fe", output_dir)

        # Check directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()

        # Check file was written
        expected_file = output_dir / "feff.inp"
        assert expected_file.exists()
        assert expected_file.read_text() == "FEFF INPUT CONTENT"

        # Check return value
        assert result == expected_file

    @patch("larch_cli_wrapper.wrapper.cif2feffinp")
    def test_generate_feff_input_method_selection(
        self, mock_cif2feff, wrapper, mock_structure_file, tmp_path
    ):
        """Test FEFF input generation method selection."""
        mock_cif2feff.return_value = "FEFF CONTENT"

        # Test auto method selection
        result = wrapper.generate_feff_input(
            mock_structure_file, "Fe", tmp_path, method="auto"
        )
        assert result.exists()

        # Test explicit larixite method
        result = wrapper.generate_feff_input(
            mock_structure_file, "Fe", tmp_path, method="larixite"
        )
        assert result.exists()

    @patch("larch_cli_wrapper.wrapper.cif2feffinp")
    def test_generate_feff_input_custom_parameters(
        self, mock_cif2feff, wrapper, tmp_path
    ):
        """Test FEFF input generation with custom parameters."""
        mock_cif2feff.return_value = "CUSTOM FEFF"

        # Create a simple CIF with Cu instead of Fe
        cif_file = tmp_path / "test_cu.cif"
        cif_file.write_text(
            """data_test
_cell_length_a 5.0
_cell_length_b 5.0  
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu 0.0 0.0 0.0
"""
        )

        result = wrapper.generate_feff_input(
            cif_file,
            "Cu",
            tmp_path,
            spectrum_type="EXAFS",
            edge="K",
            radius=12.0,
            method="larixite",
        )

        assert result.exists()
        mock_cif2feff.assert_called_once()

        # Check that parameters were used correctly
        call_args = mock_cif2feff.call_args
        assert "Cu" in str(call_args)

    @patch("larch_cli_wrapper.wrapper.ase_read")
    @patch("larch_cli_wrapper.wrapper.ase_write")
    def test_create_temp_cif(
        self, mock_ase_write, mock_ase_read, wrapper, mock_xyz_file
    ):
        """Test temporary CIF creation from non-CIF files."""
        mock_atoms = Mock()
        mock_ase_read.return_value = mock_atoms

        result = wrapper._create_temp_cif(mock_xyz_file)

        mock_ase_read.assert_called_once_with(str(mock_xyz_file))
        mock_ase_write.assert_called_once()
        assert result in wrapper._temp_files
        assert result.suffix == ".cif"

    @pytest.mark.skipif(not PYMATGEN_AVAILABLE, reason="Requires pymatgen")
    @patch("larch_cli_wrapper.wrapper.Structure")
    def test_load_structure_with_pymatgen(
        self, mock_structure, wrapper, mock_structure_file
    ):
        """Test structure loading with pymatgen."""
        mock_structure.from_file.return_value = Mock()

        result = wrapper._load_structure_with_pymatgen(mock_structure_file)

        mock_structure.from_file.assert_called_once_with(str(mock_structure_file))
        assert result is not None


class TestLarchWrapperFeffExecution:
    """Test FEFF execution methods."""

    @patch("larch_cli_wrapper.wrapper.feff8l")
    def test_run_feff_success(self, mock_feff8l, wrapper, tmp_path):
        """Test successful FEFF run."""
        mock_feff8l.return_value = None

        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        input_file = feff_dir / "feff.inp"
        input_file.write_text("FEFF INPUT CONTENT")

        # Create chi.dat to simulate successful FEFF run
        chi_file = feff_dir / "chi.dat"
        chi_file.write_text("# k chi\n1.0 0.1\n2.0 0.2\n")

        result = wrapper.run_feff(feff_dir, verbose=False)

        assert result is True
        mock_feff8l.assert_called_once_with(
            folder=str(feff_dir), feffinp="feff.inp", verbose=False
        )

    @patch("larch_cli_wrapper.wrapper.feff8l")
    def test_run_feff_failure(self, mock_feff8l, wrapper, tmp_path):
        """Test FEFF run failure."""
        mock_feff8l.side_effect = Exception("FEFF failed")

        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        input_file = feff_dir / "feff.inp"
        input_file.write_text("FEFF INPUT")

        result = wrapper.run_feff(feff_dir, verbose=False)

        assert result is False

    @patch("larch_cli_wrapper.wrapper.feff8l")
    def test_run_feff_custom_input_filename(self, mock_feff8l, wrapper, tmp_path):
        """Test FEFF run with custom input filename."""
        mock_feff8l.return_value = None

        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        input_file = feff_dir / "custom.inp"
        input_file.write_text("FEFF INPUT")

        # Create chi.dat to simulate successful FEFF run
        chi_file = feff_dir / "chi.dat"
        chi_file.write_text("# k chi\n1.0 0.1\n")

        result = wrapper.run_feff(feff_dir, input_filename="custom.inp", verbose=False)

        assert result is True
        mock_feff8l.assert_called_once_with(
            folder=str(feff_dir), feffinp="custom.inp", verbose=False
        )

    def test_run_feff_missing_input(self, wrapper, tmp_path):
        """Test FEFF run with missing input file."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="feff.inp not found"):
            wrapper.run_feff(feff_dir)


class TestLarchWrapperFeffProcessing:
    """Test FEFF output processing methods."""

    @patch("larch_cli_wrapper.wrapper.read_ascii")
    @patch("larch_cli_wrapper.wrapper.xftf")
    @patch("larch_cli_wrapper.wrapper.Group")
    def test_process_feff_output_success(
        self, mock_group_class, mock_xftf, mock_read_ascii, wrapper, tmp_path
    ):
        """Test successful FEFF output processing."""
        # Create chi.dat file
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        chi_file = feff_dir / "chi.dat"
        chi_file.write_text("# k chi\n1.0 0.1\n2.0 0.2\n")

        # Mock the larch functions
        mock_data = Mock()
        mock_data.k = np.array([1.0, 2.0])
        mock_data.chi = np.array([0.1, 0.2])
        mock_read_ascii.return_value = mock_data

        mock_group = Mock()
        mock_group_class.return_value = mock_group

        result = wrapper.process_feff_output(feff_dir, kweight=3, kmin=1.5)

        # Verify calls
        mock_read_ascii.assert_called_once_with(str(chi_file))
        mock_group_class.assert_called_once()
        mock_xftf.assert_called_once_with(
            mock_group, kweight=3, window="hanning", dk=1.0, kmin=1.5, kmax=14.0
        )

        # Check that data was assigned
        assert mock_group.k is mock_data.k
        assert mock_group.chi is mock_data.chi

        # Check return value and last_exafs_group
        assert result == mock_group
        assert wrapper.last_exafs_group == mock_group

    def test_process_feff_output_file_not_found(self, wrapper, tmp_path):
        """Test process_feff_output when chi.dat doesn't exist."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="chi.dat not found"):
            wrapper.process_feff_output(feff_dir)

    @patch("larch_cli_wrapper.wrapper.read_ascii")
    @patch("larch_cli_wrapper.wrapper.xftf")
    @patch("larch_cli_wrapper.wrapper.Group")
    def test_process_feff_output_custom_ft_params(
        self, mock_group_class, mock_xftf, mock_read_ascii, wrapper, tmp_path
    ):
        """Test FEFF output processing with custom FT parameters."""
        # Setup
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        chi_file = feff_dir / "chi.dat"
        chi_file.write_text("# k chi\n")

        mock_data = Mock()
        mock_data.k = np.array([1.0])
        mock_data.chi = np.array([0.1])
        mock_read_ascii.return_value = mock_data
        mock_group = Mock()
        mock_group_class.return_value = mock_group

        # Test with custom parameters
        wrapper.process_feff_output(
            feff_dir, kweight=3, window="kaiser", dk=0.5, kmin=3.0, kmax=12.0
        )

        mock_xftf.assert_called_once_with(
            mock_group, kweight=3, window="kaiser", dk=0.5, kmin=3.0, kmax=12.0
        )


class TestLarchWrapperPlotting:
    """Test plotting functionality."""

    @patch("larch_cli_wrapper.wrapper.plt")
    def test_plot_fourier_transform(self, mock_plt, wrapper, tmp_path):
        """Test Fourier transform plotting."""
        # Create mock EXAFS data
        mock_group = Mock()
        mock_group.r = np.array([1.0, 2.0, 3.0])
        mock_group.chir_mag = np.array([0.1, 0.2, 0.1])
        mock_group.chir_re = np.array([0.05, 0.15, 0.05])
        mock_group.chir_im = np.array([0.02, 0.08, 0.02])

        # Mock plt methods
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        pdf_path, svg_path = wrapper.plot_fourier_transform(
            mock_group, output_dir=tmp_path, filename_base="test_plot", show_plot=False
        )

        # Check plot was created
        mock_plt.subplots.assert_called_once()
        mock_ax.plot.assert_called()

        # Check that fig.savefig was called with the correct paths
        expected_pdf = tmp_path / "test_plot.pdf"
        expected_svg = tmp_path / "test_plot.svg"
        mock_fig.savefig.assert_has_calls(
            [
                call(expected_pdf, format="pdf", bbox_inches="tight"),
                call(expected_svg, format="svg", transparent=True, bbox_inches="tight"),
            ]
        )

        assert pdf_path == expected_pdf
        assert svg_path == expected_svg
        # Check return values
        assert pdf_path.suffix == ".pdf"
        assert svg_path.suffix == ".svg"
        assert "test_plot" in pdf_path.name

    @patch("larch_cli_wrapper.wrapper.plt")
    def test_plot_fourier_transform_show(self, mock_plt, wrapper):
        """Test plotting with show_plot=True."""
        mock_group = Mock()
        mock_group.r = np.array([1.0])
        mock_group.chir_mag = np.array([0.1])
        mock_group.chir_re = np.array([0.05])
        mock_group.chir_im = np.array([0.02])

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        wrapper.plot_fourier_transform(mock_group, show_plot=True)

        mock_plt.show.assert_called_once()


class TestLarchWrapperPipeline:
    """Test complete pipeline functionality."""

    @patch.object(LarchWrapper, "generate_feff_input")
    @patch.object(LarchWrapper, "run_feff")
    @patch.object(LarchWrapper, "process_feff_output")
    @patch.object(LarchWrapper, "plot_fourier_transform")
    def test_run_full_pipeline_success(
        self,
        mock_plot,
        mock_process,
        mock_run,
        mock_gen_input,
        wrapper,
        mock_structure_file,
        tmp_path,
    ):
        """Test successful full pipeline execution."""
        # Configure mocks
        mock_gen_input.return_value = tmp_path / "feff.inp"
        mock_run.return_value = True
        mock_exafs_group = Mock()
        mock_process.return_value = mock_exafs_group
        mock_plot.return_value = (tmp_path / "plot.pdf", tmp_path / "plot.svg")

        # Execute pipeline
        exafs_group, plot_paths = wrapper.run_full_pipeline(
            structure_path=mock_structure_file, absorber="Fe", output_dir=tmp_path
        )

        # Verify call sequence - check the actual call
        mock_gen_input.assert_called_once()
        mock_run.assert_called_once_with(tmp_path)
        mock_process.assert_called_once()
        mock_plot.assert_called_once()

        # Check return values
        assert exafs_group == mock_exafs_group
        assert len(plot_paths) == 2

    @patch.object(LarchWrapper, "generate_feff_input")
    @patch.object(LarchWrapper, "run_feff")
    def test_run_full_pipeline_feff_failure(
        self, mock_run, mock_gen_input, wrapper, mock_structure_file, tmp_path
    ):
        """Test full pipeline with FEFF failure."""
        mock_gen_input.return_value = tmp_path / "feff.inp"
        mock_run.return_value = False

        with pytest.raises(RuntimeError, match="FEFF calculation failed"):
            wrapper.run_full_pipeline(mock_structure_file, "Fe", tmp_path)


# class TestLarchWrapperTrajectory:
#     """Test trajectory processing functionality."""
#     # TODO: Implement trajectory processing tests


class TestLarchWrapperUtilities:
    """Test utility methods."""

    def test_get_feff_version(self, wrapper):
        """Test FEFF version retrieval."""
        with patch("larch_cli_wrapper.wrapper.find_exe", return_value="/path/to/feff"):
            version = wrapper.get_feff_version()
            assert isinstance(version, str)
            assert "FEFF" in version

    def test_get_available_feff_executables(self, wrapper):
        """Test FEFF executable detection."""
        executables = wrapper.get_available_feff_executables()
        assert isinstance(executables, dict)

    def test_select_method_auto(self, wrapper):
        """Test automatic method selection."""
        # Should prefer larixite for EXAFS
        method = wrapper._select_method("EXAFS")
        assert method == "larixite"

        # Should prefer pymatgen for XANES if available
        if PYMATGEN_AVAILABLE:
            method = wrapper._select_method("XANES")
            assert method == "pymatgen"
        else:
            method = wrapper._select_method("XANES")
            assert method == "larixite"

    def test_quick_exafs(self, wrapper, mock_structure_file, tmp_path):
        """Test quick EXAFS analysis."""
        with patch.object(wrapper, "run_full_pipeline") as mock_pipeline:
            mock_pipeline.return_value = (
                Mock(),
                (tmp_path / "plot.pdf", tmp_path / "plot.svg"),
            )
            wrapper.quick_exafs(mock_structure_file, "Fe", output_dir=tmp_path)
            mock_pipeline.assert_called_once()


# Integration tests
class TestLarchWrapperIntegration:
    """Integration tests with minimal mocking."""

    def test_end_to_end_input_generation(self, wrapper, mock_structure_file, tmp_path):
        """Test end-to-end FEFF input generation."""
        with patch(
            "larch_cli_wrapper.wrapper.cif2feffinp", return_value="MOCK FEFF INPUT"
        ):
            result = wrapper.generate_feff_input(
                mock_structure_file,
                "Fe",
                tmp_path,
                method="larixite",
                spectrum_type="EXAFS",
                edge="K",
            )

            assert result.exists()
            assert result.read_text() == "MOCK FEFF INPUT"
            assert result.parent == tmp_path


@pytest.fixture
def sample_cif_content():
    """Sample CIF file content for testing."""
    return """
data_test
_cell_length_a 5.0
_cell_length_b 5.0  
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe 0.0 0.0 0.0
"""
