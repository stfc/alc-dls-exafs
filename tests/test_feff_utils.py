"""Comprehensive tests for the feff_utils module."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from ase import Atoms

from larch_cli_wrapper.feff_utils import (
    LARGE_NUMBER_OF_SITES,
    PRESETS,
    EdgeType,
    FeffConfig,
    SpectrumType,
    WindowType,
    average_chi_spectra,
    cleanup_feff_output,
    generate_multi_site_feff_inputs,
    generate_pymatgen_input,
    get_absorber_element_from_index,
    normalize_absorbers,
    read_feff_output,
    run_feff_calculation,
    run_multi_site_feff_calculations,
    validate_absorber,
    validate_absorber_indices,
)


@pytest.fixture
def sample_atoms():
    """Create a sample atomic structure for testing."""
    return Atoms(
        "Fe2O3",
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [0.0, 1.8, 0.0],
            [1.8, 1.8, 0.0],
            [0.9, 0.9, 1.5],
        ],
    )


@pytest.fixture
def sample_config():
    """Create a sample FeffConfig for testing."""
    return FeffConfig(
        spectrum_type="EXAFS",
        edge="K",
        radius=6.0,
        kmin=3.0,
        kmax=15.0,
        kweight=2,
    )


class TestConstants:
    """Test module constants."""

    def test_max_sites_constant(self):
        """Test MAX_SITES constant."""
        assert isinstance(LARGE_NUMBER_OF_SITES, int)
        assert LARGE_NUMBER_OF_SITES > 0

    def test_presets_structure(self):
        """Test PRESETS dictionary structure."""
        assert isinstance(PRESETS, dict)
        assert len(PRESETS) > 0
        assert "quick" in PRESETS
        assert "publication" in PRESETS

        for _preset_name, preset_config in PRESETS.items():
            assert isinstance(preset_config, dict)
            assert "spectrum_type" in preset_config
            assert "edge" in preset_config
            assert "radius" in preset_config


class TestEnums:
    """Test enum definitions."""

    def test_spectrum_type_enum(self):
        """Test SpectrumType enum."""
        assert SpectrumType.EXAFS == "EXAFS"
        assert "EXAFS" in SpectrumType.__members__

    def test_edge_type_enum(self):
        """Test EdgeType enum."""
        assert EdgeType.K == "K"
        assert EdgeType.L1 == "L1"
        assert EdgeType.L2 == "L2"
        assert EdgeType.L3 == "L3"
        assert EdgeType.M1 == "M1"

    def test_window_type_enum(self):
        """Test WindowType enum."""
        assert WindowType.HANNING == "hanning"
        assert WindowType.PARZEN == "parzen"
        assert WindowType.WELCH == "welch"
        assert WindowType.GAUSSIAN == "gaussian"
        assert WindowType.SINE == "sine"
        assert WindowType.KAISER == "kaiser"


class TestFeffConfig:
    """Test FeffConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FeffConfig()
        assert config.spectrum_type == "EXAFS"
        assert config.edge == "K"
        assert config.radius == 4.0
        assert config.kmin == 2.0
        assert config.kmax == 12.0
        assert config.kweight == 2
        assert config.dk == 1.0
        assert config.window == WindowType.HANNING
        assert config.parallel is False
        assert config.force_recalculate is False
        assert config.cleanup_feff_files is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FeffConfig(
            edge="L3",
            radius=6.0,
            spectrum_type="EXAFS",
            kmin=3.0,
            kmax=15.0,
            kweight=3,
            parallel=True,
        )
        assert config.edge == "L3"
        assert config.radius == 6.0
        assert config.spectrum_type == "EXAFS"
        assert config.kmin == 3.0
        assert config.kmax == 15.0
        assert config.kweight == 3
        assert config.parallel is True

    def test_validation_invalid_radius(self):
        """Test validation of invalid radius values."""
        with pytest.raises(ValueError, match="Radius must be positive"):
            FeffConfig(radius=-1.0)

    def test_validation_invalid_energy_range(self):
        """Test validation of invalid energy range."""
        with pytest.raises(ValueError, match="kmin.*must be less than kmax"):
            FeffConfig(kmin=15.0, kmax=10.0)

        with pytest.raises(ValueError, match="kmin must be positive"):
            FeffConfig(kmin=-1.0)

    def test_validation_invalid_dk(self):
        """Test validation of invalid dk values."""
        with pytest.raises(ValueError, match="dk must be positive"):
            FeffConfig(dk=-0.5)

    def test_validation_invalid_n_workers(self):
        """Test validation of invalid n_workers."""
        with pytest.raises(ValueError, match="Invalid n_workers"):
            FeffConfig(n_workers=-1)

    def test_validation_invalid_sample_interval(self):
        """Test validation of invalid sample_interval."""
        with pytest.raises(ValueError, match="sample_interval must be >= 1"):
            FeffConfig(sample_interval=0)

    def test_fourier_params_property(self):
        """Test fourier_params property."""
        config = FeffConfig(kmin=3.0, kmax=15.0, kweight=2, dk=0.5)
        params = config.fourier_params

        assert isinstance(params, dict)
        assert params["kmin"] == 3.0
        assert params["kmax"] == 15.0
        assert params["kweight"] == 2
        assert params["dk"] == 0.5
        assert params["window"] == "hanning"

        # Check None values are filtered out
        assert "dk2" not in params  # Should be None by default
        assert "nfft" not in params  # Should be None by default

    def test_feff_params_property(self):
        """Test feff_params property."""
        config = FeffConfig(
            spectrum_type="EXAFS",
            edge="L3",
            radius=6.0,
            print="1 0 0 0 0 3",
            s02=0.9,
        )
        params = config.feff_params

        assert isinstance(params, dict)
        assert params["spectrum_type"] == "EXAFS"
        assert params["edge"] == "L3"
        assert params["radius"] == 6.0
        assert "PRINT" in params
        assert "S02" in params
        assert params["PRINT"] == "1 0 0 0 0 3"
        assert params["S02"] == "0.9"

    def test_from_preset(self):
        """Test creation from preset."""
        config = FeffConfig.from_preset("quick")
        assert config.spectrum_type == "EXAFS"
        assert config.edge == "K"
        assert config.radius == 4.0

        config = FeffConfig.from_preset("publication")
        assert config.radius == 8.0

    def test_from_preset_invalid(self):
        """Test creation from invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            FeffConfig.from_preset("nonexistent")

    @patch("larch_cli_wrapper.feff_utils.YAML_AVAILABLE", True)
    @patch("larch_cli_wrapper.feff_utils.yaml")
    def test_from_yaml(self, mock_yaml):
        """Test loading from YAML file."""
        mock_yaml.safe_load.return_value = {
            "spectrum_type": "EXAFS",
            "edge": "L3",
            "radius": 6.0,
        }

        # Use mock_open which is designed for this
        from unittest.mock import mock_open

        with patch("builtins.open", mock_open()):
            config = FeffConfig.from_yaml(Path("test.yaml"))

        assert config.spectrum_type == SpectrumType.EXAFS
        assert config.edge == EdgeType.L3
        assert config.radius == 6.0

    def test_from_yaml_unavailable(self):
        """Test YAML loading when PyYAML not available."""
        with patch("larch_cli_wrapper.feff_utils.YAML_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyYAML required"):
                FeffConfig.from_yaml(Path("test.yaml"))

    def test_as_dict(self):
        """Test conversion to dictionary."""
        config = FeffConfig(edge="L3", radius=6.0)
        config_dict = config.as_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["edge"] == "L3"
        assert config_dict["radius"] == 6.0
        assert "spectrum_type" in config_dict

    def test_repr_json(self):
        """Test JSON representation."""
        config = FeffConfig(edge="L3", radius=6.0)
        json_str = config.__repr_json__()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["edge"] == "L3"
        assert parsed["radius"] == 6.0


class TestAbsorberValidation:
    """Test absorber validation functions."""

    def test_validate_absorber_string(self, sample_atoms):
        """Test validate_absorber with string input."""
        result = validate_absorber(sample_atoms, "Fe")
        assert result == "Fe"

        result = validate_absorber(sample_atoms, "fe")  # lowercase
        assert result == "Fe"

    def test_validate_absorber_index(self, sample_atoms):
        """Test validate_absorber with index input."""
        result = validate_absorber(sample_atoms, 0)  # First Fe atom
        assert result == "Fe"

        result = validate_absorber(sample_atoms, 2)  # First O atom
        assert result == "O"

    def test_validate_absorber_invalid_element(self, sample_atoms):
        """Test validate_absorber with invalid element."""
        with pytest.raises(ValueError, match="Absorber element Cu not found"):
            validate_absorber(sample_atoms, "Cu")

    def test_validate_absorber_invalid_index(self, sample_atoms):
        """Test validate_absorber with invalid index."""
        with pytest.raises(ValueError, match="Absorber index .* out of range"):
            validate_absorber(sample_atoms, 10)

    def test_normalize_absorbers_string(self, sample_atoms):
        """Test normalize_absorbers with string input."""
        indices = normalize_absorbers(sample_atoms, "Fe")
        assert isinstance(indices, list)
        assert len(indices) == 2  # Two Fe atoms
        assert all(isinstance(i, int) for i in indices)

    def test_normalize_absorbers_int(self, sample_atoms):
        """Test normalize_absorbers with int input."""
        indices = normalize_absorbers(sample_atoms, 1)
        assert indices == [1]

    def test_normalize_absorbers_list_int(self, sample_atoms):
        """Test normalize_absorbers with list of ints."""
        indices = normalize_absorbers(sample_atoms, [0, 1])
        assert indices == [0, 1]

    def test_normalize_absorbers_list_str(self, sample_atoms):
        """Test normalize_absorbers with list of strings."""
        indices = normalize_absorbers(sample_atoms, ["Fe", "O"])
        assert len(indices) == 5  # 2 Fe + 3 O atoms
        # Check that duplicates are removed and order is preserved
        assert len(set(indices)) == len(indices)

    def test_normalize_absorbers_invalid_element(self, sample_atoms):
        """Test normalize_absorbers with invalid element."""
        with pytest.raises(ValueError, match="Element Cu not found"):
            normalize_absorbers(sample_atoms, "Cu")

    def test_normalize_absorbers_invalid_index(self, sample_atoms):
        """Test normalize_absorbers with invalid index."""
        with pytest.raises(ValueError, match="Absorber index .* out of range"):
            normalize_absorbers(sample_atoms, 10)

    def test_normalize_absorbers_empty_list(self, sample_atoms):
        """Test normalize_absorbers with empty list."""
        with pytest.raises(ValueError, match="Absorber list cannot be empty"):
            normalize_absorbers(sample_atoms, [])

    def test_get_absorber_element_from_index(self, sample_atoms):
        """Test get_absorber_element_from_index function."""
        assert get_absorber_element_from_index(sample_atoms, 0) == "Fe"
        assert get_absorber_element_from_index(sample_atoms, 2) == "O"

    def test_get_absorber_element_invalid_index(self, sample_atoms):
        """Test get_absorber_element_from_index with invalid index."""
        with pytest.raises(ValueError, match="Absorber index .* out of range"):
            get_absorber_element_from_index(sample_atoms, 10)

    def test_validate_absorber_indices(self, sample_atoms):
        """Test validate_absorber_indices function."""
        # Valid indices for Fe atoms
        element = validate_absorber_indices(sample_atoms, [0, 1])
        assert element == "Fe"

    def test_validate_absorber_indices_different_elements(self, sample_atoms):
        """Test validate_absorber_indices with different elements."""
        with pytest.raises(ValueError, match="must all correspond to the same element"):
            validate_absorber_indices(sample_atoms, [0, 2])  # Fe and O

    def test_validate_absorber_indices_empty(self, sample_atoms):
        """Test validate_absorber_indices with empty list."""
        with pytest.raises(ValueError, match="At least one absorber index"):
            validate_absorber_indices(sample_atoms, [])


class TestSpectraAveraging:
    """Test chi spectra averaging functionality."""

    def test_average_chi_spectra_single_spectrum(self):
        """Test averaging with single spectrum."""
        k = np.linspace(2, 14, 50)
        chi = np.sin(k)

        chi_avg, k_avg = average_chi_spectra([k], [chi])

        np.testing.assert_array_equal(chi_avg, chi)
        np.testing.assert_array_equal(k_avg, k)

    def test_average_chi_spectra_multiple_same_grid(self):
        """Test averaging multiple spectra on same k-grid."""
        k = np.linspace(2, 14, 50)
        chi1 = np.sin(k)
        chi2 = np.cos(k)

        chi_avg, k_avg = average_chi_spectra([k, k], [chi1, chi2])

        expected_avg = (chi1 + chi2) / 2
        np.testing.assert_array_almost_equal(chi_avg, expected_avg)
        np.testing.assert_array_equal(k_avg, k)

    def test_average_chi_spectra_different_grids(self):
        """Test averaging spectra on different k-grids."""
        k1 = np.linspace(2, 14, 50)
        k2 = np.linspace(3, 13, 40)
        chi1 = np.sin(k1)
        chi2 = np.cos(k2)

        chi_avg, k_avg = average_chi_spectra([k1, k2], [chi1, chi2])

        # Should use first spectrum's grid
        np.testing.assert_array_equal(k_avg, k1)
        assert len(chi_avg) == len(k1)

    def test_average_chi_spectra_common_range(self):
        """Test averaging with common range restriction."""
        k1 = np.linspace(2, 14, 50)
        k2 = np.linspace(4, 12, 30)
        chi1 = np.sin(k1)
        chi2 = np.cos(k2)

        chi_avg, k_avg = average_chi_spectra(
            [k1, k2], [chi1, chi2], restrict_to_common_range=True
        )

        # Should restrict to overlapping range [4, 12]
        assert k_avg[0] >= 4.0
        assert k_avg[-1] <= 12.0
        assert len(k_avg) == min(len(k1), len(k2))

    def test_average_chi_spectra_with_weights(self):
        """Test weighted averaging."""
        k = np.linspace(2, 14, 50)
        chi1 = np.ones_like(k)
        chi2 = np.zeros_like(k)

        # Weight heavily toward first spectrum
        chi_avg, k_avg = average_chi_spectra([k, k], [chi1, chi2], weights=[0.8, 0.2])

        # Should be closer to chi1
        expected = 0.8 * chi1 + 0.2 * chi2
        np.testing.assert_array_almost_equal(chi_avg, expected)

    def test_average_chi_spectra_complex_data(self):
        """Test averaging with complex chi data."""
        k = np.linspace(2, 14, 50)
        chi1 = np.sin(k) + 1j * np.cos(k)
        chi2 = np.cos(k) + 1j * np.sin(k)

        chi_avg, k_avg = average_chi_spectra([k, k], [chi1, chi2])

        expected_avg = (chi1 + chi2) / 2
        np.testing.assert_array_almost_equal(chi_avg, expected_avg)

    def test_average_chi_spectra_empty_input(self):
        """Test averaging with empty inputs."""
        with pytest.raises(ValueError, match="Empty input arrays"):
            average_chi_spectra([], [])

    def test_average_chi_spectra_mismatched_lengths(self):
        """Test averaging with mismatched k and chi array counts."""
        k = np.linspace(2, 14, 50)
        chi = np.sin(k)

        with pytest.raises(ValueError, match="Number of k and chi arrays must match"):
            average_chi_spectra([k], [chi, chi])

    def test_average_chi_spectra_no_overlap(self):
        """Test averaging with no overlapping k-range."""
        k1 = np.linspace(2, 5, 20)
        k2 = np.linspace(8, 12, 20)
        chi1 = np.sin(k1)
        chi2 = np.cos(k2)

        with pytest.raises(ValueError, match="No overlapping k-range"):
            average_chi_spectra([k1, k2], [chi1, chi2], restrict_to_common_range=True)


class TestFeffInputGeneration:
    """Test FEFF input file generation."""

    @patch("larch_cli_wrapper.feff_utils.AseAtomsAdaptor")
    @patch("larch_cli_wrapper.feff_utils.MPEXAFSSet")
    def test_generate_pymatgen_input(
        self, mock_mpexafs, mock_adaptor, sample_atoms, sample_config
    ):
        """Test generate_pymatgen_input function."""
        # Mock the pymatgen structure conversion
        mock_structure = Mock()
        mock_adaptor.return_value.get_structure.return_value = mock_structure

        # Mock the FEFF set
        mock_feff_set = Mock()
        mock_mpexafs.return_value = mock_feff_set

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = generate_pymatgen_input(sample_atoms, 0, output_dir, sample_config)

            # Check that the right calls were made
            mock_adaptor.assert_called_once()
            mock_mpexafs.assert_called_once()
            mock_feff_set.write_input.assert_called_once_with(str(output_dir))

            # Check return value
            assert result == output_dir / "feff.inp"

    def test_generate_pymatgen_input_invalid_absorber(
        self, sample_atoms, sample_config
    ):
        """Test generate_pymatgen_input with invalid absorber index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            with pytest.raises(ValueError, match="Absorber index .* out of range"):
                generate_pymatgen_input(sample_atoms, 10, output_dir, sample_config)

    @patch("larch_cli_wrapper.feff_utils.generate_pymatgen_input")
    def test_generate_multi_site_feff_inputs(
        self, mock_generate, sample_atoms, sample_config
    ):
        """Test generate_multi_site_feff_inputs function."""

        # Mock the single-site generation to return proper path and create directories
        def mock_generate_func(atoms, idx, output_dir, config):
            output_dir.mkdir(parents=True, exist_ok=True)
            feff_inp = output_dir / "feff.inp"
            feff_inp.write_text("# Mock feff.inp")
            return feff_inp

        mock_generate.side_effect = mock_generate_func

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            absorber_indices = [0, 1]

            results = generate_multi_site_feff_inputs(
                sample_atoms, absorber_indices, base_dir, sample_config
            )

            # Should have called generate_pymatgen_input for each site
            assert mock_generate.call_count == 2
            assert len(results) == 2

            # Check that site directories were created
            assert (base_dir / "site_0000").exists()
            assert (base_dir / "site_0001").exists()

            # Check that feff.inp files were created
            assert (base_dir / "site_0000" / "feff.inp").exists()
            assert (base_dir / "site_0001" / "feff.inp").exists()

    def test_generate_multi_site_feff_inputs_empty_indices(
        self, sample_atoms, sample_config
    ):
        """Test generate_multi_site_feff_inputs with empty indices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            with pytest.raises(ValueError, match="At least one absorber index"):
                generate_multi_site_feff_inputs(
                    sample_atoms, [], base_dir, sample_config
                )


class TestFeffCalculation:
    """Test FEFF calculation execution."""

    def test_run_feff_calculation_missing_input(self):
        """Test run_feff_calculation with missing input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir)

            with pytest.raises(FileNotFoundError, match="FEFF input file .* not found"):
                run_feff_calculation(feff_dir, verbose=False)

    @patch("subprocess.run")
    def test_run_feff_calculation_success(self, mock_run):
        """Test successful FEFF calculation via subprocess."""

        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir)

            # Create required input file
            input_file = feff_dir / "feff.inp"
            input_file.write_text("TITLE Test FEFF calculation\n")

            def mock_run_side_effect(*args, **kwargs):
                # Simulate feff8l creating expected output
                chi_file = feff_dir / "chi.dat"
                chi_file.write_text("# Mock chi data\n3.0 0.1\n4.0 0.2\n")
                return subprocess.CompletedProcess(args=args[0], returncode=0)

            mock_run.side_effect = mock_run_side_effect

            result = run_feff_calculation(feff_dir, verbose=False)

            assert result is True
            mock_run.assert_called_once()
            # chi.dat should persist when cleanup keeps essential outputs
            assert (feff_dir / "chi.dat").exists()

    @patch("larch_cli_wrapper.feff_utils.run_feff_calculation")
    def test_run_multi_site_feff_calculations(self, mock_run_feff):
        """Test run_multi_site_feff_calculations function."""
        mock_run_feff.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            # Create mock input files
            input_files = []
            for i in range(3):
                site_dir = base_dir / f"site_{i:04d}"
                site_dir.mkdir()
                input_file = site_dir / "feff.inp"
                input_file.write_text(f"TITLE Site {i}\n")
                input_files.append(input_file)

            results = run_multi_site_feff_calculations(
                input_files, cleanup=False, parallel=False
            )

            assert len(results) == 3
            assert all(success for feff_dir, success in results)
            assert mock_run_feff.call_count == 3


class TestFeffOutput:
    """Test FEFF output reading and processing."""

    def test_read_feff_output_file_not_found(self):
        """Test read_feff_output when chi.dat doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir)

            with pytest.raises(FileNotFoundError, match="FEFF output .* not found"):
                read_feff_output(feff_dir)

    @patch("larch_cli_wrapper.feff_utils.read_ascii")
    def test_read_feff_output_larch_success(self, mock_read_ascii):
        """Test successful read_feff_output with larch."""

        # Create a simple object with only the attributes we want
        class MockData:
            def __init__(self):
                self.chi = np.array([0.1 + 0.2j, 0.3 + 0.4j])
                self.k = np.array([3.0, 4.0])

        mock_data = MockData()
        mock_read_ascii.return_value = mock_data

        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir)
            chi_file = feff_dir / "chi.dat"
            chi_file.write_text("# Mock chi.dat\n3.0 0.1 0.2\n4.0 0.3 0.4\n")

            chi, k = read_feff_output(feff_dir)

            np.testing.assert_array_equal(chi, mock_data.chi)
            np.testing.assert_array_equal(k, mock_data.k)

    @patch("larch_cli_wrapper.feff_utils.read_ascii")
    def test_read_feff_output_mag_phase_format(self, mock_read_ascii):
        """Test read_feff_output with FEFF mag/phase format."""

        # Create mock data for FEFF format with mag and phase
        class MockFeffData:
            def __init__(self):
                self.k = np.array([3.0, 4.0])
                self.mag = np.array([0.1118, 0.2236])  # |0.1+0.05j|, |0.2+0.1j|
                self.phase = np.array([0.4636, 0.4636])  # angle of complex numbers

        mock_data = MockFeffData()
        mock_read_ascii.return_value = mock_data

        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir)
            chi_file = feff_dir / "chi.dat"
            chi_file.write_text(
                "# FEFF format\n3.0 0.1 0.1118 0.4636\n4.0 0.2 0.2236 0.4636\n"
            )

            chi, k = read_feff_output(feff_dir)

            # Should reconstruct complex chi from mag * exp(1j * phase)
            expected_chi = mock_data.mag * np.exp(1j * mock_data.phase)
            np.testing.assert_array_equal(k, mock_data.k)
            np.testing.assert_array_almost_equal(chi, expected_chi)

    def test_cleanup_feff_output(self):
        """Test cleanup_feff_output function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir)

            # Create some FEFF output files
            (feff_dir / "feff0001.dat").write_text("test")
            (feff_dir / "feff0002.dat").write_text("test")
            (feff_dir / "chi.dat").write_text("test")
            (feff_dir / "feff.inp").write_text("test")

            files_removed = cleanup_feff_output(feff_dir, keep_essential=True)

            # Should remove numbered feff files but keep essential ones
            assert files_removed == 2
            assert not (feff_dir / "feff0001.dat").exists()
            assert not (feff_dir / "feff0002.dat").exists()
            assert (feff_dir / "chi.dat").exists()
            assert (feff_dir / "feff.inp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
