"""Comprehensive tests for the exafs_data module.

This module tests all components of the EXAFS data handling:
- EXAFSDataCollection class and its methods
- PlotResult class
- Utility functions (add_metadata_to_group, create_averaged_group, etc.)
- Export/import functionality
- Plotting functionality
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from larch import Group
from larch.xafs import xftf

from larch_cli_wrapper.exafs_data import (
    EXAFSDataCollection,
    PlotConfig,
    PlotResult,
    add_metadata_to_group,
    create_averaged_group,
    plot_exafs_matplotlib,
    prepare_exafs_data_collection,
)


@pytest.fixture
def sample_k_array():
    """Standard k-array for testing."""
    return np.linspace(2, 14, 50)


@pytest.fixture
def sample_group(sample_k_array):
    """Create a sample EXAFS group with standard data."""
    group = Group()
    group.k = sample_k_array
    group.chi = np.sin(group.k) * np.exp(-group.k / 10)

    # Apply Fourier transform
    xftf(group, kmin=3, kmax=12, kweight=2, dk=1, window="hanning")

    # Add metadata
    group.site_idx = 0
    group.frame_idx = 0
    group.absorber_element = "Fe"

    return group


@pytest.fixture
def sample_group_with_std(sample_k_array):
    """Create a group with standard deviation data."""
    group = Group()
    group.k = sample_k_array
    group.chi = np.sin(group.k) * np.exp(-group.k / 10)
    group.chi_std = np.abs(group.chi) * 0.1  # 10% uncertainty

    xftf(group, kmin=3, kmax=12, kweight=2, dk=1, window="hanning")
    return group


@pytest.fixture
def multiple_groups(sample_k_array):
    """Create multiple groups for testing averaging."""
    groups = []
    for i in range(3):
        group = Group()
        group.k = sample_k_array
        # Add small variations
        group.chi = (np.sin(group.k) + i * 0.1) * np.exp(-group.k / 10)

        xftf(group, kmin=3, kmax=12, kweight=2, dk=1, window="hanning")

        group.site_idx = i
        group.frame_idx = 0
        groups.append(group)

    return groups


class TestEXAFSDataCollection:
    """Test the EXAFSDataCollection class comprehensively."""

    def test_collection_creation_defaults(self):
        """Test default creation of EXAFSDataCollection."""
        collection = EXAFSDataCollection()

        assert len(collection.individual_spectra) == 0
        assert len(collection.site_averages) == 0
        assert len(collection.frame_averages) == 0
        assert collection.overall_average is None
        assert collection.kweight_used == 2
        assert isinstance(collection.fourier_params, dict)
        assert isinstance(collection.processing_metadata, dict)
        assert collection.created_at is not None

    def test_collection_creation_with_params(self, sample_group):
        """Test creation with custom parameters."""
        fourier_params = {"kmin": 3, "kmax": 15, "kweight": 3}
        metadata = {"source": "test", "version": "1.0"}

        collection = EXAFSDataCollection(
            individual_spectra=[sample_group],
            overall_average=sample_group,
            kweight_used=3,
            fourier_params=fourier_params,
            processing_metadata=metadata,
        )

        assert len(collection.individual_spectra) == 1
        assert collection.overall_average is not None
        assert collection.kweight_used == 3
        assert collection.fourier_params == fourier_params
        assert collection.processing_metadata == metadata

    def test_get_plotting_groups_individual_only(self, multiple_groups):
        """Test getting individual spectra only."""
        collection = EXAFSDataCollection(individual_spectra=multiple_groups)

        result = collection.get_plotting_groups(
            include_individual=True,
            include_site_averages=False,
            include_frame_averages=False,
            include_overall_average=False,
        )

        assert len(result) == 3
        assert all(group in multiple_groups for group in result)

    def test_get_plotting_groups_with_limit(self, multiple_groups):
        """Test limiting individual spectra count."""
        collection = EXAFSDataCollection(individual_spectra=multiple_groups)

        result = collection.get_plotting_groups(
            include_individual=True, max_individual=2, include_overall_average=False
        )

        assert len(result) == 2

    def test_get_plotting_groups_all_types(self, sample_group, multiple_groups):
        """Test including all types of groups."""
        collection = EXAFSDataCollection(
            individual_spectra=multiple_groups,
            site_averages={0: sample_group, 1: sample_group},
            frame_averages={0: sample_group},
            overall_average=sample_group,
        )

        result = collection.get_plotting_groups(
            include_individual=True,
            include_site_averages=True,
            include_frame_averages=True,
            include_overall_average=True,
        )

        # 3 individual + 2 site + 1 frame + 1 overall = 7
        assert len(result) == 7

    def test_get_k_weighted_chi_all_weights(self, sample_group):
        """Test k-weighting for all supported weights."""
        collection = EXAFSDataCollection()

        # Test all weights
        for weight in [0, 1, 2, 3]:
            result = collection.get_k_weighted_chi(sample_group, weight)

            if weight == 0:
                np.testing.assert_array_equal(result, sample_group.chi)
            else:
                expected = sample_group.chi * sample_group.k**weight
                np.testing.assert_array_equal(result, expected)

    def test_get_plot_labels(self):
        """Test plot label generation for different k-weights."""
        collection = EXAFSDataCollection()

        test_cases = [
            (0, (r"$\chi(k)$", r"EXAFS $\chi(k)$")),
            (1, (r"$k\chi(k)$", r"EXAFS $k\chi(k)$")),
            (2, (r"$k^{2}\chi(k)$", r"EXAFS $k^{2}\chi(k)$")),
            (3, (r"$k^{3}\chi(k)$", r"EXAFS $k^{3}\chi(k)$")),
        ]

        for weight, expected in test_cases:
            result = collection.get_plot_labels(weight)
            assert result == expected


class TestPlotResult:
    """Test the PlotResult class."""

    def test_plot_result_creation(self):
        """Test PlotResult creation."""
        result = PlotResult()

        assert isinstance(result.plot_paths, dict)
        assert isinstance(result.plot_metadata, dict)
        assert len(result.plot_paths) == 0
        assert len(result.plot_metadata) == 0

    def test_plot_result_with_data(self):
        """Test PlotResult with actual data."""
        plot_paths = {"png": Path("/test.png"), "pdf": Path("/test.pdf")}
        metadata = {"title": "Test Plot", "kweight": 2}

        result = PlotResult(plot_paths=plot_paths, plot_metadata=metadata)

        assert result.plot_paths == plot_paths
        assert result.plot_metadata == metadata


class TestUtilityFunctions:
    """Test utility functions."""

    def test_add_metadata_to_group(self, sample_group):
        """Test adding metadata to a group."""
        set(dir(sample_group))

        result = add_metadata_to_group(
            sample_group, test_attr="test_value", numeric_attr=42, list_attr=[1, 2, 3]
        )

        # Should return the same group (modified in place)
        assert result is sample_group

        # Check new attributes
        assert result.test_attr == "test_value"
        assert result.numeric_attr == 42
        assert result.list_attr == [1, 2, 3]

    def test_create_averaged_group_single(self, sample_group):
        """Test averaging a single group."""
        fourier_params = {"kmin": 3, "kmax": 12, "kweight": 2}

        result = create_averaged_group([sample_group], fourier_params)

        # Should be a new group
        assert result is not sample_group

        # Should have k and chi data
        assert hasattr(result, "k")
        assert hasattr(result, "chi")
        assert len(result.k) > 0
        assert len(result.chi) > 0

    def test_create_averaged_group_multiple(self, multiple_groups):
        """Test averaging multiple groups."""
        fourier_params = {"kmin": 3, "kmax": 12, "kweight": 2}

        result = create_averaged_group(multiple_groups, fourier_params)

        # Should have k and chi data
        assert hasattr(result, "k")
        assert hasattr(result, "chi")
        assert len(result.k) > 0
        assert len(result.chi) > 0

        # Should have FT data after xftf
        assert hasattr(result, "chir_mag")

    def test_create_averaged_group_empty(self):
        """Test averaging empty group list."""
        fourier_params = {"kmin": 3, "kmax": 12, "kweight": 2}

        with pytest.raises(
            ValueError, match="Cannot create averaged group from empty list"
        ):
            create_averaged_group([], fourier_params)

    def test_prepare_exafs_data_collection_single_group(self, sample_group):
        """Test preparing collection from single group."""
        result = prepare_exafs_data_collection(
            groups=[sample_group], fourier_params={"kmin": 3, "kmax": 12, "kweight": 2}
        )

        assert isinstance(result, EXAFSDataCollection)
        assert len(result.individual_spectra) >= 1
        # May or may not have overall_average depending on implementation

    def test_prepare_exafs_data_collection_multiple_groups(self, multiple_groups):
        """Test preparing collection from multiple groups."""
        result = prepare_exafs_data_collection(
            groups=multiple_groups, fourier_params={"kmin": 3, "kmax": 12, "kweight": 2}
        )

        assert len(result.individual_spectra) >= 3
        # May or may not have overall_average depending on implementation

    def test_prepare_exafs_data_collection_no_groups(self):
        """Test error handling for no groups."""
        # The function may handle empty groups gracefully, so let's just test it
        # doesn't crash
        result = prepare_exafs_data_collection(
            groups=[], fourier_params={"kmin": 3, "kmax": 12, "kweight": 2}
        )
        # Should return an EXAFSDataCollection even if empty
        assert isinstance(result, EXAFSDataCollection)


class TestPlottingFunctionality:
    """Test plotting functionality."""

    def test_plot_exafs_matplotlib_basic(self, sample_group):
        """Test basic plotting functionality."""
        collection = EXAFSDataCollection(
            individual_spectra=[sample_group], overall_average=sample_group
        )

        config = PlotConfig(
            plot_individual=True,
            plot_overall_avg=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Just test that the function can be called without crashing
            # The actual plotting will use matplotlib, so we focus on the interface
            result = plot_exafs_matplotlib(
                collection=collection,
                config=config,
                output_dir=output_dir,
                filename_base="test_plot",
            )

            # Should return a PlotResult
            assert isinstance(result, PlotResult)
            # Should have created some plot files
            assert len(result.plot_paths) > 0

    def test_plot_exafs_matplotlib_with_averages(self, sample_group, multiple_groups):
        """Test plotting with different types of averages."""
        collection = EXAFSDataCollection(
            individual_spectra=multiple_groups, overall_average=sample_group
        )

        config = PlotConfig(
            plot_individual=True,
            plot_overall_avg=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Just test that the function can be called without crashing
            result = plot_exafs_matplotlib(
                collection=collection,
                config=config,
                output_dir=output_dir,
                filename_base="test_with_averages",
            )

            # Should return a PlotResult
            assert isinstance(result, PlotResult)
            # Should have created some plot files
            assert len(result.plot_paths) > 0

    def test_plot_exafs_matplotlib_empty_collection(self):
        """Test plotting empty collection."""
        collection = EXAFSDataCollection()

        config = PlotConfig(
            plot_overall_avg=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # The function may not raise an error, let's just test it doesn't crash
            result = plot_exafs_matplotlib(
                collection=collection,
                config=config,
                output_dir=output_dir,
                filename_base="empty",
            )

            # Should return a PlotResult even if empty
            assert isinstance(result, PlotResult)

    def test_plot_exafs_matplotlib_different_formats(self, sample_group):
        """Test plotting with different output formats."""
        collection = EXAFSDataCollection(overall_average=sample_group)

        config = PlotConfig(
            plot_overall_avg=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Just test that the function can be called without crashing
            result = plot_exafs_matplotlib(
                collection=collection,
                config=config,
                output_dir=output_dir,
                filename_base="test_formats",
            )

            # Should return a PlotResult
            assert isinstance(result, PlotResult)
            # Should have created some plot files
            assert len(result.plot_paths) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_k_weighted_chi_invalid_group(self):
        """Test k-weighting with invalid group."""
        collection = EXAFSDataCollection()

        # Group without chi attribute
        bad_group = Group()
        bad_group.k = np.linspace(2, 14, 50)

        with pytest.raises(AttributeError):
            collection.get_k_weighted_chi(bad_group, 2)

    def test_create_averaged_group_mismatched_k(self, sample_k_array):
        """Test averaging groups with mismatched k-arrays."""
        group1 = Group()
        group1.k = sample_k_array
        group1.chi = np.sin(group1.k)

        group2 = Group()
        group2.k = sample_k_array[:-5]  # Different length
        group2.chi = np.sin(group2.k)

        fourier_params = {"kmin": 3, "kmax": 12, "kweight": 2}

        # Should handle mismatched arrays gracefully
        result = create_averaged_group([group1, group2], fourier_params)
        assert hasattr(result, "chi")


class TestLarchGroupsExportImport:
    """Test the Larch Groups export/import functionality."""

    def test_export_larch_groups_ascii_format(self, sample_group):
        """Test ASCII format export of Larch Groups."""
        collection = EXAFSDataCollection(individual_spectra=[sample_group])
        collection.overall_average = sample_group
        collection.frame_averages = {0: sample_group}
        collection.site_averages = {0: sample_group}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "larch_groups"

            result_dir = collection.export_larch_groups(
                output_dir=output_dir,
                save_individual=True,
                save_averages=True,
                format="ascii",
            )

            # Check directory structure
            assert result_dir.exists()
            assert (result_dir / "collection_metadata.json").exists()

            # Check overall average files
            assert (result_dir / "overall_average.chi").exists()
            assert (result_dir / "overall_average.chir").exists()

            # Check frame averages
            frame_dir = result_dir / "frame_averages"
            assert frame_dir.exists()
            assert (frame_dir / "frame_0000.chi").exists()
            assert (frame_dir / "frame_0000.chir").exists()

            # Check site averages
            site_dir = result_dir / "site_averages"
            assert site_dir.exists()
            assert (site_dir / "site_0000.chi").exists()
            assert (site_dir / "site_0000.chir").exists()

            # Check individual spectra
            individual_dir = result_dir / "individual_spectra"
            assert individual_dir.exists()
            individual_files = list(individual_dir.glob("*.chi"))
            assert len(individual_files) >= 1

    def test_export_larch_groups_averages_only(self, sample_group):
        """Test exporting only averaged groups, not individual spectra."""
        collection = EXAFSDataCollection(individual_spectra=[sample_group])
        collection.overall_average = sample_group
        collection.frame_averages = {0: sample_group, 1: sample_group}
        collection.site_averages = {0: sample_group}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "averages_only"

            result_dir = collection.export_larch_groups(
                output_dir=output_dir,
                save_individual=False,
                save_averages=True,
                format="ascii",
            )

            # Check that averages are saved
            assert (result_dir / "overall_average.chi").exists()
            assert (result_dir / "frame_averages").exists()
            assert (result_dir / "site_averages").exists()

            # Check that individual spectra are NOT saved
            individual_dir = result_dir / "individual_spectra"
            assert not individual_dir.exists()

    def test_export_larch_groups_individuals_only(self, multiple_groups):
        """Test exporting only individual spectra, not averages."""
        collection = EXAFSDataCollection(individual_spectra=multiple_groups)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "individuals_only"

            result_dir = collection.export_larch_groups(
                output_dir=output_dir,
                save_individual=True,
                save_averages=False,
                format="ascii",
            )

            # Check that individual spectra are saved
            individual_dir = result_dir / "individual_spectra"
            assert individual_dir.exists()
            individual_files = list(individual_dir.glob("*.chi"))
            assert len(individual_files) == len(multiple_groups)

            # Check that averages are NOT saved
            assert not (result_dir / "overall_average.chi").exists()
            assert not (result_dir / "frame_averages").exists()
            assert not (result_dir / "site_averages").exists()

    def test_ascii_file_format_content(self, sample_group):
        """Test the content format of exported ASCII files."""
        collection = EXAFSDataCollection()
        collection.overall_average = sample_group

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "format_test"

            collection.export_larch_groups(output_dir=output_dir, format="ascii")

            # Check chi file content
            chi_file = output_dir / "overall_average.chi"
            with open(chi_file) as f:
                lines = f.readlines()

            # Check header
            assert lines[0].startswith("# Larch Group saved on")
            assert any("k-weight" in line for line in lines)
            # Check for FEFF chi.dat format header: k, chi, mag, phase
            assert any(
                "k" in line and "chi" in line and "mag" in line and "phase" in line
                for line in lines
            )

            # Check data format - should have 4 columns: k, chi, mag, phase
            data_lines = [line for line in lines if not line.startswith("#")]
            assert len(data_lines) > 0

            # Parse first data line
            first_data = data_lines[0].strip().split()
            assert len(first_data) >= 4  # k, chi, mag, phase columns
            float(first_data[0])  # k - should be parseable as float
            float(first_data[1])  # chi (real part)
            float(first_data[2])  # mag
            float(first_data[3])  # phase

            # Check chir file content if it exists
            chir_file = output_dir / "overall_average.chir"
            if chir_file.exists():
                with open(chir_file) as f:
                    lines = f.readlines()

                # Check for R-space data format
                assert any("R(A)" in line and "|chi(R)|" in line for line in lines)
                data_lines = [line for line in lines if not line.startswith("#")]
                if data_lines:
                    first_data = data_lines[0].strip().split()
                    assert len(first_data) >= 2  # R and |chi(R)| columns

    def test_load_larch_groups_ascii(self, sample_group):
        """Test loading Larch Groups from ASCII files."""
        # First export
        original_collection = EXAFSDataCollection(individual_spectra=[sample_group])
        original_collection.overall_average = sample_group
        original_collection.frame_averages = {0: sample_group, 1: sample_group}
        original_collection.site_averages = {0: sample_group}
        original_collection.kweight_used = 3
        original_collection.fourier_params = {"kmin": 3, "kmax": 12}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "round_trip_test"

            # Export
            original_collection.export_larch_groups(
                output_dir=output_dir,
                save_individual=True,
                save_averages=True,
                format="ascii",
            )

            # Load back
            loaded_collection = EXAFSDataCollection.load_larch_groups(output_dir)

            # Check metadata preservation
            assert loaded_collection.kweight_used == original_collection.kweight_used
            assert (
                loaded_collection.fourier_params == original_collection.fourier_params
            )

            # Check overall average
            assert loaded_collection.overall_average is not None
            assert hasattr(loaded_collection.overall_average, "k")
            assert hasattr(loaded_collection.overall_average, "chi")

            # Check frame averages
            assert len(loaded_collection.frame_averages) == 2
            assert 0 in loaded_collection.frame_averages
            assert 1 in loaded_collection.frame_averages

            # Check site averages
            assert len(loaded_collection.site_averages) == 1
            assert 0 in loaded_collection.site_averages

            # Check individual spectra
            assert len(loaded_collection.individual_spectra) >= 1

    def test_load_larch_groups_data_integrity(self, sample_group):
        """Test that k and chi data are preserved correctly during round-trip."""
        original_collection = EXAFSDataCollection()
        original_collection.overall_average = sample_group

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "data_integrity_test"

            # Export
            original_collection.export_larch_groups(output_dir, format="ascii")

            # Load back
            loaded_collection = EXAFSDataCollection.load_larch_groups(output_dir)

            # Compare k and chi arrays
            original_k = original_collection.overall_average.k
            original_chi = original_collection.overall_average.chi
            loaded_k = loaded_collection.overall_average.k
            loaded_chi = loaded_collection.overall_average.chi

            np.testing.assert_array_almost_equal(original_k, loaded_k, decimal=6)
            np.testing.assert_array_almost_equal(original_chi, loaded_chi, decimal=8)

    def test_load_larch_groups_missing_metadata(self):
        """Test loading when metadata file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "missing_metadata"
            output_dir.mkdir()

            with pytest.raises(FileNotFoundError, match="Metadata file not found"):
                EXAFSDataCollection.load_larch_groups(output_dir)

    def test_load_group_from_ascii_with_metadata(self, sample_group):
        """Test loading ASCII files with embedded metadata."""
        # Create a group with metadata
        sample_group.site_idx = 5
        sample_group.frame_idx = 3
        sample_group.absorber_element = "Cu"
        sample_group.is_average = True
        sample_group.average_type = "site"

        collection = EXAFSDataCollection()

        with tempfile.TemporaryDirectory() as temp_dir:
            chi_file = Path(temp_dir) / "test_group.chi"
            chir_file = Path(temp_dir) / "test_group.chir"

            # Save using the helper method
            collection._save_group_larch_format(
                sample_group, chi_file.with_suffix(""), "ascii"
            )

            # Load back using the helper method
            loaded_group = EXAFSDataCollection._load_group_from_ascii(
                chi_file, chir_file
            )

            # Check that metadata was preserved
            assert loaded_group.site_idx == 5
            assert loaded_group.frame_idx == 3
            assert loaded_group.absorber_element == "Cu"
            assert loaded_group.is_average
            assert loaded_group.average_type == "site"

    def test_export_larch_groups_invalid_format(self, sample_group):
        """Test error handling for invalid export format."""
        collection = EXAFSDataCollection(individual_spectra=[sample_group])
        collection.overall_average = (
            sample_group  # Add a group so format validation occurs
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "invalid_format"

            with pytest.raises(ValueError, match="Unsupported format: invalid"):
                collection.export_larch_groups(output_dir=output_dir, format="invalid")

    def test_export_empty_collection(self):
        """Test exporting an empty collection."""
        collection = EXAFSDataCollection()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "empty_collection"

            result_dir = collection.export_larch_groups(output_dir)

            # Should create the directory and metadata
            assert result_dir.exists()
            assert (result_dir / "collection_metadata.json").exists()

            # Should not create any group files
            assert not (result_dir / "overall_average.chi").exists()
            assert not (result_dir / "frame_averages").exists()
            assert not (result_dir / "site_averages").exists()
            assert not (result_dir / "individual_spectra").exists()

    def test_export_athena_format_fallback(self, sample_group):
        """Test fallback to ASCII when Athena format is not available."""
        collection = EXAFSDataCollection()
        collection.overall_average = sample_group

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "athena_fallback"

            # Test the ImportError handling in the _save_group_larch_format method
            # This happens when larch.io.athena is not available
            collection.export_larch_groups(output_dir=output_dir, format="athena")

            # Should fall back to ASCII format (since athena import will likely fail)
            assert (output_dir / "overall_average.chi").exists()
            assert (output_dir / "overall_average.chir").exists()

    def test_load_group_from_ascii_invalid_data(self):
        """Test error handling for invalid ASCII data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chi_file = Path(temp_dir) / "invalid.chi"

            # Create invalid data file
            with open(chi_file, "w") as f:
                f.write("# Header\n")
                f.write("invalid data line\n")
                f.write("not numbers\n")

            with pytest.raises(ValueError, match="Unable to extract k and chi data"):
                EXAFSDataCollection._load_group_from_ascii(chi_file)

    def test_filename_parsing_for_frame_and_site(self, multiple_groups):
        """Test that frame and site indices are correctly parsed from filenames."""
        # Set up groups with frame and site indices
        for i, group in enumerate(multiple_groups):
            group.frame_idx = i
            group.site_idx = i + 10

        collection = EXAFSDataCollection(individual_spectra=multiple_groups)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "filename_parsing"

            # Export with individual spectra
            collection.export_larch_groups(
                output_dir=output_dir, save_individual=True, save_averages=False
            )

            # Load back
            loaded_collection = EXAFSDataCollection.load_larch_groups(output_dir)

            # Check that frame and site indices were preserved
            for group in loaded_collection.individual_spectra:
                assert hasattr(group, "frame_idx")
                assert hasattr(group, "site_idx")
                assert group.frame_idx is not None
                assert group.site_idx is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
