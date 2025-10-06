"""Tests for the EXAFSDataCollection class and plotting functionality."""

import numpy as np
from larch import Group
from larch.xafs import xftf

from larch_cli_wrapper.exafs_data import (
    EXAFSDataCollection,
    prepare_exafs_data_collection,
)


class TestEXAFSDataCollection:
    """Test EXAFSDataCollection class."""

    def create_mock_group(self, site_idx=0, frame_idx=0, chi_offset=0):
        """Create a mock EXAFS group."""
        group = Group()
        group.k = np.linspace(2, 14, 50)
        group.chi = (np.sin(group.k) + chi_offset * 0.1) * np.exp(-group.k / 10)
        xftf(group, kmin=3, kmax=12, kweight=2, dk=1, window="hanning")
        group.site_idx = site_idx
        group.frame_idx = frame_idx
        return group

    def test_collection_creation(self):
        """Test basic collection creation."""
        collection = EXAFSDataCollection()
        assert len(collection.individual_spectra) == 0
        assert len(collection.site_averages) == 0
        assert len(collection.frame_averages) == 0
        assert collection.overall_average is None
        assert collection.kweight_used == 2  # Default


class TestEXAFSDataPreparation:
    """Test the EXAFS data preparation methods."""

    def create_mock_group(self, k_offset=0, chi_offset=0, site_idx=0, frame_idx=0):
        """Create a mock EXAFS group."""
        group = Group()
        group.k = np.linspace(2, 14, 50) + k_offset * 0.1
        group.chi = (np.sin(group.k) + chi_offset * 0.1) * np.exp(-group.k / 10)
        xftf(group, kmin=3, kmax=12, kweight=2, dk=1, window="hanning")
        group.site_idx = site_idx
        group.frame_idx = frame_idx
        return group

    def test_single_spectrum_preparation(self):
        """Test preparing data for single spectrum."""
        group = self.create_mock_group()

        collection = prepare_exafs_data_collection(
            groups=[group],
            fourier_params={
                "kweight": 2,
                "kmin": 3,
                "kmax": 12,
                "dk": 1,
                "window": "hanning",
            },
        )

        assert len(collection.individual_spectra) == 1
        # For single group, no overall average is computed
        assert collection.overall_average is None
        assert collection.kweight_used == 2

    def test_multiple_spectra_preparation(self):
        """Test preparing data for multiple spectra to get overall average."""
        group1 = self.create_mock_group(chi_offset=0, site_idx=0, frame_idx=0)
        group2 = self.create_mock_group(chi_offset=1, site_idx=0, frame_idx=1)
        group3 = self.create_mock_group(chi_offset=2, site_idx=0, frame_idx=2)

        collection = prepare_exafs_data_collection(
            groups=[group1, group2, group3],
            fourier_params={
                "kweight": 2,
                "kmin": 3,
                "kmax": 12,
                "dk": 1,
                "window": "hanning",
            },
        )

        assert len(collection.individual_spectra) == 3
        # For multiple groups, overall average should be computed
        assert collection.overall_average is not None
        assert collection.kweight_used == 2
        assert hasattr(collection.overall_average, "k")
        assert hasattr(collection.overall_average, "chi")

    def test_multi_site_preparation(self):
        """Test preparing data for multi-site analysis."""
        site0_groups = [
            self.create_mock_group(chi_offset=0, site_idx=0, frame_idx=0),
            self.create_mock_group(chi_offset=1, site_idx=0, frame_idx=1),
        ]
        site1_groups = [
            self.create_mock_group(chi_offset=2, site_idx=1, frame_idx=0),
            self.create_mock_group(chi_offset=3, site_idx=1, frame_idx=1),
        ]

        groups_dict = {0: site0_groups, 1: site1_groups}

        collection = prepare_exafs_data_collection(
            groups=groups_dict,
            fourier_params={
                "kweight": 3,
                "kmin": 2,
                "kmax": 14,
                "dk": 1.5,
                "window": "gaussian",
            },
        )

        assert len(collection.individual_spectra) == 4  # 2 sites × 2 frames each
        assert (
            collection.overall_average is not None
        )  # Multiple groups, so average computed
        assert collection.kweight_used == 3

    def test_no_averages_computation(self):
        """Test preparing data without computing averages."""
        group1 = self.create_mock_group()
        group2 = self.create_mock_group(chi_offset=1)

        collection = prepare_exafs_data_collection(
            groups=[group1, group2],
            compute_averages=False,
        )

        assert len(collection.individual_spectra) == 2
        # Even with multiple groups, no average computed when compute_averages=False
        assert collection.overall_average is None

    def test_default_fourier_params(self):
        """Test that default Fourier parameters are applied correctly."""
        group = self.create_mock_group()

        collection = prepare_exafs_data_collection(groups=[group])

        # Check default parameters are set
        expected_params = {
            "kweight": 2,
            "kmin": 3,
            "kmax": 12,
            "dk": 1,
            "window": "hanning",
        }

        assert collection.fourier_params == expected_params
        assert collection.kweight_used == 2
