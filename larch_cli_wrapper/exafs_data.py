"""EXAFS data structures using Larch Groups.

This module defines the Group-centric data structures that replace
the old PlotData/PlotDataCollection approach. All EXAFS data is stored
in Larch Groups to maintain consistency with k-weighting and Fourier transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from larch import Group
from larch.xafs import xftf

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = [
    "EXAFSDataCollection",
    "PlotResult",
    "add_metadata_to_group",
    "create_averaged_group",
    "prepare_exafs_data_collection",
]


@dataclass
class EXAFSDataCollection:
    """Collection of Larch Groups representing EXAFS spectra.

    Each Group contains:
    - k, chi: raw unweighted data from FEFF
    - r, chir_mag: FT results from k^kweight * chi
    - metadata attributes: site_idx, frame_idx, etc.
    """

    # Core data - all Groups processed with same kweight
    individual_spectra: list[Group] = field(default_factory=list)
    site_averages: dict[int, Group] = field(default_factory=dict)
    frame_averages: dict[int, Group] = field(default_factory=dict)
    overall_average: Group | None = None

    # Processing metadata
    kweight_used: int = 2  # The kweight used for all FTs
    fourier_params: dict[str, Any] = field(default_factory=dict)
    processing_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def get_plotting_groups(
        self,
        include_individual: bool = True,
        include_site_averages: bool = False,
        include_frame_averages: bool = False,
        include_overall_average: bool = True,
        max_individual: int = 100,
    ) -> list[Group]:
        """Get Groups for plotting based on preferences.

        Args:
            include_individual: Include individual spectra
            include_site_averages: Include site-averaged spectra
            include_frame_averages: Include frame-averaged spectra
            include_overall_average: Include overall average
            max_individual: Maximum number of individual spectra to return

        Returns:
            List of Groups for plotting
        """
        result = []

        if include_individual:
            # Limit number of individual spectra to avoid overwhelming plots
            individual_subset = self.individual_spectra[:max_individual]
            result.extend(individual_subset)

        if include_site_averages:
            result.extend(self.site_averages.values())

        if include_frame_averages:
            result.extend(self.frame_averages.values())

        if include_overall_average and self.overall_average:
            result.append(self.overall_average)

        return result

    def get_k_weighted_chi(self, group: Group, target_weight: int) -> np.ndarray:
        """Get chi with different k-weighting for visualization/comparison.

        Args:
            group: Larch Group containing raw chi data
            target_weight: Target k-weighting (0, 1, 2, or 3)

        Returns:
            K-weighted chi array
        """
        if target_weight == 0:
            return group.chi
        else:
            return group.chi * group.k**target_weight

    def get_plot_labels(self, target_weight: int) -> tuple[str, str]:
        """Get appropriate plot labels for given k-weighting.

        Args:
            target_weight: K-weighting for labels

        Returns:
            Tuple of (ylabel, title)
        """
        if target_weight == 1:
            return r"$k\chi(k)$", r"EXAFS $k\chi(k)$"
        elif target_weight == 2:
            return r"$k^{2}\chi(k)$", r"EXAFS $k^{2}\chi(k)$"
        elif target_weight == 3:
            return r"$k^{3}\chi(k)$", r"EXAFS $k^{3}\chi(k)$"
        else:  # target_weight == 0
            return r"$\chi(k)$", r"EXAFS $\chi(k)$"

    def export_larch_groups(
        self,
        output_dir: Path,
        save_individual: bool = False,
        save_averages: bool = True,
        format: str = "ascii",
    ) -> Path:
        """Export Larch Groups to files for later loading in Larch.

        Args:
            output_dir: Directory to save group files
            save_individual: Whether to save individual spectra groups
            save_averages: Whether to save averaged groups
            format: Export format ("ascii" for Larch-compatible text files,
                   "athena" for Athena project files)

        Returns:
            Path to output directory
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        if save_averages:
            # Save overall average
            if self.overall_average:
                avg_file = output_dir / "overall_average"
                self._save_group_larch_format(self.overall_average, avg_file, format)
                saved_files.append(avg_file)

            # Save frame averages
            if self.frame_averages:
                frame_dir = output_dir / "frame_averages"
                frame_dir.mkdir(exist_ok=True)
                for frame_idx, group in self.frame_averages.items():
                    frame_file = frame_dir / f"frame_{frame_idx:04d}"
                    self._save_group_larch_format(group, frame_file, format)
                    saved_files.append(frame_file)

            # Save site averages
            if self.site_averages:
                site_dir = output_dir / "site_averages"
                site_dir.mkdir(exist_ok=True)
                for site_idx, group in self.site_averages.items():
                    site_file = site_dir / f"site_{site_idx:04d}"
                    self._save_group_larch_format(group, site_file, format)
                    saved_files.append(site_file)

        if save_individual and self.individual_spectra:
            individual_dir = output_dir / "individual_spectra"
            individual_dir.mkdir(exist_ok=True)
            for i, group in enumerate(self.individual_spectra):
                # Use metadata if available
                if hasattr(group, "frame_idx") and hasattr(group, "site_idx"):
                    filename = f"frame_{group.frame_idx:04d}_site_{group.site_idx:04d}"
                else:
                    filename = f"spectrum_{i:04d}"

                spec_file = individual_dir / filename
                self._save_group_larch_format(group, spec_file, format)
                saved_files.append(spec_file)

        # Save metadata
        metadata = {
            "kweight_used": self.kweight_used,
            "fourier_params": self.fourier_params,
            "processing_metadata": self.processing_metadata,
            "created_at": self.created_at.isoformat(),
            "format": format,
            "saved_files": [str(f.relative_to(output_dir)) for f in saved_files],
        }

        metadata_file = output_dir / "collection_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_dir

    def _save_group_larch_format(
        self, group: Group, base_path: Path, format: str = "ascii"
    ) -> None:
        """Save a Larch Group in Larch-compatible format.

        Args:
            group: Larch Group to save
            base_path: Base path for output files (without extension)
            format: Format to use ("ascii" or "athena")
        """
        if format == "ascii":
            from larch.io import write_ascii

            # Prepare header with metadata
            header_lines = [
                f"Larch Group saved on {self.created_at.isoformat()}",
                f"k-weight used for FT: {self.kweight_used}",
                f"Fourier parameters: {self.fourier_params}",
            ]

            # Add group metadata as comments
            for attr_name in [
                "site_idx",
                "frame_idx",
                "absorber_element",
                "is_average",
                "average_type",
            ]:
                if hasattr(group, attr_name):
                    header_lines.append(f"{attr_name}: {getattr(group, attr_name)}")

            # Save k-space data in standard FEFF chi.dat format: k, chi, mag, phase
            # where chi is the real part
            chi_file = base_path.with_suffix(".chi")
            chi_header = header_lines + [
                "",
                "#       k          chi          mag           phase @#",
            ]

            # Handle complex chi values - compute real, magnitude, phase
            chi_data = group.chi
            if np.iscomplexobj(chi_data):
                chi_real = np.real(chi_data)
                chi_mag = np.abs(chi_data)
                chi_phase = np.angle(chi_data)
            else:
                # If chi is real, magnitude is absolute value, phase is 0
                chi_real = chi_data
                chi_mag = np.abs(chi_data)
                chi_phase = np.zeros_like(chi_data)

            write_ascii(
                str(chi_file),
                group.k,
                chi_real,
                chi_mag,
                chi_phase,
                header=chi_header,
                label="k chi mag phase",
            )

            # Save R-space data if available
            if hasattr(group, "r") and hasattr(group, "chir_mag"):
                chir_file = base_path.with_suffix(".chir")
                chir_header = header_lines + [
                    "",
                    "R(A)         |chi(R)|       Re[chi(R)]     Im[chi(R)]",
                ]

                # Get real and imaginary parts if available
                chir_re = getattr(group, "chir_re", np.zeros_like(group.r))
                chir_im = getattr(group, "chir_im", np.zeros_like(group.r))

                write_ascii(
                    str(chir_file),
                    group.r,
                    group.chir_mag,
                    chir_re,
                    chir_im,
                    header=chir_header,
                    label="r chir_mag chir_re chir_im",
                )

        elif format == "athena":
            # Note: Athena project format support would need proper implementation
            # For now, fallback to ASCII format
            print(
                "Warning: Athena format not fully implemented, saving as ASCII instead"
            )
            self._save_group_larch_format(group, base_path, "ascii")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'ascii' or 'athena'.")

    @classmethod
    def load_larch_groups(cls, input_dir: Path) -> EXAFSDataCollection:
        """Load Larch Groups from files saved by export_larch_groups.

        Args:
            input_dir: Directory containing saved group files

        Returns:
            EXAFSDataCollection with loaded groups
        """
        import json

        input_dir = Path(input_dir)
        metadata_file = input_dir / "collection_metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file) as f:
            metadata = json.load(f)

        collection = cls(
            kweight_used=metadata.get("kweight_used", 2),
            fourier_params=metadata.get("fourier_params", {}),
            processing_metadata=metadata.get("processing_metadata", {}),
        )

        # Get the format used for saving
        format_type = metadata.get("format", "ascii")

        # Load overall average
        if format_type == "ascii":
            chi_file = input_dir / "overall_average.chi"
            chir_file = input_dir / "overall_average.chir"
            if chi_file.exists():
                collection.overall_average = cls._load_group_from_ascii(
                    chi_file, chir_file
                )
        elif format_type == "athena":
            prj_file = input_dir / "overall_average.prj"
            if prj_file.exists():
                collection.overall_average = cls._load_group_from_athena(prj_file)

        # Load frame averages
        frame_dir = input_dir / "frame_averages"
        if frame_dir.exists():
            for frame_file in frame_dir.iterdir():
                if frame_file.suffix == ".chi" and format_type == "ascii":
                    frame_idx = int(frame_file.stem.split("_")[1])
                    chir_file = frame_file.with_suffix(".chir")
                    group = cls._load_group_from_ascii(frame_file, chir_file)
                    if group:
                        group.frame_idx = frame_idx
                        group.is_average = True
                        group.average_type = "frame"
                        collection.frame_averages[frame_idx] = group
                elif frame_file.suffix == ".prj" and format_type == "athena":
                    frame_idx = int(frame_file.stem.split("_")[1])
                    group = cls._load_group_from_athena(frame_file)
                    if group:
                        group.frame_idx = frame_idx
                        group.is_average = True
                        group.average_type = "frame"
                        collection.frame_averages[frame_idx] = group

        # Load site averages
        site_dir = input_dir / "site_averages"
        if site_dir.exists():
            for site_file in site_dir.iterdir():
                if site_file.suffix == ".chi" and format_type == "ascii":
                    site_idx = int(site_file.stem.split("_")[1])
                    chir_file = site_file.with_suffix(".chir")
                    group = cls._load_group_from_ascii(site_file, chir_file)
                    if group:
                        group.site_idx = site_idx
                        group.is_average = True
                        group.average_type = "site"
                        collection.site_averages[site_idx] = group
                elif site_file.suffix == ".prj" and format_type == "athena":
                    site_idx = int(site_file.stem.split("_")[1])
                    group = cls._load_group_from_athena(site_file)
                    if group:
                        group.site_idx = site_idx
                        group.is_average = True
                        group.average_type = "site"
                        collection.site_averages[site_idx] = group

        # Load individual spectra
        individual_dir = input_dir / "individual_spectra"
        if individual_dir.exists():
            for spec_file in sorted(individual_dir.iterdir()):
                if spec_file.suffix == ".chi" and format_type == "ascii":
                    chir_file = spec_file.with_suffix(".chir")
                    group = cls._load_group_from_ascii(spec_file, chir_file)
                elif spec_file.suffix == ".prj" and format_type == "athena":
                    group = cls._load_group_from_athena(spec_file)
                else:
                    continue

                if group:
                    # Parse frame and site indices from filename if present
                    stem = spec_file.stem
                    if "frame_" in stem and "site_" in stem:
                        parts = stem.split("_")
                        frame_idx = int(parts[1])
                        site_idx = int(parts[3])
                        group.frame_idx = frame_idx
                        group.site_idx = site_idx

                    group.is_average = False
                    collection.individual_spectra.append(group)

        return collection

    @staticmethod
    def _load_group_from_ascii(chi_file: Path, chir_file: Path = None) -> Group:
        """Load a Larch Group from ASCII format files.

        Uses Larch's native file reading capabilities for robust parsing of both
        standard ASCII files and FEFF-format files with enhanced metadata support.

        Args:
            chi_file: Path to chi(k) data file
            chir_file: Optional path to chi(R) data file

        Returns:
            Larch Group with loaded data
        """
        import numpy as np
        from larch import Group
        from larch.io import read_ascii
        from larch.xafs.feffdat import FeffDatFile

        if not chi_file.exists():
            raise FileNotFoundError(f"Chi file not found: {chi_file}")

        # First, try to detect if this is a FEFF format file by checking header
        # Skip any comments (lines starting with '#') and read first few lines
        is_feff_format = False
        try:
            with open(chi_file) as f:
                first_lines = [f.readline().strip() for _ in range(3)]

            # Check for FEFF format indicators
            is_feff_format = any(
                line.strip().endswith(("mag", "phase", "real[p]@#"))
                or "feff" in line.lower()
                or line.strip().startswith("#       k")
                for line in first_lines
            )
        except (ValueError, IndexError, AttributeError):
            is_feff_format = False

        # Create new group
        group = Group()

        if is_feff_format:
            try:
                # Use Larch's FeffDatFile for robust FEFF file parsing
                feff_data = FeffDatFile(filename=str(chi_file))

                # Extract data arrays
                group.k = feff_data.k

                # Handle complex chi from mag/phase or real/imag
                if hasattr(feff_data, "mag_feff") and hasattr(feff_data, "pha_feff"):
                    # Standard FEFF format: reconstruct complex chi from mag/phase
                    group.chi = feff_data.mag_feff * np.exp(1j * feff_data.pha_feff)
                elif hasattr(feff_data, "real_phc"):
                    # Use real part if available (FeffDatFile provides this)
                    group.chi = feff_data.real_phc

                # Copy FEFF metadata if available
                if hasattr(feff_data, "absorber"):
                    group.absorber_element = feff_data.absorber
                if hasattr(feff_data, "shell"):
                    group.shell = feff_data.shell
                if hasattr(feff_data, "edge"):
                    group.edge = feff_data.edge
                if hasattr(feff_data, "title"):
                    group.title = feff_data.title

                # FEFF files may not have our custom metadata, so use fallback
                use_feff = True

            except (ValueError, OSError) as e:
                print(
                    f"Warning: Failed to parse as FEFF file, "
                    f"falling back to read_ascii: {e}"
                )
                use_feff = False
        else:
            use_feff = False

        if not use_feff:
            # Use Larch's standard read_ascii for general ASCII files
            chi_group = read_ascii(str(chi_file))

            # Extract k and chi data with flexible column detection
            # First check if we have the full FEFF format: k, chi_real, mag, phase
            if hasattr(chi_group, "col1") and hasattr(chi_group, "col4"):
                # Standard FEFF chi.dat format with 4 columns
                group.k = chi_group.col1
                _chi_real = chi_group.col2  # Real part (for reference)
                chi_mag = chi_group.col3  # Magnitude
                chi_phase = chi_group.col4  # Phase
                # Reconstruct complex chi from magnitude and phase
                group.chi = chi_mag * np.exp(1j * chi_phase)
            # Method 1: Try named attributes (k, chi)
            elif hasattr(chi_group, "k") and hasattr(chi_group, "chi"):
                group.k = chi_group.k
                group.chi = chi_group.chi
            # Method 2: Try column access (col1, col2) - 2 columns only
            elif hasattr(chi_group, "col1") and hasattr(chi_group, "col2"):
                group.k = chi_group.col1
                group.chi = chi_group.col2
            # Method 3: Use data array directly (first 2 columns)
            elif (
                hasattr(chi_group, "data")
                and hasattr(chi_group.data, "shape")
                and chi_group.data.shape[0] >= 2
            ):
                group.k = chi_group.data[0]
                group.chi = chi_group.data[1]
            # Method 4: Try array_labels to find actual column names
            elif (
                hasattr(chi_group, "array_labels") and len(chi_group.array_labels) >= 2
            ):
                col1_name = chi_group.array_labels[0]
                col2_name = chi_group.array_labels[1]
                if hasattr(chi_group, col1_name) and hasattr(chi_group, col2_name):
                    group.k = getattr(chi_group, col1_name)
                    group.chi = getattr(chi_group, col2_name)
                else:
                    raise ValueError(
                        f"Unable to extract k and chi data from {chi_file}"
                    )
            else:
                raise ValueError(f"Unable to extract k and chi data from {chi_file}")

            # Use Larch's built-in attribute parsing from read_ascii
            if hasattr(chi_group, "attrs"):
                for attr_name in dir(chi_group.attrs):
                    if not attr_name.startswith("_"):
                        attr_value = getattr(chi_group.attrs, attr_name)

                        # Convert string representations to appropriate types
                        if attr_name in ["site_idx", "frame_idx"]:
                            try:
                                setattr(group, attr_name, int(attr_value))
                            except (ValueError, TypeError):
                                setattr(group, attr_name, attr_value)
                        elif attr_name == "is_average":
                            try:
                                setattr(
                                    group, attr_name, str(attr_value).lower() == "true"
                                )
                            except (ValueError, TypeError):
                                setattr(group, attr_name, attr_value)
                        else:
                            setattr(group, attr_name, attr_value)

            # Fallback: manual header parsing if attrs parsing didn't work well
            # or if we need to override incorrectly parsed metadata
            if hasattr(chi_group, "header"):
                for line in chi_group.header:
                    clean_line = line.lstrip("#").strip()
                    if ":" in clean_line:
                        key, value = clean_line.split(":", 1)
                        key = key.strip().replace(" ", "_").replace("-", "_")
                        value = value.strip()

                        # Parse known metadata fields with type conversion
                        if key in ["site_idx", "frame_idx"]:
                            try:
                                setattr(group, key, int(value))
                            except ValueError:
                                setattr(group, key, value)
                        elif key == "is_average":
                            setattr(group, key, value.lower() == "true")
                        else:
                            setattr(group, key, value)

        # Load R-space data if available using read_ascii
        if chir_file and chir_file.exists():
            try:
                chir_group = read_ascii(str(chir_file))

                # Use flexible attribute detection for R-space data
                r_attrs = ["r", "chir_mag", "chir_re", "chir_im"]
                col_attrs = ["col1", "col2", "col3", "col4"]

                for _i, (r_attr, col_attr) in enumerate(
                    zip(r_attrs, col_attrs, strict=False)
                ):
                    if hasattr(chir_group, r_attr):
                        setattr(group, r_attr, getattr(chir_group, r_attr))
                    elif hasattr(chir_group, col_attr):
                        setattr(group, r_attr, getattr(chir_group, col_attr))

            except (OSError, ValueError, AttributeError, ImportError) as e:
                print(f"Warning: Could not load R-space data from {chir_file}: {e}")

        return group

    @staticmethod
    def _load_group_from_athena(prj_file: Path) -> Group:
        """Load a Larch Group from Athena project file.

        Args:
            prj_file: Path to Athena project file

        Returns:
            Larch Group with loaded data
        """
        try:
            from larch.io import athena

            groups = athena.read_athena(str(prj_file))

            # Return first group if multiple groups in project
            if isinstance(groups, list) and len(groups) > 0:
                return groups[0]
            elif hasattr(groups, "__dict__"):
                return groups
            else:
                raise ValueError(f"No valid groups found in {prj_file}")

        except ImportError:
            raise ImportError(
                "Larch athena support not available. Cannot load .prj files."
            ) from None
        except (ValueError, OSError, TypeError) as e:
            raise ValueError(f"Failed to load Athena project {prj_file}: {e}") from e


@dataclass
class PlotResult:
    """Results of plotting operations."""

    plot_paths: dict[str, Path] = field(default_factory=dict)
    plot_metadata: dict[str, Any] = field(default_factory=dict)


# Utility functions for Group manipulation
def add_metadata_to_group(group: Group, **metadata) -> Group:
    """Add metadata attributes to a Larch Group.

    Args:
        group: Larch Group to modify
        **metadata: Key-value pairs to add as attributes

    Returns:
        The modified Group (for chaining)
    """
    for key, value in metadata.items():
        setattr(group, key, value)
    return group


def create_averaged_group(groups: list[Group], fourier_params: dict) -> Group:
    """Create an averaged Group from a list of individual Groups.

    This delegates the k-grid reconciliation to
    :func:`larch_cli_wrapper.feff_utils.average_chi_spectra` to keep the
    interpolation logic in a single place while still returning a Larch Group
    ready for Fourier transformation.

    Args:
        groups: List of EXAFS Groups to average
        fourier_params: Parameters for Fourier transform

    Returns:
        New Group containing averaged EXAFS data
    """
    if not groups:
        raise ValueError("Cannot create averaged group from empty list")

    from .feff_utils import average_chi_spectra

    k_arrays = [np.asarray(group.k) for group in groups]
    chi_arrays = [np.asarray(group.chi) for group in groups]

    chi_avg, k_common = average_chi_spectra(
        k_arrays,
        chi_arrays,
        restrict_to_common_range=True,
    )

    avg_group = Group()
    avg_group.k = k_common
    avg_group.chi = chi_avg

    xftf(avg_group, **fourier_params)

    return avg_group


def prepare_exafs_data_collection(
    groups: list[Group] | dict[int, list[Group]],
    fourier_params: dict | None = None,
    compute_averages: bool = True,
) -> EXAFSDataCollection:
    """Prepare EXAFS data collection from Groups.

    Args:
        groups: Either list of Groups or dict mapping site_idx to list of Groups
        fourier_params: Fourier transform parameters
        compute_averages: Whether to compute averaged data

    Returns:
        EXAFSDataCollection containing organized EXAFS data
    """
    # Default Fourier parameters
    if fourier_params is None:
        fourier_params = {
            "kweight": 2,
            "kmin": 3,
            "kmax": 12,
            "dk": 1,
            "window": "hanning",
        }

    # Ensure all groups have FT applied
    all_groups = []
    if isinstance(groups, dict):
        # Multi-site case
        for site_idx, site_groups in groups.items():
            for frame_idx, group in enumerate(site_groups):
                # Apply FT if not already done
                if not hasattr(group, "chir_mag"):
                    xftf(group, **fourier_params)
                # Add metadata
                add_metadata_to_group(group, site_idx=site_idx, frame_idx=frame_idx)
                all_groups.append(group)
    else:
        # Single site case
        for frame_idx, group in enumerate(groups):
            # Apply FT if not already done
            if not hasattr(group, "chir_mag"):
                xftf(group, **fourier_params)
            # Add metadata
            add_metadata_to_group(group, frame_idx=frame_idx)
            all_groups.append(group)

    # Create collection
    collection = EXAFSDataCollection(
        individual_spectra=all_groups,
        kweight_used=fourier_params.get("kweight", 2),
        fourier_params=fourier_params,
    )

    # Compute averages if requested
    if compute_averages and len(all_groups) > 1:
        avg_group = create_averaged_group(all_groups, fourier_params)
        add_metadata_to_group(
            avg_group,
            is_average=True,
            average_type="overall",
            n_components=len(all_groups),
        )
        collection.overall_average = avg_group

    return collection


def _get_style_path(style: str) -> str | Path:
    """Get the path to a matplotlib style file."""
    if style in ["presentation", "publication"]:
        from pathlib import Path

        current_dir = Path(__file__).parent
        style_file = current_dir / "styles" / f"exafs_{style}.mplstyle"
        if not style_file.exists():
            raise FileNotFoundError(f"Style file not found: {style_file}")
        return style_file

    style_path = Path(style)
    return style_path if style_path.exists() else style


@dataclass
class PlotConfig:
    """Shared configuration for EXAFS plots."""

    # What to plot
    plot_individual: bool = False
    plot_overall_avg: bool = True
    plot_frame_avg: bool = False
    plot_site_avg: bool = False

    # Metadata
    absorber: str = "X"
    edge: str = "K"
    kweight: int | None = None

    # Display options
    show_legend: bool = True
    title_prefix: str = ""

    # Style themes
    style: Literal["presentation", "publication"] = "presentation"


@dataclass
class PlotStyles:
    """Common style definitions for both backends."""

    # Colors (same for both backends)
    colors = {
        "individual": "lightgray",
        "frame_avg": "dimgray",
        "site_avg": "coral",
        "overall_avg": "black",
        "site_colors": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ],
    }

    # Line properties
    individual = {"alpha": 0.4, "linewidth": 1.0}
    frame_avg = {"alpha": 0.6, "linewidth": 1.5}
    site_avg = {"alpha": 0.7, "linewidth": 1.5}
    overall_avg = {"alpha": 1.0, "linewidth": 2.5}

    @classmethod
    def get_style(cls, key: str, custom_color: str | None = None):
        """Get style dict with color."""
        style = getattr(cls, key).copy()
        style["color"] = custom_color or cls.colors[key]
        return style


def prepare_plot_data(
    collection: EXAFSDataCollection,
    config: PlotConfig,
    max_individual: int = 50,
    max_frames: int = 50,
) -> dict:
    """Prepare data for plotting - shared between matplotlib and plotly.

    Returns a dict with categorized spectra and metadata.

    Args:
        collection: EXAFSDataCollection with spectra
        config: PlotConfig controlling what to plot
        max_individual: Maximum number of individual spectra to plot
            (subsamples if exceeded)
        max_frames: Maximum number of frame averages to plot
            (subsamples if exceeded)
    """
    kweight = config.kweight or collection.kweight_used

    plot_data = {
        "kweight": kweight,
        "chi_label": None,
        "chir_label": None,
        "individual": [],
        "frame_avg": [],
        "site_avg": [],
        "overall_avg": None,
    }

    # Get labels
    chi_label, chir_label = collection.get_plot_labels(kweight)
    plot_data["chi_label"] = chi_label
    plot_data["chir_label"] = chir_label

    # Collect individual spectra (with subsampling if too many)
    if config.plot_individual and collection.individual_spectra:
        n_individual = len(collection.individual_spectra)

        # Subsample if too many
        if n_individual > max_individual:
            # Evenly spaced indices
            step = n_individual / max_individual
            indices = [int(i * step) for i in range(max_individual)]
            spectra_to_plot = [collection.individual_spectra[i] for i in indices]
        else:
            spectra_to_plot = collection.individual_spectra

        for group in spectra_to_plot:
            chi = collection.get_k_weighted_chi(group, kweight)
            plot_data["individual"].append(
                {
                    "k": group.k,
                    "chi": np.real(chi),
                    "r": group.r,
                    "chir_mag": group.chir_mag,
                    "label": "Individual",
                    "frame_idx": getattr(group, "frame_idx", None),
                    "site_idx": getattr(group, "site_idx", None),
                }
            )

    # Collect frame averages (with subsampling if too many)
    if config.plot_frame_avg and hasattr(collection, "frame_averages"):
        frame_items = list(collection.frame_averages.items())
        n_frames = len(frame_items)

        # Subsample if too many
        if n_frames > max_frames:
            # Evenly spaced indices
            step = n_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            frames_to_plot = [(frame_items[i][0], frame_items[i][1]) for i in indices]
        else:
            frames_to_plot = frame_items

        for frame_idx, group in frames_to_plot:
            chi = collection.get_k_weighted_chi(group, kweight)
            plot_data["frame_avg"].append(
                {
                    "k": group.k,
                    "chi": np.real(chi),
                    "r": group.r,
                    "chir_mag": group.chir_mag,
                    "label": "Frame averages",
                    "frame_idx": frame_idx,  # Store frame index for hover labels
                }
            )

    # Collect site averages
    if config.plot_site_avg and hasattr(collection, "site_averages"):
        n_sites = len(collection.site_averages)

        for idx, (site_idx, group) in enumerate(collection.site_averages.items()):
            label = f"Site {site_idx}" if n_sites < 5 else "Site averages"
            color_idx = idx if n_sites < 5 else None
            chi = collection.get_k_weighted_chi(group, kweight)
            plot_data["site_avg"].append(
                {
                    "k": group.k,
                    "chi": np.real(chi),
                    "r": group.r,
                    "chir_mag": group.chir_mag,
                    "label": label,
                    "color_idx": color_idx,
                    "many_sites": n_sites >= 5,
                }
            )

    # Overall average
    if config.plot_overall_avg and collection.overall_average is not None:
        avg_group = collection.overall_average
        chi = collection.get_k_weighted_chi(avg_group, kweight)
        plot_data["overall_avg"] = {
            "k": avg_group.k,
            "chi": np.real(chi),
            "r": avg_group.r,
            "chir_mag": avg_group.chir_mag,
            "label": "Overall average",
        }

    return plot_data


# ============================================================================
# MATPLOTLIB IMPLEMENTATION
# ============================================================================


def plot_exafs_matplotlib(
    collection: EXAFSDataCollection,
    config: PlotConfig,
    output_dir: Path | str | None = None,
    filename_base: str = "EXAFS_FT",
    show_plot: bool = False,
) -> PlotResult:
    """Plot EXAFS data using matplotlib."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Get prepared data
    data = prepare_plot_data(collection, config)
    styles = PlotStyles()

    # Apply style
    style_path = _get_style_path(config.style)

    with plt.style.context(style_path):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        legends_shown = set()

        def add_trace(ax, x, y, label, style_dict):
            """Add a trace to the axis."""
            show_label = label if label not in legends_shown else None
            ax.plot(x, y, label=show_label, **style_dict)
            if show_label:
                legends_shown.add(label)

        # Plot individual spectra
        for spec in data["individual"]:
            style = styles.get_style("individual")
            add_trace(ax1, spec["k"], spec["chi"], spec["label"], style)
            add_trace(ax2, spec["r"], spec["chir_mag"], spec["label"], style)

        # Plot frame averages
        for spec in data["frame_avg"]:
            style = styles.get_style("frame_avg")
            add_trace(ax1, spec["k"], spec["chi"], spec["label"], style)
            add_trace(ax2, spec["r"], spec["chir_mag"], spec["label"], style)

        # Plot site averages
        for spec in data["site_avg"]:
            if spec["color_idx"] is not None:
                color = styles.colors["site_colors"][spec["color_idx"]]
                style = styles.get_style("site_avg", color)
            else:
                style = styles.get_style("site_avg")
                if spec["many_sites"]:
                    style["alpha"] = 0.3

            add_trace(ax1, spec["k"], spec["chi"], spec["label"], style)
            add_trace(ax2, spec["r"], spec["chir_mag"], spec["label"], style)

        # Plot overall average
        if data["overall_avg"]:
            spec = data["overall_avg"]
            style = styles.get_style("overall_avg")
            add_trace(ax1, spec["k"], spec["chi"], spec["label"], style)
            add_trace(ax2, spec["r"], spec["chir_mag"], spec["label"], style)

        # Format axes
        ax1.set_xlabel("k (Å⁻¹)")
        ax1.set_ylabel(data["chi_label"])
        ax1.set_title(f"{config.absorber} {config.edge}-edge EXAFS")
        if config.show_legend:
            ax1.legend()

        ax2.set_xlabel("R (Å)")
        ax2.set_ylabel("|χ(R)| (Å⁻³)")
        ax2.set_title(f"{config.absorber} {config.edge}-edge Fourier Transform")
        if config.show_legend:
            ax2.legend()

        plt.tight_layout()

        # Save if output directory provided
        plot_paths = {}
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for fmt in ["png", "pdf", "svg"]:
                path = output_dir / f"{filename_base}.{fmt}"
                plt.savefig(path, bbox_inches="tight")
                plot_paths[fmt] = path

        plt.show() if show_plot else plt.close(fig)

        return PlotResult(
            plot_paths=plot_paths,
            plot_metadata={
                "absorber": config.absorber,
                "edge": config.edge,
                "kweight": data["kweight"],
                "style": config.style,
            },
        )


# ============================================================================
# PLOTLY IMPLEMENTATION
# ============================================================================


def plot_exafs_plotly(
    collection: EXAFSDataCollection,
    config: PlotConfig,
) -> go.Figure:
    """Create side-by-side plotly subplots for χ(k) and χ(R).

    This function creates a figure with two subplots side-by-side:
    - Left: k-space χ(k) plot
    - Right: R-space |χ(R)| plot

    Args:
        collection: EXAFSDataCollection with spectra to plot
        config: PlotConfig controlling what to plot and styling

    Returns:
        Plotly Figure with subplots
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Get prepared data
    data = prepare_plot_data(collection, config)
    styles = PlotStyles()
    kweight = data["kweight"]

    # Create subplot titles
    if kweight == 1:
        chi_title = f"{config.absorber} {config.edge}-edge EXAFS k×χ(k)"
        chi_label = "k×χ(k)"
    elif kweight == 2:
        chi_title = f"{config.absorber} {config.edge}-edge EXAFS k²×χ(k)"
        chi_label = "k²×χ(k)"
    elif kweight == 3:
        chi_title = f"{config.absorber} {config.edge}-edge EXAFS k³×χ(k)"
        chi_label = "k³×χ(k)"
    else:
        chi_title = f"{config.absorber} {config.edge}-edge EXAFS χ(k)"
        chi_label = "χ(k)"

    # Create side-by-side subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            chi_title,
            f"{config.absorber} {config.edge}-edge Fourier Transform",
        ),
        horizontal_spacing=0.1,
    )

    # Common layout styling
    common_layout = {
        "font": {"family": "Times New Roman", "size": 18},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "margin": {"l": 100, "r": 30, "t": 80, "b": 120},
        "showlegend": config.show_legend,
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.45,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "black",
            "borderwidth": 1,
        },
    }

    legends_shown = set()

    def add_subplot_trace(
        x, y, label, style_dict, row, col, show_in_legend=True, hover_extra=""
    ):
        """Add a trace to a subplot with hover labels."""
        # Convert matplotlib style to plotly
        color = style_dict["color"]
        alpha = style_dict.get("alpha", 1.0)
        linewidth = style_dict.get("linewidth", 2)

        # Convert color to rgba if needed
        if not color.startswith("rgba"):
            if color.startswith("#"):
                # Hex to rgba
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                color = f"rgba({r},{g},{b},{alpha})"
            else:
                # Named color - approximate alpha
                color = f"rgba(128,128,128,{alpha})" if "gray" in color else color

        # Determine if we should show this in legend
        show_label = (
            config.show_legend and show_in_legend and label not in legends_shown
        )

        # Create hover template based on subplot (k-space or R-space)
        # Add hover_extra info if provided (e.g., frame index)
        label_with_extra = f"{label}{hover_extra}" if hover_extra else label

        if col == 1:
            # k-space
            hovertemplate = (
                f"{label_with_extra}<br>k: %{{x:.3f}} Å⁻¹<br>"
                f"{chi_label}: %{{y:.6f}}<extra></extra>"
            )
        else:
            # R-space
            hovertemplate = (
                f"{label_with_extra}<br>R: %{{x:.3f}} Å<br>"
                "|χ(R)|: %{y:.6f} Å⁻³<extra></extra>"
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=label if show_label else None,
                line={"width": linewidth, "color": color},
                opacity=alpha,
                showlegend=show_label
                and row == 1
                and col == 1,  # Only show legend for first subplot
                legendgroup=label,  # Group traces with same label
                hovertemplate=hovertemplate,
            ),
            row=row,
            col=col,
        )

        if show_label:
            legends_shown.add(label)

    # Plot individual spectra
    for i, spec in enumerate(data["individual"]):
        style = styles.get_style("individual")
        show_in_legend = i == 0  # Only show first individual in legend

        # Build hover label with frame and site info
        hover_parts = []
        if spec.get("frame_idx") is not None:
            hover_parts.append(f"Frame {spec['frame_idx']}")
        if spec.get("site_idx") is not None:
            hover_parts.append(f"Site {spec['site_idx']}")
        hover_extra = f" ({', '.join(hover_parts)})" if hover_parts else ""

        # k-space
        add_subplot_trace(
            spec["k"],
            spec["chi"],
            "Individual",
            style,
            row=1,
            col=1,
            show_in_legend=show_in_legend,
            hover_extra=hover_extra,
        )
        # R-space
        add_subplot_trace(
            spec["r"],
            spec["chir_mag"],
            "Individual",
            style,
            row=1,
            col=2,
            show_in_legend=False,
            hover_extra=hover_extra,
        )

    # Plot frame averages
    for i, spec in enumerate(data["frame_avg"]):
        style = styles.get_style("frame_avg")
        # Only show first frame average in legend to avoid overwhelming legend
        show_in_legend = i == 0

        # Get frame index for hover label
        frame_idx = spec.get("frame_idx", i)
        hover_label = f" {frame_idx}"  # Will be appended to "Frame"

        # k-space
        add_subplot_trace(
            spec["k"],
            spec["chi"],
            "Frame",
            style,
            row=1,
            col=1,
            show_in_legend=show_in_legend,
            hover_extra=hover_label,
        )
        # R-space
        add_subplot_trace(
            spec["r"],
            spec["chir_mag"],
            "Frame",
            style,
            row=1,
            col=2,
            show_in_legend=False,
            hover_extra=hover_label,
        )

    # Plot site averages
    for i, spec in enumerate(data["site_avg"]):
        if spec["color_idx"] is not None:
            color = styles.colors["site_colors"][spec["color_idx"]]
            style = styles.get_style("site_avg", color)
        else:
            style = styles.get_style("site_avg")
            if spec["many_sites"]:
                style["alpha"] = 0.3

        # Only show first site average in legend if many sites
        show_in_legend = not spec.get("many_sites", False) or (i == 0)

        # k-space
        add_subplot_trace(
            spec["k"],
            spec["chi"],
            spec["label"],
            style,
            row=1,
            col=1,
            show_in_legend=show_in_legend,
        )
        # R-space
        add_subplot_trace(
            spec["r"],
            spec["chir_mag"],
            spec["label"],
            style,
            row=1,
            col=2,
            show_in_legend=False,
        )

    # Plot overall average
    if data["overall_avg"]:
        spec = data["overall_avg"]
        style = styles.get_style("overall_avg")

        # k-space
        add_subplot_trace(spec["k"], spec["chi"], spec["label"], style, row=1, col=1)
        # R-space
        add_subplot_trace(
            spec["r"],
            spec["chir_mag"],
            spec["label"],
            style,
            row=1,
            col=2,
            show_in_legend=False,
        )

    # Update axis labels and styling
    fig.update_xaxes(
        title_text="k (Å⁻¹)",
        row=1,
        col=1,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgray",
        tickwidth=2,
    )
    fig.update_yaxes(
        title_text=chi_label,
        row=1,
        col=1,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgray",
        tickwidth=2,
        title_standoff=10,
    )
    fig.update_xaxes(
        title_text="R (Å)",
        row=1,
        col=2,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgray",
        tickwidth=2,
    )
    fig.update_yaxes(
        title_text="|χ(R)| (Å⁻³)",
        row=1,
        col=2,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgray",
        tickwidth=2,
        title_standoff=10,
    )

    # Apply overall styling
    fig.update_layout(**common_layout)
    fig.update_layout(height=400, width=900)

    return fig
