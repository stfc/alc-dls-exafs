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
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

__all__ = [
    "EXAFSDataCollection",
    "PlotResult",
    "add_metadata_to_group",
    "create_averaged_group",
    "prepare_exafs_data_collection",
    "PathContribution",
    "PathAggregator",
    "aggregate_store_paths",
    "filter_path_contributions",
    "make_path_key",
    "PlotConfig",
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

    # Path contributions (populated when FEFF path files are kept)
    # Maps path_key (e.g. "SS_Cl_2.86") -> PathContribution
    path_contributions: dict[str, PathContribution] = field(default_factory=dict)

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


# ---------------------------------------------------------------------------
# Path contribution data structures
# ---------------------------------------------------------------------------


def _has_angle(sample: dict) -> bool:
    """True if ``sample["angle"]`` is a usable (non-``None``, non-NaN) value.

    HDF5 float datasets have no native "missing" value, so ``angle`` is
    stored on disk as NaN for 2-/4-leg paths and converted back to Python
    ``None`` by ``ExafsHDF5Store.iter_path_contributions``. This helper
    treats both representations as "no angle" so any caller that bypasses
    that conversion still degrades gracefully instead of treating NaN as a
    real angle value (which would silently poison downstream clustering —
    every comparison against NaN is False).
    """
    angle = sample.get("angle")
    return bool(angle is not None and np.isfinite(angle))


def make_path_key(scatterer: str, nlegs: int, r_eff: float, r_bin: float = 0.15) -> str:
    """Create a display label for a (already-pooled) FEFF path population.

    This is now purely a *labeling* helper — the actual pooling of raw
    per-frame FEFF paths into populations is done by
    :meth:`PathAggregator.finalize` using proper clustering (see
    :func:`larch_cli_wrapper.debye_waller_core.cluster_1d_sorted`), not by
    rounding ``r_eff`` into a fixed-width bin. This function just formats a
    stable-looking label from the *already-clustered* mean ``r_eff``.

    Examples::

        make_path_key("Cl",  2, 2.84)  -> "SS_Cl_2.85"
        make_path_key("K",   2, 4.03)  -> "SS_K_3.90"
        make_path_key("Cl-K", 3, 6.12) -> "MS3_Cl-K_6.15"

    Args:
        scatterer: Scatterer element string (e.g. ``"Cl"`` or ``"Cl-K"``)
        nlegs: Number of scattering legs (2 = single-scattering)
        r_eff: Effective path length in ångströms
        r_bin: Bin width used only for rounding the displayed label
            (default 0.15 Å) — has no effect on grouping.

    Returns:
        Key string suitable for dict lookup / display.
    """
    prefix = "SS" if nlegs == 2 else f"MS{nlegs}"
    r_binned = round(round(r_eff / r_bin) * r_bin, 3)
    return f"{prefix}_{scatterer}_{r_binned:.2f}"


@dataclass
class PathContribution:
    """Averaged chi(k) and chi(R) for a single FEFF scattering path type.

    Produced by :class:`PathAggregator` after accumulating individual path
    results from multiple FEFF calculations and averaging them.
    """

    path_key: str
    """Stable identifier, e.g. ``"SS_Cl_2.85"``."""
    scatterer: str
    """Element label(s) of the scattering atom(s), e.g. ``"Cl"``."""
    nlegs: int
    """Number of scattering legs (2 = single-scattering)."""
    r_eff: float
    """Mean effective path length across all samples (Å)."""
    degeneracy: float
    """Mean degeneracy across all samples."""
    n_samples: int
    """Number of individual path calculations averaged."""
    # Averaged k-space data
    k: np.ndarray
    chi: np.ndarray
    # Fourier-transformed data (set after finalize)
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    chir_mag: np.ndarray = field(default_factory=lambda: np.array([]))
    chir_re: np.ndarray = field(default_factory=lambda: np.array([]))
    chir_im: np.ndarray = field(default_factory=lambda: np.array([]))
    cw_ratio: float = 0.0
    """Mean curved-wave chi amplitude ratio (0–100, relative to strongest path)."""
    angle: float | None = None
    """Mean 3-body scattering angle in degrees, or ``None`` for 2-/4-leg
    paths (or older data predating angle extraction)."""
    r_eff_std: float = 0.0
    """Standard deviation of ``r_eff`` across the samples pooled into this
    path (Å) — a pooling-quality diagnostic: how tightly did the per-frame
    FEFF geometries that were merged into this population actually agree.
    ``0.0`` for a single-sample population."""
    angle_std: float | None = None
    """Standard deviation of the 3-body angle across pooled samples
    (degrees), or ``None`` for 2-/4-leg paths (or no usable angle samples)."""
    source_frames: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    """Frame indices that contributed to this averaged path."""
    source_sites: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    """Site indices that contributed to this averaged path."""
    contribution_pct: float = 0.0
    """Percentage contribution to the total MD-averaged χ(k) amplitude."""
    # Averaged raw FEFF parameters for on-the-fly χ recomputation
    amp: np.ndarray = field(default_factory=lambda: np.array([]))
    """Averaged scattering amplitude on FEFF's native coarse k-grid."""
    pha: np.ndarray = field(default_factory=lambda: np.array([]))
    """Averaged total phase on FEFF's native coarse k-grid."""
    lam: np.ndarray = field(default_factory=lambda: np.array([]))
    """Averaged mean free path on FEFF's native coarse k-grid."""
    rep: np.ndarray = field(default_factory=lambda: np.array([]))
    """Averaged real part of complex momentum on FEFF's native coarse k-grid."""
    k_param: np.ndarray = field(default_factory=lambda: np.array([]))
    """Native coarse k-grid that amp/pha/lam/rep live on (Å⁻¹)."""


class PathAggregator:
    """Accumulate per-path chi(k) contributions then average them.

    Usage::

        agg = PathAggregator()
        # For each FEFF run that kept its per-path files:
        agg.add(paths_dict)   # {path_key: {"k": ..., "chi": ..., "r_eff": ...,
        #                                   "nlegs": ..., "degeneracy": ...,
        #                                   "scatterer": ...}}
        path_contribs = agg.finalize(fourier_params)

    Internally, samples are bucketed by a *canonical* category
    (``(sorted scatterer elements, nlegs)``) rather than by the caller-
    supplied key, then split into physically-distinct path populations by
    clustering on effective distance (and, for 3-leg paths, angle) in
    :meth:`finalize`. This avoids two failure modes of a fixed-width R-bin
    key: (1) the same physical path landing in different bins purely from
    r_eff straddling a bin edge across frames, and (2) two geometrically
    distinct 3-body paths (different angle) at a coincidentally similar
    r_eff being silently averaged together, corrupting the averaged chi(k).
    """

    def __init__(self, r_bin: float = 0.15, angle_tol: float = 15.0) -> None:
        """Initialize the PathAggregator with clustering tolerances.

        Args:
            r_bin: Distance clustering tolerance (Å). Kept as the
                historical parameter name for backward compatibility;
                despite the name this is now a clustering tolerance (see
                :func:`larch_cli_wrapper.debye_waller_core.cluster_1d_sorted`),
                not a fixed bin width.
            angle_tol: Angle clustering tolerance (degrees) for 3-leg paths.
        """
        self._r_bin = r_bin
        self._angle_tol = angle_tol
        # (canonical_scatterer, nlegs) -> list of sample dicts
        self._samples: dict[tuple[tuple[str, ...], int], list[dict]] = {}

    def add(self, paths_dict: dict[str, dict]) -> None:
        """Accumulate path samples from one FEFF calculation.

        The caller-supplied key is not used for grouping (grouping is
        determined internally from each sample's own ``scatterer``/``nlegs``
        plus clustering at :meth:`finalize` time); it is accepted purely for
        backward API compatibility with existing call sites.

        Args:
            paths_dict: Maps an arbitrary label to a dict with keys:
                ``k``, ``chi`` (arrays) and ``r_eff``, ``nlegs``,
                ``degeneracy``, ``scatterer`` (scalars), optionally ``angle``.
        """
        from .debye_waller_core import canonical_scatterer_key

        for info in paths_dict.values():
            cat = (
                canonical_scatterer_key(str(info.get("scatterer", "?"))),
                int(info.get("nlegs", 2)),
            )
            self._samples.setdefault(cat, []).append(info)

    def _cluster_category(self, nlegs: int, samples: list[dict]) -> list[list[dict]]:
        """Split one (scatterer, nlegs) category's samples into path populations."""
        from .debye_waller_core import cluster_1d_sorted

        if nlegs != 3:
            r_effs = np.array([float(s["r_eff"]) for s in samples])
            return [
                [samples[i] for i in idx_arr]
                for idx_arr in cluster_1d_sorted(r_effs, self._r_bin)
            ]

        # 3-leg paths: cluster by angle first, then by r_eff within each
        # angle cluster (mirroring the MD-side 3-body clustering in
        # calculate_grouped_msrd). Samples without a usable angle (e.g. from
        # older/failed geometry extraction) fall back to distance-only
        # clustering as their own sub-population. ``_has_angle`` treats both
        # a missing/None value and a stray NaN (the on-disk HDF5 sentinel,
        # which should normally already be converted to None by
        # ``iter_path_contributions`` before reaching here) as "no angle".
        with_angle = [s for s in samples if _has_angle(s)]
        without_angle = [s for s in samples if not _has_angle(s)]

        clusters: list[list[dict]] = []
        if with_angle:
            angles = np.array([float(s["angle"]) for s in with_angle])
            for angle_idx_arr in cluster_1d_sorted(angles, self._angle_tol):
                angle_cluster = [with_angle[i] for i in angle_idx_arr]
                r_effs = np.array([float(s["r_eff"]) for s in angle_cluster])
                for dist_idx_arr in cluster_1d_sorted(r_effs, self._r_bin):
                    clusters.append([angle_cluster[i] for i in dist_idx_arr])
        if without_angle:
            r_effs = np.array([float(s["r_eff"]) for s in without_angle])
            for idx_arr in cluster_1d_sorted(r_effs, self._r_bin):
                clusters.append([without_angle[i] for i in idx_arr])
        return clusters

    def finalize(self, fourier_params: dict) -> dict[str, PathContribution]:
        """Cluster, average, and Fourier-transform accumulated path samples.

        The chi(R) data is derived from ``xftf`` applied to the *averaged*
        chi(k), not from averaging individual chi(R) arrays.

        Args:
            fourier_params: Parameters forwarded to :func:`larch.xafs.xftf`.

        Returns:
            ``{path_key: PathContribution}`` sorted by mean ``r_eff``, where
            ``path_key`` is a display label generated (post-clustering) by
            :func:`make_path_key`.
        """
        result: dict[str, PathContribution] = {}

        for (canon_scatterer, nlegs), cat_samples in self._samples.items():
            if not cat_samples:
                continue
            scatterer_label = "-".join(canon_scatterer)
            for samples in self._cluster_category(nlegs, cat_samples):
                if not samples:
                    continue
                mean_reff = float(np.mean([s["r_eff"] for s in samples]))
                key = make_path_key(scatterer_label, nlegs, mean_reff, self._r_bin)
                # Disambiguate labels in the rare case two clusters round to
                # the same display key (e.g. two very close but genuinely
                # distinct populations).
                if key in result:
                    key = f"{key}_{len(result)}"
                self._finalize_one(
                    key, scatterer_label, nlegs, samples, fourier_params, result
                )

        # Sort by mean r_eff
        return dict(sorted(result.items(), key=lambda kv: kv[1].r_eff))

    def _finalize_one(
        self,
        key: str,
        scatterer_label: str,
        nlegs: int,
        samples: list[dict],
        fourier_params: dict,
        result: dict[str, PathContribution],
    ) -> None:
        """Build and store one :class:`PathContribution` from a clustered population."""
        from .feff_utils import average_chi_spectra

        k_arrays = [np.asarray(s["k"], dtype=np.float64) for s in samples]
        chi_arrays = [np.asarray(s["chi"], dtype=np.float64) for s in samples]
        r_effs = [float(s["r_eff"]) for s in samples]
        degs = [float(s["degeneracy"]) for s in samples]
        angles = [float(s["angle"]) for s in samples if _has_angle(s)]

        chi_avg, k_common = average_chi_spectra(
            k_arrays,
            chi_arrays,
            restrict_to_common_range=True,
        )

        # Apply xftf once to the averaged chi(k)
        g = Group(k=k_common, chi=chi_avg)
        xftf(g, **fourier_params)

        first = samples[0]
        # Store *unique* contributing indices — one entry per distinct
        # frame/site rather than one per sample.  This is 100× smaller
        # for high-degeneracy paths while still giving full provenance.
        source_frames = np.unique(
            np.array(
                [
                    int(s.get("frame_index", -1))
                    for s in samples
                    if s.get("frame_index", -1) >= 0
                ],
                dtype=np.int64,
            )
        )
        source_sites = np.unique(
            np.array(
                [
                    int(s.get("site_index", -1))
                    for s in samples
                    if s.get("site_index", -1) >= 0
                ],
                dtype=np.int64,
            )
        )

        # Average raw FEFF parameters for on-the-fly χ recomputation
        _param_names = ("amp", "pha", "lam", "rep")
        _param_avgs: dict[str, np.ndarray] = {}
        _k_param: np.ndarray = np.array([])
        for _pname in _param_names:
            _parrays = [
                np.asarray(s[_pname], dtype=np.float64) for s in samples if _pname in s
            ]
            if _parrays:
                _param_avgs[_pname] = np.mean(_parrays, axis=0)
        # k_param is the native coarse FEFF grid; all paths share it
        if "k_param" in first:
            _k_param = np.asarray(first["k_param"], dtype=np.float64)
        elif "k" in first and _param_avgs:
            _k = np.asarray(first["k"], dtype=np.float64)
            _first_amp = _param_avgs.get("amp")
            if _first_amp is not None and len(_k) == len(_first_amp):
                _k_param = _k

        pc = PathContribution(
            path_key=key,
            scatterer=scatterer_label,
            nlegs=nlegs,
            r_eff=float(np.mean(r_effs)),
            degeneracy=float(np.mean(degs)),
            n_samples=len(samples),
            k=k_common,
            chi=chi_avg,
            r=np.asarray(g.r) if hasattr(g, "r") else np.array([]),
            chir_mag=np.asarray(g.chir_mag) if hasattr(g, "chir_mag") else np.array([]),
            chir_re=np.asarray(g.chir_re) if hasattr(g, "chir_re") else np.array([]),
            chir_im=np.asarray(g.chir_im) if hasattr(g, "chir_im") else np.array([]),
            cw_ratio=float(
                np.mean([s["cw_ratio"] for s in samples]) if samples else 0.0
            ),
            angle=float(np.mean(angles)) if angles else None,
            r_eff_std=float(np.std(r_effs)) if len(r_effs) > 1 else 0.0,
            angle_std=float(np.std(angles))
            if len(angles) > 1
            else (0.0 if angles else None),
            source_frames=source_frames,
            source_sites=source_sites,
            amp=_param_avgs.get("amp", np.array([])),
            pha=_param_avgs.get("pha", np.array([])),
            lam=_param_avgs.get("lam", np.array([])),
            rep=_param_avgs.get("rep", np.array([])),
            k_param=_k_param,
        )
        result[key] = pc


def filter_path_contributions(
    contribs: dict[str, PathContribution],
    top_n: int | None = None,
    min_cw_ratio: float | None = None,
) -> dict[str, PathContribution]:
    """Filter path contributions by curved-wave amplitude ratio.

    Paths are ranked by ``cw_ratio`` (the FEFF curved-wave chi amplitude ratio,
    relative to the strongest path = 100).  Two independent filters can be
    applied in combination:

    - ``min_cw_ratio`` keeps only paths whose mean ``cw_ratio`` is at or above
      the threshold (e.g. ``5.0`` retains paths with ≥5% of the peak amplitude).
    - ``top_n`` keeps at most the N strongest paths after the ratio filter.

    The relative ordering among surviving paths (by ``r_eff``) is preserved.

    Args:
        contribs: Mapping of path_key → :class:`PathContribution` as returned
            by :meth:`PathAggregator.finalize`.
        top_n: Maximum number of paths to keep (strongest first).  ``None``
            means no cap.
        min_cw_ratio: Minimum curved-wave ratio threshold (0–100).  Paths
            below this value are excluded.  ``None`` means no threshold.

    Returns:
        Filtered dict with the same key type, ordered by ``r_eff``.
    """
    if not contribs:
        return {}

    # Rank all paths by cw_ratio descending to apply top_n / threshold
    ranked = sorted(contribs.values(), key=lambda pc: pc.cw_ratio, reverse=True)

    if min_cw_ratio is not None:
        ranked = [pc for pc in ranked if pc.cw_ratio >= min_cw_ratio]

    if top_n is not None:
        ranked = ranked[:top_n]

    # Restore r_eff ordering among survivors
    surviving_keys = {pc.path_key for pc in ranked}
    return {k: v for k, v in contribs.items() if k in surviving_keys}


# ---------------------------------------------------------------------------
# Fast vectorized / parallel path aggregation
#
# ``PathAggregator`` builds one Python dict per stored path row and averages
# per-cluster Python lists of arrays, which is prohibitively slow for large
# trajectories (hundreds of thousands to millions of path rows).  The routines
# below reproduce its clustering/averaging semantics exactly but:
#   * assign clusters with vectorized numpy grouping (no per-row dicts),
#   * stream the large chi/amp/pha/lam/rep arrays from HDF5 in worker
#     *processes* (h5py holds the GIL during gzip decompression, so process
#     parallelism is required to use multiple cores), and
#   * accumulate per-cluster sums in the workers, returning only the small
#     reduced arrays.
# The result is numerically identical to ``PathAggregator.finalize`` (verified
# per-cluster to full float precision on real data).
# ---------------------------------------------------------------------------

_PATH_ARRAY_NAMES = ("chi", "amp", "pha", "lam", "rep")

# Upper bound on aggregation worker processes.  Beyond a handful of concurrent
# readers the shared-file HDF5 reads saturate and extra processes only add
# scheduler/lock contention, so a large FEFF ``-w`` must not propagate here.
_AGG_MAX_WORKERS = 8


def _assign_path_clusters(
    canon_codes: np.ndarray,
    nlegs: np.ndarray,
    r_eff: np.ndarray,
    angle: np.ndarray,
    pool_idx: np.ndarray,
    r_bin: float,
    angle_tol: float,
) -> tuple[np.ndarray, int]:
    """Assign a local cluster id to every row in ``pool_idx``.

    Reproduces :meth:`PathAggregator._cluster_category`: rows are first grouped
    by canonical ``(scatterer, nlegs)`` category, then clustered by ``r_eff``
    (and, for 3-leg paths, by ``angle`` first, then ``r_eff``) using
    :func:`~larch_cli_wrapper.debye_waller_core.cluster_1d_sorted`.

    Args:
        canon_codes: Integer code of the canonical scatterer label per global
            row (dense codes ``0..n_canon-1``); avoids per-row Python strings.
        nlegs: Number of legs per global row.
        r_eff: Effective path length per global row.
        angle: 3-body angle per global row (NaN where absent).
        pool_idx: Sorted global row indices to cluster (all rows, or one site).
        r_bin: Distance clustering tolerance (Å).
        angle_tol: Angle clustering tolerance (degrees).

    Returns:
        ``(local_ids, n_clusters)`` where ``local_ids[j]`` is the cluster id of
        ``pool_idx[j]`` (contiguous ids ``0..n_clusters-1``).
    """
    from .debye_waller_core import cluster_1d_sorted

    out = np.full(len(pool_idx), -1, dtype=np.int64)
    n_clusters = 0

    # Combine (canon_code, nlegs) into a single integer category key instead of
    # building an object array of ``"canon|nlegs"`` strings per row.
    nl_mult = int(nlegs.max()) + 1 if len(nlegs) else 1
    keys = canon_codes[pool_idx].astype(np.int64) * nl_mult + nlegs[pool_idx]
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    bounds = np.searchsorted(inv_sorted, np.arange(len(uniq_keys) + 1))

    for ui in range(len(uniq_keys)):
        local_pos = order[bounds[ui] : bounds[ui + 1]]
        grp = pool_idx[local_pos]
        nl = int(nlegs[grp[0]])
        if nl != 3:
            for cl in cluster_1d_sorted(r_eff[grp], r_bin):
                out[local_pos[cl]] = n_clusters
                n_clusters += 1
        else:
            ang = angle[grp]
            has = np.isfinite(ang)
            wa = np.where(has)[0]
            wo = np.where(~has)[0]
            if len(wa):
                for acl in cluster_1d_sorted(ang[wa], angle_tol):
                    sub = wa[acl]
                    for dcl in cluster_1d_sorted(r_eff[grp[sub]], r_bin):
                        out[local_pos[sub[dcl]]] = n_clusters
                        n_clusters += 1
            if len(wo):
                for dcl in cluster_1d_sorted(r_eff[grp[wo]], r_bin):
                    out[local_pos[wo[dcl]]] = n_clusters
                    n_clusters += 1

    return out, n_clusters


def _limit_native_threads() -> None:
    """Pin BLAS/OpenMP thread pools to a single thread in this process.

    Aggregation workers are I/O- and lock-bound, not BLAS-bound.  Without this,
    each worker process lets numpy's native backend open one thread *per core*,
    so ``N`` worker processes on a ``C``-core box create ``N * C`` threads that
    fight over the scheduler.  On large machines that pushes the load average
    into the thousands and makes the box appear hung even though little real
    work is happening.  Setting the env vars covers ``spawn``-started workers;
    ``threadpoolctl`` (if importable) additionally clamps an already-imported
    numpy backend in ``fork``-started workers.
    """
    import os

    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_var, "1")
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:  # noqa: BLE001, S110 - best effort; env vars still apply
        pass


def _sum_path_arrays_worker(args: tuple) -> tuple[dict, dict]:
    """Worker: stream a row-range from HDF5 and accumulate per-cluster sums.

    Runs in a separate process.  Reads the large path arrays block-wise (each
    gzip chunk is decompressed exactly once) and scatters them into overall and
    per-site cluster accumulators.  Only the small reduced arrays are returned.
    """
    import h5py
    import numpy as np

    _limit_native_threads()

    path, start, end, oid_chunk, sid_chunk, n_over, n_site, array_names = args
    with h5py.File(path, "r", locking=False) as f:
        pr = f["path_results"]
        widths = {a: pr[a].shape[1] for a in array_names}
        o_acc = {a: np.zeros((n_over, widths[a])) for a in array_names}
        s_acc = {a: np.zeros((n_site, widths[a])) for a in array_names}
        blk = pr["chi"].chunks[0] if pr["chi"].chunks else 1024
        for s in range(start, end, blk):
            e = min(s + blk, end)
            o = oid_chunk[s - start : e - start]
            si = sid_chunk[s - start : e - start]
            for a in array_names:
                d = pr[a][s:e]
                np.add.at(o_acc[a], o, d)
                np.add.at(s_acc[a], si, d)
    return o_acc, s_acc


def _reduce_group_metadata(
    sort_order: np.ndarray,
    bounds: np.ndarray,
    n_clusters: int,
    *,
    r_eff: np.ndarray,
    degeneracy: np.ndarray,
    cw_ratio: np.ndarray,
    angle: np.ndarray,
    frame_index: np.ndarray,
    site_index: np.ndarray,
    canon_codes: np.ndarray,
    canon_labels: np.ndarray,
    nlegs: np.ndarray,
) -> list[dict]:
    """Compute per-cluster scalar/metadata reductions via sorted-group slices.

    Iterating clusters with a fresh boolean mask over all rows would be
    O(n_rows * n_clusters); instead rows are pre-sorted by cluster id once and
    each cluster is a contiguous slice.
    """
    meta: list[dict] = [{} for _ in range(n_clusters)]
    for c in range(n_clusters):
        rows = sort_order[bounds[c] : bounds[c + 1]]
        if len(rows) == 0:
            continue
        re = r_eff[rows]
        ang = angle[rows]
        ang = ang[np.isfinite(ang)]
        meta[c] = {
            "rows": rows,
            "n_samples": int(len(rows)),
            "r_eff": float(re.mean()),
            "r_eff_std": float(re.std()) if len(re) > 1 else 0.0,
            "degeneracy": float(degeneracy[rows].mean()),
            "cw_ratio": float(cw_ratio[rows].mean()),
            "angle": float(ang.mean()) if len(ang) else None,
            "angle_std": (
                float(ang.std()) if len(ang) > 1 else (0.0 if len(ang) else None)
            ),
            "source_frames": np.unique(frame_index[rows]).astype(np.int64),
            "source_sites": np.unique(site_index[rows]).astype(np.int64),
            "scatterer": str(canon_labels[canon_codes[rows[0]]]),
            "nlegs": int(nlegs[rows[0]]),
        }
    return meta


def _build_contributions(
    meta: list[dict],
    sums: dict[str, np.ndarray],
    counts: np.ndarray,
    k_paths: np.ndarray,
    k_param: np.ndarray,
    fourier_params: dict,
    r_bin: float,
) -> dict[str, PathContribution]:
    """Turn per-cluster sums + metadata into a ``{key: PathContribution}`` dict."""
    result: dict[str, PathContribution] = {}
    order = sorted(
        range(len(meta)),
        key=lambda c: meta[c].get("r_eff", np.inf) if meta[c] else np.inf,
    )
    for c in order:
        m = meta[c]
        if not m:
            continue
        n = counts[c]
        if n <= 0:
            continue
        chi_avg = sums["chi"][c] / n
        g = Group(k=k_paths, chi=chi_avg)
        xftf(g, **fourier_params)

        key = make_path_key(m["scatterer"], m["nlegs"], m["r_eff"], r_bin)
        if key in result:
            key = f"{key}_{len(result)}"

        result[key] = PathContribution(
            path_key=key,
            scatterer=m["scatterer"],
            nlegs=m["nlegs"],
            r_eff=m["r_eff"],
            degeneracy=m["degeneracy"],
            n_samples=m["n_samples"],
            k=k_paths,
            chi=chi_avg,
            r=np.asarray(g.r) if hasattr(g, "r") else np.array([]),
            chir_mag=np.asarray(g.chir_mag) if hasattr(g, "chir_mag") else np.array([]),
            chir_re=np.asarray(g.chir_re) if hasattr(g, "chir_re") else np.array([]),
            chir_im=np.asarray(g.chir_im) if hasattr(g, "chir_im") else np.array([]),
            cw_ratio=m["cw_ratio"],
            angle=m["angle"],
            r_eff_std=m["r_eff_std"],
            angle_std=m["angle_std"],
            source_frames=m["source_frames"],
            source_sites=m["source_sites"],
            amp=(sums["amp"][c] / n) if "amp" in sums else np.array([]),
            pha=(sums["pha"][c] / n) if "pha" in sums else np.array([]),
            lam=(sums["lam"][c] / n) if "lam" in sums else np.array([]),
            rep=(sums["rep"][c] / n) if "rep" in sums else np.array([]),
            k_param=k_param,
        )
    return result


def aggregate_store_paths(
    h5_path,
    fourier_params: dict,
    *,
    r_bin: float = 0.15,
    angle_tol: float = 15.0,
    max_workers: int | None = None,
    progress_callback=None,
) -> tuple[dict[str, PathContribution], dict[int, dict[str, PathContribution]]]:
    """Aggregate all stored per-path contributions into averaged paths.

    Fast, vectorized, process-parallel replacement for feeding
    :class:`PathAggregator` from :meth:`ExafsHDF5Store.iter_path_contributions`.
    Produces the overall average and per-site averages in a single read pass.

    Args:
        h5_path: Path to the results HDF5 file (with a ``path_results`` group).
        fourier_params: Parameters forwarded to :func:`larch.xafs.xftf`.
        r_bin: Distance clustering tolerance (Å).
        angle_tol: Angle clustering tolerance (degrees) for 3-leg paths.
        max_workers: Number of worker processes for the streaming sum. ``None``
            or ``<=1`` runs serially in-process.
        progress_callback: Optional ``(completed, total, phase)`` callback.

    Returns:
        ``(overall, per_site)`` where ``overall`` is ``{key: PathContribution}``
        and ``per_site`` is ``{site_index: {key: PathContribution}}``.
    """
    import h5py

    from .debye_waller_core import canonical_scatterer_key

    h5_path = str(h5_path)
    with h5py.File(h5_path, "r", locking=False) as f:
        pr = f.get("path_results")
        if pr is None or "chi" not in pr:
            return {}, {}
        n = pr["frame_index"].shape[0]
        if n == 0:
            return {}, {}
        r_eff = pr["r_eff"][:]
        nlegs = pr["nlegs"][:].astype(np.int64)
        angle = (
            pr["angle"][:] if "angle" in pr else np.full(n, np.nan, dtype=np.float64)
        )
        site_index = pr["site_index"][:].astype(np.int64)
        frame_index = pr["frame_index"][:].astype(np.int64)
        degeneracy = pr["degeneracy"][:]
        cw_ratio = pr["cw_ratio"][:]
        scat_raw = pr["scatterer"][:]
        k_paths = pr["k_grid_paths"][:]
        k_param = pr["k_grid_params"][:] if "k_grid_params" in pr else np.array([])
        array_names = [a for a in _PATH_ARRAY_NAMES if a in pr]

    if progress_callback:
        progress_callback(0, 1, "aggregating")

    # Canonicalize scatterer labels over *unique* labels only, then represent
    # every row by a small integer code into ``canon_labels`` rather than an
    # object array of one Python string per row (which, for millions of rows,
    # dominates memory and is slow to build/index).
    uniq_scat, inv_scat = np.unique(scat_raw, return_inverse=True)
    uniq_canon = np.array(
        [
            "-".join(
                canonical_scatterer_key(
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                )
            )
            for s in uniq_scat
        ],
        dtype=object,
    )
    # Distinct canonical labels get dense codes; several raw labels may collapse
    # onto the same canonical label.
    canon_labels, canon_of_uniqscat = np.unique(uniq_canon, return_inverse=True)
    canon_codes = canon_of_uniqscat[inv_scat].astype(np.int64)

    all_idx = np.arange(n)
    oid, n_over = _assign_path_clusters(
        canon_codes, nlegs, r_eff, angle, all_idx, r_bin, angle_tol
    )

    # Per-site cluster ids, globally unique, plus the site value per site-cluster.
    sid = np.full(n, -1, dtype=np.int64)
    n_site = 0
    site_of_cluster: list[int] = []
    for sv in np.unique(site_index):
        pool = np.where(site_index == sv)[0]
        sub, nc = _assign_path_clusters(
            canon_codes, nlegs, r_eff, angle, pool, r_bin, angle_tol
        )
        sid[pool] = sub + n_site
        n_site += nc
        site_of_cluster.extend([int(sv)] * nc)

    # Streaming per-cluster sums (optionally across worker processes).
    def _width(a: str) -> int:
        return k_paths.shape[0] if a == "chi" else k_param.shape[0]

    o_sums = {a: np.zeros((n_over, _width(a))) for a in array_names}
    s_sums = {a: np.zeros((n_site, _width(a))) for a in array_names}

    import os

    # Aggregation is I/O- and lock-bound (every worker reopens and reads the
    # *same* HDF5 file), not compute-bound, so a large FEFF ``-w`` does more
    # harm than good here: too many processes oversubscribe cores and contend
    # on HDF5 file locks.  Cap the pool independently of the FEFF worker count.
    cpu = os.cpu_count() or 2
    if max_workers is None:
        n_workers = max(1, min(cpu // 2, _AGG_MAX_WORKERS))
    else:
        n_workers = max(1, min(int(max_workers), cpu, _AGG_MAX_WORKERS))
    n_workers = max(1, min(n_workers, n))
    step = (n + n_workers - 1) // n_workers
    ranges = [
        (
            h5_path,
            s,
            min(s + step, n),
            oid[s : min(s + step, n)],
            sid[s : min(s + step, n)],
            n_over,
            n_site,
            array_names,
        )
        for s in range(0, n, step)
    ]

    done_frac = 0
    total_frac = len(ranges)
    if n_workers == 1:
        for r in ranges:
            o_acc, s_acc = _sum_path_arrays_worker(r)
            for a in array_names:
                o_sums[a] += o_acc[a]
                s_sums[a] += s_acc[a]
            done_frac += 1
            if progress_callback:
                progress_callback(done_frac, total_frac, "aggregating")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_sum_path_arrays_worker, r) for r in ranges]
                for fut in as_completed(futs):
                    o_acc, s_acc = fut.result()
                    for a in array_names:
                        o_sums[a] += o_acc[a]
                        s_sums[a] += s_acc[a]
                    done_frac += 1
                    if progress_callback:
                        progress_callback(done_frac, total_frac, "aggregating")
        except Exception:  # noqa: BLE001 - fall back to serial on any pool failure
            for a in array_names:
                o_sums[a][:] = 0
                s_sums[a][:] = 0
            for r in ranges:
                o_acc, s_acc = _sum_path_arrays_worker(r)
                for a in array_names:
                    o_sums[a] += o_acc[a]
                    s_sums[a] += s_acc[a]

    # Per-cluster counts and metadata via sorted-group slices.
    o_order = np.argsort(oid, kind="stable")
    o_bounds = np.searchsorted(oid[o_order], np.arange(n_over + 1))
    o_counts = np.diff(o_bounds)
    o_meta = _reduce_group_metadata(
        o_order,
        o_bounds,
        n_over,
        r_eff=r_eff,
        degeneracy=degeneracy,
        cw_ratio=cw_ratio,
        angle=angle,
        frame_index=frame_index,
        site_index=site_index,
        canon_codes=canon_codes,
        canon_labels=canon_labels,
        nlegs=nlegs,
    )
    overall = _build_contributions(
        o_meta, o_sums, o_counts, k_paths, k_param, fourier_params, r_bin
    )

    s_order = np.argsort(sid, kind="stable")
    s_bounds = np.searchsorted(sid[s_order], np.arange(n_site + 1))
    s_counts = np.diff(s_bounds)
    s_meta = _reduce_group_metadata(
        s_order,
        s_bounds,
        n_site,
        r_eff=r_eff,
        degeneracy=degeneracy,
        cw_ratio=cw_ratio,
        angle=angle,
        frame_index=frame_index,
        site_index=site_index,
        canon_codes=canon_codes,
        canon_labels=canon_labels,
        nlegs=nlegs,
    )

    # Split site-clusters back out by their originating site value.
    per_site: dict[int, list[int]] = {}
    for c in range(n_site):
        per_site.setdefault(site_of_cluster[c], []).append(c)
    per_site_result: dict[int, dict[str, PathContribution]] = {}
    for sv, cluster_ids in per_site.items():
        sub_meta = [s_meta[c] for c in cluster_ids]
        sub_sums = {a: s_sums[a][cluster_ids] for a in array_names}
        sub_counts = s_counts[cluster_ids]
        per_site_result[sv] = _build_contributions(
            sub_meta, sub_sums, sub_counts, k_paths, k_param, fourier_params, r_bin
        )

    if progress_callback:
        progress_callback(total_frac, total_frac, "aggregating")

    return overall, per_site_result


# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------


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

    # Path contributions panel
    plot_paths: bool = False
    """Add a 2×2 paths panel showing individual path k-space and R-space."""
    max_paths: int | None = None
    """Maximum number of path contributions to show (ranked by max chir_mag).

    ``None`` means all paths are shown.
    """

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


def _plot_paths_panel(
    axk: plt.Axes,
    axr: plt.Axes,
    path_contributions: dict[str, PathContribution],
    max_paths: int | None,
    kweight: int,
    overall_average: Group | None = None,
    fourier_params: dict | None = None,
    n_sites_total: int | None = None,
) -> None:
    """Populate an existing pair of axes with per-path contributions.

    Paths are ranked by ``max(chir_mag)`` and the top ``max_paths`` are drawn.
    Single-scattering paths are solid; 3-leg MS paths are dashed; 4+-leg MS
    paths are dotted.  Line colour is taken from the ``plasma`` colormap and
    scales with the path's peak |chi(R)| amplitude relative to the strongest
    path — the dominant contribution is bright yellow, weaker paths fade toward
    deep purple.

    A sanity-check overlay is added at the end: the weighted sum of *all* path
    chi(k) contributions (``Σ (n_j / n_sites) * chi_j``) is plotted as a thick
    dashed black line.  The ``overall_average`` (if supplied) is plotted as a
    thick solid red line so the two can be visually compared.

    The weighted formula is required because multiple FEFF paths from the same
    calculation can share the same path key (e.g. many Cu–Cu single-scattering
    paths at slightly different distances in a bulk crystal).  Each
    :class:`PathContribution` averages *all* samples of that path type, so its
    chi must be weighted by ``n_samples / n_sites_total`` before summing.

    Args:
        axk: Matplotlib Axes for chi(k) (k-space panel)
        axr: Matplotlib Axes for chi(R) (R-space panel)
        path_contributions: Mapping of path_key → :class:`PathContribution`
        max_paths: Maximum number of paths to draw
        kweight: k-weighting to apply when displaying chi(k)
        overall_average: Optional overall-average Larch Group for comparison.
        fourier_params: Fourier-transform parameters forwarded to
            :func:`larch.xafs.xftf` when computing chi(R) for the path sum.
        n_sites_total: Total number of individual FEFF calculations that
            contributed to the path statistics.  Used to correctly weight the
            per-path-type averages when computing the sum::

                chi_sum = sum_j(n_samples_j / n_sites_total * chi_j)

            Defaults to ``max(n_samples)`` over all path types (valid when at
            least one path type is present in every calculation).
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt  # noqa: F401 – needed for type hints

    if not path_contributions:
        axk.text(
            0.5, 0.5, "No path data", ha="center", va="center", transform=axk.transAxes
        )
        axr.text(
            0.5, 0.5, "No path data", ha="center", va="center", transform=axr.transAxes
        )
        return

    # Rank by peak chi(R)
    def _rank_key(pc: PathContribution) -> float:
        return float(np.max(pc.chir_mag)) if len(pc.chir_mag) else 0.0

    ranked = sorted(path_contributions.values(), key=_rank_key, reverse=True)[
        :max_paths
    ]

    # Map peak chi(R) amplitude → colour on the plasma colormap.
    # Normalise so the strongest path = 1.0; clamp lower end at 0.15 so even
    # the weakest plotted path remains visible (not pure black).
    strengths = np.array([_rank_key(pc) for pc in ranked])
    max_strength = strengths[0] if strengths[0] > 0 else 1.0
    norm_strengths = np.clip(strengths / max_strength, 0.0, 1.0)

    cmap = plt.colormaps.get_cmap("viridis")  # "plasma" is another good option
    cmap_lo, cmap_hi = 0.15, 1.0  # usable range within the colormap
    path_colors = {
        pc.path_key: cmap(cmap_lo + (cmap_hi - cmap_lo) * ns)
        for pc, ns in zip(ranked, norm_strengths, strict=False)
    }

    # Add a colorbar to show the amplitude scale
    scalar_mappable = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=0.0, vmax=max_strength),
        cmap=mcolors.LinearSegmentedColormap.from_list(
            "plasma_clipped",
            [cmap(cmap_lo + (cmap_hi - cmap_lo) * x) for x in np.linspace(0, 1, 256)],
        ),
    )
    scalar_mappable.set_array([])
    fig = axr.get_figure()
    if fig is not None:
        fig.colorbar(scalar_mappable, ax=axr, label="|χ(R)| peak amplitude", pad=0.02)

    # Linestyle by scattering type
    def _linestyle(nlegs: int) -> str:
        if nlegs == 2:
            return "solid"
        if nlegs == 3:
            return "dashed"
        return "dotted"

    for pc in ranked:
        color = path_colors[pc.path_key]
        ls = _linestyle(pc.nlegs)
        lw = 1.5
        label = f"{pc.path_key} (n={pc.n_samples})"

        if len(pc.k) and len(pc.chi):
            kw_chi = pc.chi * pc.k**kweight
            axk.plot(pc.k, kw_chi, color=color, linestyle=ls, linewidth=lw, label=label)

        if len(pc.r) and len(pc.chir_mag):
            axr.plot(
                pc.r, pc.chir_mag, color=color, linestyle=ls, linewidth=lw, label=label
            )

    # ------------------------------------------------------------------
    # Sanity-check overlay: sum of ALL paths vs overall average
    # ------------------------------------------------------------------
    _plot_paths_sum_vs_average(
        axk,
        axr,
        path_contributions,
        kweight,
        overall_average,
        fourier_params,
        n_sites_total,
    )


def _plot_paths_sum_vs_average(
    axk: plt.Axes,
    axr: plt.Axes,
    path_contributions: dict[str, PathContribution],
    kweight: int,
    overall_average: Group | None,
    fourier_params: dict | None,
    n_sites_total: int | None,
) -> None:
    """Overlay the weighted sum of all path chi(k) vs the overall average.

    The correct formula accounting for multiple paths per FEFF calculation
    mapping to the same path key is::

        chi_sum = sum_j( n_samples_j / n_sites_total * chi_j )

    This equals the overall average when every path key appears the same
    number of times per FEFF calculation (which is true for dilute/molecular
    systems, but not for bulk crystals where many bond-length-equivalent
    paths share a key).
    """
    if not path_contributions:
        return

    # Use n_sites_total to correctly weight per-path-type averaged chi.
    # Multiple FEFF paths from the same calculation can share a path key;
    # PathContribution.chi is their mean, so we must weight by n_samples.
    # If n_sites_total is not provided, fall back to max(n_samples), which is
    # exact when at least one path type is present in every calculation.
    n_total = (
        n_sites_total
        if n_sites_total and n_sites_total > 0
        else max(pc.n_samples for pc in path_contributions.values())
    )

    k_min = max(float(pc.k[0]) for pc in path_contributions.values() if len(pc.k))
    k_max = min(float(pc.k[-1]) for pc in path_contributions.values() if len(pc.k))
    if k_min >= k_max:
        return
    n_pts = int(round((k_max - k_min) / 0.05)) + 1
    k_common = np.linspace(k_min, k_max, max(n_pts, 50))

    # Weighted sum: chi_j is scaled by n_samples_j / n_total before summing.
    chi_sum = np.zeros_like(k_common)
    for pc in path_contributions.values():
        if len(pc.k) < 2 or len(pc.chi) < 2:
            continue
        weight = pc.n_samples / n_total
        chi_sum += weight * np.interp(k_common, pc.k, pc.chi, left=0.0, right=0.0)

    axk.plot(
        k_common,
        chi_sum * k_common**kweight,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="paths sum",
        zorder=5,
    )

    # chi(R) of the summed chi(k) — FT must be applied to the sum, not summed
    # from individual chi(R) because |FT(a+b)| ≠ |FT(a)| + |FT(b)|.
    try:
        g_sum = Group(k=k_common, chi=chi_sum)
        xftf(g_sum, **(fourier_params or {}))
        if hasattr(g_sum, "r") and hasattr(g_sum, "chir_mag"):
            axr.plot(
                np.asarray(g_sum.r),
                np.asarray(g_sum.chir_mag),
                color="black",
                linewidth=2.5,
                linestyle="--",
                label="paths sum",
                zorder=5,
            )
    except Exception:  # noqa: BLE001, S110
        pass  # FT failure is non-fatal; k-space comparison is still shown

    # Overlay the overall average for direct comparison.
    if overall_average is not None:
        avg_k = np.asarray(getattr(overall_average, "k", []))
        avg_chi = np.asarray(getattr(overall_average, "chi", []))
        if len(avg_k) and len(avg_chi):
            axk.plot(
                avg_k,
                avg_chi * avg_k**kweight,
                color="red",
                linewidth=2.5,
                linestyle="-",
                label="average",
                zorder=6,
            )
        avg_r = np.asarray(getattr(overall_average, "r", []))
        avg_chir_mag = np.asarray(getattr(overall_average, "chir_mag", []))
        if len(avg_r) and len(avg_chir_mag):
            axr.plot(
                avg_r,
                avg_chir_mag,
                color="red",
                linewidth=2.5,
                linestyle="-",
                label="average",
                zorder=6,
            )


def plot_exafs_matplotlib(
    collection: EXAFSDataCollection,
    config: PlotConfig,
    output_dir: Path | str | None = None,
    filename_base: str = "EXAFS_FT",
    show_plot: bool = False,
) -> PlotResult:
    """Plot EXAFS data using matplotlib.

    When ``config.plot_paths`` is ``True`` **and** the collection has path
    contributions, a 2×2 figure is produced:

    * Row 1: full-spectrum chi(k) and chi(R) (same as the default 1×2 layout)
    * Row 2: per-path chi(k) and chi(R) contributions (top ``max_paths`` paths
      ranked by peak |chi(R)|)

    Args:
        collection: EXAFSDataCollection with spectra to plot
        config: PlotConfig controlling what to plot and styling
        output_dir: Output directory for saving plots (should be absolute path)
        filename_base: Base name for output files
        show_plot: Whether to display the plot interactively

    Returns:
        PlotResult with paths to saved plots
    """
    import matplotlib.pyplot as plt

    # Get prepared data
    data = prepare_plot_data(collection, config)
    styles = PlotStyles()
    kweight = data["kweight"]

    # Determine whether to add path contributions panel
    show_paths = config.plot_paths and bool(collection.path_contributions)

    # Apply style
    style_path = _get_style_path(config.style)

    with plt.style.context(style_path):
        if show_paths:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            ax1, ax2 = axes[0]
            ax3, ax4 = axes[1]
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2)

        legends_shown: set[str] = set()

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

        # Format top-row axes
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

        # Path contributions panel (bottom row)
        if show_paths:
            _plot_paths_panel(
                ax3,
                ax4,
                collection.path_contributions,
                config.max_paths,
                kweight,
                overall_average=collection.overall_average,
                fourier_params=collection.fourier_params,
                n_sites_total=len(collection.individual_spectra) or None,
            )
            ax3.set_xlabel("k (Å⁻¹)")
            ax3.set_ylabel(data["chi_label"])
            ax3.set_title("Path contributions – k-space")
            ax3.legend(fontsize="x-small", ncol=2)

            ax4.set_xlabel("R (Å)")
            ax4.set_ylabel("|χ(R)| (Å⁻³)")
            ax4.set_title("Path contributions – R-space")
            ax4.legend(fontsize="x-small", ncol=2)

        plt.tight_layout()

        # Save if output directory provided
        plot_paths = {}
        if output_dir:
            # Convert to Path and ensure it exists
            # Caller should provide absolute path to avoid CWD issues
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
                "kweight": kweight,
                "style": config.style,
                "paths_plotted": show_paths,
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
