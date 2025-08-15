"""
Matplotlib style files for EXAFS plotting.

This module contains matplotlib style files that provide consistent,
publication-quality formatting for EXAFS plots matching the marimo
notebook aesthetic.

Available styles:
- exafs_publication.mplstyle: High-quality publication plots (300 DPI)
- exafs_presentation.mplstyle: Large fonts for presentations (150 DPI)
"""

from pathlib import Path

STYLES_DIR = Path(__file__).parent

# Available style files
PUBLICATION_STYLE = STYLES_DIR / "exafs_publication.mplstyle"
PRESENTATION_STYLE = STYLES_DIR / "exafs_presentation.mplstyle"

__all__ = ["STYLES_DIR", "PUBLICATION_STYLE", "PRESENTATION_STYLE"]
