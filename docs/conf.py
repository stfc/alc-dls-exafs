"""Sphinx configuration for MD-EXAFS documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Add package source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

project = "MD-EXAFS"
copyright = "2026, Kane Shenton, Joshua Elliott, Alin M. Elena"
author = "Kane Shenton, Joshua Elliott, Alin M. Elena"
release = "0.2.0"
version = "0.2.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

# Source suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "adr", "REFACTORING_PLAN.md"]

# -- MyST Parser settings ----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
    "html_admonition",
]
myst_heading_anchors = 3

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- HTML output settings ----------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "MD-EXAFS"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "repository_url": "https://github.com/stfc/alc-dls-exafs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "navigation_with_keys": True,
}
