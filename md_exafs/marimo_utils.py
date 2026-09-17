"""Utility functions for working with Marimo notebooks and EXAFS data processing."""

import tempfile
from pathlib import Path

import marimo as mo
from ase.io import read
from weas_widget.atoms_viewer import AtomsViewer
from weas_widget.base_widget import BaseWidget
from weas_widget.utils import ASEAdapter


def view_atoms(
    atoms,
    model_style=1,
    boundary=None,
    show_bonded_atoms=True,
    highlight_indices=None,
):
    """Function to visualise an ASE Atoms object(or list of them) using weas_widget.

    using weas_widget.
    """
    guiConfig = {"controls": {"enabled": False}}
    if boundary is None:
        boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
    v = AtomsViewer(BaseWidget(guiConfig=guiConfig))
    v.atoms = ASEAdapter.to_weas(atoms)
    v.model_style = model_style
    v.boundary = boundary
    v.show_bonded_atoms = show_bonded_atoms
    v.color_type = "VESTA"
    v.cell.settings["showAxes"] = True
    if highlight_indices is not None:
        v.highlight.settings["selected"] = {
            "type": "sphere",
            "indices": highlight_indices,
            "color": "yellow",
            "scale": 1.2,
        }

    return v._widget


file_upload = mo.ui.file(label="Upload File", multiple=False)
input_kwargs_text = mo.ui.text_area(
    label="Input Kwargs (as dict)",
    value="",
    placeholder="e.g., {'format': 'xyz', 'index': 0}",
)
read_button = mo.ui.run_button(
    label="📁 Read Structure",
    kind="success",
    tooltip="Parse the uploaded file with current settings",
)
model_style = mo.ui.dropdown(
    options={"Ball": 0, "Ball and Stick": 1, "Polyhedral": 2, "Stick": 3},
    label="Model Style",
    value="Ball and Stick",
)
show_bonded_atoms = mo.ui.checkbox(label="Show atoms bonded beyond cell", value=True)


def process_uploaded_structure(structure_file, input_kwargs):
    """Process new structure/trajectory file."""
    suffix = f".{structure_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(structure_file.contents)
        temp_path = Path(tmp.name)
    atoms = read(temp_path, **input_kwargs)
    temp_path.unlink()  # Delete the temporary file
    return atoms
