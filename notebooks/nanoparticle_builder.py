"""Nanoparticle builder."""
# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import io

    import marimo as mo
    from ase.io import write
    from weas_widget.atoms_viewer import AtomsViewer
    from weas_widget.base_widget import BaseWidget
    from weas_widget.utils import ASEAdapter

    # I disabled the controls in the GUi, because the style is not loaded
    # properly inside Marimo notebook
    guiConfig = {"controls": {"enabled": False}}
    return ASEAdapter, AtomsViewer, BaseWidget, ast, guiConfig, io, mo, write


@app.cell
def _(ASEAdapter, AtomsViewer, BaseWidget, guiConfig):
    def view_atoms(atoms, model_style=1):
        v = AtomsViewer(BaseWidget(guiConfig=guiConfig))
        v.atoms = ASEAdapter.to_weas(atoms)
        v.model_style = model_style
        return v._widget

    return (view_atoms,)


@app.function
def setup_cluster(
    # Basic parameters
    symbol1="Au",  # Shell/side1 symbol
    symbol2="Ag",  # Core/side2 symbol
    method="cutoff",  # Method: 'cutoff', 'hemisphere', or 'mixed'
    crystal_structure="fcc",  # Crystal structure: 'fcc', 'bcc', or 'sc'
    # Structure parameters - surfaces and layers
    surface_specs=None,  # List of (surface_tuple, n_layers) pairs
    latticeconstant=4.09,  # Lattice constant in Angstroms
    # Method-specific parameters
    cutoff=3.0,  # Cutoff distance for 'cutoff' method (Angstroms)
    axis_x=1.0,  # X component of hemisphere axis
    axis_y=0.0,  # Y component of hemisphere axis
    axis_z=0.0,  # Z component of hemisphere axis
    ratio=0.5,  # Ratio of symbol2 atoms for 'mixed' method
    vacuum=10.0,  # Vacuum padding around the cluster (Angstroms)
):
    """Create a bimetallic nanoparticle cluster with different composition methods.

    Parameters:
    -----------
    symbol1, symbol2 : str
        Chemical symbols for the two elements
    method : str
        Method for atom distribution: 'cutoff', 'hemisphere', or 'mixed'
    crystal_structure : str
        Crystal structure type: 'fcc', 'bcc', or 'sc'
    latticeconstant : float
        Lattice constant in Angstroms
    surface_specs : list of tuples
        List of ((h,k,l), n_layers) pairs defining surfaces and their layer counts
        Default: [((1,0,0), 6), ((1,1,0), 9), ((1,1,1), 5)]
    cutoff : float
        Distance cutoff for core-shell method (Angstroms)
    axis_x, axis_y, axis_z : float
        Components of axis vector for hemisphere method
    ratio : float
        Fraction of atoms that should be symbol2 for mixed method (0-1)
    vacuum : float
        Vacuum padding around the cluster (Angstroms)
    """
    import numpy as np
    from ase.cluster import BodyCenteredCubic, FaceCenteredCubic, SimpleCubic

    # Normalize crystal structure input
    crystal_structure = crystal_structure.lower().strip()

    # Define crystal structure classes
    structure_map = {
        "fcc": FaceCenteredCubic,
        "bcc": BodyCenteredCubic,
        "sc": SimpleCubic,
    }

    # Validate crystal structure
    if crystal_structure not in structure_map:
        valid_structures = ", ".join(f"'{k}'" for k in structure_map.keys())
        raise ValueError(
            f"Invalid crystal_structure '{crystal_structure}'. "
            f"Valid options are: {valid_structures}"
        )

    # Get the appropriate cluster class
    ClusterClass = structure_map[crystal_structure]

    # Default surface specifications if none provided
    if surface_specs is None:
        surface_specs = [((1, 0, 0), 6), ((1, 1, 0), 9), ((1, 1, 1), 5)]

    # Extract surfaces and layers from surface_specs
    if surface_specs:
        surfaces = [spec[0] for spec in surface_specs]  # Extract (h,k,l) tuples
        layers = [spec[1] for spec in surface_specs]  # Extract layer counts
        print(f"Crystal structure: {crystal_structure.upper()}")
        print(f"Surfaces: {surfaces}, Layers: {layers}")
    else:
        surfaces = None
        layers = None

    # Create symbols list from individual parameters
    symbols = [symbol1, symbol2]

    # Create axis vector from components
    axis_vec = (axis_x, axis_y, axis_z)

    # Create the cluster with the appropriate crystal structure
    atoms = ClusterClass(symbol1, surfaces, layers, latticeconstant=latticeconstant)

    atoms.center(vacuum=vacuum, axis=(0, 1, 2))
    atoms.set_pbc([True, True, True])

    positions = atoms.positions
    center = atoms.get_center_of_mass()

    if method == "cutoff":
        dist = np.linalg.norm(positions - center, axis=1)
        shell = np.where(dist >= cutoff)[0]
        core = np.where(dist < cutoff)[0]
        atoms.symbols[:] = symbols[0]
        atoms.symbols[core] = symbols[1]
        print(f"Method: cutoff   cutoff = {cutoff:.2f} Å")
        print(f"  shell atoms ({symbol1}): {len(shell)}")
        print(f"  core  atoms ({symbol2}): {len(core)}")

    elif method == "hemisphere":
        axis = np.array(axis_vec, float)
        axis /= np.linalg.norm(axis)
        dots = np.dot(positions - center, axis)
        side1 = np.where(dots >= 0)[0]
        side2 = np.where(dots < 0)[0]
        atoms.symbols[:] = symbols[0]
        atoms.symbols[side2] = symbols[1]
        print(f"Method: hemisphere along {axis_vec}")
        print(f"  side1 ({symbol1}): {len(side1)}")
        print(f"  side2 ({symbol2}): {len(side2)}")

    elif method == "mixed":
        N = len(atoms)
        indices = np.arange(N)
        np.random.shuffle(indices)
        n2 = int(round(ratio * N))
        atoms.symbols[:] = symbols[0]
        atoms.symbols[indices[:n2]] = symbols[1]
        print(
            f"Method: mixed nanoparticle (ratio={ratio})  "
            f"{symbol1}: {N - n2}, {symbol2}: {n2}"
        )

    else:
        valid_methods = "'cutoff', 'hemisphere', 'mixed'"
        raise ValueError(
            f"Invalid method '{method}'. Valid options are: {valid_methods}"
        )

    return atoms


@app.cell
def _(mo):
    # Create reactive state for surfaces list
    get_surfaces, set_surfaces = mo.state(
        [
            {"h": 1, "k": 0, "l": 0, "layers": 6},
            {"h": 1, "k": 1, "l": 0, "layers": 9},
            {"h": 1, "k": 1, "l": 1, "layers": 5},
        ],
        allow_self_loops=True,
    )

    # Basic parameter UI elements
    crystal_strucuture_ui = mo.ui.dropdown(
        ["fcc", "bcc", "sc"], value="fcc", label="Crystal Structure"
    )
    symbol1_ui = mo.ui.dropdown(
        ["Au", "Ag", "Cu", "Pt", "Pd"], value="Au", label="Symbol 1"
    )
    symbol2_ui = mo.ui.dropdown(
        ["Au", "Ag", "Cu", "Pt", "Pd"], value="Ag", label="Symbol 2"
    )
    method_ui = mo.ui.dropdown(
        ["cutoff", "hemisphere", "mixed"], value="cutoff", label="Method"
    )
    lattice_ui = mo.ui.number(
        start=2.0, value=4.09, step=0.01, label="Lattice constant (Å)"
    )
    vacuum_ui = mo.ui.number(
        start=0.0, value=10.0, step=0.5, label="Vacuum padding (Å)"
    )

    # Method-specific parameters
    cutoff_ui = mo.ui.slider(
        1.0, 10.0, value=3.0, step=0.1, label="Cutoff distance (Å)"
    )
    axis_x_ui = mo.ui.slider(-1.0, 1.0, value=1.0, step=0.1, label="Axis X")
    axis_y_ui = mo.ui.slider(-1.0, 1.0, value=0.0, step=0.1, label="Axis Y")
    axis_z_ui = mo.ui.slider(-1.0, 1.0, value=0.0, step=0.1, label="Axis Z")
    ratio_ui = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="Symbol2 ratio")

    # Surface management buttons
    add_surface_btn = mo.ui.button(
        label="Add Surface",
        on_change=lambda _: set_surfaces(
            lambda v: v + [{"h": 1, "k": 0, "l": 0, "layers": 5}]
        ),
    )

    remove_surface_btn = mo.ui.button(
        label="Remove Last Surface",
        on_change=lambda _: set_surfaces(lambda v: v[:-1] if len(v) > 1 else v),
    )

    # helpers
    def update_surface(i, key, new_value):
        set_surfaces(
            lambda arr: [
                ({**s, key: int(new_value)} if idx == i else s)
                for idx, s in enumerate(arr)
            ]
        )

    def remove_at(i):
        set_surfaces(lambda arr: arr[:i] + arr[i + 1 :])

    return (
        add_surface_btn,
        axis_x_ui,
        axis_y_ui,
        axis_z_ui,
        crystal_strucuture_ui,
        cutoff_ui,
        get_surfaces,
        lattice_ui,
        method_ui,
        ratio_ui,
        remove_at,
        remove_surface_btn,
        symbol1_ui,
        symbol2_ui,
        update_surface,
        vacuum_ui,
    )


@app.cell
def _(get_surfaces, mo, update_surface):
    def render_surface_ui():
        rows = []
        for idx, surf in enumerate(get_surfaces()):
            rows.append(
                mo.hstack(
                    [
                        mo.ui.number(
                            label="h",
                            value=surf["h"],
                            on_change=lambda v, i=idx: update_surface(i, "h", v),
                        ),
                        mo.ui.number(
                            label="k",
                            value=surf["k"],
                            on_change=lambda v, i=idx: update_surface(i, "k", v),
                        ),
                        mo.ui.number(
                            label="l",
                            value=surf["l"],
                            on_change=lambda v, i=idx: update_surface(i, "l", v),
                        ),
                        mo.ui.number(
                            label="layers",
                            value=surf["layers"],
                            on_change=lambda v, i=idx: update_surface(i, "layers", v),
                        ),
                    ]
                )
            )
        return mo.vstack(rows)

    return


@app.cell
def _(
    axis_x_ui,
    axis_y_ui,
    axis_z_ui,
    crystal_strucuture_ui,
    cutoff_ui,
    lattice_ui,
    method_ui,
    mo,
    ratio_ui,
    symbol1_ui,
    symbol2_ui,
    vacuum_ui,
):
    # Display the UI
    # display the axis_x_ui etc only if method is hemisphere
    method_ui_options = []
    if method_ui.value == "cutoff":
        method_ui_options = [cutoff_ui]
    elif method_ui.value == "hemisphere":
        method_ui_options = [axis_x_ui, axis_y_ui, axis_z_ui]
    elif method_ui.value == "mixed":
        method_ui_options = [ratio_ui]

    mo.hstack(
        [
            mo.vstack(
                [
                    crystal_strucuture_ui,
                    mo.hstack([symbol1_ui, symbol2_ui], justify="start"),
                    method_ui,
                    lattice_ui,
                    vacuum_ui,
                ]
            ),
            mo.vstack(method_ui_options),
        ]
    )
    return


@app.cell
def _(get_surfaces, mo, remove_at, update_surface):
    rows = mo.ui.array(
        [
            mo.ui.array(
                [  # one row per surface
                    mo.ui.number(
                        step=1,
                        label="h",
                        value=surf["h"],
                        on_change=lambda v, i=i: update_surface(i, "h", v),
                    ),
                    mo.ui.number(
                        step=1,
                        label="k",
                        value=surf["k"],
                        on_change=lambda v, i=i: update_surface(i, "k", v),
                    ),
                    mo.ui.number(
                        step=1,
                        label="l",
                        value=surf["l"],
                        on_change=lambda v, i=i: update_surface(i, "l", v),
                    ),
                    mo.ui.number(
                        step=1,
                        label="layers",
                        value=surf["layers"],
                        on_change=lambda v, i=i: update_surface(i, "layers", v),
                    ),
                    mo.ui.button(label="Remove", on_change=lambda _, i=i: remove_at(i)),
                ]
            )
            for i, surf in enumerate(get_surfaces())
        ]
    )
    return (rows,)


@app.cell
def _(add_surface_btn, mo, remove_surface_btn, rows):
    mo.vstack(
        [
            mo.vstack([mo.hstack(row) for row in rows]),
            mo.hstack([add_surface_btn, remove_surface_btn]),
        ]
    )
    return


@app.cell
def _(get_surfaces, mo):
    current_surfaces = [((s["h"], s["k"], s["l"]), s["layers"]) for s in get_surfaces()]
    mo.md(f"""
    **Current Surface Configuration:**
    {
        chr(10).join(
            [
                f"- Surface {i}: ({spec[0][0]},{spec[0][1]},{spec[0][2]}) "
                + f"with {spec[1]} layers"
                for i, spec in enumerate(current_surfaces)
            ]
        )
    }
    """)
    return


@app.cell
def _(
    axis_x_ui,
    axis_y_ui,
    axis_z_ui,
    crystal_strucuture_ui,
    cutoff_ui,
    get_surfaces,
    lattice_ui,
    method_ui,
    ratio_ui,
    symbol1_ui,
    symbol2_ui,
    vacuum_ui,
):
    # Get current values directly from UI elements for the cluster creation
    surface_specs = [
        ((surface["h"], surface["k"], surface["l"]), surface["layers"])
        for surface in get_surfaces()
    ]

    atoms = setup_cluster(
        symbol1=symbol1_ui.value,
        symbol2=symbol2_ui.value,
        method=method_ui.value,
        crystal_structure=crystal_strucuture_ui.value,
        surface_specs=surface_specs,
        latticeconstant=lattice_ui.value,
        cutoff=cutoff_ui.value,
        axis_x=axis_x_ui.value,
        axis_y=axis_y_ui.value,
        axis_z=axis_z_ui.value,
        ratio=ratio_ui.value,
        vacuum=vacuum_ui.value,
    )
    atoms  # This will display the ASE atoms object
    return (atoms,)


@app.cell
def _(atoms, view_atoms):
    v = view_atoms(atoms)
    v
    return


@app.cell
def _(atoms, mo):
    mo.md(
        f"""
    ### Cluster Summary

    - **Total atoms:** {len(atoms)}
    - **Composition:** {atoms.get_chemical_formula()}
    - **Diameter (volume):** {atoms.get_diameter(method="volume"):.2f}"  Å
      (diameter of a sphere with the same volume as the atoms.
    - **Diameter (shape):** {atoms.get_diameter(method="shape"):.2f} Å
      (averaged diameter calculated from the directions given by the defined surfaces.)
    """
    )
    return


@app.cell
def _(write):
    def save_atoms_to_file(atoms, filename, write_kwargs):
        write(filename, atoms, **write_kwargs)
        print(f"Cluster saved to {filename}")

    return


@app.cell
def _(ast):
    def parse_kwargs_string(text: str):
        """Safely parse user input as a dict.

        Accepts Python-style dicts (single or double quotes, True/False)
        and JSON-style dicts.

        Returns {} if parsing fails.
        """
        text = text.strip()
        if not text:
            return {}
        # literal_eval is safe: no arbitrary code execution
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
        else:
            return {}

    return (parse_kwargs_string,)


@app.cell
def _(io, write):
    def atoms_to_contents(atoms, fmt: str, **write_kwargs):
        """Try to write Atoms in text mode; fall back to binary if needed.

        Returns str (for text) or bytes (for binary).
        """

        def _handle_single_atom_error(error_msg: str, format_name: str) -> ValueError:
            """Handle the specific error when format can only store 1 Atoms object."""
            if "can only store 1 Atoms object" in error_msg:
                return ValueError(
                    f"Format '{format_name}' can only store 1 Atoms object. "
                    f"Try adding 'index': 0 to the write kwargs to select "
                    f"the first structure, "
                    f"or use a format that supports multiple structures"
                    f" (e.g., 'xyz', 'traj')."
                )
            return ValueError(
                f"Unable to write atoms in format '{format_name}': {error_msg}"
            )

        def _try_write(buffer_type, buffer_class):
            """Attempt to write to the given buffer type."""
            try:
                buf = buffer_class()
                write(buf, atoms, format=fmt, **write_kwargs)
                return buf.getvalue()
            except (TypeError, UnicodeDecodeError):
                raise
            except ValueError as e:
                # Format-specific errors should be handled immediately
                raise _handle_single_atom_error(str(e), fmt) from e
            except Exception as e:
                # Any other unexpected error
                raise ValueError(
                    f"Unexpected error writing {buffer_type}"
                    + f"format '{fmt}': {str(e)}"
                ) from e

        # Try text mode first
        try:
            return _try_write("text", io.StringIO)
        except (TypeError, UnicodeDecodeError):
            # Fall back to binary mode for encoding issues
            pass
        except ValueError:
            # Format errors should not fall back to binary - re-raise immediately
            raise

        # Try binary mode
        return _try_write("binary", io.BytesIO)

    return (atoms_to_contents,)


@app.cell
def _(atoms, atoms_to_contents, mo):
    write_formats = ["xyz", "cif", "vasp", "extxyz"]
    fmt_dropdown = mo.ui.dropdown(
        options=write_formats, value="cif", label="Output format"
    )

    def make_download(fmt_choice, file_prefix="converted_structure", write_kwargs=None):
        contents = atoms_to_contents(atoms, fmt_choice, **write_kwargs)
        return mo.download(contents, filename=f"{file_prefix}")

    output_kwargs_text = mo.ui.text_area(label="Output kwargs", value="{}")
    return fmt_dropdown, make_download, output_kwargs_text


@app.cell
def _(fmt_dropdown, mo):
    output_file_name = mo.ui.text(
        label="Output File Name", value=f"converted_file.{fmt_dropdown.value}"
    )
    return (output_file_name,)


@app.cell
def _(
    atoms,
    fmt_dropdown,
    make_download,
    mo,
    output_file_name,
    output_kwargs_text,
    parse_kwargs_string,
):
    output_kwargs = parse_kwargs_string(output_kwargs_text.value)
    download_link = (
        make_download(
            fmt_dropdown.value,
            file_prefix=output_file_name.value,
            write_kwargs=output_kwargs,
        )
        if atoms
        else mo.md("Create a cluster to enable download.")
    )

    mo.vstack(
        [
            mo.md("### 💾 Save Cluster to File"),
            fmt_dropdown,
            output_file_name,
            output_kwargs_text,
            download_link,
        ],
        justify="center",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
