"""EXAFS processing pipeline using Marimo app."""

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium", app_title="EXAFS Pipeline")


@app.cell
def _():
    import ast
    import tempfile
    import traceback
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import plotly.graph_objects as go
    from ase import Atoms
    from ase.io import read
    from weas_widget.atoms_viewer import AtomsViewer
    from weas_widget.base_widget import BaseWidget
    from weas_widget.utils import ASEAdapter

    # Package imports
    try:
        from larch.io import write_ascii

        from larch_cli_wrapper.feff_utils import (
            PRESETS,
            EdgeType,
            FeffConfig,
            WindowType,
        )
        from larch_cli_wrapper.wrapper import LarchWrapper, ProcessingMode
    except ImportError:
        mo.stop(
            mo.output.append(
                mo.md("""
            **❌ Import Error**: `larch_cli_wrapper` not found.

            Make sure:
            - The package is installed (`pip install larch-cli-wrapper`)
            - Or run `pip install -e .` if this is a local package
            """)
            )
        )

    # Constants
    CACHE_DIR = Path.home() / ".larch_cache"
    DEFAULT_OUTPUT_DIR = "outputs/exafs_pipeline"

    # I disabled the controls in the GUi, because the style is not loaded
    # properly inside Marimo notebook
    guiConfig = {"controls": {"enabled": False}}
    return (
        ASEAdapter,
        Atoms,
        AtomsViewer,
        BaseWidget,
        CACHE_DIR,
        DEFAULT_OUTPUT_DIR,
        EdgeType,
        FeffConfig,
        LarchWrapper,
        PRESETS,
        Path,
        ProcessingMode,
        SimpleNamespace,
        WindowType,
        ast,
        go,
        guiConfig,
        mo,
        read,
        tempfile,
        traceback,
        write_ascii,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # EXAFS Pipeline Processing

    Interactive EXAFS processing using the larch wrapper.
    Process single structures or trajectories with customizable parameters.
    """
    )
    return


@app.cell
def _(
    Atoms,
    file_upload,
    get_sampling_config,
    input_kwargs_text,
    mo,
    parse_kwargs_string,
    process_uploaded_structure,
    read_button,
):
    structure_list = []
    input_kwargs = {}
    parse_kwargs_msg = mo.md("")
    reading_structure_message = mo.md("")
    if read_button.value and file_upload.value:
        input_kwargs, parse_kwargs_msg = parse_kwargs_string(
            input_kwargs_text.value, get_sampling_config()
        )
        if file_upload.value:
            try:
                structure_list = process_uploaded_structure(
                    file_upload.value[0], input_kwargs=input_kwargs
                )
                if isinstance(structure_list, Atoms):
                    structure_list = [structure_list]
            except (OSError, ValueError, KeyError, TypeError) as e:
                # OSError: file reading issues, ValueError: parsing errors,
                # KeyError/TypeError: invalid input_kwargs
                structure_list = None
                reading_structure_message = mo.md(
                    f"**❌ Error reading structure:** {e}"
                )
    return reading_structure_message, structure_list


@app.cell
def _(in_form2, mo, reading_structure_message):
    mo.vstack([in_form2, reading_structure_message])
    return


@app.cell
def _(vis):
    vis
    return


@app.cell
def _(
    EdgeType,
    PRESETS,
    dk_input,
    enable_parallel,
    force_recalc_input,
    k_weight,
    kmax_input,
    kmin_input,
    mo,
    num_workers,
    output_dir_ui,
    process_existing_input,
    radius_input,
    species_list,
    window_type,
):
    form = (
        mo.md(r"""
        <style>
          .mo-tabs input {{ display: none; }}
          .mo-tabs .tab-labels {{
            display: flex;
            gap: 0.5rem;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1rem;
          }}
          .mo-tabs .tab-label {{
            padding: 0.5rem 1rem;
            cursor: pointer;
            border: 1px solid #e5e7eb;
            border-radius: 0.375rem 0.375rem 0 0;
            background: var(--background);
          }}
          .mo-tabs .tab-label:hover {{ background: var(--background); }}
          #tab-run:checked ~ .tab-labels label[for="tab-run"],
          #tab-analysis:checked ~ .tab-labels label[for="tab-analysis"],
          #tab-par:checked ~ .tab-labels label[for="tab-par"] {{
            background: var(--background);
            border-bottom-color: white;
            font-weight: 600;
          }}
          .mo-tabs .tab-panels {{
            border: 1px solid #e5e7eb;
            border-radius: 0 0.375rem 0.375rem 0.375rem;
            padding: 1rem;
            background: var(--background);
          }}
          .mo-tabs .panel {{ display: none; }}
          #tab-run:checked ~ .tab-panels .panel-run,
          #tab-analysis:checked ~ .tab-panels .panel-analysis,
          #tab-par:checked ~ .tab-panels .panel-par {{ display: block; }}
          .settings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
          }}
          .main-config {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: var(--gray-1);
            border-radius: 0.5rem;
          }}
        </style>

        **EXAFS Processing Pipeline**

        <div class="main-config">
          {absorber}
          {edge}
          {preset}
        </div>

        <div class="mo-tabs">
          <input type="radio" name="tabset" id="tab-run" checked>
          <input type="radio" name="tabset" id="tab-analysis">
          <input type="radio" name="tabset" id="tab-par">

          <div class="tab-labels">
            <label class="tab-label" for="tab-run">FEFF Run</label>
            <label class="tab-label" for="tab-analysis">Analysis</label>
            <label class="tab-label" for="tab-par">Parallelization</label>
          </div>

          <div class="tab-panels">
            <div class="panel panel-run">
              <div class="settings-grid">
                {radius_input}
                {process_existing_input}
                {force_recalc_input}
                {output_dir_ui}
              </div>
            </div>

            <div class="panel panel-analysis">
              <div class="settings-grid">
                {k_weight}
                {window_type}
                {dk_input}
                {kmin_input}
                {kmax_input}
              </div>
            </div>

            <div class="panel panel-par">
              <div class="settings-grid">
                {enable_parallel}
                {num_workers}
              </div>
            </div>
          </div>
        </div>
        """)
        .batch(
            # Main configuration
            absorber=mo.ui.dropdown(
                options=species_list,
                value=species_list[0] if species_list else None,
                label="Absorbing Species",
            ),
            edge=mo.ui.dropdown(
                options=[e.name for e in EdgeType], value="K", label="Edge"
            ),
            preset=mo.ui.dropdown(
                options={name.title(): name for name in PRESETS.keys()},
                value="Quick",
                label="Configuration Preset",
            ),
            # FEFF Run parameters
            radius_input=radius_input,
            process_existing_input=process_existing_input,
            force_recalc_input=force_recalc_input,
            output_dir_ui=output_dir_ui,
            # Analysis parameters
            k_weight=k_weight,
            window_type=window_type,
            dk_input=dk_input,
            kmin_input=kmin_input,
            kmax_input=kmax_input,
            # Parallelization parameters
            enable_parallel=enable_parallel,
            num_workers=num_workers,
        )
        .form(submit_button_label="Run EXAFS Processing", bordered=True)
    )
    form
    return (form,)


@app.cell
def _(form, mo, run_exafs_processing):
    mo.stop(form.value is None)

    message, result = run_exafs_processing()
    return message, result


@app.cell(hide_code=True)
def _(mo):
    plot_type = mo.ui.radio(["χ(k)", "k²χ(k)", "k³χ(k)", "|χ(R)|"], value="χ(k)")
    return (plot_type,)


@app.cell
def _(plot_output):
    plot_output
    return


@app.cell
def _(form):
    settings = form.value or {}
    return (settings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## 🗑️ Cache Management""")
    return


@app.cell(hide_code=True)
def _(clear_cache_button, mo, show_cache_info_button):
    mo.hstack([show_cache_info_button, clear_cache_button], justify="start")
    return


@app.cell(hide_code=True)
def _(clear_cache, mo, show_cache):
    show_cache_info_button = mo.ui.button(
        label="🔍 Show Cache Info", kind="neutral", on_click=show_cache
    )
    clear_cache_button = mo.ui.button(
        label="🗑️ Clear Cache", kind="danger", on_click=clear_cache
    )
    return clear_cache_button, show_cache_info_button


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 Usage Instructions

    1. **Upload** a structure (CIF, XYZ, etc.) or trajectory
    2. **Specify absorber** (e.g., `Fe`, `Cu`)
    3. **Choose processing mode**:
       - Single structure: One-off EXAFS
       - Trajectory: Average over frames
    4. **Select configuration preset**:
       - Quick: Fast results for testing
       - Publication: High-quality for final analysis
    5. **Run processing** or use existing outputs
    6. **Explore results** using the plot options

    💡 **Tips**:
    - Use "Process output only" to reanalyze existing FEFF results
    - Enable "Force recalculate" to bypass cache
    - Cache speeds up repeated calculations
    """
    )
    return


@app.cell
def _(ASEAdapter, AtomsViewer, BaseWidget, guiConfig):
    def view_atoms(
        atoms,
        model_style=1,
        boundary=None,
        show_bonded_atoms=True,
    ):
        """Function to visualise an ASE Atoms object(or list of them) using weas_widget.

        using weas_widget.
        """
        if boundary is None:
            boundary = [[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]]
        v = AtomsViewer(BaseWidget(guiConfig=guiConfig))
        v.atoms = ASEAdapter.to_weas(atoms)
        v.model_style = model_style
        v.boundary = boundary
        v.show_bonded_atoms = show_bonded_atoms
        v.color_type = "VESTA"
        v.cell.settings["showAxes"] = True
        return v._widget

    return (view_atoms,)


@app.cell
def _(ast, mo):
    # Functions to read in structures

    def parse_kwargs_string(
        text: str, existing_kwargs: dict | None = None
    ) -> tuple[dict, mo.md]:
        """Safely parse user input as a dict.

        Accepts Python-style dicts (single or double quotes, True/False)
        and JSON-style dicts.
        Returns existing_kwargs (or {}) if parsing fails.
        """
        kwargs = existing_kwargs.copy() if existing_kwargs else {}
        text = text.strip()

        if not text:
            return kwargs, mo.md("No extra kwargs provided.")

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                kwargs.update(parsed)
                return kwargs, mo.md(f"Parsed ASE read kwargs: `{parsed}`")
            else:
                return kwargs, mo.md(
                    "**Warning**: Input is not a dict, using existing kwargs."
                )
        except (ValueError, SyntaxError) as e:
            # ValueError: invalid literal, SyntaxError: malformed expression
            return kwargs, mo.md(
                f"**Error parsing kwargs**: {e}. Using existing kwargs."
            )

    return (parse_kwargs_string,)


@app.cell
def _(mo):
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
    return file_upload, input_kwargs_text, read_button


@app.cell
def _(
    file_upload,
    input_kwargs_text,
    mo,
    parameter_input,
    read_button,
    sampling_method,
):
    in_form2 = mo.vstack(
        [
            file_upload,
            mo.hstack([sampling_method, parameter_input]),
            input_kwargs_text,
            read_button,
        ],
        justify="space-around",
        align="start",
    )
    return (in_form2,)


@app.cell
def _(mo):
    model_style = mo.ui.dropdown(
        options={"Ball": 0, "Ball and Stick": 1, "Polyhedral": 2, "Stick": 3},
        label="Model Style",
        value="Ball and Stick",
    )
    show_bonded_atoms = mo.ui.checkbox(
        label="Show atoms bonded beyond cell", value=True
    )
    return model_style, show_bonded_atoms


@app.cell
def _(Path, read, tempfile):
    def process_uploaded_structure(structure_file, input_kwargs):
        """Process new structure/trajectory file."""
        suffix = f".{structure_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(structure_file.contents)
            temp_path = Path(tmp.name)
        atoms = read(temp_path, **input_kwargs)
        temp_path.unlink()  # Delete the temporary file
        return atoms

    return (process_uploaded_structure,)


@app.cell
def _(structure_list):
    species_list = sorted(
        {sym for atoms in structure_list for sym in atoms.get_chemical_symbols()}
    )
    return (species_list,)


@app.cell
def _(mo, model_style, show_bonded_atoms, structure_list, view_atoms):
    try:
        v = (
            view_atoms(
                structure_list,
                model_style=model_style.value,
                show_bonded_atoms=show_bonded_atoms.value,
            )
            if structure_list
            else mo.md("Upload a file to view the structure.")
        )
    except (ValueError, AttributeError, TypeError) as e:
        # ValueError: incompatible structures, AttributeError: missing properties,
        # TypeError: invalid input types for visualization
        # If we're dealing with the exception that weas can't display multiple
        # structures with different atoms types, then we can show the first
        # structure only and warn the user
        if (
            "All atoms must have the same species" in str(e)
            and isinstance(structure_list, list)
            and len(structure_list) > 1
        ):
            try:
                v = view_atoms(
                    structure_list[0],
                    model_style=model_style.value,
                    show_bonded_atoms=show_bonded_atoms.value,
                )
                v = mo.vstack(
                    [
                        v,
                        mo.md(
                            "**Warning**: Displaying only the first structure due to "
                            "differing atom types in trajectory."
                        ),
                    ]
                )
            except (ValueError, AttributeError, TypeError) as e2:
                # Same visualization errors for fallback attempt
                v = mo.md(f"**Error displaying structure(s):** {e2}")
        else:
            v = mo.md(f"**Error displaying structure(s):** {e}")

    vis = mo.vstack(
        [
            v,
            mo.hstack(
                [model_style, show_bonded_atoms], justify="space-around", align="center"
            ),
        ]
    )
    return (vis,)


@app.cell
def _(mo):
    # Create the sampling method dropdown
    sampling_method = mo.ui.dropdown(
        options=["all", "single", "every Nth"], value="all", label="Sampling Method"
    )
    return (sampling_method,)


@app.cell
def _(mo, sampling_method):
    # Create the appropriate input based on selection
    if sampling_method.value == "single":
        parameter_input = mo.ui.number(
            step=1, value=-1, label="Frame Index (0-based, -1 for last frame)"
        )
    elif sampling_method.value == "every Nth":
        parameter_input = mo.ui.number(
            start=1, step=1, value=1, label="N (every Nth frame)"
        )
    else:  # "all"
        parameter_input = mo.md("")

    def get_sampling_config() -> dict:
        """Construct an ASE-compatible index kwarg based on the options."""
        method = sampling_method.value

        if method == "all":
            index = ":"
        if method == "single":
            index = str(parameter_input.value)
        elif sampling_method.value == "every Nth":
            index = f"::{int(parameter_input.value)}"
        return {"index": index}

    return get_sampling_config, parameter_input


@app.cell
def _(FeffConfig):
    def create_feff_config(settings) -> FeffConfig:
        """Create a FeffConfig object from the current UI settings.

        Maps UI values to the appropriate FeffConfig fields.

        settings should be the result of form.value
        """
        return FeffConfig(
            # Map radius from UI
            radius=settings.get("radius_input"),
            # Map FEFF analysis parameters
            kmin=settings.get("kmin_input"),
            kmax=settings.get("kmax_input"),
            kweight=settings.get("k_weight"),
            dk=settings.get("dk_input"),
            window=settings.get("window_type"),
            # Map parallel settings
            parallel=settings.get("enable_parallel"),
            n_workers=settings.get("num_workers"),
            # Map force recalculate setting
            force_recalculate=settings.get("force_recalc_input"),
            # Map cleanup setting
            cleanup_feff_files=settings.get("cleanup_feff_files"),
        )

    return (create_feff_config,)


@app.cell
def _(ProcessingMode, SimpleNamespace, mo, traceback):
    def process_existing_outputs(wrapper, config, output_dir, absorber, is_traj):
        """Process existing FEFF outputs without running new calculations."""
        if is_traj:
            frame_dirs = sorted(
                [
                    d
                    for d in output_dir.iterdir()
                    if d.is_dir() and d.name.startswith("frame_")
                ]
            )
            if not frame_dirs:
                return mo.md(f"**❌ No trajectory frames found in {output_dir}**"), None

            # Use the new trajectory FEFF output processing method
            result = wrapper.process_trajectory_feff_outputs(
                frame_dirs=frame_dirs,
                output_dir=output_dir,
                config=config,
                plot_individual_frames=True,
                chi_weighting="chi",
            )
            return success_message(result, is_traj, output_dir), result
        else:
            result = process_single_existing(wrapper, config, output_dir, absorber)
            return mo.md(f"### ✅ Processed existing output in `{output_dir}`"), result

    def process_single_existing(wrapper, config, output_dir, absorber):
        """Process single existing FEFF output."""
        exafs_group = wrapper.process_feff_output(output_dir, config)
        wrapper.plot_results(
            exafs_group,
            output_dir,
            absorber=absorber,
            edge=config.edge,
            chi_weighting="chi",
        )
        return SimpleNamespace(
            exafs_group=exafs_group,
            processing_mode=ProcessingMode.SINGLE_FRAME,
            nframes=1,
        )

    def process_new_structures(
        wrapper, structures, config, output_dir, absorber, is_traj
    ):
        """Process new structure(s). Input is a list of ASE Atoms objects."""
        try:
            with mo.status.progress_bar(
                total=100,
                title="Processing EXAFS...",
                subtitle="Starting...",
                completion_title="✅ Processing Complete",
                remove_on_exit=True,
            ) as bar:

                def progress_callback(completed, total, desc):
                    bar.total = total
                    if completed == 0:
                        bar.total = total
                    bar.update(increment=1, subtitle=desc)

                result = wrapper.process(
                    structure=structures,
                    absorber=absorber,
                    output_dir=output_dir,
                    config=config,
                    trajectory=is_traj,
                    plot_individual_frames=True,
                    progress_callback=progress_callback,
                )
            return success_message(result, is_traj, output_dir), result
        except (OSError, ValueError, RuntimeError, FileNotFoundError) as e:
            # OSError: file operations, ValueError: invalid parameters,
            # RuntimeError: processing failures, FileNotFoundError: missing files
            return mo.md(f"""
                ### ❌ Processing Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """), None

    def success_message(result, is_traj, output_dir):
        """Generate success message based on processing mode."""
        if is_traj:
            return mo.md(f"""
                ### ✅ Trajectory Processing Complete
                - **Frames processed**: {result.nframes}
                - **Output**: `{output_dir}`
            """)
        return mo.md(f"""
            ### ✅ Single Structure Processed
            - **Output**: `{output_dir}`
        """)

    return process_existing_outputs, process_new_structures


@app.cell
def _(WindowType, mo):
    # Create individual UI elements first (these will be accessible)
    radius_input = mo.ui.number(label="Radius (Å)", value=6.0, start=1.0)
    process_existing_input = mo.ui.checkbox(
        label="Process existing FEFF outputs (skip FEFF run)", value=False
    )
    force_recalc_input = mo.ui.checkbox(
        label="Force recalculate (ignore cache)", value=False
    )
    output_dir_ui = mo.ui.text(
        label="Output Directory",
        value="outputs/exafs_pipeline",
        placeholder="Directory for output files",
    )

    k_weight = mo.ui.number(label="k-weight", value=2, start=0, step=1)
    window_type = mo.ui.dropdown(
        options=[w.value for w in WindowType],
        value=WindowType.HANNING,
        label="Window type for FT",
    )
    dk_input = mo.ui.number(label="dk (Å⁻¹)", value=0.3, start=0.1, step=0.1)
    kmin_input = mo.ui.number(label="kmin (Å⁻¹)", value=3.0, start=0.0, step=0.1)
    kmax_input = mo.ui.number(label="kmax (Å⁻¹)", value=12.0, start=0.1, step=0.1)

    cleanup_feff_files = mo.ui.checkbox(
        label="Clean up unnecessary FEFF output files", value=True
    )
    enable_parallel = mo.ui.checkbox(label="Enable parallel processing", value=True)
    num_workers = mo.ui.number(label="Number of workers (auto if blank)", value=None)

    # Now create the layout using the individual elements
    feff_run_settings = mo.vstack(
        [
            radius_input,
            process_existing_input,
            force_recalc_input,
            output_dir_ui,
        ]
    )

    feff_analysis_settings = mo.vstack(
        [
            k_weight,
            window_type,
            dk_input,
            kmin_input,
            kmax_input,
        ]
    )

    parallel_settings = mo.vstack([enable_parallel, num_workers])

    # # Accordion for the different sections
    # settings_tabs = mo.tabs(
    #     {
    #         "FEFF Run": feff_run_settings,
    #         "FEFF Analysis": feff_analysis_settings,
    #         "Parallelisation": parallel_settings,
    #     }
    # )
    return (
        dk_input,
        enable_parallel,
        force_recalc_input,
        k_weight,
        kmax_input,
        kmin_input,
        num_workers,
        output_dir_ui,
        process_existing_input,
        radius_input,
        window_type,
    )


@app.cell
def _(form):
    form.value
    return


@app.cell
def _(
    CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    LarchWrapper,
    Path,
    create_feff_config,
    mo,
    process_existing_input,
    process_existing_outputs,
    process_new_structures,
    settings,
    structure_list,
    traceback,
):
    def run_exafs_processing():
        """Function to execute the whole EXAFS processing pipeline."""
        if not settings or not structure_list or not settings.get("absorber"):
            return mo.md(
                "### ❌ Processing Failed: Missing inputs or configuration."
            ), None

        processing_absorber = settings["absorber"].strip()
        output_dir = Path(settings.get("output_dir", DEFAULT_OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create configuration
        config = create_feff_config(settings)
        config.edge = settings.get("edge", "K")

        is_traj = len(structure_list) > 1

        try:
            with LarchWrapper(verbose=False, cache_dir=CACHE_DIR) as processing_wrapper:
                if process_existing_input.value:
                    message, result = process_existing_outputs(
                        processing_wrapper,
                        config,
                        output_dir,
                        processing_absorber,
                        is_traj,
                    )
                else:
                    message, result = process_new_structures(
                        processing_wrapper,
                        structure_list,
                        config,
                        output_dir,
                        processing_absorber,
                        is_traj,
                    )
        except (OSError, ValueError, RuntimeError, FileNotFoundError) as e:
            # OSError: file operations, ValueError: invalid parameters,
            # RuntimeError: processing failures, FileNotFoundError: missing files
            message = mo.md(f"""
                ### ❌ Processing Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """)
            result = None

        return message, result

    return (run_exafs_processing,)


@app.cell
def _(
    DEFAULT_OUTPUT_DIR,
    Path,
    go,
    message,
    mo,
    plot_type,
    result,
    settings,
    write_ascii,
):
    def create_plot(
        exafs_group, individual_frames, plot_type, show_legend, absorber, edge
    ):
        """Create plot based on selected options."""
        fig = go.Figure()
        common_layout = {
            "font": {"family": "Times New Roman", "size": 18},
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "margin": {"l": 80, "r": 30, "t": 50, "b": 70},
            "xaxis": {
                "showline": True,
                "linewidth": 2,
                "linecolor": "black",
                "mirror": True,
                "gridcolor": "lightgray",
                "tickwidth": 2,
            },
            "yaxis": {
                "showline": True,
                "linewidth": 2,
                "linecolor": "black",
                "mirror": True,
                "gridcolor": "lightgray",
                "tickwidth": 2,
            },
            "legend": {"x": 0.02, "y": 0.95, "bgcolor": "rgba(0,0,0,0)"},
        }

        if plot_type != "|χ(R)|":
            # k-space plot
            add_chi_plot(fig, exafs_group, individual_frames, plot_type, show_legend)
            # Set title and axis labels based on chi weighting
            if plot_type == "χ(k)":
                title = "EXAFS χ(k)"
                yaxis_title = "χ(k)"
            elif plot_type == "k²χ(k)":
                title = "EXAFS k²χ(k)"
                yaxis_title = "k²χ(k)"
            elif plot_type == "k³χ(k)":
                title = "EXAFS k³χ(k)"
                yaxis_title = "k³χ(k)"

            fig.update_layout(
                title=title,
                xaxis_title="k [Å⁻¹]",
                yaxis_title=yaxis_title,
                **common_layout,
            )
        else:
            add_ft_plot(fig, exafs_group, individual_frames, show_legend)
            fig.update_layout(
                title="Fourier Transform |χ(R)|",
                xaxis_title="R [Å]",
                yaxis_title="|χ(R)|",
                **common_layout,
            )

        fig.add_annotation(
            text=f"{absorber} {edge} edge",
            x=0.9,
            y=0.98,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16},
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )
        return fig

    def add_chi_plot(fig, exafs_group, individual_frames, chi_weighting, show_legend):
        """Add chi(k) plot traces."""

        # Apply weighting to data
        def apply_weighting(k, chi):
            if chi_weighting == "k²χ(k)":
                return k**2 * chi
            elif chi_weighting == "k³χ(k)":
                return k**3 * chi
            else:  # "χ(k)"
                return chi

        should_plot_frames = individual_frames and len(individual_frames) > 1

        if should_plot_frames:
            for i, frame in enumerate(individual_frames):
                weighted_chi = apply_weighting(frame.k, frame.chi)
                fig.add_trace(
                    go.Scatter(
                        x=frame.k,
                        y=weighted_chi,
                        name=f"Frame {i + 1}",
                        line={"width": 1, "color": "rgba(128,128,128,0.3)"},
                        showlegend=show_legend and i == 0,
                    )
                )

        weighted_chi = apply_weighting(exafs_group.k, exafs_group.chi)
        k = exafs_group.k

        if hasattr(exafs_group, "chi_std") and should_plot_frames:
            weighted_std = apply_weighting(
                k, exafs_group.chi_std
            )  # TODO: Check if this is correct
            fig.add_trace(
                go.Scatter(
                    x=list(k) + list(k[::-1]),
                    y=list(weighted_chi + weighted_std)
                    + list((weighted_chi - weighted_std)[::-1]),
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.1)",
                    line={"color": "rgba(255,255,255,0)"},
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=k,
                y=weighted_chi,
                name=f"{chi_weighting} Average ± σ"
                if should_plot_frames
                else chi_weighting,
                line={"width": 2.5, "color": "black"},
            )
        )

    def add_ft_plot(fig, exafs_group, individual_frames, show_legend):
        """Add Fourier transform plot traces."""
        should_plot_frames = individual_frames and len(individual_frames) > 1

        if should_plot_frames:
            for i, frame in enumerate(individual_frames):
                fig.add_trace(
                    go.Scatter(
                        x=frame.r,
                        y=frame.chir_mag,
                        name=f"Frame {i + 1}",
                        line={"width": 1, "color": "rgba(128,128,128,0.3)"},
                        showlegend=show_legend and i == 0,
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=exafs_group.r,
                y=exafs_group.chir_mag,
                name="|χ(R)| Average" if should_plot_frames else "|χ(R)|",
                line={"width": 2.5, "color": "black"},
            )
        )

    def save_raw_data(exa, iframes):
        # this can be probably passed but better in case we want to change it.
        outdir = Path(settings.get("output_dir", DEFAULT_OUTPUT_DIR))
        exafs_chir = outdir / "exafs.chir"
        write_ascii(exafs_chir, exa.r, exa.chir_mag, label="r [Å]      |χ(r)| [a.u.]")
        exafs_chik = outdir / "exafs.chik"
        write_ascii(
            exafs_chik,
            exa.k,
            exa.chi,
            exa.k**2 * exa.chi,
            exa.k**3 * exa.chi,
            label="k [1/Å]      χ(k) [a.u.]   k²χ(k) [a.u.]   k³χ(k) [a.u.]",
        )
        for i, frame in enumerate(iframes):
            write_ascii(
                outdir / f"exafs{i}.chir",
                frame.r,
                frame.chir_mag,
                label="r [Å]      |χ(r)| [a.u.]",
            )
            write_ascii(
                outdir / f"exafs{i}.chik",
                frame.k,
                frame.chi,
                frame.k**2 * frame.chi,
                frame.k**3 * frame.chi,
                label="k [1/Å]      χ(k) [a.u.]   k²χ(k) [a.u.]   k³χ(k) [a.u.]",
            )

    # Skip if no result
    if result is None:
        plot_output = message
    else:
        # Main plot rendering
        exafs_group = result.exafs_group
        edge = settings.get("edge", "K")
        plot_absorber = settings.get("absorber", "")
        show_legend = False

        # Get individual frames if they exist
        individual_frames = getattr(result, "individual_frame_groups", None)
        save_raw_data(exafs_group, individual_frames)
        fig = create_plot(
            exafs_group,
            individual_frames,
            plot_type.value,
            show_legend,
            plot_absorber,
            edge,
        )

        plot_output = mo.vstack([mo.hstack([plot_type, fig], justify="start"), message])
    return (plot_output,)


@app.cell(hide_code=True)
def _(CACHE_DIR, LarchWrapper, mo):
    def clear_cache(button_value=None):
        """Clear the Larch cache directory."""
        try:
            with LarchWrapper(cache_dir=CACHE_DIR) as wrapper:
                wrapper.clear_cache()
                message = mo.md("### 🗑️ Cache Cleared Successfully")

        except (OSError, PermissionError, FileNotFoundError) as e:
            # OSError: file system errors, PermissionError: access denied,
            # FileNotFoundError: cache directory missing
            message = mo.md(f"### ❌ Cache Error\n{str(e)}")
        mo.output.append(message)
        # return None

    def show_cache(button_value=None):
        """Show the Larch cache information."""
        try:
            with LarchWrapper(cache_dir=CACHE_DIR) as wrapper:
                info = wrapper.get_cache_info()
                message = mo.md(f"""
                        ### 📊 Cache Status
                        | Property | Value |
                        |----------|-------|
                        | **Status** | ✅ Enabled |
                        | **Directory** | `{info["cache_dir"]}` |
                        | **Files** | {info["files"]} |
                        | **Size** | {info["size_mb"]} MB |
                    """)

        except (OSError, PermissionError, FileNotFoundError) as e:
            # OSError: file system errors, PermissionError: access denied,
            # FileNotFoundError: cache directory missing
            message = mo.md(f"### ❌ Cache Error\n{str(e)}")
        mo.output.append(message)
        # return message

    return clear_cache, show_cache


if __name__ == "__main__":
    app.run()
