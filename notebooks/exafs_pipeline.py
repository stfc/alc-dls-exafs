"""EXAFS processing pipeline using Marimo app."""

import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium", app_title="EXAFS Pipeline")


@app.cell
def _():
    import ast
    import tempfile
    import traceback
    from pathlib import Path

    import marimo as mo
    from ase import Atoms
    from ase.io import read

    from larch_cli_wrapper.marimo_utils import (
        file_upload,
        input_kwargs_text,
        model_style,
        read_button,
        show_bonded_atoms,
        view_atoms,
    )

    # Package imports
    try:
        from larch_cli_wrapper.exafs_data import (
            PlotConfig,
            plot_exafs_plotly,
        )
        from larch_cli_wrapper.feff_utils import (
            EdgeType,
            FeffConfig,
            WindowType,
        )
        from larch_cli_wrapper.pipeline import PipelineProcessor
        # from larch_cli_wrapper.wrapper import LarchWrapper, ProcessingMode
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
    return (
        Atoms,
        CACHE_DIR,
        DEFAULT_OUTPUT_DIR,
        EdgeType,
        FeffConfig,
        Path,
        PipelineProcessor,
        PlotConfig,
        WindowType,
        ast,
        file_upload,
        input_kwargs_text,
        mo,
        model_style,
        plot_exafs_plotly,
        read,
        read_button,
        show_bonded_atoms,
        tempfile,
        traceback,
        view_atoms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # EXAFS Pipeline Processing

    Interactive EXAFS processing using the new PipelineProcessor architecture.

    **Three-Stage Workflow:**
    - **Stage A: Input Generation** - Create FEFF input files
    - **Stage B: FEFF Execution** - Run FEFF calculations
    - **Stage C: Analysis** - Process results and create plots
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
def _(file_upload_form, mo, reading_structure_message):
    mo.vstack([file_upload_form, reading_structure_message])
    return


@app.cell
def _(mo, model_style, show_bonded_atoms, structure_list, v):
    mo.vstack(
        [
            mo.md(f"Loaded {len(structure_list)} structure(s)."),
            v,
            mo.hstack(
                [model_style, show_bonded_atoms], justify="space-around", align="center"
            ),
        ]
    )
    return


@app.cell
def _(input_form, mo):
    mo.output.append(input_form)
    return


@app.cell
def _(input_message, mo):
    # Display Input Generation results
    input_message if input_message is not None else mo.md(
        "### 📝 Generate input files to proceed"
    )
    return


@app.cell
def _(feff_form, mo):
    mo.output.append(feff_form)
    return


@app.cell
def _(feff_message, mo):
    # Display FEFF Execution results
    feff_message if feff_message is not None else mo.md(
        "### 🔬 Run FEFF calculations on inputs from Stage A"
    )
    return


@app.cell
def _(analysis_form, mo):
    mo.output.append(analysis_form)
    return


@app.cell
def _(mo, plot_output):
    if plot_output is not None:
        mo.output.append(plot_output)
    return


@app.cell
def _(EdgeType, mo, output_dir_ui, radius_input, species_list):
    # Stage A: Input Generation Form
    input_form = (
        mo.md(r"""
        <style>
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

        **Input File Generation**

        Generate FEFF input files for your structures. This creates all necessary
        feff.inp files without running the calculations.

        **Absorber Options:**
        - **Species name**: e.g., `Fe`, `Cu`, `Ag` (processes all sites of that element)
        - **Site indices**: e.g., `0,1,2,3` (processes specific sites by index)
        - **Combined**: e.g., `Fe:0,1` (first two Fe sites only)

        <div class="main-config">
          {absorber}
          {edge}
        </div>

        <div class="settings-grid">
          {radius_input}
          {output_dir_ui}
          {all_sites}
          {all_frames}
        </div>
        """)
        .batch(
            # Main configuration
            absorber=mo.ui.text(
                label="Absorbing Species or Indices",
                value=species_list[0] if species_list else "",
                placeholder="e.g., 'Fe' or '0,1,2,3' for first 4 sites",
            ),
            edge=mo.ui.dropdown(
                options=[e.name for e in EdgeType], value="K", label="Edge"
            ),
            # Input generation parameters
            radius_input=radius_input,
            output_dir_ui=output_dir_ui,
            all_sites=mo.ui.checkbox(
                label="Process all sites of selected element", value=True
            ),
            all_frames=mo.ui.checkbox(
                label="Process all frames (for trajectories)", value=True
            ),
        )
        .form(submit_button_label="📝 Generate Input Files", bordered=True)
    )
    return (input_form,)


@app.cell
def _(enable_parallel, force_recalc_input, input_form, mo, num_workers):
    # Stage B: FEFF Execution Form
    feff_form = (
        mo.md(r"""
        **Run FEFF Calculations**

        Execute FEFF calculations on the input files generated in Stage A.
        This will run the actual FEFF computations to generate χ(k) spectra
        for the specific structures and sites from the previous step.

        <div class="settings-grid">
          {enable_parallel}
          {num_workers}
          {force_recalc_input}
          {cleanup_files}
        </div>
        """)
        .batch(
            enable_parallel=enable_parallel,
            num_workers=num_workers,
            force_recalc_input=force_recalc_input,
            cleanup_files=mo.ui.checkbox(
                label="Clean up intermediate files", value=True
            ),
        )
        .form(
            submit_button_label="🔬 Run FEFF Calculations",
            bordered=True,
            submit_button_disabled=input_form.value is None,
        )
    )
    return (feff_form,)


@app.cell
def _(feff_form, mo, run_feff_execution):
    # Stage B: FEFF Execution Processing
    mo.stop(feff_form.value is None)

    feff_message, feff_result = run_feff_execution()
    return feff_message, feff_result


@app.cell
def _(dk_input, feff_form, k_weight, kmax_input, kmin_input, mo, window_type):
    # Stage C: Analysis Form
    analysis_form = (
        mo.md(r"""
        **Analyze Results**

        Process FEFF outputs from Stage B to create averaged spectra and
        publication-ready plots. This stage takes the FEFF calculation results
        and applies Fourier transforms with your specified parameters to generate
        side-by-side χ(k) and χ(R) plots.

        The k-weight setting controls both the Fourier transform and the χ(k)
        plot type display.

        <div class="settings-grid">
          {plot_mode}
          {k_weight}
          {window_type}
          {dk_input}
          {kmin_input}
          {kmax_input}
          {show_plots}
          {plot_style}
        </div>
        """)
        .batch(
            # Plot configuration
            plot_mode=mo.ui.dropdown(
                options={
                    "Overall Average": "overall",
                    "Frame Averages": "frames",
                    "Site Averages": "sites",
                },
                value="Overall Average",
                label="Plot Mode",
            ),
            plot_style=mo.ui.dropdown(
                options={"Publication": "publication", "Presentation": "presentation"},
                value="Publication",
                label="Plot Style",
            ),
            show_plots=mo.ui.checkbox(label="Show interactive plots", value=True),
            # Fourier transform parameters
            k_weight=k_weight,
            window_type=window_type,
            dk_input=dk_input,
            kmin_input=kmin_input,
            kmax_input=kmax_input,
        )
        .form(
            submit_button_label="📊 Analyze Results",
            bordered=True,
            submit_button_disabled=feff_form.value is None,
        )
    )
    return (analysis_form,)


@app.cell
def _(input_form, mo, run_input_generation):
    # Stage A: Input Generation Processing
    mo.stop(input_form.value is None)

    input_message, input_result = run_input_generation()
    return input_message, input_result


@app.cell
def _(analysis_form, mo, run_analysis):
    # Analysis Processing
    mo.stop(analysis_form.value is None)

    analysis_message, analysis_result = run_analysis()
    return analysis_message, analysis_result


@app.cell
def _():
    return


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

    **📝 Input Generation**


    1. **Upload** a structure (CIF, XYZ, etc.) or trajectory
    3. **Specify absorber** (e.g., `Fe`, `Cu`)
    4. **Configure** radius and generation settings
    5. **Generate input files** - creates all feff.inp files needed

    **🔬 FEFF Execution**

    1. **Uses input files from Stage A automatically** - no directory scanning
    2. **Configure** parallel execution and caching options
    3. **Run FEFF calculations** to generate χ(k) spectra
    4. **Monitor progress** of calculations

    **📊 Analysis**

    1. **Uses FEFF results from Stage B automatically** - no directory scanning
    2. **Choose plot mode** (overall/frames/sites averages)
    3. **Adjust** Fourier transform parameters (k-weight, window, etc.)
    4. **Generate side-by-side χ(k) and χ(R) plots** - k-weight controls χ(k) plot type
    """
    )
    return


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
def _(
    file_upload,
    input_kwargs_text,
    mo,
    parameter_input,
    read_button,
    sampling_method,
):
    file_upload_form = mo.vstack(
        [
            file_upload,
            mo.hstack([sampling_method, parameter_input]),
            input_kwargs_text,
            read_button,
        ],
        justify="space-around",
        align="start",
    )
    return (file_upload_form,)


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
    return (v,)


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
    def create_feff_config(settings):
        """Create a FeffConfig object from the current UI settings.

        Maps UI values to the appropriate FeffConfig fields.
        Provides sensible defaults for missing parameters.

        settings should be the result of form.value
        """
        return FeffConfig(
            # Map radius from UI (FEFF calculation parameter)
            radius=settings.get("radius_input", 4.0),
            # Map FEFF analysis parameters (with defaults for missing values)
            kmin=settings.get("kmin_input", 3.0),
            kmax=settings.get("kmax_input", 12.0),
            kweight=settings.get("k_weight", 2),
            dk=settings.get("dk_input", 0.3),
            window=settings.get("window_type", "hanning"),
            # Map parallel settings (with defaults)
            parallel=settings.get("enable_parallel", False),
            n_workers=settings.get("num_workers", None),
            # Map force recalculate setting (with default)
            force_recalculate=settings.get("force_recalc_input", False),
            # Map cleanup setting (with default)
            cleanup_feff_files=settings.get("cleanup_feff_files", True),
        )

    def update_config_from_settings(config, settings, setting_type="both"):
        """Update an existing FeffConfig with parameters from settings.

        Args:
            config: Existing FeffConfig to update
            settings: Dictionary of settings from UI forms
            setting_type: "feff", "analysis", or "both" to specify which
                parameters to update
        """
        if setting_type in ("analysis", "both"):
            # Analysis/Fourier transform parameters
            config.kmin = settings.get("kmin_input", config.kmin)
            config.kmax = settings.get("kmax_input", config.kmax)
            config.kweight = settings.get("k_weight", config.kweight)
            config.dk = settings.get("dk_input", config.dk)
            config.window = settings.get("window_type", config.window)

        if setting_type in ("feff", "both"):
            # FEFF calculation parameters
            config.radius = settings.get("radius_input", config.radius)
            config.parallel = settings.get("enable_parallel", config.parallel)
            config.n_workers = settings.get("num_workers", config.n_workers)
            config.force_recalculate = settings.get(
                "force_recalc_input", config.force_recalculate
            )
            config.edge = settings.get("edge", getattr(config, "edge", "K"))

        return config

    return create_feff_config, update_config_from_settings


@app.cell
def _(PipelineProcessor, mo, traceback):
    def run_input_generation_only(
        structures, config, output_dir, absorber, all_sites, all_frames, cache_dir=None
    ):
        """Stage A: Generate FEFF input files using PipelineProcessor."""
        try:
            with mo.status.progress_bar(
                total=100,
                title="Generating FEFF Input Files...",
                subtitle="Initializing...",
                completion_title="✅ Input Generation Complete",
                remove_on_exit=True,
            ) as bar:
                # Initialize PipelineProcessor with cache
                processor = PipelineProcessor(config=config, cache_dir=cache_dir)
                bar.update(increment=20, subtitle="Initialized processor...")

                # Parse absorber specification using first structure as reference
                # Import the CLI parsing function for consistency
                from larch_cli_wrapper.cli import parse_absorber_specification

                reference_atoms = structures[0]
                absorber_spec = parse_absorber_specification(
                    absorber, reference_atoms, all_sites
                )

                bar.update(
                    increment=10,
                    subtitle=f"Parsed absorber: {absorber_spec['description']}",
                )

                # Determine structure type and generate inputs
                if len(structures) == 1 and not all_frames:
                    # Single structure
                    batch = processor.input_generator.generate_single_site_inputs(
                        structure=structures[0],
                        absorber=absorber_spec["absorber"],
                        output_dir=output_dir,
                    )
                    bar.update(
                        increment=50, subtitle="Generated single structure inputs..."
                    )
                else:
                    # Trajectory or multiple frames
                    batch = processor.input_generator.generate_trajectory_inputs(
                        structures=structures,
                        absorber=absorber_spec["absorber"],
                        output_dir=output_dir,
                    )
                    bar.update(increment=50, subtitle="Generated trajectory inputs...")

                bar.update(increment=20, subtitle="Complete!")

                return mo.md(f"""
                    ### ✅ Input Generation Complete
                    - **Generated:** {len(batch.tasks)} FEFF input files
                    - **Frames:** {len(structures)}
                    - **Sites:** {len({task.site_index for task in batch.tasks})}
                    - **Output:** {output_dir}
                    - **Absorber:** {absorber_spec["description"]}

                    Ready for FEFF calculations!
                    """), batch

        except (OSError, ValueError, RuntimeError) as e:
            return mo.md(f"""
                ### ❌ Input Generation Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """), None

    def run_feff_execution_only(batch, config, parallel=True, cache_dir=None):
        """Stage B: Execute FEFF calculations using PipelineProcessor."""
        if batch is None or not batch.tasks:
            return mo.md("### ❌ FEFF Execution Failed: No input tasks provided."), None

        try:
            with mo.status.progress_bar(
                total=len(batch.tasks),
                title="Running FEFF Calculations...",
                subtitle="Initializing...",
                completion_title="✅ FEFF Execution Complete",
                remove_on_exit=True,
            ) as bar:
                # Initialize executor with cache
                processor = PipelineProcessor(config=config, cache_dir=cache_dir)
                bar.update(increment=0, subtitle="Initialized executor...")

                # Create progress callback for marimo progress bar
                def progress_callback(completed: int, total: int):
                    """Update marimo progress bar with FEFF calculation progress."""
                    # Calculate how much to advance (marimo uses incremental updates)
                    current_progress = getattr(progress_callback, "_current", 0)
                    increment = completed - current_progress
                    progress_callback._current = completed

                    bar.update(
                        increment=increment,
                        subtitle=f"FEFF calculations: {completed}/{total} complete",
                    )

                # Initialize progress tracking
                progress_callback._current = 0

                # Execute all FEFF calculations with real-time progress
                task_results = processor.feff_executor.execute_batch(
                    batch=batch,
                    parallel=parallel,
                    progress_callback=progress_callback,
                )

                # Final update
                bar.update(increment=0, subtitle="Complete!")

                # Count successful calculations
                successful_tasks = sum(
                    1 for success in task_results.values() if success
                )

                # Get batch information for display
                frames = len({task.frame_index for task in batch.tasks})
                sites = len({task.site_index for task in batch.tasks})

                return mo.md(f"""
                    ### ✅ FEFF Execution Complete
                    - **Processed:** {successful_tasks}/{len(batch.tasks)} calculations
                    - **Frames:** {frames}
                    - **Sites:** {sites}
                    - **Parallel:** {"Yes" if parallel else "No"}
                    - **Cache:** {
                    "Enabled" if processor.feff_executor.cache_dir else "Disabled"
                }
                    - **Source:** Generated from Stage A inputs

                    Ready for analysis!
                    """), (batch, task_results)

        except (OSError, ValueError, RuntimeError) as e:
            return mo.md(f"""
                ### ❌ FEFF Execution Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """), None

    def run_analysis_only(
        batch,
        task_results,
        config,
        plot_mode="overall",
        show_plot=False,
        plot_style="publication",
        cache_dir=None,
    ):
        """Stage C: Analyze results using PipelineProcessor."""
        try:
            with mo.status.progress_bar(
                total=100,
                title="Analyzing FEFF Results...",
                subtitle="Loading results...",
                completion_title="✅ Analysis Complete",
                remove_on_exit=True,
            ) as bar:
                # Initialize processor with cache
                processor = PipelineProcessor(config=config, cache_dir=cache_dir)
                bar.update(increment=20, subtitle="Loading results...")

                # Load successful results
                groups = processor.result_processor.load_successful_results(
                    batch=batch,
                    task_results=task_results,
                )
                bar.update(increment=30, subtitle="Creating averages...")

                # Create averages based on data structure
                frame_averages = processor.result_processor.create_frame_averages(
                    groups=groups, batch=batch
                )
                site_averages = processor.result_processor.create_site_averages(
                    groups=groups, batch=batch
                )
                overall_average = processor.result_processor.create_overall_average(
                    all_groups=list(groups.values())
                )
                bar.update(increment=30, subtitle="Generating plots...")

                # Create data collection for plotting
                from larch_cli_wrapper.exafs_data import EXAFSDataCollection

                data_collection = EXAFSDataCollection(
                    individual_spectra=list(groups.values()),
                    overall_average=overall_average,
                    frame_averages=frame_averages,
                    site_averages=site_averages,
                    kweight_used=config.kweight,
                    fourier_params=config.fourier_params,
                )
                bar.update(increment=20, subtitle="Complete!")

                # Prepare result summary
                n_frames = len(frame_averages)
                n_sites = len(site_averages)
                n_successful = len(groups)

                return mo.md(f"""
                    ### ✅ Analysis Complete
                    - **Processed:** {n_successful} calculations from Stage B
                    - **Frames:** {n_frames}
                    - **Sites:** {n_sites}
                    - **Plot mode:** {plot_mode}
                    - **k-weighting:** {config.kweight}
                    - **Source:** FEFF results from Stage B

                    Data collection created successfully!
                    """), data_collection

        except (OSError, ValueError, RuntimeError) as e:
            return mo.md(f"""
                ### ❌ Analysis Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """), None

    return (
        run_analysis_only,
        run_feff_execution_only,
        run_input_generation_only,
    )


@app.cell
def _(WindowType, mo):
    # Create individual UI elements first (these will be accessible)
    radius_input = mo.ui.number(
        label="Radius (Å)", value=4.0, start=1.0
    )  # temporary default 2.0
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

    # Plot control checkboxes
    show_individual_ui = mo.ui.checkbox(label="Show individual spectra", value=False)
    show_overall_average_ui = mo.ui.checkbox(label="Show overall average", value=True)
    show_frame_averages_ui = mo.ui.checkbox(label="Show frame averages", value=False)
    show_site_averages_ui = mo.ui.checkbox(label="Show site averages", value=False)
    show_legend_ui = mo.ui.checkbox(label="Show legend", value=True)

    enable_parallel = mo.ui.checkbox(label="Enable parallel processing", value=True)
    num_workers = mo.ui.number(label="Number of workers (auto if blank)", value=None)
    return (
        dk_input,
        enable_parallel,
        force_recalc_input,
        k_weight,
        kmax_input,
        kmin_input,
        num_workers,
        output_dir_ui,
        radius_input,
        show_frame_averages_ui,
        show_individual_ui,
        show_legend_ui,
        show_overall_average_ui,
        show_site_averages_ui,
        window_type,
    )


@app.cell
def _(
    CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    Path,
    create_feff_config,
    input_form,
    mo,
    run_input_generation_only,
    structure_list,
    traceback,
):
    def run_input_generation():
        """Stage A: Generate FEFF input files from uploaded structures."""
        if not input_form or not structure_list:
            return mo.md("### ❌ Input Generation Failed: Missing inputs."), None

        input_settings = input_form.value or {}
        processing_absorber = input_settings.get("absorber", "").strip()
        if not processing_absorber:
            return mo.md("### ❌ Input Generation Failed: No absorber specified."), None

        output_dir = Path(input_settings.get("output_dir_ui", DEFAULT_OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create configuration from settings
        config = create_feff_config(input_settings)

        all_sites = input_settings.get("all_sites", True)
        all_frames = input_settings.get("all_frames", True)

        try:
            message, result = run_input_generation_only(
                structures=structure_list,
                config=config,
                output_dir=output_dir,
                absorber=processing_absorber,
                all_sites=all_sites,
                all_frames=all_frames,
                cache_dir=CACHE_DIR,
            )
        except (OSError, ValueError, RuntimeError) as e:
            message = mo.md(f"""
                ### ❌ Input Generation Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """)
            result = None

        return message, result

    return (run_input_generation,)


@app.cell
def _(
    CACHE_DIR,
    create_feff_config,
    feff_form,
    input_result,
    mo,
    run_feff_execution_only,
    traceback,
    update_config_from_settings,
):
    def run_feff_execution():
        """Stage B: Execute FEFF calculations on generated input files."""
        if not feff_form or not input_result:
            return mo.md(
                "### ❌ FEFF Execution Failed: Missing inputs. Please run "
                "Stage A (Input Generation) first."
            ), None

        feff_settings = feff_form.value or {}

        # Display information about what will be processed
        len({task.frame_index for task in input_result.tasks})
        len({task.site_index for task in input_result.tasks})

        # Create configuration from settings
        config = create_feff_config(feff_settings)
        config = update_config_from_settings(config, feff_settings, "feff")

        parallel = feff_settings.get("enable_parallel", True)

        try:
            message, result = run_feff_execution_only(
                batch=input_result,  # FeffBatch from input generation
                config=config,
                parallel=parallel,
                cache_dir=CACHE_DIR,
            )
        except (OSError, ValueError, RuntimeError) as e:
            message = mo.md(f"""
                ### ❌ FEFF Execution Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """)
            result = None

        return message, result

    return (run_feff_execution,)


@app.cell
def _(
    CACHE_DIR,
    analysis_form,
    create_feff_config,
    feff_result,
    mo,
    run_analysis_only,
    traceback,
    update_config_from_settings,
):
    def run_analysis():
        """Stage C: Analyze FEFF results and create plots."""
        if not analysis_form or not feff_result:
            return mo.md(
                "### ❌ Analysis Failed: Missing FEFF results. Please run "
                "Stage B (FEFF Execution) first."
            ), None

        analysis_settings = analysis_form.value or {}

        # Create configuration from settings
        config = create_feff_config(analysis_settings)
        config = update_config_from_settings(config, analysis_settings, "analysis")

        plot_mode = analysis_settings.get("plot_mode", "overall")
        show_plot = analysis_settings.get("show_plots", True)
        plot_style = analysis_settings.get("plot_style", "publication")

        # Extract batch and task_results from feff_result
        batch, task_results = feff_result

        try:
            message, result = run_analysis_only(
                batch=batch,
                task_results=task_results,
                config=config,
                plot_mode=plot_mode,
                show_plot=show_plot,
                plot_style=plot_style,
                cache_dir=CACHE_DIR,
            )
        except (OSError, ValueError, RuntimeError) as e:
            message = mo.md(f"""
                ### ❌ Analysis Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """)
            result = None

        return message, result

    return (run_analysis,)


@app.cell
def _(analysis_form, feff_form, input_form):
    # Extract settings from forms
    input_settings = input_form.value or {}
    feff_settings = feff_form.value or {}
    analysis_settings = analysis_form.value or {}
    return analysis_settings, feff_settings, input_settings


@app.cell
def _(
    PlotConfig,
    analysis_message,
    analysis_result,
    analysis_settings,
    create_feff_config,
    feff_message,
    feff_result,
    feff_settings,
    input_settings,
    mo,
    plot_exafs_plotly,
    show_frame_averages_ui,
    show_individual_ui,
    show_legend_ui,
    show_overall_average_ui,
    show_site_averages_ui,
    update_config_from_settings,
):
    # Extract plot control settings
    show_individual = show_individual_ui.value
    show_overall_average = show_overall_average_ui.value
    show_frame_averages = show_frame_averages_ui.value
    show_site_averages = show_site_averages_ui.value
    show_legend = show_legend_ui.value

    # Determine which result to use for plotting
    if analysis_result is not None:
        result = analysis_result
        settings = analysis_settings
        message = analysis_message
        # Create config with defaults and update with analysis parameters
        config = create_feff_config({})
        config = update_config_from_settings(config, settings, "analysis")
    elif feff_result is not None:
        result = feff_result
        settings = feff_settings
        message = feff_message
        # Create config with defaults and update with FEFF parameters
        config = create_feff_config({})
        config = update_config_from_settings(config, settings, "feff")
    else:
        result = None
        settings = {}
        message = mo.md("### ℹ️ Run FEFF calculations or analysis to see plots")
        config = None

    # Skip if no result
    if result is None:
        plot_output = message
    else:
        # Main plot rendering using plot_exafs_plotly and interactive plots
        edge = input_settings.get("edge", "")
        plot_absorber = settings.get("absorber", "")

        # Use existing plot data collection if available, otherwise create it
        if (
            hasattr(result, "plot_data_collection")
            and result.plot_data_collection is not None
        ):
            plot_data_collection = result.plot_data_collection
        else:
            # Fallback: assume result is already EXAFSDataCollection
            plot_data_collection = result

        try:
            plot_kweight = (
                config.kweight if config and hasattr(config, "kweight") else 2
            )

            # Create PlotConfig for the interactive plotly subplots
            plot_config = PlotConfig(
                plot_individual=show_individual,
                plot_overall_avg=show_overall_average,
                plot_frame_avg=show_frame_averages,
                plot_site_avg=show_site_averages,
                absorber=plot_absorber,
                edge=edge,
                kweight=plot_kweight,
                style=None,
                show_legend=show_legend,
            )

            # Create interactive plot using the centralized function
            interactive_fig = plot_exafs_plotly(
                plot_data_collection,
                plot_config,
            )

            plot_output = mo.vstack(
                [
                    mo.hstack(
                        [
                            show_overall_average_ui,
                            show_individual_ui,
                            show_frame_averages_ui,
                            show_site_averages_ui,
                        ],
                        justify="start",
                    ),
                    show_legend_ui,
                    interactive_fig,
                    message,
                ]
            )

        except (ValueError, TypeError, RuntimeError) as e:
            # If plotting fails, show error but keep the message
            plot_output = mo.vstack(
                [
                    mo.md(f"""
                ### ❌ Plotting Error
                **Error:** {str(e)}

                Data is available but plotting failed. Check the plotting functions.
                """),
                    message,
                ]
            )
    return (plot_output,)


@app.cell(hide_code=True)
def _(CACHE_DIR, FeffConfig, PipelineProcessor, mo):
    def clear_cache(button_value=None):
        """Clear the cache directory using PipelineProcessor."""
        try:
            processor = PipelineProcessor(config=FeffConfig(), cache_dir=CACHE_DIR)
            files_cleared = processor.clear_cache()
            message = mo.md(
                f"### 🗑️ Cache Cleared Successfully\n{files_cleared} files removed"
            )

        except (OSError, PermissionError, FileNotFoundError) as e:
            # OSError: file system errors, PermissionError: access denied,
            # FileNotFoundError: cache directory missing
            message = mo.md(f"### ❌ Cache Error\n{str(e)}")
        mo.output.append(message)

    def show_cache(button_value=None):
        """Show cache information using PipelineProcessor."""
        try:
            processor = PipelineProcessor(config=FeffConfig(), cache_dir=CACHE_DIR)
            info = processor.get_cache_info()
            message = mo.md(f"""
                        ### 📊 Cache Status
                        - **Enabled:** {"✓" if info["enabled"] else "✗"}
                        - **Directory:** {info.get("cache_dir", "N/A")}
                        - **Files:** {info.get("files", 0)}
                        - **Size:** {info.get("size_mb", 0):.2f} MB
                        """)

        except (OSError, PermissionError, FileNotFoundError) as e:
            # OSError: file system errors, PermissionError: access denied,
            # FileNotFoundError: cache directory missing
            message = mo.md(f"### ❌ Cache Error\n{str(e)}")
        mo.output.append(message)

    return clear_cache, show_cache


if __name__ == "__main__":
    app.run()
