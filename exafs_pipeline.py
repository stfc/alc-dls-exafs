import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import plotly.graph_objects as go
    import tempfile
    import traceback
    from types import SimpleNamespace

    # Package imports
    try:
        from larch_cli_wrapper.wrapper import LarchWrapper, ProcessingMode
        from larch_cli_wrapper.feff_utils import FeffConfig, PRESETS, EdgeType
    except ImportError:
        mo.stop(
            mo.output.append(mo.md("""
            **❌ Import Error**: `larch_cli_wrapper` not found.

            Make sure:
            - The package is installed (`pip install larch-cli-wrapper`)
            - Or run `pip install -e .` if this is a local package
            """))
        )

    # Constants
    CACHE_DIR = Path.home() / ".larch_cache"
    DEFAULT_OUTPUT_DIR = "outputs/exafs_pipeline"

    return (
        CACHE_DIR,
        DEFAULT_OUTPUT_DIR,
        EdgeType,
        FeffConfig,
        LarchWrapper,
        PRESETS,
        Path,
        ProcessingMode,
        SimpleNamespace,
        go,
        mo,
        os,
        tempfile,
        traceback,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # EXAFS Pipeline Processing

    Interactive EXAFS processing using the streamlined larch wrapper. 
    Process single structures or trajectories with customizable parameters.
    """
    )
    return


@app.cell
def _(EdgeType, PRESETS, mo):
    form = (
        mo.md('''
        **EXAFS Processing Pipeline**

        Structure/Trajectory File: {structure_file}

        {absorber}

        {edge}

        {processing_mode}

        {sample_interval}

        **Configuration**

        Configuration Preset: {preset}

        **Processing Options**

        {parallel_settings}

        Output Directory: {output_dir}

        {run_options}
        ''')
        .batch(
            structure_file=mo.ui.file(label="Structure/Trajectory File", multiple=False),
            absorber=mo.ui.text(label="Absorbing Species", placeholder="e.g. Fe, Cu, O"),
            edge=mo.ui.dropdown(
                options=[e.name for e in EdgeType],
                value="K",
                label="Edge"
            ),
            processing_mode=mo.ui.radio(
                options=["Single structure", "Trajectory (all frames)"],
                value="Single structure",
                label="Processing Mode"
            ),
            sample_interval=mo.ui.number(
                label="Sample interval (trajectories only)",
                value=1,
                start=1,
            ),
            preset=mo.ui.dropdown(
                options={name.title(): name for name in PRESETS.keys()},
                value="Quick",
                label="Configuration Preset"
            ),
            parallel_settings=mo.ui.dictionary({
                "parallel": mo.ui.checkbox(label="Enable parallel processing", value=True),
                "n_workers": mo.ui.number(label="Number of workers (auto if blank)", value=None)
            }),
            output_dir=mo.ui.text(
                label="Output Directory",
                value="outputs/exafs_pipeline",
                placeholder="Directory for output files"
            ),
            run_options=mo.ui.dictionary({
                "process_output_only": mo.ui.checkbox(label="Process existing FEFF outputs (skip FEFF run)", value=False),
                "force_recalculate": mo.ui.checkbox(label="Force recalculate (ignore cache)", value=False),
            })
        )
        .form(
            submit_button_label="Run EXAFS Processing",
            bordered=True
        ))
    form
    return (form,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    plot_type =  mo.ui.radio(["χ(k)", "|χ(R)|"], value="χ(k)")
    # mo.center(plot_type)
    return (plot_type,)


@app.cell
def _(form, mo):
    settings = form.value or {}
    settings_message = (
        mo.md("*Configure and submit the form above to start processing.*")
        if not settings
        else mo.md(f"""
            **⚙️ Current Settings:**
            | Setting | Value |
            |--------|-------|
            | 📁 Structure | {settings['structure_file'][0].name} |
            | 🎯 Absorber | {settings.get('absorber', 'Not set')} |
            | ⚡ Mode | {settings.get('processing_mode', 'Not set')} |
            | 🔧 Preset | {settings.get('preset', 'Not set').title()} |
            | 🚀 Parallel | {'Yes' if settings['parallel_settings']['parallel'] else 'No'} |
            | 📝 Process Output Only | {'Yes' if settings['run_options']['process_output_only'] else 'No'} |
            | 💾 Force Recalculate | {'Yes' if settings['run_options']['force_recalculate'] else 'No'} |
        """)
    )
    return (settings,)


@app.cell
def _(
    CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    FeffConfig,
    LarchWrapper,
    Path,
    ProcessingMode,
    SimpleNamespace,
    mo,
    os,
    settings,
    tempfile,
    traceback,
):
    def process_existing_outputs(wrapper, config, output_dir, absorber, is_traj):
        """Process existing FEFF outputs without running new calculations"""
        if is_traj:
            frame_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')]
            if not frame_dirs:
                return mo.md(f"**❌ No trajectory frames found in {output_dir}**"), None

            result = wrapper.process(
                output_dir, absorber, output_dir, config,
                trajectory=True, plot_individual_frames=True
            )
            return success_message(result, is_traj, output_dir), result
        else:
            result = process_single_existing(wrapper, config, output_dir, absorber)
            return mo.md(f"### ✅ Processed existing output in `{output_dir}`"), result

    def process_single_existing(wrapper, config, output_dir, absorber):
        """Process single existing FEFF output"""
        exafs_group = wrapper.process_feff_output(output_dir, config)
        wrapper.plot_results(
            exafs_group, output_dir, absorber=absorber, edge=config.edge
        )
        return SimpleNamespace(
            exafs_group=exafs_group,
            processing_mode=ProcessingMode.SINGLE_FRAME,
            nframes=1
        )

    def process_new_file(wrapper, structure_file, config, output_dir, absorber, is_traj):
        """Process new structure/trajectory file"""
        suffix = f".{structure_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(structure_file.contents)
            temp_path = Path(tmp.name)

        try:
            with mo.status.progress_bar(
                total=None,
                title="Processing EXAFS...",
                subtitle="Starting...",
                completion_title="✅ Processing Complete",
                remove_on_exit=True,
            ) as bar:
                def progress_callback(completed, total, desc):
                    bar.total = total
                    if completed == 0: bar.total = total
                    bar.update(increment=1, subtitle=desc)

                result = wrapper.process(
                    structure=temp_path,
                    absorber=absorber,
                    output_dir=output_dir,
                    config=config,
                    trajectory=is_traj,
                    plot_individual_frames=True,
                    progress_callback=progress_callback
                )
            return success_message(result, is_traj, output_dir), result
        finally:
            if temp_path.exists(): os.unlink(temp_path)

    def success_message(result, is_traj, output_dir):
        """Generate success message based on processing mode"""
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

    # Main processing logic
    if not settings or not settings.get("structure_file") or not settings.get("absorber"):
        message, result = None, None
    else:
        structure_file = settings["structure_file"][0]
        processing_absorber = settings["absorber"].strip()
        output_dir = Path(settings.get("output_dir", DEFAULT_OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create configuration
        config = FeffConfig.from_preset(settings["preset"])
        config.edge = settings.get("edge", "K")
        config.force_recalculate = settings["run_options"]["force_recalculate"]

        # Configure trajectory settings
        is_traj = settings["processing_mode"] == "Trajectory (all frames)"
        if is_traj:
            config.sample_interval = settings["sample_interval"]
            config.parallel = settings["parallel_settings"]["parallel"]
            config.n_workers = settings["parallel_settings"]["n_workers"]

        try:
            with LarchWrapper(verbose=False, cache_dir=CACHE_DIR) as processing_wrapper:
                if settings["run_options"]["process_output_only"]:
                    message, result = process_existing_outputs(processing_wrapper, config, output_dir, processing_absorber, is_traj)
                else:
                    message, result = process_new_file(processing_wrapper, structure_file, config, output_dir, processing_absorber, is_traj)
        except Exception as e:
            message = mo.md(f"""
                ### ❌ Processing Failed
                **Error:** {str(e)}
                ```
                {traceback.format_exc()}
                ```
                """)
            result = None

    return message, result


@app.cell
def _(go, message, mo, plot_type, result, settings):
    def create_plot(exafs_group, individual_frames, plot_type, show_legend, absorber, edge):
        """Create plot based on selected options"""
        fig = go.Figure()
        common_layout = {
            "font": {"family": "Times New Roman", "size": 18},
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "margin": {"l": 80, "r": 30, "t": 50, "b": 70},
            "xaxis": {"showline": True, "linewidth": 2, "linecolor": "black", "mirror": True, "gridcolor":"lightgray", "tickwidth":2},
            "yaxis": {"showline": True, "linewidth": 2, "linecolor": "black", "mirror": True, "gridcolor":"lightgray", "tickwidth":2},
            "legend": {"x": 0.02, "y": 0.95, "bgcolor": "rgba(0,0,0,0)"}
        }

        if plot_type == "χ(k)":
            add_chi_plot(fig, exafs_group, individual_frames, show_legend)
            fig.update_layout(
                title="EXAFS χ(k)",
                xaxis_title="k [Å⁻¹]",
                yaxis_title="χ(k)",
                **common_layout
            )
        else:
            add_ft_plot(fig, exafs_group, individual_frames, show_legend)
            fig.update_layout(
                title="Fourier Transform |χ(R)|",
                xaxis_title="R [Å]",
                yaxis_title="|χ(R)|",
                **common_layout
            )

        fig.add_annotation(
            text=f"{absorber} {edge} edge",
            x=0.9, y=0.98, xref="paper", yref="paper",
            showarrow=False, font={"size": 16},
            bgcolor="white", bordercolor="black", borderwidth=1
        )
        return fig

    def add_chi_plot(fig, exafs_group, individual_frames, show_legend):
        """Add chi(k) plot traces"""
        if individual_frames:
            for i, frame in enumerate(individual_frames[:50]):  # Limit to 50 frames
                fig.add_trace(go.Scatter(
                    x=frame.k, y=frame.chi, name=f"Frame {i+1}",
                    line={"width": 1, "color": "rgba(128,128,128,0.3)"},
                    showlegend=show_legend and i == 0
                ))

        if hasattr(exafs_group, 'chi_std') and individual_frames:
            k, chi, std = exafs_group.k, exafs_group.chi, exafs_group.chi_std
            fig.add_trace(go.Scatter(
                x=list(k) + list(k[::-1]),
                y=list(chi + std) + list((chi - std)[::-1]),
                fill='toself', fillcolor='rgba(0,0,0,0.1)',
                line={"color": 'rgba(255,255,255,0)'},
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=exafs_group.k, y=exafs_group.chi,
            name="χ(k) Average ± σ" if hasattr(exafs_group, 'chi_std') else "χ(k)",
            line={"width": 2.5, "color": "black"}
        ))

    def add_ft_plot(fig, exafs_group, individual_frames, show_legend):
        """Add Fourier transform plot traces"""
        if individual_frames:
            for i, frame in enumerate(individual_frames[:50]):  # Limit to 50 frames
                fig.add_trace(go.Scatter(
                    x=frame.r, y=frame.chir_mag, name=f"Frame {i+1}",
                    line={"width": 1, "color": "rgba(128,128,128,0.3)"},
                    showlegend=show_legend and i == 0
                ))

        fig.add_trace(go.Scatter(
            x=exafs_group.r, y=exafs_group.chir_mag,
            name="|χ(R)| Average" if individual_frames else "|χ(R)|",
            line={"width": 2.5, "color": "black"}
        ))

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
        individual_frames = getattr(result, 'individual_frame_groups', None)

        fig = create_plot(
            exafs_group, 
            individual_frames,
            plot_type.value,
            show_legend,
            plot_absorber,
            edge
        )

        plot_output = mo.vstack([mo.hstack([plot_type, fig], justify="start"), message])

    return (plot_output,)


@app.cell
def _(plot_output):
    plot_output
    return


@app.cell(hide_code=True)
def _(clear_cache, mo, show_cache):
    show_cache_info_button =  mo.ui.button(label="🔍 Show Cache Info", kind="neutral", on_click=show_cache)
    clear_cache_button =  mo.ui.button(label="🗑️ Clear Cache", kind="danger", on_click=clear_cache)
    return clear_cache_button, show_cache_info_button


@app.cell(hide_code=True)
def _(CACHE_DIR, LarchWrapper, mo):
    def clear_cache(button_value=None):
        """Clear the Larch cache directory"""
        try:
            with LarchWrapper(cache_dir=CACHE_DIR) as wrapper:
                wrapper.clear_cache()
                message = mo.md("### 🗑️ Cache Cleared Successfully")

        except Exception as e:
            message = mo.md(f"### ❌ Cache Error\n{str(e)}")
        mo.output.append(message)
        # return None

    def show_cache(button_value=None):
        """Show the Larch cache information"""
        try:
            with LarchWrapper(cache_dir=CACHE_DIR) as wrapper:
                info = wrapper.get_cache_info()
                message = mo.md(f"""
                        ### 📊 Cache Status
                        | Property | Value |
                        |----------|-------|
                        | **Status** | ✅ Enabled |
                        | **Directory** | `{info['cache_dir']}` |
                        | **Files** | {info['files']} |
                        | **Size** | {info['size_mb']} MB |
                    """)

        except Exception as e:
            message = mo.md(f"### ❌ Cache Error\n{str(e)}")
        mo.output.append(message)
        # return message
    return clear_cache, show_cache


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(f"""## 🗑️ Cache Management""")
    return


@app.cell(hide_code=True)
def _(clear_cache_button, mo, show_cache_info_button):
    mo.hstack([show_cache_info_button, clear_cache_button], justify="start")
    return


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


if __name__ == "__main__":
    app.run()
