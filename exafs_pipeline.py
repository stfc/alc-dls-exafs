import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import plotly.graph_objects as go
    import tempfile

    # ⚠️ Make sure this package is installed or in PYTHONPATH
    try:
        from larch_cli_wrapper.wrapper import LarchWrapper, FeffConfig, PRESETS, EdgeType, ProcessingMode
    except ImportError:
        mo.stop(
            mo.md("""
            **❌ Import Error**: `larch_cli_wrapper` not found.

            Make sure:
            - The package is installed (`pip install larch-cli-wrapper` or similar)
            - Or the `src/` folder is in your Python path
            - Or run `pip install -e .` if this is a local package
            """)
        )

    return (
        EdgeType,
        FeffConfig,
        LarchWrapper,
        PRESETS,
        Path,
        ProcessingMode,
        go,
        mo,
        os,
        tempfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # EXAFS Pipeline Processing

    This app provides an interactive interface for EXAFS processing using the streamlined larch wrapper.
    You can process single structures or trajectories with customizable parameters and enhanced plotting options.
    """
    )
    return


@app.cell
def _(EdgeType, PRESETS, mo):
    # Create the main form
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
                label="Edge (for single structures)"
            ),
            processing_mode=mo.ui.radio(
                options={"Single structure": "single", "Trajectory (all frames)": "trajectory"},
                value="Single structure",
                label="Processing Mode"
            ),
            sample_interval=mo.ui.number(
                label="Sample interval (for trajectories)",
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
            })
        )
        .form(
            submit_button_label="Run EXAFS Processing",
            show_clear_button=True,
            bordered=True
        )
    )


    form
    return (form,)


@app.cell(hide_code=True)
def _(mo):
    plot_type = mo.ui.radio(["χ(k)", "|χ(R)|"], value="χ(k)")
    show_individual_frame_legend = mo.ui.checkbox(
        label="Show legend for individual frames",
        value=False,
    )

    # wrap the plot type and legend settings in a form
    plot_options_form = mo.md(
        """
        ## 📊 Plot Options

        **Plot Type**: {plot_type}

        **Show Individual Frame Legend**: {show_individual_frame_legend}
        """.format(plot_type=plot_type, show_individual_frame_legend=show_individual_frame_legend))
    plot_options_form
    return plot_type, show_individual_frame_legend


@app.cell(hide_code=True)
def _(
    absorber,
    go,
    mo,
    plot_type,
    result,
    settings,
    show_individual_frame_legend,
):
    """Interactive plotting cell with support for individual trajectory frames."""
    mo.stop(result is None)

    exafs_group = result.exafs_group
    individual_frames = result.individual_frame_groups
    edge = settings.get("edge", "K")

    # Define common style
    layout_common = dict(
        font=dict(family="Times New Roman", size=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        margin=dict(l=80, r=30, t=50, b=70),
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            gridcolor='lightgray',
            ticks='outside',
            tickwidth=2,
            tickcolor='black',
            title_font=dict(size=20)
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            gridcolor='lightgray',
            ticks='outside',
            tickwidth=2,
            tickcolor='black',
            title_font=dict(size=20)
        ),
        legend=dict(
            x=0.02, y=0.95,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=16)
        )
    )

    # Build figure based on plot type
    fig = go.Figure()

    if plot_type.value == "χ(k)":
        # Plot individual frames first (if available) so they appear in background
        if individual_frames:
            for i, frame in enumerate(individual_frames):
                fig.add_trace(go.Scatter(
                    x=frame.k,
                    y=frame.chi,
                    mode="lines",
                    name=f"Frame {i+1}",
                    line=dict(width=1, color=f"rgba(128,128,128,0.3)"),
                    showlegend=show_individual_frame_legend.value  # Only show legend for first 3 frames
                ))

        # Plot main/averaged spectrum
        line_props = dict(width=2.5, color="black")
        if hasattr(exafs_group, 'chi_std') and individual_frames:
            # Add standard deviation envelope if available
            k = exafs_group.k
            chi = exafs_group.chi
            std = exafs_group.chi_std

            # Standard deviation envelope
            fig.add_trace(go.Scatter(
                x=list(k) + list(k[::-1]),
                y=list(chi + std) + list((chi - std)[::-1]),
                fill='toself',
                fillcolor='rgba(0,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo="skip"
            ))
            name = f"χ(k) Average ± σ"
        else:
            name = "χ(k)"

        fig.add_trace(go.Scatter(
            x=exafs_group.k,
            y=exafs_group.chi,
            mode="lines",
            name=name,
            line=line_props
        ))
        fig.update_layout(
            title="EXAFS χ(k)",
            xaxis_title="k [Å⁻¹]",
            yaxis_title="χ(k)",
            **layout_common
        )
    else:
        # Plot individual frames first (if available)
        if individual_frames:
            for i, frame in enumerate(individual_frames):
                fig.add_trace(go.Scatter(
                    x=frame.r,
                    y=frame.chir_mag,
                    mode="lines",
                    name=f"Frame {i+1}",
                    line=dict(width=1, color=f"rgba(128,128,128,0.3)"),
                    showlegend=show_individual_frame_legend.value
                ))

        # Plot main/averaged spectrum
        fig.add_trace(go.Scatter(
            x=exafs_group.r,
            y=exafs_group.chir_mag,
            mode="lines",
            name="|χ(R)| Average" if individual_frames else "|χ(R)|",
            line=dict(width=2.5, color="black")
        ))
        fig.update_layout(
            title="Fourier Transform |χ(R)|",
            xaxis_title="R [Å]",
            yaxis_title="|χ(R)|",
            **layout_common
        )

    fig.add_annotation(
        text=f"{absorber} {edge} edge",
        xref="paper",
        yref="paper",
        x=0.9,
        y=0.98,
        showarrow=False,
        font=dict(family="Times New Roman", size=16, color="black"),
        align="center",
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )


    fig

    return


@app.cell
def _(
    FeffConfig,
    LarchWrapper,
    Path,
    ProcessingMode,
    mo,
    os,
    settings,
    tempfile,
):
    """Main processing cell - handles both full processing and output-only processing."""
    mo.stop(not settings or not settings.get("structure_file") or not settings.get("absorber"))

    structure_file = settings["structure_file"][0]
    absorber = settings["absorber"].strip()

    if not absorber:
        mo.stop(mo.md("**❌ Absorbing atom is required.**"))

    # Setup configuration
    config = FeffConfig.from_preset(settings["preset"])
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    is_traj = settings["processing_mode"] == "trajectory"
    if is_traj:
        config.sample_interval = settings["sample_interval"]
        config.parallel = settings["parallel_settings"]["parallel"]
        if settings["parallel_settings"]["n_workers"] is not None:
            config.n_workers = settings["parallel_settings"]["n_workers"]

    plot_individual_frames = is_traj
    show_plot = False # we handle this in the app instead
    process_output_only = settings["run_options"]["process_output_only"]

    try:
        wrapper = LarchWrapper(verbose=False)

        if process_output_only:
            # Process existing FEFF outputs
            if is_traj:
                frame_dirs = [d for d in output_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('frame_')]
                if not frame_dirs:
                    result = None
                    message = mo.md(f"**❌ No trajectory frames found in {output_dir}**")
                else:
                    result = wrapper.process_trajectory_output(
                        output_dir, config, 
                        plot_individual_frames=plot_individual_frames,
                        show_plot=show_plot,
                        output_dir=output_dir
                    )
                    message = mo.md(f"""
                    ### ✅ Processed Existing Trajectory Outputs
                    - **Frames processed**: {result.nframes}
                    - **Individual frame plots**: {'Yes' if plot_individual_frames else 'No'}
                    - **Output**: `{output_dir}`
                    """)
            else:
                chi_file = output_dir / "chi.dat"
                if not chi_file.exists():
                    result = None
                    message = mo.md(f"**❌ No chi.dat found in {output_dir}**")
                else:
                    exafs_group_temp = wrapper.process_feff_output(output_dir, config)
                    plot_paths = wrapper.plot_results(exafs_group_temp, output_dir, show_plot=show_plot)

                    # Create ProcessingResult-like object for consistency
                    from types import SimpleNamespace
                    result = SimpleNamespace(
                        exafs_group=exafs_group_temp,
                        plot_paths=plot_paths,
                        processing_mode=ProcessingMode.SINGLE_FRAME,
                        nframes=1,
                        individual_frame_groups=None
                    )
                    message = mo.md(f"""
                    ### ✅ Processed Existing Single Frame Output
                    - **Output**: `{output_dir}`
                    """)
        else:
            # Full processing from structure file
            # Create temporary file
            suffix = f".{structure_file.name.split('.')[-1]}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(structure_file.contents)
                temp_path = Path(tmp.name)

            try:
                # For progress bar, use a reasonable initial estimate
                # The wrapper will provide exact counts via progress callback
                initial_total = 100 if is_traj else 1
                
                with mo.status.progress_bar(
                    total=initial_total,
                    title="Processing EXAFS...",
                    subtitle="Starting...",
                    completion_title="✅ Processing Complete",
                    completion_subtitle="EXAFS analysis finished.",
                ) as bar:

                    def progress_callback(completed, total, desc):
                        # Update progress bar total on first call if needed
                        if completed == 0 and total != initial_total:
                            bar.total = total
                        # Only increment if we've actually completed work (completed > 0)
                        elif completed > 0:
                            bar.update(increment=1, subtitle=desc)

                    result = wrapper.process(
                        structure=temp_path,
                        absorber=absorber,
                        output_dir=output_dir,
                        config=config,
                        trajectory=is_traj,
                        show_plot=show_plot,
                        plot_individual_frames=plot_individual_frames,
                        progress_callback=progress_callback
                    )

                # Show results based on processing mode
                if result.processing_mode == ProcessingMode.TRAJECTORY:
                    message = mo.md(f"""
                    ### ✅ Trajectory Processing Complete
                    - **Frames processed**: {result.nframes}
                    - **Individual frame plots**: {'Yes' if plot_individual_frames else 'No'}
                    - **Output**: `{output_dir}`
                    - **PDF Plot**: `{result.plot_paths[0].name}`
                    - **SVG Plot**: `{result.plot_paths[1].name}`
                    - **Mode**: Averaged trajectory
                    """)
                else:
                    message = mo.md(f"""
                    ### ✅ Single Structure Processed
                    - **Output**: `{output_dir}`
                    - **PDF**: `{result.plot_paths[0].name}`
                    - **SVG**: `{result.plot_paths[1].name}`
                    """)

            finally:
                # Cleanup temp file
                if 'temp_path' in locals() and temp_path.exists():
                    os.unlink(temp_path)

    except Exception as e:
        import traceback
        result = None
        message = mo.md(f"""
        ### ❌ Processing Failed
        **Error:** {str(e)}
        ```
        {traceback.format_exc()}
        ```
        """)

    message
    return absorber, result


@app.cell(hide_code=True)
def _(form, mo):
    settings = form.value or {}
    if not settings:
        settings_message = "*Configure and submit the form above to start processing.*"

    else:
        is_trajectory = settings["processing_mode"] == "trajectory"

        settings_message = f"""**⚙️ Current Settings:**

        | Setting | Value |
        |--------|-------|
        | 📁 Structure | {settings['structure_file'][0].name if settings.get('structure_file') else 'None'} |
        | 🎯 Absorber | {settings.get('absorber', 'Not set')} |
        | ⚡ Mode | {settings.get('processing_mode', 'Not set')} |
        | 🔧 Preset | {settings.get('preset', 'Not set').title()} |
        | 🚀 Parallel | {'Yes' if is_trajectory and settings['parallel_settings']['parallel'] else 'No' if is_trajectory else 'N/A'} |
        | 📝 Process Output Only | {'Yes' if settings['run_options']['process_output_only'] else 'No'} |
            """
    mo.md(settings_message)

    return (settings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## 📚 Usage Instructions

    1. **Upload** a structure (CIF, XYZ, POSCAR, etc.) or trajectory.
    2. **Set absorber** (e.g., `Fe`, `Cu`).
    3. **Choose mode**:
       - *Single structure*: One-off EXAFS
       - *Trajectory*: Average over frames
    4. **Tweak settings**:
       - Use **"Quick"** preset for testing, **"Publication"** for final results
       - Enable **parallel** for large trajectories
       - Enable **"Individual frames"** to overlay individual trajectory frames in plots
    5. **Processing options**:
       - **Normal**: Full processing from structure file
       - **"Process output only"**: Reprocess existing FEFF outputs with different analysis parameters
    6. Click **Run**.

    💡 **Tips**: 
    - Use "Process output only" to reprocess with different Fourier transform parameters
    - Individual frame plotting helps visualize trajectory dynamics
    - Try different presets to optimize for speed vs. accuracy
    """
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
