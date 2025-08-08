import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from ase.io import read as ase_read
    from ase.io import write as ase_write
    from ase.data import atomic_numbers
    from math import floor

    import marimo as mo
    import os
    from pathlib import Path
    import plotly.graph_objects as go
    import tempfile




    # \u26a0\ufe0f Make sure this package is installed or in PYTHONPATH
    try:
        from larch_cli_wrapper.wrapper import LarchWrapper, FeffConfig, PRESETS, EdgeType
    except ImportError:
        mo.stop(
            mo.md("""
            **\u274c Import Error**: `larch_cli_wrapper` not found.

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

    This app provides an interactive interface for EXAFS processing using the larch wrapper.
    You can process single structures or trajectories with customizable parameters.
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
                value="Quick",  # Match actual key
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
                "input_only": mo.ui.checkbox(label="Generate FEFF inputs only (don't run calculations)"),
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
    plot_type

    return (plot_type,)


@app.cell(hide_code=True)
def _(absorber, edge, go, mo, plot_type, result):
    mo.stop(result is None)

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
    if plot_type.value == "χ(k)":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.k,
            y=result.chi,
            mode="lines",
            name="χ(k)",
            line=dict(width=2.5, color="black")
        ))
        fig.update_layout(
            title="EXAFS χ(k)",
            xaxis_title="k [Å⁻¹]",
            yaxis_title="χ(k)",
            **layout_common
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.r,
            y=result.chir_mag,
            mode="lines",
            name="|χ(R)|",
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
def _(EdgeType, FeffConfig, LarchWrapper, Path, mo, os, settings, tempfile):
    mo.stop(not settings or not settings.get("structure_file") or not settings.get("absorber"))

    structure_file = settings["structure_file"][0]
    absorber = settings["absorber"].strip()

    if not absorber:
        mo.stop(mo.md("**❌ Absorbing atom is required.**"))

    # Create temp file
    suffix = f".{structure_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(structure_file.contents)
        temp_path = Path(tmp.name)

    try:
        wrapper = LarchWrapper(verbose=False)
        config = FeffConfig.from_preset(settings["preset"])
        # Update with chosen edge
        config.edge = EdgeType[settings["edge"]]

        output_dir = Path(settings["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        is_traj = settings["processing_mode"] == "trajectory"
        if is_traj:
            config.sample_interval = settings["sample_interval"]
            config.parallel = settings["parallel_settings"]["parallel"]
            if settings["parallel_settings"]["n_workers"] is not None:
                config.n_workers = settings["parallel_settings"]["n_workers"]

        # Generate inputs
        input_info = wrapper.generate_inputs(
            structure=temp_path,
            absorber=absorber,
            output_dir=output_dir,
            config=config,
            trajectory=is_traj
        )

        nframes = input_info.get("nframes", 1)
        msg = f"✅ Generated inputs for {nframes} frame(s)" if is_traj else "✅ Generated FEFF input"

        message = mo.md(f"""
        ### {msg}
        - **Mode**: {settings['processing_mode']}
        - **Absorber**: {absorber}
        - **Preset**: {settings['preset']}
        - **Output**: `{input_info['output_dir']}`
        - **Parallel**: {'Yes' if is_traj and config.parallel else 'No'}
        """)

    except Exception as e:
        message = mo.md(f"**❌ Error during input generation:** {str(e)}")
        input_info = None
        wrapper = None

    finally:
        if 'temp_path' in locals() and temp_path.exists():
            os.unlink(temp_path)

    return absorber, input_info, nframes, wrapper


@app.cell
def _(input_info, is_trajectory, mo, nframes, settings, wrapper):
    mo.stop(input_info is None or wrapper is None)

    input_only = settings["run_options"]["input_only"]
    edge = settings["edge"]
    settings['run_options']['show_plot'] = False  # We plot it in marimo instead
    result = None

    if input_only:
        # nframes = input_info.get("nframes", 1)
        message2 = mo.md(f"""
        ### \u2705 Input Generation Complete

        - Generated inputs for {nframes} structure(s)
        - Output directory: `{input_info['output_dir']}`
        - Preset: {settings['preset']}
        - Mode: {settings['processing_mode']}

        \u2705 You can now disable "input only" to run full processing.
        """)
    else:
        try:
            with mo.status.progress_bar(
                total=input_info.get("nframes", 1),
                title="Processing frames...",
                subtitle="Starting...",
                completion_title="✅ Processing Complete",
                completion_subtitle="All frames processed."
            ) as bar:

                def progress_callback(completed, total, desc):
                    bar.update(
                        increment=1,  # one step per frame
                    )

                if is_trajectory:
                    result = wrapper._process_trajectory_from_inputs(
                        input_info,
                        show_plot=settings["run_options"]["show_plot"],
                        progress_callback=progress_callback
                    )
                else:
                    result = wrapper._process_single_frame_from_inputs(
                        input_info,
                        show_plot=settings["run_options"]["show_plot"]
                    )

            # After bar closes, show results
            if result.is_averaged:
                message2 = mo.md(f"""
                ### ✅ Trajectory Processing Complete

                - Frames processed: {result.nframes}
                - Output: `{input_info['output_dir']}`
                - PDF Plot: `{result.plot_paths[0]}`
                - SVG Plot: `{result.plot_paths[1]}`
                - Mode: Averaged trajectory
                """)
            else:
                message2 = mo.md(f"""
                ### ✅ Single Structure Processed

                - Output: `{input_info['output_dir']}`
                - PDF: `{result.plot_paths[0]}`
                - SVG: `{result.plot_paths[1]}`
                """)

        except Exception as e:
            import traceback
            message2 = mo.md(f"""
            ### ❌ Processing Failed

            **Error:** {str(e)}

            Inputs were generated. Try running FEFF manually.

            ```
            {traceback.format_exc()}
            ```
            """)
    message2

    return edge, result


@app.cell
def _():
    # # Display the svg created
    # if result and result.plot_paths:
    #     # Render an image from a URL
    #     svg_plot = mo.image(
    #         src=result.plot_paths[1],
    #         alt="Generated EXAFS plot",
    #         width=800,
    #         height=400,
    #         rounded=False,
    #         caption="Generated EXAFS plot",
    #     )
    # svg_plot
    return


@app.cell(hide_code=True)
def _(form, mo):
    settings = form.value or {}
    if not settings:
        settings_message = "*Configure and submit the form above to start processing.*"

    else:
        is_trajectory = settings["processing_mode"] == "trajectory"

        settings_message = f"""**\u2699\ufe0f Current Settings:**

        | Setting | Value |
        |--------|-------|
        | 📁 Structure | {settings['structure_file'][0].name if settings.get('structure_file') else 'None'} |
        | 🎯 Absorber | {settings.get('absorber', 'Not set')} |
        | ⚡ Mode | {settings.get('processing_mode', 'Not set')} |
        | 🔧 Preset | {settings.get('preset', 'Not set').title()} |
        | 🚀 Parallel | {'Yes' if is_trajectory and settings['parallel_settings']['parallel'] else 'No' if is_trajectory else 'N/A'} |
        | 📝 Input only | {'Yes' if settings['run_options']['input_only'] else 'No'} |
            """
    mo.md(settings_message)

    return is_trajectory, settings


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
       - Use **"Quick"** preset for testing
       - Enable **parallel** for large trajectories
    5. Check **"Input only"** to debug before full run.
    6. Click **Run**.

    💡 Tip: Always test with "input only" first!
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
