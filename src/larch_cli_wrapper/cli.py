#!/usr/bin/env python3
"""CLI interface for Larch Wrapper - EXAFS processing pipeline"""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .wrapper import LarchWrapper

app = typer.Typer(
    name="larch-cli",
    help="A CLI wrapper for larch EXAFS processing with configuration file support",
    invoke_without_command=True,
    no_args_is_help=True,
    add_completion=True,
)
console = Console()


def create_progress() -> Progress:
    """Create a consistent progress bar for all commands."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def create_simple_progress() -> Progress:
    """Create a simple progress bar for indeterminate tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


@app.command("info")
def show_info():
    """Show diagnostic information."""
    wrapper = LarchWrapper(verbose=False)
    wrapper.print_diagnostics()


@app.command("pipeline")
def process_pipeline(
    trajectory_file: Path = typer.Argument(
        ...,
        help="Path to trajectory file (XYZ, PDB, DCD, etc.) or single structure file",
    ),
    absorber: str = typer.Argument(
        ..., help="Absorbing atom symbol or site index (e.g., Fe, Cu, 0)"
    ),
    output_dir: Path = typer.Option(
        Path("outputs/trajectory"), "--output-dir", "-o", help="Base output directory"
    ),
    sample_interval: int = typer.Option(
        1,
        "--interval",
        "-i",
        help="Process every Nth frame (ignored if frame_index is specified)",
    ),
    frame_index: int = typer.Option(
        None,
        "--frame",
        "-f",
        help="Process specific frame index only (overrides interval processing)",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel/--sequential",
        help="Enable parallel processing for improved performance",
    ),
    n_workers: int = typer.Option(
        None,
        "--workers",
        "-j",
        help="Number of parallel workers (auto-detect if not specified)",
    ),
    input_only: bool = typer.Option(
        False, "--input-only", help="Generate FEFF inputs only, don't run calculations"
    ),
    method: str = typer.Option(
        "auto", "--method", "-m", help="Method to use: auto, larixite, or pymatgen"
    ),
    spectrum_type: str = typer.Option(
        "EXAFS", "--spectrum", "-s", help="Spectrum type: EXAFS, XANES, etc."
    ),
    edge: str = typer.Option(
        "K", "--edge", help="Absorption edge (K, L1, L2, L3, etc.)"
    ),
    radius: float = typer.Option(
        10.0, "--radius", "-r", help="Cluster radius in Angstroms"
    ),
    kweight: int = typer.Option(
        2, "--kweight", "-k", help="k-weighting for Fourier transform"
    ),
    window: str = typer.Option(
        "hanning", "--window", "-w", help="Window function for FT"
    ),
    dk: float = typer.Option(1.0, "--dk", help="dk parameter for FT"),
    kmin: float = typer.Option(2.0, "--kmin", help="Minimum k value for FT"),
    kmax: float = typer.Option(14.0, "--kmax", help="Maximum k value for FT"),
    config_file: Path = typer.Option(
        None,
        "--config",
        help="Configuration file (YAML format, overrides other parameters)",
    ),
    show_plot: bool = typer.Option(
        False, "--show/--no-show", help="Display plot interactively"
    ),
):
    """Process an MD trajectory for time-averaged EXAFS analysis or a single structure/frame with optional parallel processing."""
    if not trajectory_file.exists():
        console.print(f"[red]Error: Trajectory file {trajectory_file} not found[/red]")
        raise typer.Exit(1)

    try:
        wrapper = LarchWrapper()

        # Display processing mode information
        if parallel:
            if n_workers is None:
                import multiprocessing as mp

                n_workers_display = min(mp.cpu_count(), 8)
                console.print(
                    f"[cyan]Using parallel processing (auto-detecting {n_workers_display} workers)[/cyan]"
                )
            else:
                console.print(
                    f"[cyan]Using parallel processing with {n_workers} workers[/cyan]"
                )
        else:
            console.print("[yellow]Using sequential processing[/yellow]")

        # Handle frame selection logic
        import tempfile

        from ase.io import read as ase_read
        from ase.io import write as ase_write

        if frame_index is not None:
            # Process specific frame only
            console.print(f"[yellow]Processing specific frame {frame_index}[/yellow]")
            structures = [ase_read(str(trajectory_file), index=frame_index)]
            processing_mode = "single_frame"
        else:
            # Process according to sample interval
            if sample_interval == 1:
                console.print("[yellow]Processing all frames[/yellow]")
            else:
                console.print(
                    f"[yellow]Processing every {sample_interval} frames[/yellow]"
                )
            structures = ase_read(str(trajectory_file), index=f"::{sample_interval}")
            if not isinstance(structures, list):
                structures = [structures]
            processing_mode = "pipeline"

        if input_only:
            # Generate FEFF inputs for selected frames without running calculations
            console.print(
                f"Processing {len(structures)} frame{'s' if len(structures) != 1 else ''}..."
            )

            with create_progress() as progress:

                task = progress.add_task(
                    "Generating FEFF inputs...", total=len(structures)
                )

                for i, atoms in enumerate(structures):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        if frame_index is not None:
                            struct_path = Path(tmpdir) / f"frame_{frame_index}.cif"
                            frame_output = output_dir / f"frame_{frame_index:04d}"
                        else:
                            struct_path = Path(tmpdir) / f"frame_{i}.cif"
                            frame_output = output_dir / f"frame_{i:04d}"

                        ase_write(str(struct_path), atoms, format="cif")

                        wrapper.generate_feff_input(
                            structure_path=struct_path,
                            absorber=absorber,
                            output_dir=frame_output,
                            filename="feff.inp",
                            spectrum_type=spectrum_type,
                            edge=edge,
                            radius=radius,
                            method=method,
                        )

                        progress.update(
                            task,
                            advance=1,
                            description=f"Generating FEFF inputs... ({i+1}/{len(structures)})",
                        )

            console.print(
                f"\n[bold green]FEFF inputs generated for {len(structures)} frame{'s' if len(structures) != 1 else ''}![/bold green]"
            )
            console.print(f"  Output directory: {output_dir}")

        else:
            # Run full processing
            total_frames = len(structures)

            if processing_mode == "single_frame":
                # Process single frame directly (similar to old structure command)
                console.print(f"Processing single frame {frame_index}...")

                with create_progress() as progress:
                    task = progress.add_task("Processing single frame...", total=3)

                    # Create temporary structure file
                    with tempfile.TemporaryDirectory() as tmpdir:
                        struct_path = Path(tmpdir) / f"frame_{frame_index}.cif"
                        ase_write(str(struct_path), structures[0], format="cif")

                        frame_output = output_dir / f"frame_{frame_index:04d}"

                        # Step 1: Generate FEFF input
                        progress.update(task, description="Generating FEFF input...")
                        input_file = wrapper.generate_feff_input(
                            structure_path=struct_path,
                            absorber=absorber,
                            output_dir=frame_output,
                            filename="feff.inp",
                            spectrum_type=spectrum_type,
                            edge=edge,
                            radius=radius,
                            method=method,
                            config_parameters=config_file,
                        )
                        progress.advance(task, 1)

                        # Step 2: Run FEFF calculation
                        progress.update(task, description="Running FEFF calculation...")
                        if not wrapper.run_feff(frame_output):
                            raise RuntimeError("FEFF calculation failed")
                        progress.advance(task, 1)

                        # Step 3: Process output and create plots
                        progress.update(
                            task,
                            description="Processing output and generating plots...",
                        )
                        exafs_group = wrapper.process_feff_output(
                            frame_output,
                            kweight=kweight,
                            window=window,
                            dk=dk,
                            kmin=kmin,
                            kmax=kmax,
                        )
                        plot_paths = wrapper.plot_fourier_transform(
                            exafs_group,
                            frame_output,
                            f"{trajectory_file.stem}_frame_{frame_index}_EXAFS_FT",
                            show_plot=show_plot,
                        )
                        progress.advance(task, 1)

                        progress.update(
                            task,
                            description="[green]✓[/green] Single frame processing complete!",
                        )

                console.print(
                    "\n[bold green]Single frame processing completed![/bold green]"
                )
                console.print(f"  Processed frame: {frame_index}")
                console.print(f"  Output directory: {frame_output}")
                console.print(f"  PDF plot: {plot_paths[0]}")
                console.print(f"  SVG plot: {plot_paths[1]}")

            else:
                # Run trajectory processing for time-averaged analysis
                with create_progress() as progress:

                    task = progress.add_task(
                        "Processing trajectory frames...", total=total_frames
                    )

                    def progress_callback(completed, total, description):
                        """Callback to update progress bar."""
                        progress.update(
                            task, completed=completed, description=description
                        )

                    avg_group = wrapper.process_trajectory(
                        trajectory_path=trajectory_file,
                        absorber=absorber,
                        output_base=output_dir,
                        sample_interval=sample_interval,
                        parallel=parallel,
                        n_workers=n_workers,
                        progress_callback=progress_callback,
                        method=method,
                        spectrum_type=spectrum_type,
                        edge=edge,
                        radius=radius,
                        kweight=kweight,
                        window=window,
                        dk=dk,
                        kmin=kmin,
                        kmax=kmax,
                        config_parameters=config_file,
                    )

                    progress.update(
                        task,
                        description="Generating averaged plots...",
                        completed=total_frames,
                    )

                    # Generate plots for averaged data
                    plot_paths = wrapper.plot_fourier_transform(
                        avg_group,
                        output_dir,
                        f"{trajectory_file.stem}_trajectory_avg",
                        show_plot=show_plot,
                    )

                    progress.update(
                        task,
                        description="[green]✓[/green] Trajectory processing complete!",
                    )

                console.print(
                    "\n[bold green]Trajectory processing completed![/bold green]"
                )
                console.print(f"  Processed {avg_group.nframes} frames")
                if parallel:
                    console.print(
                        f"  Processing mode: [cyan]Parallel ({n_workers} workers)[/cyan]"
                    )
                else:
                    console.print("  Processing mode: [yellow]Sequential[/yellow]")
                console.print(f"  Sample interval: {sample_interval}")
                console.print(f"  Output directory: {output_dir}")
                console.print(f"  PDF plot: {plot_paths[0]}")
                console.print(f"  SVG plot: {plot_paths[1]}")

    except Exception as e:
        console.print(f"\n[red]Error processing trajectory: {e}[/red]")
        raise typer.Exit(1)


@app.command("run-feff")
def run_feff_calculation(
    feff_dir: Path = typer.Argument(..., help="Directory containing FEFF input file"),
    input_file: str = typer.Option(
        "feff.inp", "--input", "-i", help="FEFF input filename"
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Verbose output"),
):
    """Run FEFF calculation."""
    if not feff_dir.exists():
        console.print(f"[red]Error: Directory {feff_dir} not found[/red]")
        raise typer.Exit(1)

    input_path = feff_dir / input_file
    if not input_path.exists():
        console.print(f"[red]Error: FEFF input file {input_path} not found[/red]")
        raise typer.Exit(1)

    try:
        wrapper = LarchWrapper()
        with create_simple_progress() as progress:
            task = progress.add_task("Running FEFF calculation...", total=None)
            success = wrapper.run_feff(feff_dir, input_file, verbose)

        if success:
            console.print("[green]✓[/green] FEFF calculation completed successfully")
        else:
            console.print("[red]✗[/red] FEFF calculation failed")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error running FEFF: {e}[/red]")
        raise typer.Exit(1)


@app.command("process")
def process_feff_output(
    feff_dir: Path = typer.Argument(..., help="Directory containing FEFF output files"),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: same as feff_dir)",
    ),
    kweight: int = typer.Option(
        2, "--kweight", "-k", help="k-weighting for Fourier transform"
    ),
    window: str = typer.Option(
        "hanning", "--window", "-w", help="Window function for FT"
    ),
    dk: float = typer.Option(1.0, "--dk", help="dk parameter for FT"),
    kmin: float = typer.Option(2.0, "--kmin", help="Minimum k value for FT"),
    kmax: float = typer.Option(14.0, "--kmax", help="Maximum k value for FT"),
    show_plot: bool = typer.Option(
        False, "--show/--no-show", help="Display plot interactively"
    ),
):
    """Process FEFF output and generate Fourier transform plot."""
    if not feff_dir.exists():
        console.print(f"[red]Error: Directory {feff_dir} not found[/red]")
        raise typer.Exit(1)

    chi_file = feff_dir / "chi.dat"
    if not chi_file.exists():
        console.print(f"[red]Error: FEFF output file {chi_file} not found[/red]")
        raise typer.Exit(1)

    if output_dir is None:
        output_dir = feff_dir

    try:
        wrapper = LarchWrapper()

        console.print("Processing FEFF output...")
        exafs_group = wrapper.process_feff_output(
            feff_dir, kweight=kweight, window=window, dk=dk, kmin=kmin, kmax=kmax
        )

        console.print("Generating plots...")
        plot_paths = wrapper.plot_fourier_transform(
            exafs_group, output_dir, "EXAFS_FT", show_plot=show_plot
        )

        console.print("[green]✓[/green] Processing complete!")
        console.print(f"  PDF plot: {plot_paths[0]}")
        console.print(f"  SVG plot: {plot_paths[1]}")

    except Exception as e:
        console.print(f"[red]Error processing FEFF output: {e}[/red]")
        raise typer.Exit(1)


@app.command("config-example")
def create_config_example(
    output_file: Path = typer.Option(
        Path("example_config.yaml"),
        "--output",
        "-o",
        help="Output configuration file path",
    ),
    preset: str = typer.Option(
        "publication",
        "--preset",
        "-p",
        help="Preset to use as base (quick, publication, xanes)",
    ),
):
    """Create an example configuration file."""
    from .wrapper import PRESETS, save_config_parameter_set

    if preset not in PRESETS:
        console.print(
            f"[red]Error: Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}[/red]"
        )
        raise typer.Exit(1)

    try:
        parameters = PRESETS[preset].copy()
        description = f"Example configuration file based on '{preset}' preset"
        save_config_parameter_set(parameters, output_file, description)

        console.print(
            f"[bold green]✓[/bold green] Example configuration file created: {output_file}"
        )
        console.print(f"[dim]  Based on preset: {preset}[/dim]")
        console.print("[dim]  Edit the file to customize FEFF parameters[/dim]")

    except ImportError:
        console.print(
            "[red]Error: PyYAML is required for configuration support. Install with: pip install pyyaml[/red]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error creating configuration file: {e}[/red]")
        raise typer.Exit(1)


@app.command("config-validate")
def validate_config(
    config_file: Path = typer.Argument(
        ..., help="Path to configuration file to validate"
    )
):
    """Validate a configuration file."""
    from .wrapper import load_config_parameter_set

    if not config_file.exists():
        console.print(f"[red]Error: Configuration file {config_file} not found[/red]")
        raise typer.Exit(1)

    try:
        parameters = load_config_parameter_set(config_file)

        console.print(
            f"[bold green]✓[/bold green] Configuration file is valid: {config_file}"
        )
        console.print("\n[bold]Loaded parameters:[/bold]")

        # Display parameters in organized way
        if "spectrum_type" in parameters:
            console.print(
                f"  [cyan]Spectrum Type:[/cyan] {parameters['spectrum_type']}"
            )
        if "edge" in parameters:
            console.print(f"  [cyan]Edge:[/cyan] {parameters['edge']}")
        if "radius" in parameters:
            console.print(f"  [cyan]Radius:[/cyan] {parameters['radius']} Å")

        if "user_tag_settings" in parameters:
            console.print("  [cyan]FEFF Parameters:[/cyan]")
            for key, value in parameters["user_tag_settings"].items():
                console.print(f"    {key}: {value}")

    except ImportError:
        console.print(
            "[red]Error: PyYAML is required for configuration support. Install with: pip install pyyaml[/red]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error validating configuration file: {e}[/red]")
        raise typer.Exit(1)


@app.command("version")
def show_version():
    """Show version and diagnostic information."""
    wrapper = LarchWrapper(verbose=False)
    wrapper.print_diagnostics()


if __name__ == "__main__":
    app()
