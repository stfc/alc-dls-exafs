#!/usr/bin/env python3
"""Streamlined CLI interface for Larch Wrapper - EXAFS processing pipeline."""

from pathlib import Path

import typer
from ase import Atoms
from ase.io import read as ase_read
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .feff_utils import PRESETS, EdgeType, FeffConfig, generate_feff_input
from .wrapper import (
    EXAFSProcessingError,
    LarchWrapper,
    ProcessingMode,
    ProcessingResult,
)

app = typer.Typer(
    name="larch-cli",
    help="Streamlined CLI for EXAFS processing with larch",
    invoke_without_command=True,
    no_args_is_help=True,
)
console = Console()


def _setup_config(
    config_file: Path | None = None,
    preset: str | None = None,
    force_recalculate: bool = False,
) -> FeffConfig:
    """Setup configuration from file, preset, or defaults."""
    if config_file:
        config = FeffConfig.from_yaml(config_file)
        console.print(f"[dim]Loaded configuration from {config_file}[/dim]")
    elif preset:
        config = FeffConfig.from_preset(preset)
        console.print(f"[dim]Using '{preset}' preset[/dim]")
    else:
        config = FeffConfig()
        console.print("[dim]Using default configuration (larixite defaults)[/dim]")

    config.force_recalculate = force_recalculate
    return config


def create_progress() -> Progress:
    """Create progress bar for processing tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )


@app.command("info")  # type: ignore[misc]
def show_info() -> None:
    """Show system and dependency information."""
    wrapper = LarchWrapper(verbose=True, cache_dir=Path.home() / ".larch_cache")
    wrapper.print_diagnostics()


def load_structure(structure_path: Path) -> Atoms:
    """Load structure from file using ASE."""
    return ase_read(str(structure_path))


@app.command("generate")  # type: ignore[misc]
def generate_inputs(
    structure: Path = typer.Argument(..., help="Path to structure file"),
    absorber: str = typer.Argument(..., help="Absorbing atom symbol or site index"),
    edge: str = typer.Option(
        EdgeType.K.value, help=f"Absorption edge: {[e.value for e in EdgeType]}"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output", "-o", help="Output directory"
    ),
    config_file: Path = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"
    ),
    method: str = typer.Option(
        "auto", "--method", "-m", help="Method: auto, larixite, pymatgen"
    ),
) -> None:
    """Generate FEFF input files only."""
    if not structure.exists():
        console.print(f"[red]Error: Structure file {structure} not found[/red]")
        raise typer.Exit(1)

    try:
        config = _setup_config(config_file, preset)
        config.edge = edge
        config.method = method

        # Load structure file
        atoms = load_structure(structure)

        with LarchWrapper(verbose=True, cache_dir=Path.home() / ".larch_cache"):
            console.print(
                f"[cyan]Generating FEFF input for {absorber} using "
                f"{config.method} method...[/cyan]"
            )

            # Generate FEFF input using the unified method
            input_path = generate_feff_input(atoms, absorber, output_dir, config)

            console.print(f"[green]✓ FEFF input generated: {input_path}[/green]")
            console.print(f"  Check {input_path / 'feff.inp'} for details")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("run-feff")  # type: ignore[misc]
def run_feff_calculation(
    feff_dir: Path,
    verbose: bool = False,
) -> None:
    """Run FEFF calculation in specified directory."""
    # Apply defaults
    if feff_dir is None:
        console.print("[red]Error: feff_dir argument is required[/red]")
        raise typer.Exit(1)
    if not feff_dir.exists():
        console.print(f"[red]Error: Directory {feff_dir} not found[/red]")
        raise typer.Exit(1)

    if not (feff_dir / "feff.inp").exists():
        console.print(f"[red]Error: No feff.inp found in {feff_dir}[/red]")
        raise typer.Exit(1)

    try:
        with LarchWrapper(
            verbose=True, cache_dir=Path.home() / ".larch_cache"
        ) as wrapper:
            console.print(f"[cyan]Running FEFF calculation in {feff_dir}...[/cyan]")
            success = wrapper.run_feff(feff_dir)
            if success:
                console.print(
                    "[green]✓ FEFF calculation completed successfully[/green]"
                )
                console.print(f"  Output files in: {feff_dir}")
                # Show key output files
                chi_file = feff_dir / "chi.dat"
                log_file = feff_dir / "feff.log"
                if chi_file.exists():
                    console.print(f"  Chi data: {chi_file}")
                if log_file.exists():
                    console.print(f"  Log file: {log_file}")
            else:
                console.print("[red]✗ FEFF calculation failed[/red]")
                log_file = feff_dir / "feff.log"
                if log_file.exists():
                    console.print(f"  Check log file: {log_file}")
                raise typer.Exit(1)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("process")  # type: ignore[misc]
def process(
    structure: Path = typer.Argument(..., help="Path to structure file or trajectory"),
    absorber: str = typer.Argument(..., help="Absorbing atom symbol or site index"),
    # Edge options from EdgeType enum keys
    edge: str = typer.Option(
        EdgeType.K.value, help=f"Absorption edge: {[e.value for e in EdgeType]}"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output", "-o", help="Output directory"
    ),
    trajectory: bool = typer.Option(
        False, "--trajectory", "-t", help="Process as trajectory"
    ),
    sample_interval: int = typer.Option(
        1, "--interval", "-i", help="Process every Nth frame"
    ),
    config_file: Path = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    preset: str = typer.Option(
        None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"
    ),
    method: str = typer.Option(
        "auto", "--method", "-m", help="Method: auto, larixite, pymatgen"
    ),
    show_plot: bool = typer.Option(False, "--show", help="Display plots interactively"),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Enable parallel processing"
    ),
    n_workers: int = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers"
    ),
    plot_individual_frames: bool = typer.Option(
        False, "--plot-frames", help="Plot individual trajectory frames"
    ),
    plot_style: str = typer.Option(
        "publication",
        "--plot-style",
        help="Plot style: publication, presentation, quick",
    ),
    force_recalculate: bool = typer.Option(
        False, "--force", help="Skip cache and recalculate"
    ),
) -> None:
    """Process structure or trajectory for EXAFS analysis."""
    # Explicitly indicate no return value for CLI commands
    # (Typer handles exits and prints internally)
    if not structure.exists():
        console.print(f"[red]Error: Structure file {structure} not found[/red]")
        raise typer.Exit(1)

    try:
        config = _setup_config(config_file, preset, force_recalculate=force_recalculate)
        config.sample_interval = sample_interval
        config.parallel = parallel
        config.n_workers = n_workers
        config.edge = edge
        config.method = method

        with LarchWrapper(
            verbose=True, cache_dir=Path.home() / ".larch_cache"
        ) as wrapper:
            console.print(
                f"[cyan]Processing {structure} for {absorber} using "
                f"{config.method} method...[/cyan]"
            )

            with create_progress() as progress:
                task_id: int | None = None
                total_frames = 1

                def progress_callback(
                    current: int, total: int, description: str
                ) -> None:
                    nonlocal task_id, total_frames
                    if task_id is None:
                        # First call - create the task with correct total
                        total_frames = total
                        task_id = progress.add_task(description, total=total)
                        progress.update(task_id, completed=current)
                    else:
                        progress.update(
                            task_id, completed=current, description=description
                        )

                # Process the structure
                result = wrapper.process(
                    structure=structure,
                    absorber=absorber,
                    output_dir=output_dir,
                    config=config,
                    trajectory=trajectory,
                    show_plot=show_plot,
                    plot_style=plot_style,
                    plot_individual_frames=plot_individual_frames,
                    progress_callback=progress_callback,
                )

                # Ensure completion
                if task_id is not None:
                    progress.update(
                        task_id,
                        completed=total_frames,
                        description="[green]✓ Complete![/green]",
                    )

            # Show results
            console.print("\n[bold green]✓ Processing completed![/bold green]")
            console.print(f"  Method: {config.method}")
            console.print(f"  Output: {output_dir}")

            # Display cache statistics if available
            if hasattr(result, "cache_hits") and hasattr(result, "cache_misses"):
                total = result.cache_hits + result.cache_misses
                if total > 0:
                    hit_rate = (result.cache_hits / total) * 100
                    console.print(
                        f"  Cache: {result.cache_hits} hits / "
                        f"{result.cache_misses} misses ({hit_rate:.1f}%)"
                    )

            formats = ", ".join(f"{k.upper()}" for k in result.plot_paths.keys())
            console.print(f"  Plots: {formats}")

            if hasattr(result, "nframes"):
                console.print(f"  Frames processed: {result.nframes}")

            # Show plot paths
            console.print("\n[bold]Generated plots:[/bold]")
            for fmt, path in result.plot_paths.items():
                console.print(f"  {fmt.upper()}: {path}")

    except EXAFSProcessingError as e:
        console.print(f"[red]Processing error: {e}[/red]")
        raise typer.Exit(1) from e
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        # For debugging, could show more details in verbose mode
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        # For debugging, could show more details in verbose mode
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1) from e


@app.command("process-output")  # type: ignore[misc]
def process_feff_output(
    feff_dir: Path = typer.Argument(
        ..., help="Directory containing FEFF output (chi.dat or trajectory frames)"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory (default: same as feff_dir)"
    ),
    config_file: Path = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    preset: str = typer.Option(
        None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"
    ),
    show_plot: bool = typer.Option(False, "--show", help="Display plots interactively"),
    plot_individual_frames: bool = typer.Option(
        False, "--plot-frames", help="Plot individual trajectory frames"
    ),
    plot_style: str = typer.Option(
        "publication",
        "--plot-style",
        help="Plot style: publication, presentation, quick",
    ),
) -> None:
    """Process existing FEFF output files (single or trajectory) and generate plots."""
    if not feff_dir.exists():
        console.print(f"[red]Error: Directory {feff_dir} not found[/red]")
        raise typer.Exit(1)

    output_dir = output_dir or feff_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = _setup_config(config_file, preset)

        with LarchWrapper(
            verbose=True, cache_dir=Path.home() / ".larch_cache"
        ) as wrapper:
            chi_file = feff_dir / "chi.dat"
            frame_dirs = [
                d
                for d in feff_dir.iterdir()
                if d.is_dir() and d.name.startswith("frame_")
            ]

            if frame_dirs:
                console.print(
                    f"[cyan]Processing trajectory output with "
                    f"{len(frame_dirs)} frames...[/cyan]"
                )

                # For trajectory processing, we need to create a virtual structure path
                # that the wrapper can process as a trajectory
                with create_progress() as progress:
                    task_id: int | None = None

                    def progress_callback(
                        current: int, total: int, description: str
                    ) -> None:
                        nonlocal task_id
                        if task_id is None:
                            task_id = progress.add_task(description, total=total)
                            progress.update(task_id, completed=current)
                        else:
                            progress.update(
                                task_id, completed=current, description=description
                            )

                    # Process using the main process method
                    result = wrapper.process(
                        structure=feff_dir,  # Not used directly for trajectory output
                        absorber="auto",  # Will be determined from the data
                        output_dir=output_dir,
                        config=config,
                        trajectory=True,
                        show_plot=show_plot,
                        plot_style=plot_style,
                        plot_individual_frames=plot_individual_frames,
                        progress_callback=progress_callback,
                    )

                console.print("[green]✓ Trajectory processed[/green]")
                console.print(f"  Output: {output_dir}")
                console.print(f"  Frames processed: {result.nframes}")

                # Show plot paths
                console.print("\n[bold]Generated plots:[/bold]")
                for fmt, path in result.plot_paths.items():
                    console.print(f"  {fmt.upper()}: {path}")

            elif chi_file.exists():
                console.print("[cyan]Processing single FEFF output...[/cyan]")
                exafs_group = wrapper.process_feff_output(feff_dir, config)

                # Create a proper ProcessingResult for consistent output
                plot_paths = wrapper.plot_results(
                    exafs_group,
                    output_dir,
                    filename_base="EXAFS_FT",
                    show_plot=show_plot,
                    plot_style=plot_style,
                )

                result = ProcessingResult(
                    exafs_group=exafs_group,
                    plot_paths=plot_paths,
                    processing_mode=ProcessingMode.SINGLE_FRAME,
                )

                formats = ", ".join(f"{k.upper()}" for k in plot_paths.keys())
                console.print("[green]✓ Single output processed[/green]")
                console.print(f"  Plots: {formats}")
                console.print(f"  Output: {output_dir}")

                # Show plot paths
                console.print("\n[bold]Generated plots:[/bold]")
                for fmt, path in plot_paths.items():
                    console.print(f"  {fmt.upper()}: {path}")
            else:
                console.print(
                    f"[red]Error: No FEFF output (chi.dat or frame_*) "
                    f"found in {feff_dir}[/red]"
                )
                raise typer.Exit(1)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("config-example")  # type: ignore[misc]
def create_config_example(
    output_file: Path = typer.Option(
        Path("config.yaml"), "--output", "-o", help="Output file path"
    ),
    preset: str = typer.Option(
        "publication", "--preset", "-p", help=f"Base preset: {list(PRESETS.keys())}"
    ),
) -> None:
    """Create an example configuration file."""
    if preset not in PRESETS:
        console.print(f"[red]Error: Unknown preset '{preset}'[/red]")
        raise typer.Exit(1)

    try:
        config = FeffConfig.from_preset(preset)
        yaml_content = f"""# EXAFS Configuration (based on '{preset}' preset)
spectrum_type: {config.spectrum_type}
edge: {config.edge}
radius: {config.radius}
kmin: {config.kmin}
kmax: {config.kmax}
kweight: {config.kweight}
window: {config.window}
dk: {config.dk}
method: {config.method}
force_recalculate: false  # Set to true to skip cache

# User tag settings (empty = use larixite defaults)
user_tag_settings: {dict(config.user_tag_settings) or "{}"}

# Optional settings:
# parallel: true
# n_workers: 4
# sample_interval: 1

# Example custom FEFF parameters (overrides larixite defaults):
# user_tag_settings:
#   S02: "0.8"              # Amplitude reduction factor
#   SCF: "5.0 0 30 0.1 1"   # Self-consistent field
#   EXCHANGE: "0"           # Exchange correlation (0=Hedin-Lundqvist)
#   PRINT: "1 0 0 0 0 3"    # Verbosity levels
"""
        output_file.write_text(yaml_content)
        console.print(f"[green]✓ Configuration example created: {output_file}[/green]")
        console.print(f"  Based on '{preset}' preset")
        console.print("[dim]Note: Empty user_tag_settings uses larixite defaults[/dim]")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("cache")  # type: ignore[misc]
def cache_management(
    action: str = typer.Argument(..., help="Cache action: info, clear"),
) -> None:
    """Manage processing cache."""
    if action not in ["info", "clear"]:
        console.print(
            f"[red]Error: Unknown action '{action}'. Use 'info' or 'clear'[/red]"
        )
        raise typer.Exit(1)

    try:
        wrapper = LarchWrapper(verbose=True, cache_dir=Path.home() / ".larch_cache")

        if action == "info":
            cache_info = wrapper.get_cache_info()
            if cache_info["enabled"]:
                console.print("[cyan]Cache Status[/cyan]")
                console.print("  Enabled: ✓")
                console.print(f"  Directory: {cache_info['cache_dir']}")
                console.print(f"  Files: {cache_info['files']}")
                console.print(f"  Size: {cache_info['size_mb']} MB")
            else:
                console.print("[yellow]Cache is disabled[/yellow]")

        elif action == "clear":
            cache_info = wrapper.get_cache_info()
            if cache_info["files"] > 0:
                wrapper.clear_cache()
                console.print(
                    f"[green]✓ Cleared {cache_info['files']} cache files "
                    f"({cache_info['size_mb']} MB)[/green]"
                )
            else:
                console.print("[yellow]Cache is already empty[/yellow]")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
