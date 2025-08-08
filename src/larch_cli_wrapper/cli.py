#!/usr/bin/env python3
"""CLI interface for Larch Wrapper - EXAFS processing pipeline"""

from pathlib import Path
import sys
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

from .wrapper import LarchWrapper, FeffConfig, PRESETS, ProcessingMode

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
    wrapper = LarchWrapper(verbose=True)
    wrapper.print_diagnostics()

@app.command("process")
def process(
    structure: Path = typer.Argument(
        ...,
        help="Path to structure file (XYZ, CIF, PDB, etc.) or trajectory",
    ),
    absorber: str = typer.Argument(
        ..., help="Absorbing atom symbol or site index (e.g., Fe, Cu, 0)"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output-dir", "-o", help="Output directory"
    ),
    trajectory: bool = typer.Option(
        False, "--trajectory", "-t", help="Process as trajectory (multiple frames, supports parallel processing)"
    ),
    sample_interval: int = typer.Option(
        1, "--interval", "-i", help="Process every Nth frame (for trajectories)"
    ),
    input_only: bool = typer.Option(
        False, "--input-only", help="Generate FEFF inputs only, don't run calculations"
    ),
    config_file: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file (YAML format, overrides other parameters)",
    ),
    preset: str = typer.Option(
        None,
        "--preset",
        help=f"Configuration preset ({', '.join(PRESETS.keys())})",
    ),
    show_plot: bool = typer.Option(
        False, "--show/--no-show", help="Display plot interactively"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel", help="Enable parallel processing for trajectories"
    ),
    n_workers: int = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers (default: auto)"
    ),
):
    """Process a structure or trajectory for EXAFS analysis."""
    if not structure.exists():
        console.print(f"[red]Error: Structure file {structure} not found[/red]")
        raise typer.Exit(1)

    try:
        # Setup configuration
        if config_file:
            config = FeffConfig.from_yaml(config_file)
            console.print(f"[cyan]Loaded configuration from {config_file}[/cyan]")
        elif preset:
            config = FeffConfig.from_preset(preset)
            console.print(f"[cyan]Using preset '{preset}' configuration[/cyan]")
        else:
            config = FeffConfig()
            console.print("[cyan]Using default configuration[/cyan]")
        
        # Apply CLI-specific overrides
        if trajectory:
            config.sample_interval = sample_interval
            config.parallel = parallel
            if n_workers is not None:
                config.n_workers = n_workers
        
        # Show parallel processing status
        if trajectory and config.parallel:
            workers = config.n_workers or "auto"
            console.print(f"[dim]Parallel processing: enabled ({workers} workers)[/dim]")
        elif trajectory:
            console.print(f"[dim]Parallel processing: disabled[/dim]")
        
        wrapper = LarchWrapper(verbose=True)
        
        # Generate FEFF inputs (always the first step)
        console.print(f"[yellow]Generating FEFF input(s) for {absorber}...[/yellow]")
        
        input_info = wrapper.generate_inputs(
            structure=structure,
            absorber=absorber,
            output_dir=output_dir,
            config=config,
            trajectory=trajectory
        )
        
        # Report input generation results
        nframes = input_info['nframes']
        if trajectory:
            console.print(f"[green]✓[/green] Generated inputs for {nframes} frames")
            console.print(f"  Output directory: {output_dir}")
            console.print(f"  Frame directories: {output_dir}/frame_XXXX/")
        else:
            console.print(f"[green]✓[/green] Generated FEFF input successfully")
            console.print(f"  Output directory: {output_dir}")
            console.print(f"  Input file: {output_dir / 'feff.inp'}")
        
        if input_only:
            # Stop here if only inputs were requested
            console.print(f"\n[bold green]✓ Input generation completed successfully![/bold green]")
            console.print("[dim]Use --no-input-only to continue with full processing[/dim]")
            return
        
        # Continue with full processing
        console.print(f"\n[cyan]Processing {structure} for {absorber} (absorber)[/cyan]")
        
        with create_progress() as progress:
            task = progress.add_task(
                "Processing...", 
                total=100
            )
            
            def progress_callback(completed, total, description):
                """Callback to update progress bar."""
                progress.update(
                    task, 
                    completed=int(completed/total * 100),
                    description=description
                )
            
            result = wrapper.process(
                structure=structure,
                absorber=absorber,
                output_dir=output_dir,
                config=config,
                trajectory=trajectory,
                show_plot=show_plot,
                progress_callback=progress_callback
            )
            
            progress.update(task, completed=100, description="[green]✓ Complete![/green]")
        
        console.print("\n[bold green]✓ Processing completed successfully![/bold green]")
        console.print(f"  Output directory: {output_dir}")
        console.print(f"  PDF plot: {result.plot_paths[0]}")
        console.print(f"  SVG plot: {result.plot_paths[1]}")
        
        if result.is_averaged:
            console.print(f"  Processed frames: {result.nframes}")
            console.print(f"  Processing mode: [cyan]Averaged trajectory[/cyan]")
        else:
            console.print(f"  Processing mode: [yellow]Single structure[/yellow]")
    
    except Exception as e:
        console.print(f"\n[red]Error processing structure: {e}[/red]")
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
        help=f"Preset to use as base ({', '.join(PRESETS.keys())})",
    ),
):
    """Create an example configuration file."""
    if preset not in PRESETS:
        console.print(
            f"[red]Error: Unknown preset '{preset}'. Available: {', '.join(PRESETS.keys())}[/red]"
        )
        raise typer.Exit(1)

    try:
        config = FeffConfig.from_preset(preset)
        config.to_yaml(
            output_file, 
            f"Example configuration file based on '{preset}' preset"
        )

        console.print(
            f"[bold green]\u2713[/bold green] Example configuration file created: {output_file}"
        )
        console.print(f"[dim]  Based on preset: {preset}[/dim]")
        console.print("[dim]  Edit the file to customize FEFF parameters[/dim]")

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
    if not config_file.exists():
        console.print(f"[red]Error: Configuration file {config_file} not found[/red]")
        raise typer.Exit(1)

    try:
        config = FeffConfig.from_yaml(config_file)

        console.print(
            f"[bold green]\u2713[/bold green] Configuration file is valid: {config_file}"
        )
        console.print("\n[bold]Loaded parameters:[/bold]")

        # Display parameters in organized way
        console.print(f"  [cyan]Spectrum Type:[/cyan] {config.spectrum_type.value}")
        console.print(f"  [cyan]Edge:[/cyan] {config.edge.value}")
        console.print(f"  [cyan]Radius:[/cyan] {config.radius} �")
        console.print(f"  [cyan]kmin:[/cyan] {config.kmin}")
        console.print(f"  [cyan]kmax:[/cyan] {config.kmax}")
        console.print(f"  [cyan]kweight:[/cyan] {config.kweight}")
        
        if config.user_tag_settings:
            console.print("  [cyan]FEFF Parameters:[/cyan]")
            for key, value in config.user_tag_settings.items():
                console.print(f"    {key}: {value}")

    except Exception as e:
        console.print(f"[red]Error validating configuration file: {e}[/red]")
        raise typer.Exit(1)

@app.command("version")
def show_version():
    """Show version and diagnostic information."""
    wrapper = LarchWrapper(verbose=True)
    wrapper.print_diagnostics()

if __name__ == "__main__":
    app()