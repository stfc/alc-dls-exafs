#!/usr/bin/env python3
"""Streamlined CLI interface for Larch Wrapper - EXAFS processing pipeline"""

from pathlib import Path
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from .wrapper import LarchWrapper, FeffConfig, PRESETS

app = typer.Typer(
    name="larch-cli",
    help="Streamlined CLI for EXAFS processing with larch",
    invoke_without_command=True,
    no_args_is_help=True,
)
console = Console()

def _setup_config(config_file: Path = None, preset: str = None) -> FeffConfig:
    """Setup configuration from file, preset, or defaults."""
    if config_file:
        config = FeffConfig.from_yaml(config_file)
        console.print(f"[dim]Loaded configuration from {config_file}[/dim]")
    elif preset:
        config = FeffConfig.from_preset(preset)
        console.print(f"[dim]Using '{preset}' preset[/dim]")
    else:
        config = FeffConfig()
        console.print(f"[dim]Using default configuration[/dim]")
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

@app.command("info")
def show_info():
    """Show system and dependency information."""
    wrapper = LarchWrapper(verbose=True)
    wrapper.print_diagnostics()

@app.command("generate")
def generate_inputs(
    structure: Path = typer.Argument(..., help="Path to structure file"),
    absorber: str = typer.Argument(..., help="Absorbing atom symbol or site index"),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o", help="Output directory"),
    config_file: Path = typer.Option(None, "--config", "-c", help="YAML configuration file"),
    preset: str = typer.Option(None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"),
):
    """Generate FEFF input files only."""
    if not structure.exists():
        console.print(f"[red]Error: Structure file {structure} not found[/red]")
        raise typer.Exit(1)
    
    try:
        # Setup configuration
        config = _setup_config(config_file, preset)
        
        with LarchWrapper(verbose=True) as wrapper:
            console.print(f"[cyan]Generating FEFF input for {absorber}...[/cyan]")
            input_path = wrapper.generate_feff_input(structure, absorber, output_dir, config)
            console.print(f"[green]✓ FEFF input generated: {input_path}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("run-feff") 
def run_feff_calculation(
    feff_dir: Path = typer.Argument(..., help="Directory containing feff.inp"),
):
    """Run FEFF calculation in specified directory."""
    if not feff_dir.exists():
        console.print(f"[red]Error: Directory {feff_dir} not found[/red]")
        raise typer.Exit(1)
    
    if not (feff_dir / "feff.inp").exists():
        console.print(f"[red]Error: No feff.inp found in {feff_dir}[/red]")
        raise typer.Exit(1)
    
    try:
        with LarchWrapper(verbose=True) as wrapper:
            console.print(f"[cyan]Running FEFF calculation in {feff_dir}...[/cyan]")
            success = wrapper.run_feff(feff_dir)
            if success:
                console.print(f"[green]✓ FEFF calculation completed successfully[/green]")
                console.print(f"  Output files in: {feff_dir}")
            else:
                console.print(f"[red]✗ FEFF calculation failed[/red]")
                raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("process")
def process(
    structure: Path = typer.Argument(..., help="Path to structure file or trajectory"),
    absorber: str = typer.Argument(..., help="Absorbing atom symbol or site index"),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o", help="Output directory"),
    trajectory: bool = typer.Option(False, "--trajectory", "-t", help="Process as trajectory"),
    sample_interval: int = typer.Option(1, "--interval", "-i", help="Process every Nth frame"),
    config_file: Path = typer.Option(None, "--config", "-c", help="YAML configuration file"),
    preset: str = typer.Option(None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"),
    show_plot: bool = typer.Option(False, "--show", help="Display plots interactively"),
    parallel: bool = typer.Option(True, "--parallel/--sequential", help="Enable parallel processing"),
    n_workers: int = typer.Option(None, "--workers", "-w", help="Number of parallel workers"),
    plot_individual_frames: bool = typer.Option(False, "--plot-frames", help="Plot individual trajectory frames (trajectories only)"),
):
    """Process structure or trajectory for EXAFS analysis."""
    if not structure.exists():
        console.print(f"[red]Error: Structure file {structure} not found[/red]")
        raise typer.Exit(1)
    
    try:
        # Setup configuration
        config = _setup_config(config_file, preset)
        
        # Configure for trajectory processing
        if trajectory:
            config.sample_interval = sample_interval
            config.parallel = parallel
            config.n_workers = n_workers
        
        with LarchWrapper(verbose=True) as wrapper:
            console.print(f"[cyan]Processing {structure} for {absorber}...[/cyan]")
            
            with create_progress() as progress:
                task = progress.add_task("Processing...", total=100)
                
                def progress_callback(completed, total, description):
                    progress.update(task, completed=completed, description=description)
                    # And initialize with correct total:
                    task = progress.add_task("Processing...", total=total)

                result = wrapper.process(
                    structure=structure,
                    absorber=absorber,
                    output_dir=output_dir,
                    config=config,
                    trajectory=trajectory,
                    show_plot=show_plot,
                    plot_individual_frames=plot_individual_frames,
                    progress_callback=progress_callback
                )
                
                progress.update(task, completed=100, description="[green]✓ Complete![/green]")
            
            # Show results
            console.print(f"\n[bold green]✓ Processing completed![/bold green]")
            console.print(f"  Output: {output_dir}")
            console.print(f"  Plots: {result.plot_paths[0].name}, {result.plot_paths[1].name}")
            if trajectory:
                console.print(f"  Frames processed: {result.nframes}")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("process-output")
def process_feff_output(
    feff_dir: Path = typer.Argument(..., help="Directory containing FEFF output (chi.dat or trajectory frames)"),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Output directory (default: same as feff_dir)"),
    config_file: Path = typer.Option(None, "--config", "-c", help="YAML configuration file"),
    preset: str = typer.Option(None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"),
    show_plot: bool = typer.Option(False, "--show", help="Display plots interactively"),
    plot_individual_frames: bool = typer.Option(False, "--plot-frames", help="Plot individual trajectory frames (trajectories only)"),
):
    """Process FEFF output files and generate plots. Handles both single frames and trajectories."""
    if not feff_dir.exists():
        console.print(f"[red]Error: Directory {feff_dir} not found[/red]")
        raise typer.Exit(1)
    
    output_dir = output_dir or feff_dir
    
    try:
        config = _setup_config(config_file, preset)
        
        with LarchWrapper(verbose=True) as wrapper:
            # Check if this is a trajectory output (contains frame_XXXX directories)
            frame_dirs = [d for d in feff_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('frame_')]
            
            if frame_dirs:
                # Trajectory output processing
                console.print(f"[cyan]Processing trajectory FEFF output in {feff_dir}...[/cyan]")
                console.print(f"  Found {len(frame_dirs)} trajectory frames")
                
                result = wrapper.process_trajectory_output(
                    feff_dir, config, 
                    plot_individual_frames=plot_individual_frames,
                    show_plot=show_plot,
                    output_dir=output_dir
                )
                
                console.print(f"[green]✓ Trajectory FEFF output processed successfully[/green]")
                console.print(f"  Frames processed: {result.nframes}")
                console.print(f"  Plots: {result.plot_paths[0].name}, {result.plot_paths[1].name}")
                console.print(f"  Output: {output_dir}")
                
            else:
                # Single frame output processing
                if not (feff_dir / "chi.dat").exists():
                    console.print(f"[red]Error: No chi.dat found in {feff_dir}[/red]")
                    raise typer.Exit(1)
                
                console.print(f"[cyan]Processing single FEFF output in {feff_dir}...[/cyan]")
                
                # Process FEFF output
                exafs_group = wrapper.process_feff_output(feff_dir, config)
                
                # Generate plots
                plot_paths = wrapper.plot_results(exafs_group, output_dir, show_plot=show_plot)
                
                console.print(f"[green]✓ FEFF output processed successfully[/green]")
                console.print(f"  Plots: {plot_paths[0].name}, {plot_paths[1].name}")
                console.print(f"  Output: {output_dir}")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@app.command("config-example")
def create_config_example(
    output_file: Path = typer.Option(Path("config.yaml"), "--output", "-o", help="Output file path"),
    preset: str = typer.Option("publication", "--preset", "-p", help=f"Base preset: {list(PRESETS.keys())}"),
):
    """Create an example configuration file."""
    if preset not in PRESETS:
        console.print(f"[red]Error: Unknown preset '{preset}'[/red]")
        raise typer.Exit(1)
    
    try:
        config = FeffConfig.from_preset(preset)
        # Create a simple YAML representation
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
user_tag_settings: {dict(config.user_tag_settings) or '{}'}
"""
        
        output_file.write_text(yaml_content)
        console.print(f"[green]✓ Configuration example created: {output_file}[/green]")
        console.print(f"  Based on '{preset}' preset")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()