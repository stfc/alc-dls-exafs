#!/usr/bin/env python3
"""Streamlined CLI interface for Larch Wrapper - EXAFS processing pipeline."""

from enum import Enum
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

from . import DEFAULT_CACHE_DIR
from .exafs_data import (
    EXAFSDataCollection,
    PlotConfig,
    plot_exafs_matplotlib,
)
from .feff_utils import (
    PRESETS,
    FeffConfig,
    WindowType,
)
from .pipeline import (
    FeffBatch,
    FeffExecutor,
    FeffTask,
    PipelineProcessor,
    ResultProcessor,
)


class PlotComponent(str, Enum):
    """Available plot components."""

    INDIVIDUAL = "individual"
    AVERAGE = "average"
    FRAMES = "frames"
    SITES = "sites"
    ALL = "all"


def parse_plot_components(components_str: str) -> list[PlotComponent]:
    """Parse comma-separated plot components string."""
    if not components_str:
        return [PlotComponent.ALL]

    components = []
    for comp in components_str.split(","):
        comp = comp.strip().lower()
        try:
            components.append(PlotComponent(comp))
        except ValueError:
            available = ", ".join([c.value for c in PlotComponent])
            raise typer.BadParameter(
                f"Invalid plot component '{comp}'. Available: {available}"
            ) from None
    # If 'all' is specified, return all components
    if PlotComponent.ALL in components:
        return [
            PlotComponent.INDIVIDUAL,
            PlotComponent.AVERAGE,
            PlotComponent.FRAMES,
            PlotComponent.SITES,
        ]
    return components


app = typer.Typer(
    name="larch-cli",
    help="Streamlined CLI for EXAFS processing with larch",
    invoke_without_command=True,
    no_args_is_help=True,
)
console = Console()


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


def load_config(
    config_file: Path | None = None, preset: str | None = None
) -> FeffConfig:
    """Load configuration from file or preset."""
    if config_file:
        config = FeffConfig.from_yaml(config_file)
        console.print(f"[dim]Loaded configuration from {config_file}[/dim]")
    elif preset:
        config = FeffConfig.from_preset(preset)
        console.print(f"[dim]Using '{preset}' preset[/dim]")
    else:
        config = FeffConfig()
        console.print("[dim]Using default configuration[/dim]")
    return config


def update_config_from_cli_options(
    config: FeffConfig,
    # FEFF Input parameters
    radius: float | None = None,
    edge: str | None = None,
    # Analysis parameters
    kmin: float | None = None,
    kmax: float | None = None,
    kweight: int | None = None,
    dk: float | None = None,
    dk2: float | None = None,
    window: str | None = None,
    rmax_out: float | None = None,
    with_phase: bool | None = None,
    nfft: int | None = None,
    kstep: float | None = None,
    # Processing parameters
    parallel: bool | None = None,
    workers: int | None = None,
    force_recalculate: bool | None = None,
    cleanup: bool | None = None,
) -> FeffConfig:
    """Update configuration with CLI options (only non-None values)."""
    from dataclasses import replace

    # Build update dictionary with only non-None values
    updates = {}

    # FEFF Input parameters
    if radius is not None:
        updates["radius"] = radius
    if edge is not None:
        updates["edge"] = edge

    # Analysis parameters
    if kmin is not None:
        updates["kmin"] = kmin
    if kmax is not None:
        updates["kmax"] = kmax
    if kweight is not None:
        updates["kweight"] = kweight
    if dk is not None:
        updates["dk"] = dk
    if dk2 is not None:
        updates["dk2"] = dk2
    if window is not None:
        updates["window"] = WindowType(window)
    if rmax_out is not None:
        updates["rmax_out"] = rmax_out
    if with_phase is not None:
        updates["with_phase"] = with_phase
    if nfft is not None:
        updates["nfft"] = nfft
    if kstep is not None:
        updates["kstep"] = kstep

    # Processing parameters
    if parallel is not None:
        updates["parallel"] = parallel
    if workers is not None:
        updates["n_workers"] = workers
    if force_recalculate is not None:
        updates["force_recalculate"] = force_recalculate
    if cleanup is not None:
        updates["cleanup_feff_files"] = cleanup

    # Apply updates
    if updates:
        config = replace(config, **updates)
        console.print(f"[dim]Updated config with {len(updates)} CLI parameters[/dim]")

    return config


def parse_absorber_specification(absorber: str, atoms, all_sites: bool = False) -> dict:
    """Parse absorber specification into normalized form.

    Args:
        absorber: Either element symbol (e.g., 'Fe') or comma-separated indices
                 (e.g., '0,1,2') or combined (e.g., 'Fe:0,1') for specific element sites
        atoms: ASE Atoms object
        all_sites: If True and absorber is element symbol, use all matching sites

    Returns:
        Dict with 'absorber' (list of indices), 'element' (element symbol),
        and 'description' (for display)

    Note:
        Combined format 'Element:idx1,idx2' uses relative indexing within element sites.
        E.g., 'Fe:0,1' means the 1st and 2nd Fe sites, not absolute indices 0,1.
    """
    # Check if absorber looks like indices (contains digits and commas)
    elements = atoms.get_chemical_symbols()
    if absorber.replace(",", "").replace(" ", "").isdigit():
        # Parse as comma-separated indices
        indices = [int(x.strip()) for x in absorber.split(",")]

        # Validate indices
        max_index = len(atoms) - 1
        invalid_indices = [i for i in indices if i < 0 or i > max_index]
        if invalid_indices:
            raise ValueError(
                f"Invalid atom indices {invalid_indices}. "
                f"Must be between 0 and {max_index}"
            )

        # Get element symbols for selected indices only
        selected_elements = [elements[i] for i in indices]
        unique_elements = list(set(selected_elements))

        if len(unique_elements) == 1:
            description = f"indices {indices} ({unique_elements[0]})"
        else:
            raise ValueError(
                f"Multiple species selected: {unique_elements}. "
                "We only support running FEFF on a single element.\n"
                "If you want to choose based on the indices of a specific element, "
                " use 'Element:idx1,idx2' or use the element symbol "
                "(with the optional --all-sites flag)."
            )

        return {
            "absorber": indices,
            "element": unique_elements[0],
            "description": description,
        }
    elif ":" in absorber:
        # Parse as combined element and indices, e.g. 'Fe:0,1'
        try:
            element_part, indices_part = absorber.split(":")
            element = element_part.strip()
            chosen_indices = [int(x.strip()) for x in indices_part.split(",")]

            # Find all indices of atoms with this element
            all_element_indices = [
                i for i, atom in enumerate(atoms) if atom.symbol == element
            ]

            if not all_element_indices:
                raise ValueError(f"No atoms with symbol '{element}' found in structure")

            # Now we want the actual indices in the full atoms list
            indices = []
            for idx in chosen_indices:
                if idx < 0 or idx >= len(all_element_indices):
                    raise ValueError(
                        f"Invalid index {idx} for element '{element}'. "
                        f"Must be between 0 and {len(all_element_indices) - 1}"
                    )
                indices.append(all_element_indices[idx])

            description = f"element {element} (indices {indices})"
            return {"absorber": indices, "element": element, "description": description}
        except ValueError as e:
            raise ValueError(
                f"Invalid absorber format '{absorber}'. "
                f"Use 'Element:idx1,idx2' or 'idx1,idx2' or 'Element'. "
                "Note that multiple elements are not supported in a single run."
            ) from e

    else:
        # Parse as element symbol
        element = absorber.strip()

        # Find all matching indices
        matching_indices = [i for i, atom in enumerate(atoms) if atom.symbol == element]

        if not matching_indices:
            raise ValueError(f"No atoms with symbol '{element}' found in structure")

        if all_sites:
            selected_indices = matching_indices
            description = (
                f"element {element} (all {len(selected_indices)} sites: "
                f"{selected_indices})"
            )
        else:
            # Default: use first site but inform user about others
            selected_indices = [matching_indices[0]]  # Keep as list for consistency
            description = (
                f"element {element} (site {matching_indices[0]}, "
                f"{len(matching_indices) - 1} other sites available)"
            )
            if len(matching_indices) > 1:
                console.print(
                    f"[yellow]Note: Found {len(matching_indices)} {element} sites. "
                    f"Using first site ({matching_indices[0]}). "
                    f"Use --all-sites to process all sites: {matching_indices}[/yellow]"
                )

        return {
            "absorber": selected_indices,
            "element": element,
            "description": description,
        }


@app.command("info")
def show_info() -> None:
    """Show system and dependency information."""
    # Create processor with default cache to show diagnostics
    processor = PipelineProcessor(
        config=FeffConfig(),
        cache_dir=DEFAULT_CACHE_DIR,
    )
    diagnostics = processor.get_diagnostics()

    console.print("[bold cyan]EXAFS Processing System Info[/bold cyan]")
    console.print(f"Python version: {diagnostics['python_version']}")
    console.print(f"Platform: {diagnostics['platform']}")
    console.print(f"Cache enabled: {diagnostics['cache_enabled']}")

    if diagnostics["cache_enabled"]:
        console.print(f"Cache directory: {diagnostics['cache_dir']}")
        console.print(f"Cache files: {diagnostics['cache_files']}")
        console.print(f"Cache size: {diagnostics['cache_size_mb']:.2f} MB")


@app.command("generate")
def generate_feff_inputs(
    structure: Path = typer.Argument(..., help="Path to structure file"),
    absorber: str = typer.Argument(
        ...,
        help="Absorbing atom symbol, site indices, or combined "
        "(e.g., 'Fe', '0,1,2', or 'Fe:0,1')",
    ),
    output_dir: Path = typer.Option(
        Path("feff_inputs"), "--output", "-o", help="Output directory"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"
    ),
    all_frames: bool = typer.Option(
        False,
        "--all-frames",
        "--trajectory",
        "-t",
        help="Generate inputs for all frames in trajectory",
    ),
    all_sites: bool = typer.Option(
        False, "--all-sites", help="Generate inputs for all sites of the given element"
    ),
    cleanup: bool = typer.Option(
        True, "--cleanup/--no-cleanup", help="Clean up unnecessary files"
    ),
    # FEFF Input Parameters
    radius: float | None = typer.Option(
        None, "--radius", help="Cluster radius in Angstroms"
    ),
    edge: str | None = typer.Option(
        None, "--edge", help="Absorption edge (K, L1, L2, L3)"
    ),
    ase_read_kwargs: str | None = typer.Option(
        None,
        "--ase-kwargs",
        help=(
            "JSON string of ASE read kwargs. Examples: "
            '\'{"index":":"}\' (all frames), '
            '\'{"index":"::10"}\' (every 10th frame), '
            '\'{"format":"vasp"}\' (force format)'
        ),
    ),
) -> None:
    """Generate FEFF input files for structure.

    The absorber can be specified as:
    - Element symbol (e.g., 'Fe') - uses first site by default
    - Comma-separated indices (e.g., '0,1,2') - uses specific sites
    - Combined format (e.g., 'Fe:0,1') - specific sites of given element

    Use --all-frames with element symbol to process all frames in trajectory.
    Use --all-sites with element symbol to process all matching sites.

    Use --ase-kwargs to pass additional arguments to ase.io.read():
    - '{"index": ":"}' to read all frames
    - '{"index": "0:10"}' to read first 10 frames
    - '{"index": "::10"}' Every 10th frame
    - Other ASE read kwargs like format, parallel, etc.
    """
    if not structure.exists():
        console.print(f"[red]Error: Structure file {structure} not found[/red]")
        raise typer.Exit(1)

    try:
        config = load_config(config_file, preset)
        config = update_config_from_cli_options(
            config,
            radius=radius,
            edge=edge,
            cleanup=cleanup,
        )

        # Parse ASE read kwargs if provided
        read_kwargs = {"index": ":"} if all_frames else {"index": -1}
        if ase_read_kwargs:
            import json

            try:
                read_kwargs = json.loads(ase_read_kwargs)
                console.print(f"[dim]Using ASE read kwargs: {read_kwargs}[/dim]")
            except json.JSONDecodeError as e:
                console.print(f"[red]Error: Invalid JSON in --ase-kwargs: {e}[/red]")
                raise typer.Exit(1) from e

        # Load structures based on trajectory flag and kwargs
        structures = ase_read(str(structure), **read_kwargs)
        if not isinstance(structures, list):
            structures = [structures]
        console.print(f"[dim]Loaded {len(structures)} frames from file[/dim]")

        # Resolve output_dir to absolute path immediately
        # to avoid issues with CWD changes
        output_dir = output_dir.resolve()

        # Parse absorber specification using first structure as reference
        # Note this assumes that the indices are valid for all frames!
        absorber_spec = parse_absorber_specification(absorber, structures[0], all_sites)

        console.print(
            f"[cyan]Generating FEFF inputs for {absorber_spec['description']}...[/cyan]"
        )

        # Use default cache directory for consistency
        processor = PipelineProcessor(
            config=config,
            cache_dir=DEFAULT_CACHE_DIR,
        )

        if all_frames and len(structures) > 1:
            # Generate inputs for all frames (trajectory mode)
            console.print(
                f"[cyan]Processing trajectory: {len(structures)} frames × "
                f"{len(absorber_spec['absorber'])} sites...[/cyan]"
            )

            batch = processor.input_generator.generate_trajectory_inputs(
                structures=structures,
                absorber=absorber_spec["absorber"],
                output_dir=output_dir,
            )
        else:
            # Generate inputs for single structure (or single frame with multiple sites)
            batch = processor.input_generator.generate_single_site_inputs(
                structure=structures[0],
                absorber=absorber_spec["absorber"],
                output_dir=output_dir,
            )

        console.print(f"[green]✓ Generated {len(batch.tasks)} FEFF input files[/green]")

        # Determine structure from tasks
        if batch.tasks:
            # Count unique frames and sites
            frames_by_index = batch.get_tasks_by_frame()
            num_frames = len(frames_by_index)
            num_sites = (
                len(batch.tasks) // num_frames if num_frames > 0 else len(batch.tasks)
            )

            console.print(f"  Output directory: {output_dir}")

            if num_frames > 1:
                console.print(f"  {num_frames} frames × {num_sites} sites")
            else:
                console.print(f"  {num_sites} sites")

            # Show example directories (full frame_XXXX/site_XXXX paths)
            example_dirs = [
                f"{task.feff_dir.parent.name}/{task.feff_dir.name}"
                for task in batch.tasks[: min(5, len(batch.tasks))]
            ]
            if example_dirs:
                console.print(f"  Examples: {', '.join(example_dirs)}")
            if len(batch.tasks) > 5:
                console.print(f"  ... and {len(batch.tasks) - 5} more")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("run-feff")
def run_feff(
    directories: list[Path] = typer.Argument(
        ..., help="Directories containing feff.inp files"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Run FEFF calculations in parallel"
    ),
    workers: int | None = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers"
    ),
    cleanup: bool = typer.Option(
        True, "--cleanup/--no-cleanup", help="Clean up unnecessary files"
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Cache directory for FEFF results"
    ),
    force_recalculate: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Force recalculation even if cached results exist",
    ),
) -> None:
    """Run FEFF calculations in specified directories."""
    # Validate directories and create FeffTask objects

    valid_tasks = []
    for feff_dir in directories:
        # Convert to absolute path to avoid working directory issues
        feff_dir = feff_dir.resolve()

        if not feff_dir.exists():
            console.print(
                f"[yellow]Warning: Directory {feff_dir} not found, skipping[/yellow]"
            )
            continue

        feff_inp = feff_dir / "feff.inp"
        if not feff_inp.exists():
            console.print(
                f"[yellow]Warning: No feff.inp in {feff_dir}, skipping[/yellow]"
            )
            continue

        # Create FeffTask from directory
        # Extract frame/site indices from directory name if possible
        frame_index = 0
        site_index = 0

        # Try to parse frame_XXXX/site_XXXX pattern
        parts = feff_dir.parts
        for _i, part in enumerate(parts):
            if part.startswith("frame_") and part[6:].isdigit():
                frame_index = int(part[6:])
            if part.startswith("site_") and part[5:].isdigit():
                site_index = int(part[5:])

        task = FeffTask(
            input_file=feff_inp,
            site_index=site_index,
            frame_index=frame_index,
        )
        valid_tasks.append(task)

    if not valid_tasks:
        console.print("[red]Error: No valid FEFF directories found[/red]")
        raise typer.Exit(1)

    try:
        config = FeffConfig(cleanup_feff_files=cleanup)

        console.print(
            f"[cyan]Running FEFF calculations in "
            f"{len(valid_tasks)} directories...[/cyan]"
        )

        # Use default cache directory if not specified
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR

        # Create batch and execute using PipelineProcessor with caching
        # Use first task's parent directory as output_dir for batch
        output_dir = valid_tasks[0].feff_dir.parent if valid_tasks else Path(".")
        batch = FeffBatch(tasks=valid_tasks, output_dir=output_dir, config=config)
        executor = FeffExecutor(
            max_workers=workers,
            cache_dir=cache_dir,
            force_recalculate=force_recalculate,
        )

        with create_progress() as progress:
            task_id = progress.add_task("Running FEFF...", total=len(valid_tasks))

            # Execute all FEFF calculations
            task_results = executor.execute_batch(batch, parallel=parallel)

            progress.update(
                task_id,
                completed=len(valid_tasks),
                description="[green]✓ Complete![/green]",
            )

        # Count successes
        success_count = sum(1 for success in task_results.values() if success)
        failed_tasks = [
            task_id for task_id, success in task_results.items() if not success
        ]

        console.print(
            f"[green]✓ FEFF calculations completed: "
            f"{success_count}/{len(valid_tasks)} successful[/green]"
        )

        if failed_tasks:
            console.print(f"[yellow]Failed tasks: {failed_tasks[:5]}")
            if len(failed_tasks) > 5:
                console.print(f"  ... and {len(failed_tasks) - 5} more")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("analyze")
def analyze_feff_outputs(
    directories: list[Path] = typer.Argument(
        ..., help="Directories containing FEFF output files"
    ),
    output_dir: Path = typer.Option(
        Path("analysis"), "--output", "-o", help="Output directory for plots"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"
    ),
    show_plot: bool = typer.Option(False, "--show", help="Display plots interactively"),
    plot_style: str = typer.Option(
        "publication",
        "--style",
        help=(
            "Plot style: 'publication', 'presentation', "
            "or path to matplotlib style file"
        ),
    ),
    save_groups: bool = typer.Option(
        False,
        "--save-groups",
        help="Save averaged Larch Groups to files (ASCII format)",
    ),
    plot_include: str = typer.Option(
        "all",
        "--plot-include",
        help=(
            "Comma-separated list of plot components to include. "
            "Options: 'individual' (individual spectra), "
            "'average' (overall average across frames & sites), "
            "'frames' (one curve per frame; averaged over sites if needed), "
            "'sites' (one curve per site; averaged over frames) and 'all' "
            "(all components present). "
            "Examples: 'average', 'average,frames', 'individual,average,sites' "
            "Default is 'all'."
        ),
    ),
    # Analysis Parameters
    kmin: float | None = typer.Option(
        None, "--kmin", help="Starting k for Fourier transform (Å⁻¹)"
    ),
    kmax: float | None = typer.Option(
        None, "--kmax", help="Ending k for Fourier transform (Å⁻¹)"
    ),
    kweight: int | None = typer.Option(
        None, "--kweight", help="k-weighting exponent (1, 2, or 3)"
    ),
    dk: float | None = typer.Option(
        None, "--dk", help="Tapering parameter for FT window"
    ),
    dk2: float | None = typer.Option(
        None, "--dk2", help="Second tapering parameter for FT window"
    ),
    window: str | None = typer.Option(
        None, "--window", help=f"Window function: {[w.value for w in WindowType]}"
    ),
    rmax_out: float | None = typer.Option(
        None, "--rmax-out", help="Highest R for output data (Å)"
    ),
    with_phase: bool | None = typer.Option(
        None, "--with-phase/--no-phase", help="Output phase as well as magnitude"
    ),
    nfft: int | None = typer.Option(None, "--nfft", help="Number of FFT points"),
    kstep: float | None = typer.Option(None, "--kstep", help="k-step for FFT (Å⁻¹)"),
) -> None:
    """Analyze FEFF output files and generate plots.

    Plot components:
    - 'individual': Plot individual spectra (before averaging)
    - 'average': Plot the overall average spectrum (averaged over all sites and frames)
    - 'frames': Plot frame averages (each frame averaged over sites)
    - 'sites': Plot site averages (each site averaged over frames)
    - 'all': Plot all available components

    You can combine multiple components with commas, e.g.:
    - 'average,frames': Plot both overall average and frame averages
    - 'individual,average': Plot individual spectra with overall average overlay
    - 'average,frames,sites': Plot overall average with both frame and site averages
    """
    # Validate directories and create FeffTask objects

    valid_tasks = []
    for feff_dir in directories:
        if not feff_dir.exists():
            console.print(
                f"[yellow]Warning: Directory {feff_dir} not found, skipping[/yellow]"
            )
            continue
        if not (feff_dir / "chi.dat").exists():
            console.print(
                f"[yellow]Warning: No chi.dat in {feff_dir}, skipping[/yellow]"
            )
            continue

        # Create FeffTask from directory (similar to run-feff)
        # Extract frame/site indices from directory name if possible
        frame_index = 0
        site_index = 0

        # Try to parse frame_XXXX/site_XXXX pattern
        parts = feff_dir.parts
        for _i, part in enumerate(parts):
            if part.startswith("frame_") and part[6:].isdigit():
                frame_index = int(part[6:])
            if part.startswith("site_") and part[5:].isdigit():
                site_index = int(part[5:])

        # Use feff.inp if exists, otherwise chi.dat
        input_file = feff_dir / "feff.inp"
        if not input_file.exists():
            input_file = feff_dir / "chi.dat"

        task = FeffTask(
            input_file=input_file,
            site_index=site_index,
            frame_index=frame_index,
        )
        valid_tasks.append(task)

    if not valid_tasks:
        console.print("[red]Error: No valid FEFF output directories found[/red]")
        raise typer.Exit(1)

    try:
        config = load_config(config_file, preset)
        config = update_config_from_cli_options(
            config,
            kmin=kmin,
            kmax=kmax,
            kweight=kweight,
            dk=dk,
            dk2=dk2,
            window=window,
            rmax_out=rmax_out,
            with_phase=with_phase,
            nfft=nfft,
            kstep=kstep,
        )

        # Resolve output_dir to absolute path
        # immediately to avoid issues with CWD changes
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(
            f"[cyan]Analyzing {len(valid_tasks)} FEFF output directories...[/cyan]"
        )

        # Use PipelineProcessor ResultProcessor
        # Create batch and mark all tasks as successful
        batch_output_dir = valid_tasks[0].feff_dir.parent if valid_tasks else output_dir
        batch = FeffBatch(tasks=valid_tasks, output_dir=batch_output_dir, config=config)
        processor = ResultProcessor(config)

        # Create fake task_results marking all as successful (since we validated
        # chi.dat exists)
        task_results = {task.task_id: True for task in valid_tasks}

        with create_progress() as progress:
            task_id = progress.add_task("Processing outputs...", total=len(valid_tasks))

            # Load all successful results using ResultProcessor
            groups = processor.load_successful_results(batch, task_results)

            progress.update(
                task_id,
                completed=len(valid_tasks),
                description="[green]✓ Processing complete![/green]",
            )

            # Generate plots
            console.print("[cyan]Generating plots...[/cyan]")

            frame_averages = processor.create_frame_averages(
                groups, batch
            )  # These are frame averages (averaged over sites)
            site_averages = processor.create_site_averages(
                groups, batch
            )  # These are site averages (averaged over frames)
            overall_average = processor.create_overall_average(
                list(groups.values())
            )  # Overall average (averaged over all sites and/or frames)

            # Parse plot components and create plot
            plot_components = parse_plot_components(plot_include)

            # Create collection with all available data
            collection = EXAFSDataCollection(
                kweight_used=config.kweight,
                fourier_params=config.fourier_params,
            )

            # Set data based on what's available
            if overall_average:
                collection.overall_average = overall_average
            if frame_averages:
                collection.frame_averages = frame_averages
            if site_averages:
                collection.site_averages = site_averages
            if groups:
                collection.individual_spectra = list(groups.values())

            # Check what components are available and warn if requested but missing
            available_components = []
            missing_components = []

            for component in plot_components:
                if (
                    component == PlotComponent.INDIVIDUAL
                    and collection.individual_spectra
                ):
                    available_components.append(component)
                elif component == PlotComponent.AVERAGE and collection.overall_average:
                    available_components.append(component)
                elif component == PlotComponent.FRAMES and collection.frame_averages:
                    available_components.append(component)
                elif component == PlotComponent.SITES and collection.site_averages:
                    available_components.append(component)
                else:
                    missing_components.append(component)

            if missing_components:
                missing_str = ", ".join([c.value for c in missing_components])
                console.print(
                    f"[yellow]Warning: Requested components not available: "
                    f"{missing_str}[/yellow]"
                )

            if available_components:
                # Determine plot flags
                plot_individual = PlotComponent.INDIVIDUAL in available_components
                plot_overall = PlotComponent.AVERAGE in available_components
                plot_frames = PlotComponent.FRAMES in available_components
                plot_sites = PlotComponent.SITES in available_components

                # Generate filename based on components
                component_names = [c.value for c in available_components]
                filename_base = "_and_".join(component_names) + "_EXAFS"

                # Create plot configuration
                plot_config = PlotConfig(
                    plot_individual=plot_individual,
                    plot_overall_avg=plot_overall,
                    plot_frame_avg=plot_frames,
                    plot_site_avg=plot_sites,
                    absorber="X",  # Generic absorber for analysis
                    edge="K",
                    style=plot_style,
                )

                # Generate plot using new matplotlib function
                plot_result = plot_exafs_matplotlib(
                    collection=collection,
                    config=plot_config,
                    output_dir=output_dir,
                    filename_base=filename_base,
                    show_plot=show_plot,
                )

                # Success message
                components_str = ", ".join(component_names)
                console.print(
                    f"[green]✓ Plot generated with components: {components_str}[/green]"
                )

                # Additional info
                if plot_individual:
                    console.print(
                        f"  Individual spectra: {len(collection.individual_spectra)}"
                    )
                if plot_frames:
                    console.print(f"  Frame averages: {len(collection.frame_averages)}")
                if plot_sites:
                    console.print(f"  Site averages: {len(collection.site_averages)}")

            else:
                console.print(
                    "[yellow]Warning: No data available for any requested "
                    "plot components[/yellow]"
                )
                plot_result = None

            # Show plot information
            if plot_result:
                console.print("\n[bold]Generated plots:[/bold]")
                # Handle nested PlotResult structure
                plot_paths = None
                if hasattr(plot_result, "plot_paths"):
                    if isinstance(plot_result.plot_paths, dict):
                        plot_paths = plot_result.plot_paths
                    elif hasattr(plot_result.plot_paths, "plot_paths") and isinstance(
                        plot_result.plot_paths.plot_paths, dict
                    ):
                        # Handle nested PlotResult
                        plot_paths = plot_result.plot_paths.plot_paths

                if plot_paths:
                    for fmt, path in plot_paths.items():
                        console.print(f"  {fmt.upper()}: {path}")
                else:
                    console.print("  Plot result available but format not recognized")

            # Save averaged groups if requested
            if save_groups:
                console.print("\n[cyan]Saving averaged Larch Groups...[/cyan]")

                # Create collection with all the computed data
                collection = EXAFSDataCollection(
                    kweight_used=config.kweight,
                    fourier_params=config.fourier_params,
                )

                if overall_average:
                    collection.overall_average = overall_average

                collection.frame_averages = frame_averages
                collection.site_averages = site_averages
                collection.individual_spectra = list(groups.values())

                # Save to subdirectory
                groups_dir = output_dir / "larch_groups"
                saved_dir = collection.export_larch_groups(
                    output_dir=groups_dir, save_individual=True, save_averages=True
                )

                console.print(f"[green]✓ Larch Groups saved to {saved_dir}[/green]")
                console.print(
                    "  Use EXAFSDataCollection.load_larch_groups() to load them back"
                )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("pipeline")
def run_full_pipeline(
    structure: Path = typer.Argument(..., help="Path to structure file"),
    absorber: str = typer.Argument(
        ...,
        help="Absorbing atom symbol, site indices, or combined "
        "(e.g., 'Fe', '0,1,2', or 'Fe:0,1')",
    ),
    output_dir: Path = typer.Option(
        Path("pipeline_output"), "--output", "-o", help="Output directory"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help=f"Configuration preset: {list(PRESETS.keys())}"
    ),
    all_sites: bool = typer.Option(
        False, "--all-sites", help="Process all sites of the given element"
    ),
    all_frames: bool = typer.Option(
        False,
        "--all-frames",
        "--trajectory",
        "-t",
        help="Process all frames in trajectory",
    ),
    ase_read_kwargs: str | None = typer.Option(
        None,
        "--ase-kwargs",
        help=(
            "JSON string of ASE read kwargs. Examples: "
            '\'{"index":":"}\' (all frames), '
            '\'{"index":"::10"}\' (every 10th frame), '
            '\'{"format":"vasp"}\' (force format)'
        ),
    ),
    show_plot: bool = typer.Option(False, "--show", help="Display plots interactively"),
    plot_style: str = typer.Option(
        "publication",
        "--style",
        help=(
            "Plot style: 'publication', 'presentation', "
            "or path to matplotlib style file"
        ),
    ),
    save_groups: bool = typer.Option(
        False,
        "--save-groups",
        help="Save averaged Larch Groups to files (ASCII format)",
    ),
    plot_include: str = typer.Option(
        "all",
        "--plot-include",
        help=(
            "Comma-separated list of plot components to include. "
            "Options: 'individual' (individual spectra), 'average' (overall average), "
            "'frames' (frame averages), 'sites' (site averages) and 'all' "
            "(all components present). "
            "Examples: 'average', 'average,frames', 'individual,average,sites' "
            "Default is 'all'."
        ),
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--sequential", help="Enable parallel processing"
    ),
    workers: int | None = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers"
    ),
    cleanup: bool = typer.Option(
        True, "--cleanup/--no-cleanup", help="Clean up intermediate files"
    ),
    force_recalculate: bool = typer.Option(
        False, "--force/--no-force", help="Force recalculation even if output exists"
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Cache directory for FEFF results"
    ),
    precompute_potentials: bool = typer.Option(
        False,
        "--reuse-potentials/--no-reuse-potentials",
        help="Reuse precomputed potentials for faster calculations "
        "i.e. compute potentials once for each specified site and "
        "reuse for these for all frames. This is much faster for trajectories. "
        "You may want to also specify --potentials-structure to provide a "
        "specific structure file for the potentials generation.",
    ),
    precompute_potentials_structure_path: Path | None = typer.Option(
        None,
        "--potentials-structure",
        help="Structure file to use for precomputing potentials "
        "(defaults to average of input structures if not provided).",
    ),
    # FEFF Input Parameters
    radius: float | None = typer.Option(
        None, "--radius", help="Cluster radius in Angstroms"
    ),
    edge: str | None = typer.Option(
        None, "--edge", help="Absorption edge (K, L1, L2, L3)"
    ),
    # Analysis Parameters
    kmin: float | None = typer.Option(
        None, "--kmin", help="Starting k for Fourier transform (Å⁻¹)"
    ),
    kmax: float | None = typer.Option(
        None, "--kmax", help="Ending k for Fourier transform (Å⁻¹)"
    ),
    kweight: int | None = typer.Option(
        None, "--kweight", help="k-weighting exponent (1, 2, or 3)"
    ),
    dk: float | None = typer.Option(
        None, "--dk", help="Tapering parameter for FT window"
    ),
    dk2: float | None = typer.Option(
        None, "--dk2", help="Second tapering parameter for FT window"
    ),
    window: str | None = typer.Option(
        None, "--window", help=f"Window function: {[w.value for w in WindowType]}"
    ),
    rmax_out: float | None = typer.Option(
        None, "--rmax-out", help="Highest R for output data (Å)"
    ),
    with_phase: bool | None = typer.Option(
        None, "--with-phase/--no-phase", help="Output phase as well as magnitude"
    ),
    nfft: int | None = typer.Option(None, "--nfft", help="Number of FFT points"),
    kstep: float | None = typer.Option(None, "--kstep", help="k-step for FFT (Å⁻¹)"),
) -> None:
    """Run the complete EXAFS processing pipeline.

    The absorber can be specified as:
    - Element symbol (e.g., 'Fe') - uses first site by default
    - Comma-separated indices (e.g., '0,1,2') - uses specific sites
    - Combined format (e.g., 'Fe:0,1') - specific sites of given element

    Use --all-sites with element symbol to process all matching sites.
    Use --all-frames to process all frames in trajectory.
    Use --reuse-potentials to speed up trajectory calculations by precomputing
    potentials once and reusing them (recommended for trajectories with many frames).
    Use --potentials-structure to specify a structure file for precomputing potentials.
    Use --ase-kwargs to pass additional arguments to ase.io.read():
    - '{"index": ":"}' to read all frames
    - '{"index": "0:10"}' to read first 10 frames
    - '{"index": "::10"}' Every 10th frame
    - Other ASE read kwargs like format, parallel, etc.

    Plot components:
    - 'individual': Plot individual spectra (before averaging)
    - 'average': Plot the overall average spectrum (averaged over all sites and frames)
    - 'frames': Plot frame averages (each frame averaged over sites)
    - 'sites': Plot site averages (each site averaged over frames)
    - 'all': Plot all components (individual, average, frames, sites)

    You can combine multiple components with commas, e.g.:
    - 'average,frames': Plot both overall average and frame averages
    - 'average,frames,sites': Plot overall average components with both frame
      and site averages
    """
    if not structure.exists():
        console.print(f"[red]Error: Structure file {structure} not found[/red]")
        raise typer.Exit(1)

    try:
        config = load_config(config_file, preset)
        config = update_config_from_cli_options(
            config,
            # FEFF Input parameters
            radius=radius,
            edge=edge,
            # Analysis parameters
            kmin=kmin,
            kmax=kmax,
            kweight=kweight,
            dk=dk,
            dk2=dk2,
            window=window,
            rmax_out=rmax_out,
            with_phase=with_phase,
            nfft=nfft,
            kstep=kstep,
            # Processing parameters
            parallel=parallel,
            workers=workers,
            force_recalculate=force_recalculate,
            cleanup=cleanup,
        )

        # Parse ASE read kwargs if provided
        read_kwargs = {"index": ":"} if all_frames else {"index": -1}
        if ase_read_kwargs:
            import json

            try:
                read_kwargs = json.loads(ase_read_kwargs)
                console.print(f"[dim]Using ASE read kwargs: {read_kwargs}[/dim]")
            except json.JSONDecodeError as e:
                console.print(f"[red]Error: Invalid JSON in --ase-kwargs: {e}[/red]")
                raise typer.Exit(1) from e

        # Load structures based on trajectory flag and kwargs
        atoms = ase_read(str(structure), **read_kwargs)
        if not isinstance(atoms, list):
            atoms = [atoms]
        structures = atoms
        console.print(f"[dim]Loaded {len(structures)} frames from file[/dim]")

        # Optionally load structure to precompute potentials for
        precompute_potentials_structure: Atoms | None = None
        if precompute_potentials_structure_path:
            if not precompute_potentials_structure_path.exists():
                console.print(
                    f"[red]Error: Precompute structure file "
                    f"{precompute_potentials_structure_path} not found[/red]"
                )
                raise typer.Exit(1)

            precompute_atoms = ase_read(str(precompute_potentials_structure_path))
            if not isinstance(precompute_atoms, Atoms):
                console.print(
                    "[red]Error: Precompute structure file must contain "
                    "a single structure in a common format (i.e. ASE-readable)[/red]"
                )
                raise typer.Exit(1)

            precompute_potentials_structure = precompute_atoms
            console.print(
                "[dim]Using provided structure for precomputing potentials[/dim]"
            )

        # Resolve output_dir to absolute path
        # immediately to avoid issues with CWD changes
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use first structure for absorber parsing
        # Note this assumes that the indices are valid for all frames!
        absorber_spec = parse_absorber_specification(absorber, structures[0], all_sites)

        console.print(
            f"[cyan]Running full EXAFS pipeline for "
            f"{absorber_spec['description']}...[/cyan]"
        )

        # Use default cache directory if not specified
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR

        # The config object returned by update_config_from_cli_options
        # is already a FeffConfig
        # Update cleanup setting and other processing parameters
        config.cleanup_feff_files = cleanup
        config.parallel = parallel
        config.n_workers = workers
        config.force_recalculate = force_recalculate

        processor = PipelineProcessor(
            config=config,
            max_workers=workers,
            cache_dir=cache_dir,
            force_recalculate=force_recalculate,
        )

        with create_progress() as progress:
            feff_task_id = progress.add_task("FEFF calculations...", total=None)

            def progress_callback(completed: int, total: int):
                """Update progress bar with FEFF calculation progress."""
                progress.update(
                    feff_task_id,
                    completed=completed,
                    total=total,
                    description=f"FEFF calculations: {completed}/{total}",
                )

            # Trajectory processing
            final_group, frame_averages, actual_site_averages, individual_groups = (
                processor.process_trajectory(
                    structures=structures,
                    absorber=absorber_spec["absorber"],
                    output_dir=output_dir,
                    parallel=parallel,
                    progress_callback=progress_callback,
                    precompute_potentials=precompute_potentials,
                    precompute_potentials_structure=precompute_potentials_structure,
                )
            )

            # Mark as complete
            progress.update(
                feff_task_id,
                description="[green]✓ FEFF calculations complete![/green]",
            )

        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
        console.print(f"  Output directory: {output_dir}")
        console.print(
            f"  Processing mode: "
            f"{'trajectory' if len(structures) > 1 else 'single structure'}"
        )
        console.print(f"  Successful calculations: {len(individual_groups)}")

        # Generate plots if show_plot is enabled or plot components specified
        if show_plot or plot_include != "all":
            console.print("\n[cyan]Generating plots...[/cyan]")

            # Parse plot components and create plot
            plot_components = parse_plot_components(plot_include)

            # Create collection with all available data
            collection = EXAFSDataCollection(
                kweight_used=config.kweight,
                fourier_params=config.fourier_params,
            )

            # Set data based on what's available
            if final_group:
                collection.overall_average = final_group
            if frame_averages:
                collection.frame_averages = frame_averages
            if actual_site_averages:
                collection.site_averages = actual_site_averages
            if individual_groups:
                collection.individual_spectra = individual_groups

            # Check what components are available and warn if requested but missing
            available_components = []
            missing_components = []

            for component in plot_components:
                if (
                    component == PlotComponent.INDIVIDUAL
                    and collection.individual_spectra
                ):
                    available_components.append(component)
                elif component == PlotComponent.AVERAGE and collection.overall_average:
                    available_components.append(component)
                elif component == PlotComponent.FRAMES and collection.frame_averages:
                    available_components.append(component)
                elif component == PlotComponent.SITES and collection.site_averages:
                    available_components.append(component)
                else:
                    missing_components.append(component)

            if missing_components:
                missing_str = ", ".join([c.value for c in missing_components])
                console.print(
                    f"[yellow]Warning: Requested components not available: "
                    f"{missing_str}[/yellow]"
                )

            if available_components:
                # Determine plot flags
                plot_individual = PlotComponent.INDIVIDUAL in available_components
                plot_overall = PlotComponent.AVERAGE in available_components
                plot_frames = PlotComponent.FRAMES in available_components
                plot_sites = PlotComponent.SITES in available_components

                # Generate filename based on components
                component_names = [c.value for c in available_components]
                filename_base = "_and_".join(component_names) + "_EXAFS"

                # Create plot configuration
                plot_config = PlotConfig(
                    plot_individual=plot_individual,
                    plot_overall_avg=plot_overall,
                    plot_frame_avg=plot_frames,
                    plot_site_avg=plot_sites,
                    absorber=absorber_spec["element"],  # Generic absorber for analysis
                    edge=config.edge,
                    style=plot_style,
                )

                # Generate plot using new matplotlib function
                _plot_result = plot_exafs_matplotlib(
                    collection=collection,
                    config=plot_config,
                    output_dir=output_dir,
                    filename_base=filename_base,
                    show_plot=show_plot,
                )

                # Success message
                components_str = ", ".join(component_names)
                console.print(
                    f"[green]✓ Plot generated with components: {components_str}[/green]"
                )

                # Additional info
                if plot_individual:
                    console.print(
                        f"  Individual spectra: {len(collection.individual_spectra)}"
                    )
                if plot_frames:
                    console.print(f"  Frame averages: {len(collection.frame_averages)}")
                if plot_sites:
                    console.print(f"  Site averages: {len(collection.site_averages)}")

            else:
                console.print(
                    "[yellow]Warning: No data available for any requested "
                    "plot components[/yellow]"
                )
        else:
            console.print("  Note: Use --show or --plot-include to generate plots")

        # Save averaged groups if requested
        if save_groups:
            console.print("\n[cyan]Saving averaged Larch Groups...[/cyan]")

            # Create collection with all the computed data
            collection = EXAFSDataCollection(
                kweight_used=config.kweight,
                fourier_params=config.fourier_params,
            )

            if final_group:
                collection.overall_average = final_group

            collection.frame_averages = frame_averages
            collection.site_averages = actual_site_averages
            collection.individual_spectra = individual_groups

            # Save to subdirectory
            groups_dir = output_dir / "larch_groups"
            saved_dir = collection.export_larch_groups(
                output_dir=groups_dir, save_individual=True, save_averages=True
            )

            console.print(f"[green]✓ Larch Groups saved to {saved_dir}[/green]")
            console.print(
                "  Use EXAFSDataCollection.load_larch_groups() to load them back"
            )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("cache")
def manage_cache(
    action: str = typer.Argument(..., help="Cache action: info, clear"),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Custom cache directory"
    ),
) -> None:
    """Manage processing cache."""
    if action not in ["info", "clear"]:
        console.print(
            f"[red]Error: Unknown action '{action}'. Use 'info' or 'clear'[/red]"
        )
        raise typer.Exit(1)

    try:
        cache_path = cache_dir or DEFAULT_CACHE_DIR
        processor = PipelineProcessor(
            config=FeffConfig(),
            cache_dir=cache_path,
        )

        if action == "info":
            cache_info = processor.get_cache_info()
            console.print("[cyan]Cache Information[/cyan]")
            console.print(f"  Enabled: {'✓' if cache_info['enabled'] else '✗'}")
            if cache_info["enabled"]:
                console.print(f"  Directory: {cache_info['cache_dir']}")
                console.print(f"  Files: {cache_info['files']}")
                console.print(f"  Size: {cache_info['size_mb']:.2f} MB")
            else:
                console.print("  Cache is disabled")

        elif action == "clear":
            cache_info = processor.get_cache_info()
            if cache_info["enabled"] and cache_info["files"] > 0:
                cleared_count = processor.clear_cache()
                console.print(
                    f"[green]✓ Cleared {cleared_count} cache files "
                    f"({cache_info['size_mb']:.2f} MB)[/green]"
                )
            else:
                console.print("[yellow]Cache is empty or disabled[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command("config-example")
def create_config_example(
    output_file: Path = typer.Option(
        Path("exafs_config.yaml"), "--output", "-o", help="Output file path"
    ),
    preset: str = typer.Option(
        "publication", "--preset", "-p", help=f"Base preset: {list(PRESETS.keys())}"
    ),
) -> None:
    """Create an example configuration file by copying a preset.

    This copies the preset YAML file to your specified location,
    which you can then customize for your specific needs.
    """
    if preset not in PRESETS:
        console.print(f"[red]Error: Unknown preset '{preset}'[/red]")
        console.print(f"[dim]Available presets: {', '.join(PRESETS.keys())}[/dim]")
        raise typer.Exit(1)

    try:
        # Get the preset YAML file path
        preset_dir = Path(__file__).parent / "feff_configs"
        preset_file = preset_dir / f"{preset}.yaml"

        if not preset_file.exists():
            console.print(f"[red]Error: Preset file not found: {preset_file}[/red]")
            raise typer.Exit(1)

        # Copy the preset file to the output location
        import shutil

        shutil.copy2(preset_file, output_file)

        console.print(f"[green]✓ Configuration file created: {output_file}[/green]")
        console.print(f"  Based on '{preset}' preset")
        console.print(
            "  [dim]Edit this file to customize parameters for your analysis[/dim]"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
