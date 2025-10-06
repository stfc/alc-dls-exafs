"""Comprehensive tests for the streamlined CLI interface.

Tests cover all CLI commands, helper functions, and error handling
for the new Typer-based EXAFS processing interface.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from ase import Atoms
from typer.testing import CliRunner

from larch_cli_wrapper.cli import (
    PlotComponent,
    app,
    create_progress,
    load_config,
    parse_absorber_specification,
    parse_plot_components,
    update_config_from_cli_options,
)
from larch_cli_wrapper.feff_utils import FeffConfig, WindowType

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def cli_runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_atoms():
    """Create a sample ASE Atoms object for testing."""
    atoms = Atoms(
        symbols=["Fe", "O", "O", "Fe", "O"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        cell=[5.0, 5.0, 5.0],  # Add unit cell
        pbc=[True, True, True],  # Add periodic boundary conditions
    )
    return atoms


@pytest.fixture
def temp_structure_file(sample_atoms):
    """Create a temporary structure file."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        sample_atoms.write(f.name)
        yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file."""
    config_content = """spectrum_type: EXAFS
edge: K
radius: 6.0
kmin: 2.0
kmax: 12.0
kweight: 2
dk: 1.0
window: hanning
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_content)
        f.flush()  # Ensure content is written
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink(missing_ok=True)


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test CLI helper functions."""

    def test_create_progress(self):
        """Test progress bar creation."""
        progress = create_progress()

        assert progress is not None
        # Should have the expected columns
        assert len(progress.columns) == 5

    def test_load_config_default(self):
        """Test loading default configuration."""
        with patch("larch_cli_wrapper.cli.console") as mock_console:
            config = load_config()

            assert isinstance(config, FeffConfig)
            mock_console.print.assert_called_with(
                "[dim]Using default configuration[/dim]"
            )

    def test_load_config_from_file(self, temp_config_file):
        """Test loading configuration from file."""
        with patch("larch_cli_wrapper.cli.console") as mock_console:
            config = load_config(config_file=temp_config_file)

            assert isinstance(config, FeffConfig)
            assert config.edge == "K"
            assert config.radius == 6.0
            mock_console.print.assert_called_with(
                f"[dim]Loaded configuration from {temp_config_file}[/dim]"
            )

    def test_load_config_from_preset(self):
        """Test loading configuration from preset."""
        with patch("larch_cli_wrapper.cli.console") as mock_console:
            config = load_config(preset="publication")

            assert isinstance(config, FeffConfig)
            mock_console.print.assert_called_with(
                "[dim]Using 'publication' preset[/dim]"
            )

    def test_update_config_from_cli_options_no_updates(self):
        """Test config update with no CLI options."""
        config = FeffConfig()
        original_kmin = config.kmin

        with patch("larch_cli_wrapper.cli.console") as mock_console:
            updated_config = update_config_from_cli_options(config)

            assert updated_config.kmin == original_kmin
            mock_console.print.assert_not_called()

    def test_update_config_from_cli_options_with_updates(self):
        """Test config update with CLI options."""
        config = FeffConfig()

        with patch("larch_cli_wrapper.cli.console") as mock_console:
            updated_config = update_config_from_cli_options(
                config,
                kmin=3.0,
                kmax=15.0,
                window="gaussian",
                parallel=False,
                workers=4,
            )

            assert updated_config.kmin == 3.0
            assert updated_config.kmax == 15.0
            assert updated_config.window == WindowType.GAUSSIAN
            assert updated_config.parallel is False
            assert updated_config.n_workers == 4
            mock_console.print.assert_called_with(
                "[dim]Updated config with 5 CLI parameters[/dim]"
            )

    def test_parse_absorber_specification_element_single_site(self, sample_atoms):
        """Test parsing element symbol for single site."""
        with patch("larch_cli_wrapper.cli.console") as mock_console:
            result = parse_absorber_specification("Fe", sample_atoms)

            assert result["absorber"] == [0]  # First Fe atom (as list)
            assert "Fe" in result["description"]
            assert "site 0" in result["description"]
            # Should warn about multiple sites
            mock_console.print.assert_called_once()

    def test_parse_absorber_specification_element_all_sites(self, sample_atoms):
        """Test parsing element symbol for all sites."""
        result = parse_absorber_specification("Fe", sample_atoms, all_sites=True)

        assert result["absorber"] == [0, 3]  # Both Fe atoms
        assert "Fe" in result["description"]
        assert "all 2 sites" in result["description"]

    def test_parse_absorber_specification_indices(self, sample_atoms):
        """Test parsing comma-separated indices."""
        result = parse_absorber_specification("0,3", sample_atoms)

        assert result["absorber"] == [0, 3]
        assert "indices [0, 3]" in result["description"]
        assert "Fe" in result["description"]

    def test_parse_absorber_specification_mixed_indices(self, sample_atoms):
        """Test parsing indices with mixed elements should raise error."""
        with pytest.raises(ValueError, match="Multiple species selected"):
            parse_absorber_specification("0,1", sample_atoms)

    def test_parse_absorber_specification_invalid_element(self, sample_atoms):
        """Test parsing with non-existent element."""
        with pytest.raises(ValueError, match="No atoms with symbol 'Pt' found"):
            parse_absorber_specification("Pt", sample_atoms)

    def test_parse_absorber_specification_invalid_indices(self, sample_atoms):
        """Test parsing with invalid indices."""
        with pytest.raises(ValueError, match="Invalid atom indices"):
            parse_absorber_specification("0,10", sample_atoms)  # Index 10 doesn't exist


# ============================================================================
# Test CLI Commands
# ============================================================================


class TestCLICommands:
    """Test individual CLI commands."""

    def test_info_command(self, cli_runner):
        """Test the info command."""
        with patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.get_diagnostics.return_value = {
                "python_version": "3.12.0",
                "platform": "Linux",
                "cache_enabled": True,
                "cache_dir": "/home/user/.exafs_cache",
                "cache_files": 10,
                "cache_size_mb": 15.5,
            }
            mock_processor_class.return_value = mock_processor

            result = cli_runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert "EXAFS Processing System Info" in result.stdout
            assert "Python version: 3.12.0" in result.stdout
            assert "Platform: Linux" in result.stdout
            assert "Cache enabled: True" in result.stdout

    def test_generate_command_basic(self, cli_runner, temp_structure_file):
        """Test basic generate command."""
        with (
            patch("larch_cli_wrapper.cli.ase_read") as mock_read,
            patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class,
        ):
            # Mock ASE read
            mock_atoms = Mock()
            mock_atoms.__len__ = Mock(return_value=5)
            mock_read.return_value = [mock_atoms]

            # Mock processor and batch
            mock_processor = Mock()
            mock_batch = Mock()

            # Create proper mock tasks with Path-like feff_dir under output directory
            mock_task1 = Mock()
            mock_task1.feff_dir = Path("feff_inputs/site_0000")
            mock_task2 = Mock()
            mock_task2.feff_dir = Path("feff_inputs/site_0001")
            mock_batch.tasks = [mock_task1, mock_task2]

            mock_processor.input_generator.generate_single_site_inputs.return_value = (
                mock_batch
            )
            mock_processor_class.return_value = mock_processor

            # Mock absorber parsing
            with patch(
                "larch_cli_wrapper.cli.parse_absorber_specification"
            ) as mock_parse:
                mock_parse.return_value = {
                    "absorber": 0,
                    "description": "element Fe (site 0)",
                }

                result = cli_runner.invoke(
                    app, ["generate", str(temp_structure_file), "Fe"]
                )

                assert result.exit_code == 0
                assert "Generated 2 FEFF input files" in result.stdout

    def test_generate_command_with_options(
        self, cli_runner, temp_structure_file, temp_config_file
    ):
        """Test generate command with various options."""
        with (
            patch("larch_cli_wrapper.cli.ase_read") as mock_read,
            patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class,
        ):
            mock_atoms = Mock()
            mock_atoms.__len__ = Mock(return_value=5)
            mock_read.return_value = [mock_atoms]

            mock_processor = Mock()
            mock_batch = Mock()
            mock_task = Mock()
            mock_task.feff_dir = Path("feff_inputs/site_0000")
            mock_batch.tasks = [mock_task]
            mock_processor.input_generator.generate_single_site_inputs.return_value = (
                mock_batch
            )
            mock_processor_class.return_value = mock_processor

            with patch(
                "larch_cli_wrapper.cli.parse_absorber_specification"
            ) as mock_parse:
                mock_parse.return_value = {
                    "absorber": [0, 1],
                    "description": "indices [0, 1]",
                }

                result = cli_runner.invoke(
                    app,
                    [
                        "generate",
                        str(temp_structure_file),
                        "0,1",
                        "--config",
                        str(temp_config_file),
                        "--radius",
                        "8.0",
                        "--edge",
                        "L3",
                    ],
                )

                assert result.exit_code == 0

    def test_generate_command_trajectory(self, cli_runner, temp_structure_file):
        """Test generate command with trajectory processing."""
        with (
            patch("larch_cli_wrapper.cli.ase_read") as mock_read,
            patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class,
        ):
            # Mock multiple frames
            mock_atoms1 = Mock()
            mock_atoms1.__len__ = Mock(return_value=5)
            mock_atoms2 = Mock()
            mock_atoms2.__len__ = Mock(return_value=5)
            mock_read.return_value = [mock_atoms1, mock_atoms2]

            mock_processor = Mock()
            mock_batch = Mock()

            # Create mock tasks with proper Path objects
            mock_tasks = []
            for frame in range(2):
                for site in range(2):
                    task = Mock()
                    task.feff_dir = Path(
                        f"feff_inputs/frame_{frame:04d}/site_{site:04d}"
                    )
                    mock_tasks.append(task)

            mock_batch.tasks = mock_tasks  # 2 frames × 2 sites = 4 tasks
            mock_batch.get_tasks_by_frame.return_value = [
                mock_tasks[:2],
                mock_tasks[2:],
            ]
            mock_processor.input_generator.generate_trajectory_inputs.return_value = (
                mock_batch
            )
            mock_processor_class.return_value = mock_processor

            with patch(
                "larch_cli_wrapper.cli.parse_absorber_specification"
            ) as mock_parse:
                mock_parse.return_value = {
                    "absorber": [0, 1],
                    "description": "element Fe (all sites)",
                    "sites": [0, 1],
                }

                result = cli_runner.invoke(
                    app,
                    [
                        "generate",
                        str(temp_structure_file),
                        "Fe",
                        "--all-frames",
                        "--all-sites",
                    ],
                )

                assert result.exit_code == 0
                assert "Generated 4 FEFF input files" in result.stdout
                assert "Processing trajectory: 2 frames" in result.stdout

    def test_generate_command_file_not_found(self, cli_runner):
        """Test generate command with non-existent structure file."""
        result = cli_runner.invoke(app, ["generate", "nonexistent.xyz", "Fe"])

        assert result.exit_code == 1
        assert "Structure file nonexistent.xyz not found" in result.stdout

    def test_run_feff_command(self, cli_runner):
        """Test run-feff command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir) / "site_0000"
            feff_dir.mkdir()
            feff_inp = feff_dir / "feff.inp"
            feff_inp.write_text("EDGE K\nRADIUS 6.0")

            with patch("larch_cli_wrapper.cli.FeffExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor.execute_batch.return_value = {"task_0000": True}
                mock_executor_class.return_value = mock_executor

                result = cli_runner.invoke(app, ["run-feff", str(feff_dir)])

                assert result.exit_code == 0
                assert "FEFF calculations completed: 1/1 successful" in result.stdout

    def test_run_feff_command_no_valid_dirs(self, cli_runner):
        """Test run-feff command with no valid directories."""
        result = cli_runner.invoke(app, ["run-feff", "nonexistent_dir"])

        assert result.exit_code == 1
        assert "No valid FEFF directories found" in result.stdout

    def test_analyze_command(self, cli_runner):
        """Test analyze command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feff_dir = Path(tmpdir) / "site_0000"
            feff_dir.mkdir()
            chi_dat = feff_dir / "chi.dat"
            chi_dat.write_text("# k chi\n1.0 0.1\n2.0 0.2")

            with (
                patch("larch_cli_wrapper.cli.ResultProcessor") as mock_processor_class,
                patch("larch_cli_wrapper.cli.plot_exafs_matplotlib"),
            ):
                mock_processor = Mock()
                # load_successful_results should return a dict, not a list
                mock_processor.load_successful_results.return_value = {
                    "task_0000": Mock()
                }
                mock_processor.create_frame_averages.return_value = {}
                mock_processor.create_site_averages.return_value = {}
                mock_processor.create_overall_average.return_value = Mock()
                mock_processor_class.return_value = mock_processor

                result = cli_runner.invoke(
                    app, ["analyze", str(feff_dir), "--plot-include", "average"]
                )

                assert result.exit_code == 0

    def test_pipeline_command_basic(self, cli_runner, temp_structure_file):
        """Test basic pipeline command."""
        with (
            patch("larch_cli_wrapper.cli.ase_read") as mock_read,
            patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class,
            patch("larch_cli_wrapper.cli.plot_exafs_matplotlib"),
        ):
            # Mock ASE read
            mock_atoms = Mock()
            mock_atoms.__len__ = Mock(return_value=5)
            mock_read.return_value = mock_atoms  # Single structure

            # Mock processor
            mock_processor = Mock()
            # process_trajectory returns
            # (final_group, frame_averages, site_averages, individual_groups)
            mock_processor.process_trajectory.return_value = (Mock(), {}, {}, [Mock()])
            mock_processor_class.return_value = mock_processor

            with patch(
                "larch_cli_wrapper.cli.parse_absorber_specification"
            ) as mock_parse:
                mock_parse.return_value = {
                    "absorber": 0,
                    "description": "element Fe (site 0)",
                }

                result = cli_runner.invoke(
                    app, ["pipeline", str(temp_structure_file), "Fe"]
                )

                assert result.exit_code == 0
                assert "Pipeline completed successfully!" in result.stdout

    def test_cache_info_command(self, cli_runner):
        """Test cache info command."""
        with patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.get_cache_info.return_value = {
                "enabled": True,
                "cache_dir": "/home/user/.exafs_cache",
                "files": 5,
                "size_mb": 10.0,
            }
            mock_processor_class.return_value = mock_processor

            result = cli_runner.invoke(app, ["cache", "info"])

            assert result.exit_code == 0

    def test_cache_clear_command(self, cli_runner):
        """Test cache clear command."""
        with patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class:
            mock_processor = Mock()
            mock_processor.get_cache_info.return_value = {
                "enabled": True,
                "files": 5,
                "size_mb": 10.0,
            }
            mock_processor.clear_cache.return_value = 5
            mock_processor_class.return_value = mock_processor

            result = cli_runner.invoke(app, ["cache", "clear"])

            assert result.exit_code == 0

    def test_cache_invalid_action(self, cli_runner):
        """Test cache command with invalid action."""
        result = cli_runner.invoke(app, ["cache", "invalid"])

        assert result.exit_code == 1
        assert "Unknown action 'invalid'" in result.stdout

    def test_config_example_command(self, cli_runner):
        """Test config-example command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "test_config.yaml"

            result = cli_runner.invoke(
                app,
                [
                    "config-example",
                    "--output",
                    str(output_file),
                    "--preset",
                    "publication",
                ],
            )

            assert result.exit_code == 0
            assert output_file.exists()
            assert "Configuration example created" in result.stdout

            # Check file content
            content = output_file.read_text()
            assert "spectrum_type:" in content
            assert "kmin:" in content

    def test_config_example_invalid_preset(self, cli_runner):
        """Test config-example command with invalid preset."""
        result = cli_runner.invoke(
            app, ["config-example", "--preset", "invalid_preset"]
        )

        assert result.exit_code == 1
        assert "Unknown preset 'invalid_preset'" in result.stdout


# ============================================================================
# Test CLI Error Handling
# ============================================================================


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def test_generate_ase_kwargs_invalid_json(self, cli_runner, temp_structure_file):
        """Test generate command with invalid ASE kwargs JSON."""
        with patch("larch_cli_wrapper.cli.ase_read"):
            result = cli_runner.invoke(
                app,
                [
                    "generate",
                    str(temp_structure_file),
                    "Fe",
                    "--ase-kwargs",
                    "invalid_json",
                ],
            )

            assert result.exit_code == 1
            assert "Error:" in result.stdout

    def test_pipeline_multisite_trajectory_combination(
        self, cli_runner, temp_structure_file
    ):
        """Test pipeline with trajectory + multi-site (complex scenario)."""
        with (
            patch("larch_cli_wrapper.cli.ase_read") as mock_read,
            patch("larch_cli_wrapper.cli.PipelineProcessor") as mock_processor_class,
            patch("larch_cli_wrapper.cli.plot_exafs_matplotlib"),
        ):
            # Mock trajectory (multiple frames)
            mock_atoms1 = Mock()
            mock_atoms1.__len__ = Mock(return_value=5)
            mock_atoms2 = Mock()
            mock_atoms2.__len__ = Mock(return_value=5)
            mock_read.return_value = [mock_atoms1, mock_atoms2]

            mock_processor = Mock()
            mock_processor.process_trajectory.return_value = (
                Mock(),
                {0: Mock()},
                {0: Mock()},
                [Mock(), Mock()],
            )
            mock_processor_class.return_value = mock_processor

            with patch(
                "larch_cli_wrapper.cli.parse_absorber_specification"
            ) as mock_parse:
                mock_parse.return_value = {
                    "absorber": [0, 1],
                    "description": "element Fe (all sites)",
                }

                result = cli_runner.invoke(
                    app, ["pipeline", str(temp_structure_file), "Fe", "--all-sites"]
                )

                assert result.exit_code == 0

    def test_command_exception_handling(self, cli_runner, temp_structure_file):
        """Test that CLI commands handle exceptions gracefully."""
        with patch(
            "larch_cli_wrapper.cli.ase_read", side_effect=Exception("File read error")
        ):
            result = cli_runner.invoke(
                app, ["generate", str(temp_structure_file), "Fe"]
            )

            assert result.exit_code == 1
            assert "Error: File read error" in result.stdout


# ============================================================================
# Test CLI Integration
# ============================================================================


class TestCLIIntegration:
    """Test CLI command integration and workflows."""

    def test_full_workflow_simulation(self, cli_runner):
        """Test a simulated full workflow using CLI commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock structure file
            structure_file = Path(tmpdir) / "structure.xyz"
            structure_file.write_text("""5
Test structure
Fe 0.0 0.0 0.0
O  1.0 0.0 0.0
O  0.0 1.0 0.0
Fe 2.0 0.0 0.0
O  0.0 2.0 0.0
""")

            # Test info command
            with patch(
                "larch_cli_wrapper.cli.PipelineProcessor"
            ) as mock_processor_class:
                mock_processor = Mock()
                mock_processor.get_diagnostics.return_value = {
                    "python_version": "3.12.0",
                    "platform": "Linux",
                    "cache_enabled": True,
                    "cache_dir": tmpdir,
                    "cache_files": 0,
                    "cache_size_mb": 0.0,
                }
                mock_processor_class.return_value = mock_processor

                result = cli_runner.invoke(app, ["info"])
                assert result.exit_code == 0

            # Test config-example command
            config_file = Path(tmpdir) / "config.yaml"
            result = cli_runner.invoke(
                app, ["config-example", "--output", str(config_file)]
            )
            assert result.exit_code == 0
            assert config_file.exists()

    def test_plot_component_enum_values(self):
        """Test that PlotComponent enum has expected values."""
        assert PlotComponent.INDIVIDUAL == "individual"
        assert PlotComponent.AVERAGE == "average"
        assert PlotComponent.FRAMES == "frames"
        assert PlotComponent.SITES == "sites"
        assert PlotComponent.ALL == "all"

    def test_parse_plot_components(self):
        """Test plot components parsing function."""
        # Test single component
        result = parse_plot_components("average")
        assert result == [PlotComponent.AVERAGE]

        # Test multiple components
        result = parse_plot_components("average,frames")
        assert result == [PlotComponent.AVERAGE, PlotComponent.FRAMES]

        # Test all components
        result = parse_plot_components("individual,average,frames,sites")
        assert result == [
            PlotComponent.INDIVIDUAL,
            PlotComponent.AVERAGE,
            PlotComponent.FRAMES,
            PlotComponent.SITES,
        ]

        # Test empty string defaults to all
        result = parse_plot_components("")
        assert result == [PlotComponent.ALL]

        # Test whitespace handling
        result = parse_plot_components("  average  ,  frames  ")
        assert result == [PlotComponent.AVERAGE, PlotComponent.FRAMES]

        # Test special 'all' component expands to all other components
        result = parse_plot_components("all")
        assert result == [
            PlotComponent.INDIVIDUAL,
            PlotComponent.AVERAGE,
            PlotComponent.FRAMES,
            PlotComponent.SITES,
        ]

        # Test invalid component raises error
        with pytest.raises(typer.BadParameter):
            parse_plot_components("invalid")

    def test_cli_app_configuration(self):
        """Test that the Typer app is configured correctly."""
        assert app.info.name == "larch-cli"
        assert "EXAFS processing" in app.info.help
        assert app.info.invoke_without_command is True
        assert app.info.no_args_is_help is True


# ============================================================================
# Test CLI Parameter Validation
# ============================================================================


class TestCLIParameterValidation:
    """Test CLI parameter validation and type conversion."""

    def test_window_type_validation(self):
        """Test that window parameter is validated against WindowType enum."""
        config = FeffConfig()

        # Valid window type
        updated = update_config_from_cli_options(config, window="hanning")
        assert updated.window == WindowType.HANNING

        # Invalid window type should raise error
        with pytest.raises(ValueError):
            update_config_from_cli_options(config, window="invalid_window")

    def test_numeric_parameter_types(self):
        """Test that numeric parameters are properly typed."""
        config = FeffConfig()

        updated = update_config_from_cli_options(
            config, radius=8.5, kmin=2.0, kmax=15.0, kweight=3, workers=4
        )

        assert isinstance(updated.radius, float)
        assert isinstance(updated.kmin, float)
        assert isinstance(updated.kmax, float)
        assert isinstance(updated.kweight, int)
        assert isinstance(updated.n_workers, int)

    def test_boolean_parameter_handling(self):
        """Test boolean parameter handling."""
        config = FeffConfig()

        updated = update_config_from_cli_options(
            config,
            parallel=True,
            force_recalculate=False,
            cleanup=True,
            with_phase=False,
        )

        assert updated.parallel is True
        assert updated.force_recalculate is False
        assert updated.cleanup_feff_files is True
        assert updated.with_phase is False


class TestSaveGroupsCLI:
    """Test the --save-groups CLI functionality."""

    @patch("larch_cli_wrapper.cli.EXAFSDataCollection")
    @patch("larch_cli_wrapper.cli.PipelineProcessor")
    @patch("larch_cli_wrapper.cli.parse_absorber_specification")
    @patch("larch_cli_wrapper.cli.load_config")
    @patch("larch_cli_wrapper.cli.update_config_from_cli_options")
    @patch("larch_cli_wrapper.cli.ase_read")
    def test_pipeline_with_save_groups_flag(
        self,
        mock_ase_read,
        mock_update_config,
        mock_load_config,
        mock_parse_absorber,
        mock_pipeline,
        mock_collection_class,
        cli_runner,
        sample_atoms,
    ):
        """Test that --save-groups flag is properly handled in pipeline command."""
        # Mock configuration - create a proper FeffConfig instance
        from larch_cli_wrapper.feff_utils import FeffConfig

        mock_config = FeffConfig()
        mock_load_config.return_value = mock_config
        mock_update_config.return_value = mock_config

        # Mock ASE read
        mock_ase_read.return_value = sample_atoms

        # Mock absorber parsing - return correct dictionary format
        mock_parse_absorber.return_value = {
            "absorber": [0],
            "description": "element Fe (site 0, 0 other sites available)",
        }

        # Mock pipeline processor
        mock_processor = Mock()
        mock_pipeline.return_value = mock_processor

        # Mock process_trajectory to return the expected 4-tuple
        mock_overall = Mock()
        mock_frame_averages = {0: Mock()}
        mock_site_averages = {0: Mock()}
        mock_groups = {0: Mock()}
        mock_processor.process_trajectory.return_value = (
            mock_overall,
            mock_frame_averages,
            mock_site_averages,
            mock_groups,
        )

        # Mock EXAFSDataCollection
        mock_collection = Mock()
        mock_collection.export_larch_groups = Mock(return_value=Path("groups_dir"))
        mock_collection_class.return_value = mock_collection

        with tempfile.TemporaryDirectory() as temp_dir:
            structure_file = Path(temp_dir) / "structure.xyz"
            sample_atoms.write(structure_file)

            result = cli_runner.invoke(
                app,
                [
                    "pipeline",
                    str(structure_file),
                    "Fe",  # absorber as positional argument
                    "--output",
                    temp_dir,
                    "--save-groups",
                ],
            )

            # Should not exit with error
            assert result.exit_code == 0

            # Should call export_larch_groups
            mock_collection.export_larch_groups.assert_called_once()
            call_args = mock_collection.export_larch_groups.call_args
            assert "output_dir" in call_args.kwargs
            assert call_args.kwargs["output_dir"].name == "larch_groups"

    @patch("larch_cli_wrapper.cli.plot_exafs_matplotlib")
    @patch("larch_cli_wrapper.feff_utils.read_feff_output")
    @patch("larch_cli_wrapper.cli.FeffConfig")
    @patch("larch_cli_wrapper.cli.EXAFSDataCollection")
    def test_analyze_with_save_groups_flag(
        self, mock_collection_class, mock_config, mock_read_feff, mock_plot, cli_runner
    ):
        """Test that --save-groups flag is properly handled in analyze command."""
        # Mock the FeffConfig
        mock_config.from_preset.return_value = Mock()

        # Mock read_feff_output to return fake chi and k data
        import numpy as np

        fake_k = np.linspace(0, 15, 100)
        fake_chi = np.sin(fake_k) + 1j * np.cos(fake_k)  # Complex chi data
        mock_read_feff.return_value = (fake_chi, fake_k)

        # Mock plot_exafs_matplotlib to return a successful result
        from larch_cli_wrapper.exafs_data import PlotResult

        mock_plot.return_value = PlotResult(plot_paths={})

        # Mock EXAFSDataCollection and its methods
        mock_collection = Mock()
        mock_collection.export_larch_groups = Mock(return_value=Path("groups_dir"))
        mock_collection_class.return_value = mock_collection

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake FEFF output directory
            feff_dir = Path(temp_dir) / "feff_output"
            feff_dir.mkdir()
            (feff_dir / "chi.dat").write_text("# Fake chi.dat\n")

            result = cli_runner.invoke(
                app, ["analyze", str(feff_dir), "--output", temp_dir, "--save-groups"]
            )

            # Should not exit with error
            if result.exit_code != 0:
                print(f"Command output: {result.stdout}")
                print(f"Command stderr: {result.stderr}")
            assert result.exit_code == 0

            # Should call export_larch_groups
            mock_collection.export_larch_groups.assert_called_once()
            call_args = mock_collection.export_larch_groups.call_args
            assert "output_dir" in call_args.kwargs
            assert call_args.kwargs["output_dir"].name == "larch_groups"

    @patch("larch_cli_wrapper.cli.EXAFSDataCollection")
    @patch("larch_cli_wrapper.cli.PipelineProcessor")
    @patch("larch_cli_wrapper.cli.parse_absorber_specification")
    @patch("larch_cli_wrapper.cli.load_config")
    @patch("larch_cli_wrapper.cli.update_config_from_cli_options")
    @patch("larch_cli_wrapper.cli.ase_read")
    def test_without_save_groups_flag(
        self,
        mock_ase_read,
        mock_update_config,
        mock_load_config,
        mock_parse_absorber,
        mock_pipeline,
        mock_collection_class,
        cli_runner,
        sample_atoms,
    ):
        """Test that groups are not saved when --save-groups flag is not used."""
        # Mock configuration
        from larch_cli_wrapper.feff_utils import FeffConfig

        mock_config = FeffConfig()
        mock_load_config.return_value = mock_config
        mock_update_config.return_value = mock_config

        # Mock ASE read
        mock_ase_read.return_value = sample_atoms

        # Mock absorber parsing
        mock_parse_absorber.return_value = {
            "absorber": [0],
            "description": "element Fe (site 0, 0 other sites available)",
        }

        # Mock pipeline processor
        mock_processor = Mock()
        mock_pipeline.return_value = mock_processor

        # Mock process_trajectory to return the expected 4-tuple
        mock_overall = Mock()
        mock_frame_averages = {0: Mock()}
        mock_site_averages = {0: Mock()}
        mock_groups = {0: Mock()}
        mock_processor.process_trajectory.return_value = (
            mock_overall,
            mock_frame_averages,
            mock_site_averages,
            mock_groups,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            structure_file = Path(temp_dir) / "structure.xyz"
            sample_atoms.write(structure_file)

            result = cli_runner.invoke(
                app,
                [
                    "pipeline",
                    str(structure_file),
                    "Fe",
                    "--output",
                    temp_dir,
                    # Note: no --save-groups flag
                ],
            )

            assert result.exit_code == 0

            # Should NOT create EXAFSDataCollection when --save-groups is not provided
            mock_collection_class.assert_not_called()
