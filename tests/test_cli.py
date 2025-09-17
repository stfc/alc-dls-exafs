"""Comprehensive tests for the CLI interface."""

from unittest.mock import Mock, patch

from typer.testing import CliRunner

from larch_cli_wrapper.cli import app
from larch_cli_wrapper.wrapper import EXAFSProcessingError, ProcessingResult


class TestCLI:
    """Test cases for CLI interface."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    # ================== BASIC COMMAND TESTS ==================

    def test_help_command(self):
        """Test main help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Streamlined CLI for EXAFS processing" in result.stdout

    def test_no_args_shows_help(self):
        """Test that running without arguments shows help."""
        result = self.runner.invoke(app, [])
        # Typer returns exit code 2 for missing arguments when no_args_is_help=True
        assert result.exit_code == 2
        assert "Usage:" in result.stdout

    # ================== INFO COMMAND TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_info_command(self, mock_wrapper_class):
        """Test info command."""
        mock_wrapper = Mock()
        mock_wrapper.print_diagnostics.return_value = None
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["info"])
        assert result.exit_code == 0
        mock_wrapper.print_diagnostics.assert_called_once()

    # ================== GENERATE COMMAND TESTS ==================

    def test_generate_missing_structure(self):
        """Test generate with missing structure file."""
        result = self.runner.invoke(app, ["generate", "nonexistent.cif", "Fe"])
        assert result.exit_code == 1
        assert "Error: Structure file" in result.stdout

    def test_generate_command_cleanup_option(self):
        """Test generate command accepts cleanup options."""
        # Test --cleanup option (default)
        result = self.runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--cleanup" in result.stdout
        assert "--no-cleanup" in result.stdout
        assert "Clean up unnecessary FEFF output files" in result.stdout

    def test_process_command_cleanup_option(self):
        """Test process command accepts cleanup options."""
        # Test cleanup option exists in help
        result = self.runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0
        assert "--cleanup" in result.output
        assert "--no-cleanup" in result.output

        # Test that the cleanup option can be used (but don't actually run the command)
        # This will fail if the option isn't recognized
        result = self.runner.invoke(app, ["process", "--cleanup", "--help"])
        assert result.exit_code == 0

        result = self.runner.invoke(app, ["process", "--no-cleanup", "--help"])
        assert result.exit_code == 0

    def test_generate_success(self, mock_generate_workflow, tmp_structure_file):
        """Test successful generate command."""
        output_dir = tmp_structure_file.parent / "outputs"

        result = self.runner.invoke(
            app,
            [
                "generate",
                str(tmp_structure_file),
                "Fe",
                "--output",
                str(output_dir),
                "--edge",
                "K",
                "--method",
                "larixite",
            ],
        )

        assert result.exit_code == 0
        assert "FEFF input generated" in result.stdout
        mock_generate_workflow["generate_feff_input"].assert_called_once()

    def test_generate_with_preset(self, mock_generate_workflow, tmp_structure_file):
        """Test generate with configuration preset."""
        result = self.runner.invoke(
            app, ["generate", str(tmp_structure_file), "Fe", "--preset", "publication"]
        )

        assert result.exit_code == 0

    def test_generate_with_config_file(
        self, mock_generate_workflow, tmp_structure_file, tmp_path
    ):
        """Test generate with configuration file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
spectrum_type: EXAFS
edge: K
radius: 8.0
kmin: 2.0
kmax: 14.0
""")

        result = self.runner.invoke(
            app,
            ["generate", str(tmp_structure_file), "Fe", "--config", str(config_file)],
        )

        assert result.exit_code == 0

    # ================== RUN-FEFF COMMAND TESTS ==================

    def test_run_feff_missing_directory(self):
        """Test run-feff with missing directory."""
        result = self.runner.invoke(app, ["run-feff", "nonexistent_dir"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_run_feff_missing_input_file(self, tmp_path):
        """Test run-feff with missing input file."""
        # Create directory but no input file
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()

        result = self.runner.invoke(app, ["run-feff", str(feff_dir)])

        assert result.exit_code == 1
        assert "feff.inp" in result.stdout.lower()

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_run_feff_success(self, mock_wrapper_class, tmp_path):
        """Test successful run-feff command."""
        # Create directory and input file
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        input_file = feff_dir / "feff.inp"
        input_file.write_text("fake feff input")

        # Create mock output files
        chi_file = feff_dir / "chi.dat"
        chi_file.write_text("fake chi data")
        log_file = feff_dir / "feff.log"
        log_file.write_text("fake log")

        # Mock successful FEFF run
        mock_wrapper = Mock()
        mock_wrapper.run_feff.return_value = True
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["run-feff", str(feff_dir)])

        assert result.exit_code == 0
        assert "completed successfully" in result.stdout
        mock_wrapper.run_feff.assert_called_once()

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_run_feff_failure(self, mock_wrapper_class, tmp_path):
        """Test run-feff command failure."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        input_file = feff_dir / "feff.inp"
        input_file.write_text("fake feff input")

        # Mock failed FEFF run
        mock_wrapper = Mock()
        mock_wrapper.run_feff.return_value = False
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["run-feff", str(feff_dir)])

        assert result.exit_code == 1
        assert "failed" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_run_feff_verbose(self, mock_wrapper_class, tmp_path):
        """Test run-feff with verbose output."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        input_file = feff_dir / "feff.inp"
        input_file.write_text("fake feff input")

        mock_wrapper = Mock()
        mock_wrapper.run_feff.return_value = True
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["run-feff", str(feff_dir), "--verbose"])

        assert result.exit_code == 0
        # Check that run_feff was called (verbose is handled by wrapper internally)
        mock_wrapper.run_feff.assert_called_once_with(feff_dir)

    # ================== PROCESS COMMAND TESTS ==================

    def test_process_missing_structure(self):
        """Test process with missing structure file."""
        result = self.runner.invoke(app, ["process", "nonexistent.cif", "Fe"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_success(self, mock_wrapper_class, tmp_path):
        """Test successful process command."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake cif content")
        output_dir = tmp_path / "outputs"

        # Create mock result
        mock_result = ProcessingResult(
            exafs_group=Mock(),
            plot_paths={"pdf": output_dir / "plot.pdf", "png": output_dir / "plot.png"},
            processing_mode="single_frame",
            nframes=1,
        )
        mock_result.cache_hits = 2
        mock_result.cache_misses = 1

        mock_wrapper = Mock()
        mock_wrapper.process.return_value = mock_result
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(
            app,
            [
                "process",
                str(structure_file),
                "Fe",
                "--output",
                str(output_dir),
                "--edge",
                "K",
            ],
        )

        assert result.exit_code == 0
        assert "Processing completed" in result.stdout
        assert "Cache:" in result.stdout
        mock_wrapper.process.assert_called_once()

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_trajectory(self, mock_wrapper_class, tmp_path):
        """Test process command with trajectory."""
        structure_file = tmp_path / "trajectory.xyz"
        structure_file.write_text("fake trajectory content")

        mock_result = ProcessingResult(
            exafs_group=Mock(),
            plot_paths={"pdf": tmp_path / "plot.pdf"},
            processing_mode="trajectory",
            nframes=10,
        )

        mock_wrapper = Mock()
        mock_wrapper.process.return_value = mock_result
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(
            app,
            [
                "process",
                str(structure_file),
                "Fe",
                "--trajectory",
                "--interval",
                "2",
                "--parallel",
                "--workers",
                "4",
            ],
        )

        assert result.exit_code == 0
        assert "Frames processed: 10" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_with_all_options(self, mock_wrapper_class, tmp_path):
        """Test process command with all available options."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake cif content")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("edge: L3\nkmin: 3.0")

        mock_result = ProcessingResult(
            exafs_group=Mock(),
            plot_paths={"pdf": tmp_path / "plot.pdf"},
            processing_mode="single_frame",
        )

        mock_wrapper = Mock()
        mock_wrapper.process.return_value = mock_result
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(
            app,
            [
                "process",
                str(structure_file),
                "Fe",
                "--config",
                str(config_file),
                "--edge",
                "K",
                "--method",
                "pymatgen",
                "--show",
                "--plot-style",
                "presentation",
                "--plot-frames",
                "--force",
            ],
        )

        assert result.exit_code == 0

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_error_handling(self, mock_wrapper_class, tmp_path):
        """Test process command error handling."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake cif content")

        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = EXAFSProcessingError("Processing failed")
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["process", str(structure_file), "Fe"])

        assert result.exit_code == 1
        assert "Processing error" in result.stdout

    # ================== PROCESS-OUTPUT COMMAND TESTS ==================

    def test_process_output_missing_directory(self):
        """Test process-output with missing directory."""
        result = self.runner.invoke(app, ["process-output", "nonexistent_dir"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_output_single_file(self, mock_wrapper_class, tmp_path):
        """Test process-output with single chi.dat file."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        chi_file = feff_dir / "chi.dat"
        chi_file.write_text("fake chi data")

        mock_exafs_group = Mock()
        plot_paths = {"pdf": feff_dir / "EXAFS_FT.pdf"}

        mock_wrapper = Mock()
        mock_wrapper.process_feff_output.return_value = mock_exafs_group
        mock_wrapper.plot_results.return_value = plot_paths
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["process-output", str(feff_dir)])

        assert result.exit_code == 0
        assert "Single output processed" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_output_trajectory(self, mock_wrapper_class, tmp_path):
        """Test process-output with trajectory frames."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()

        # Create frame directories
        for i in range(3):
            frame_dir = feff_dir / f"frame_{i:04d}"
            frame_dir.mkdir()
            chi_file = frame_dir / "chi.dat"
            chi_file.write_text(f"fake chi data {i}")

        mock_result = ProcessingResult(
            exafs_group=Mock(),
            plot_paths={"pdf": str(feff_dir / "trajectory_avg.pdf")},
            processing_mode="trajectory",
            nframes=3,
        )

        mock_wrapper = Mock()
        mock_wrapper.process_trajectory_feff_outputs.return_value = mock_result
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["process-output", str(feff_dir)])

        assert result.exit_code == 0
        assert "Trajectory processed" in result.stdout
        assert "Frames processed: 3" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_output_no_data(self, mock_wrapper_class, tmp_path):
        """Test process-output with no valid data."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        # No chi.dat or frame directories

        result = self.runner.invoke(app, ["process-output", str(feff_dir)])

        assert result.exit_code == 1
        assert "No FEFF output" in result.stdout

    # ================== CONFIG-EXAMPLE COMMAND TESTS ==================

    def test_config_example_default(self, tmp_path):
        """Test config-example command with default settings."""
        output_file = tmp_path / "config.yaml"

        result = self.runner.invoke(
            app, ["config-example", "--output", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Configuration example created" in result.stdout

        # Check content
        content = output_file.read_text()
        assert "spectrum_type:" in content
        assert "edge:" in content
        assert "radius:" in content

    def test_config_example_with_preset(self, tmp_path):
        """Test config-example with specific preset."""
        output_file = tmp_path / "config.yaml"

        result = self.runner.invoke(
            app, ["config-example", "--output", str(output_file), "--preset", "quick"]
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Based on 'quick' preset" in result.stdout

    def test_config_example_invalid_preset(self):
        """Test config-example with invalid preset."""
        result = self.runner.invoke(
            app, ["config-example", "--preset", "invalid_preset"]
        )

        assert result.exit_code == 1
        assert "Unknown preset" in result.stdout

    # ================== CACHE COMMAND TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_cache_info(self, mock_wrapper_class):
        """Test cache info command."""
        mock_wrapper = Mock()
        mock_wrapper.get_cache_info.return_value = {
            "enabled": True,
            "cache_dir": "/home/user/.larch_cache",
            "files": 5,
            "size_mb": 12.5,
        }
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["cache", "info"])

        assert result.exit_code == 0
        assert "Cache Status" in result.stdout
        assert "Enabled: ✓" in result.stdout
        assert "5" in result.stdout
        assert "12.5 MB" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_cache_info_disabled(self, mock_wrapper_class):
        """Test cache info when cache is disabled."""
        mock_wrapper = Mock()
        mock_wrapper.get_cache_info.return_value = {
            "enabled": False,
            "cache_dir": None,
            "files": 0,
            "size_mb": 0,
        }
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["cache", "info"])

        assert result.exit_code == 0
        assert "Cache is disabled" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_cache_clear(self, mock_wrapper_class):
        """Test cache clear command."""
        mock_wrapper = Mock()
        mock_wrapper.get_cache_info.return_value = {
            "enabled": True,
            "files": 3,
            "size_mb": 5.2,
        }
        mock_wrapper.clear_cache.return_value = None
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["cache", "clear"])

        assert result.exit_code == 0
        assert "Cleared 3 cache files" in result.stdout
        mock_wrapper.clear_cache.assert_called_once()

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_cache_clear_empty(self, mock_wrapper_class):
        """Test cache clear when cache is empty."""
        mock_wrapper = Mock()
        mock_wrapper.get_cache_info.return_value = {
            "enabled": True,
            "files": 0,
            "size_mb": 0,
        }
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["cache", "clear"])

        assert result.exit_code == 0
        assert "already empty" in result.stdout

    def test_cache_invalid_action(self):
        """Test cache command with invalid action."""
        result = self.runner.invoke(app, ["cache", "invalid"])

        assert result.exit_code == 1
        assert "Unknown action" in result.stdout

    # ================== EDGE CASES AND ERROR HANDLING ==================

    def test_invalid_edge_type(self, tmp_path):
        """Test commands with invalid edge type."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        result = self.runner.invoke(
            app, ["generate", str(structure_file), "Fe", "--edge", "invalid_edge"]
        )

        # Should not fail at CLI level (validation happens in wrapper)
        assert result.exit_code == 1

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_wrapper_initialization_error(self, mock_wrapper_class, tmp_path):
        """Test handling of wrapper initialization errors."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        mock_wrapper_class.side_effect = Exception("Wrapper init failed")

        result = self.runner.invoke(app, ["generate", str(structure_file), "Fe"])

        assert result.exit_code == 1
        assert "Error:" in result.stdout

    # ================== PROGRESS CALLBACK TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_progress_callback_functionality(self, mock_wrapper_class, tmp_path):
        """Test that progress callbacks work correctly."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        def mock_process(*args, **kwargs):
            # Simulate calling the progress callback
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                progress_callback(0, 3, "Starting...")
                progress_callback(1, 3, "Processing...")
                progress_callback(3, 3, "Complete!")

            return ProcessingResult(
                exafs_group=Mock(),
                plot_paths={"pdf": tmp_path / "plot.pdf"},
                processing_mode="trajectory",
                nframes=3,
            )

        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = mock_process
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(
            app, ["process", str(structure_file), "Fe", "--trajectory"]
        )

        assert result.exit_code == 0
        # Verify progress callback was called
        args, kwargs = mock_wrapper.process.call_args
        assert "progress_callback" in kwargs
        assert callable(kwargs["progress_callback"])
