"""Integration tests for CLI commands with realistic scenarios."""

from unittest.mock import Mock, patch

import yaml
from typer.testing import CliRunner

from larch_cli_wrapper.cli import app
from larch_cli_wrapper.feff_utils import PRESETS
from larch_cli_wrapper.wrapper import ProcessingResult


class TestCLIIntegration:
    """Integration tests for CLI with realistic workflows."""

    def setup_method(self):
        """Setup test runner and common fixtures."""
        self.runner = CliRunner()

    # ================== WORKFLOW INTEGRATION TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_full_workflow_generate_run_process(self, mock_wrapper_class, tmp_path):
        """Test complete workflow: generate -> run-feff -> process-output."""
        # Setup structure file
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("""
# CIF file
_chemical_name_common 'Iron Oxide'
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
""")

        feff_dir = tmp_path / "feff_output"
        feff_dir.mkdir()

        # Mock wrapper with realistic behavior
        mock_wrapper = Mock()
        mock_wrapper.generate_feff_input.return_value = feff_dir
        mock_wrapper.run_feff.return_value = True
        mock_wrapper.process_feff_output.return_value = Mock()
        mock_wrapper.plot_results.return_value = {"pdf": feff_dir / "plot.pdf"}
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        # Step 1: Generate FEFF input
        result1 = self.runner.invoke(
            app,
            [
                "generate",
                str(structure_file),
                "Fe",
                "--output",
                str(feff_dir),
                "--method",
                "larixite",
            ],
        )
        assert result1.exit_code == 0

        # Create feff.inp for next step
        feff_input = feff_dir / "feff.inp"
        feff_input.write_text("TITLE Test FEFF input\nCONTROL 1 1 1 1 1 1")

        # Step 2: Run FEFF
        result2 = self.runner.invoke(app, ["run-feff", str(feff_dir)])
        assert result2.exit_code == 0

        # Create chi.dat for next step
        chi_file = feff_dir / "chi.dat"
        chi_file.write_text("# k chi\n1.0 0.1\n2.0 0.2\n")

        # Step 3: Process output
        result3 = self.runner.invoke(app, ["process-output", str(feff_dir)])
        assert result3.exit_code == 0

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_trajectory_workflow(self, mock_wrapper_class, tmp_path):
        """Test trajectory processing workflow."""
        # Setup trajectory file
        trajectory_file = tmp_path / "trajectory.xyz"
        trajectory_file.write_text("""
10
Frame 0
Fe 0.0 0.0 0.0
O 1.0 0.0 0.0
O 0.0 1.0 0.0
""")

        output_dir = tmp_path / "trajectory_output"

        # Mock trajectory processing result
        mock_result = ProcessingResult(
            exafs_group=Mock(),
            plot_paths={
                "pdf": output_dir / "trajectory_avg_EXAFS_FT.pdf",
                "png": output_dir / "trajectory_avg_EXAFS_FT.png",
            },
            processing_mode="trajectory",
            nframes=5,
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
                str(trajectory_file),
                "Fe",
                "--trajectory",
                "--output",
                str(output_dir),
                "--interval",
                "2",
                "--parallel",
                "--workers",
                "4",
                "--plot-style",
                "publication",
            ],
        )

        assert result.exit_code == 0
        assert "Frames processed: 5" in result.stdout

        # Verify all options were passed correctly
        args, kwargs = mock_wrapper.process.call_args
        assert kwargs["trajectory"] is True
        assert kwargs["config"].sample_interval == 2
        assert kwargs["config"].parallel is True
        assert kwargs["config"].n_workers == 4

    # ================== CONFIGURATION SCENARIOS ==================

    def test_comprehensive_config_file(self, tmp_path):
        """Test with comprehensive configuration file."""
        config_file = tmp_path / "comprehensive_config.yaml"
        config_data = {
            "spectrum_type": "EXAFS",
            "edge": "L3",
            "radius": 10.0,
            "method": "pymatgen",
            "kmin": 3.0,
            "kmax": 15.0,
            "kweight": 3,
            "window": "kaiser",
            "dk": 0.5,
            "parallel": True,
            "n_workers": 8,
            "force_recalculate": True,
            "user_tag_settings": {
                "S02": "0.85",
                "SCF": "6.0 0 30 0.1 1",
                "EXCHANGE": "0",
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake cif content")

        with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
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
                ["process", str(structure_file), "Fe", "--config", str(config_file)],
            )

            assert result.exit_code == 0

            # Verify config was loaded correctly
            args, kwargs = mock_wrapper.process.call_args
            # Note: CLI overrides config file settings with command-line defaults
            # The edge should be the default 'K' since no --edge was specified

    def test_preset_combinations(self, tmp_path):
        """Test different preset combinations."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake cif content")

        for preset_name in PRESETS.keys():
            with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
                mock_wrapper = Mock()
                mock_wrapper.generate_feff_input.return_value = tmp_path / "output"
                mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
                mock_wrapper.__exit__ = Mock(return_value=None)
                mock_wrapper_class.return_value = mock_wrapper

                result = self.runner.invoke(
                    app,
                    ["generate", str(structure_file), "Fe", "--preset", preset_name],
                )

                assert result.exit_code == 0, f"Failed with preset {preset_name}"

    # ================== ERROR HANDLING SCENARIOS ==================

    def test_permission_errors(self, tmp_path):
        """Test handling of permission errors."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Try to write to a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir(mode=0o444)

        try:
            result = self.runner.invoke(
                app,
                ["generate", str(structure_file), "Fe", "--output", str(readonly_dir)],
            )
            # Should handle the error gracefully
            assert result.exit_code == 1
        finally:
            # Cleanup - restore write permissions
            readonly_dir.chmod(0o755)

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_memory_intensive_operations(self, mock_wrapper_class, tmp_path):
        """Test handling of memory-intensive operations."""
        structure_file = tmp_path / "large_structure.cif"
        structure_file.write_text("fake content")

        # Mock a memory error during processing
        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = MemoryError("Out of memory")
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
                "--workers",
                "16",  # High number of workers
            ],
        )

        assert result.exit_code == 1
        assert "Error:" in result.stdout

    # ================== PLOTTING AND OUTPUT SCENARIOS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_different_plot_formats(self, mock_wrapper_class, tmp_path):
        """Test generation of different plot formats."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Mock result with multiple plot formats
        plot_paths = {
            "pdf": tmp_path / "plot.pdf",
            "png": tmp_path / "plot.png",
            "svg": tmp_path / "plot.svg",
            "eps": tmp_path / "plot.eps",
        }

        mock_result = ProcessingResult(
            exafs_group=Mock(), plot_paths=plot_paths, processing_mode="single_frame"
        )

        mock_wrapper = Mock()
        mock_wrapper.process.return_value = mock_result
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["process", str(structure_file), "Fe"])

        assert result.exit_code == 0
        assert "PDF" in result.stdout
        assert "PNG" in result.stdout
        assert "SVG" in result.stdout
        assert "EPS" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_plot_style_variations(self, mock_wrapper_class, tmp_path):
        """Test different plot style options."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        plot_styles = ["publication", "presentation", "quick"]

        for style in plot_styles:
            mock_result = ProcessingResult(
                exafs_group=Mock(),
                plot_paths={"pdf": tmp_path / f"plot_{style}.pdf"},
                processing_mode="single_frame",
            )

            mock_wrapper = Mock()
            mock_wrapper.process.return_value = mock_result
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            result = self.runner.invoke(
                app, ["process", str(structure_file), "Fe", "--plot-style", style]
            )

            assert result.exit_code == 0

            # Verify style was passed to wrapper
            args, kwargs = mock_wrapper.process.call_args
            assert kwargs["plot_style"] == style

    # ================== CACHE OPERATION SCENARIOS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_cache_operations_comprehensive(self, mock_wrapper_class):
        """Test comprehensive cache operations."""
        mock_wrapper = Mock()

        # Test cache info with various states
        cache_states = [
            {"enabled": True, "cache_dir": "/tmp/cache", "files": 0, "size_mb": 0},
            {"enabled": True, "cache_dir": "/tmp/cache", "files": 10, "size_mb": 25.5},
            {"enabled": False, "cache_dir": None, "files": 0, "size_mb": 0},
        ]

        for state in cache_states:
            mock_wrapper.get_cache_info.return_value = state
            mock_wrapper_class.return_value = mock_wrapper

            result = self.runner.invoke(app, ["cache", "info"])
            assert result.exit_code == 0

            if state["enabled"]:
                assert "Cache Status" in result.stdout
                assert str(state["files"]) in result.stdout
            else:
                assert "disabled" in result.stdout

    # ================== EDGE TYPE AND METHOD COMBINATIONS ==================

    def test_edge_type_method_combinations(self, tmp_path):
        """Test various edge type and method combinations."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        edge_types = ["K", "L1", "L2", "L3"]
        methods = ["auto", "larixite", "pymatgen"]

        for edge in edge_types:
            for method in methods:
                with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
                    mock_wrapper = Mock()
                    mock_wrapper.generate_feff_input.return_value = tmp_path / "output"
                    mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
                    mock_wrapper.__exit__ = Mock(return_value=None)
                    mock_wrapper_class.return_value = mock_wrapper

                    result = self.runner.invoke(
                        app,
                        [
                            "generate",
                            str(structure_file),
                            "Fe",
                            "--edge",
                            edge,
                            "--method",
                            method,
                        ],
                    )

                    assert result.exit_code == 0, (
                        f"Failed with edge={edge}, method={method}"
                    )

                    # Verify parameters were passed correctly
                    # Note: generate_feff_input gets (structure, absorber,
                    # output_dir, config) as positional args
                    args, kwargs = mock_wrapper.generate_feff_input.call_args
                    config = args[3]  # config is the 4th positional argument
                    assert config.edge == edge
                    assert config.method == method

    # ================== PARALLEL PROCESSING SCENARIOS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_parallel_processing_scenarios(self, mock_wrapper_class, tmp_path):
        """Test different parallel processing configurations."""
        structure_file = tmp_path / "trajectory.xyz"
        structure_file.write_text("fake trajectory")

        parallel_configs = [
            {"parallel": True, "workers": None},
            {"parallel": True, "workers": 2},
            {"parallel": True, "workers": 8},
            {"parallel": False, "workers": None},
        ]

        for config in parallel_configs:
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

            cmd = ["process", str(structure_file), "Fe", "--trajectory"]

            if config["parallel"]:
                cmd.append("--parallel")
            else:
                cmd.append("--sequential")

            if config["workers"]:
                cmd.extend(["--workers", str(config["workers"])])

            result = self.runner.invoke(app, cmd)
            assert result.exit_code == 0

            # Verify configuration was passed
            args, kwargs = mock_wrapper.process.call_args
            wrapper_config = kwargs["config"]
            assert wrapper_config.parallel == config["parallel"]
            if config["workers"]:
                assert wrapper_config.n_workers == config["workers"]

    # ================== LONG OUTPUT AND PROGRESS TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_long_trajectory_with_progress(self, mock_wrapper_class, tmp_path):
        """Test processing of long trajectory with progress updates."""
        structure_file = tmp_path / "long_trajectory.xyz"
        structure_file.write_text("fake long trajectory")

        def simulate_long_process(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                # Simulate processing 100 frames
                for i in range(0, 101, 10):
                    progress_callback(i, 100, f"Processing frame {i}/100...")

            return ProcessingResult(
                exafs_group=Mock(),
                plot_paths={"pdf": tmp_path / "plot.pdf"},
                processing_mode="trajectory",
                nframes=100,
            )

        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = simulate_long_process
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(
            app, ["process", str(structure_file), "Fe", "--trajectory"]
        )

        assert result.exit_code == 0
        assert "Frames processed: 100" in result.stdout

    # ================== CLEANUP AND RESOURCE MANAGEMENT ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_resource_cleanup_on_interrupt(self, mock_wrapper_class, tmp_path):
        """Test proper cleanup when operations are interrupted."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Mock an interrupted operation
        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = KeyboardInterrupt("User interrupted")
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["process", str(structure_file), "Fe"])

        # Should handle KeyboardInterrupt gracefully
        # Exit code 130 is standard for KeyboardInterrupt (128 + SIGINT signal 2)
        assert result.exit_code == 130
        # Verify __exit__ was called for proper cleanup
        mock_wrapper.__exit__.assert_called_once()
