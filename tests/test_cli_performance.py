"""Performance and stress tests for CLI commands."""

import time
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from larch_cli_wrapper.cli import app
from larch_cli_wrapper.wrapper import ProcessingResult


class TestCLIPerformance:
    """Performance and stress tests for CLI interface."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    # ================== PERFORMANCE TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_large_trajectory_performance(self, mock_wrapper_class, tmp_path):
        """Test performance with large trajectory files."""
        # Simulate large trajectory
        trajectory_file = tmp_path / "large_trajectory.xyz"
        trajectory_file.write_text("fake large trajectory")

        # Mock processing of many frames
        mock_result = ProcessingResult(
            exafs_group=Mock(),
            plot_paths={"pdf": tmp_path / "plot.pdf"},
            processing_mode="trajectory",
            nframes=1000,
        )

        mock_result.cache_hits = 800
        mock_result.cache_misses = 200

        def slow_process(*args, **kwargs):
            # Simulate progress updates for many frames
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                for i in range(0, 1001, 100):
                    progress_callback(i, 1000, f"Processing frame {i}/1000...")
            return mock_result

        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = slow_process
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        start_time = time.time()
        result = self.runner.invoke(
            app,
            [
                "process",
                str(trajectory_file),
                "Fe",
                "--trajectory",
                "--parallel",
                "--workers",
                "8",
            ],
        )
        end_time = time.time()

        assert result.exit_code == 0
        assert "Frames processed: 1000" in result.stdout
        assert "Cache:" in result.stdout

        # Should complete reasonably quickly (mocked, so very fast)
        assert end_time - start_time < 10.0

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_memory_usage_large_datasets(self, mock_wrapper_class, tmp_path):
        """Test memory usage with large datasets."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Mock large memory usage scenario
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
                "--workers",
                "16",  # High worker count
            ],
        )

        assert result.exit_code == 0

    # ================== STRESS TESTS ==================

    def test_rapid_sequential_commands(self, tmp_structure_file):
        """Test rapid execution of sequential commands."""
        # Execute same command multiple times rapidly
        for i in range(10):
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    str(tmp_structure_file),
                    "Fe",
                    "--output",
                    str(tmp_structure_file.parent / f"output_{i}"),
                ],
            )
            assert result.exit_code == 0

    # ================== RESOURCE EXHAUSTION TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_high_worker_count_handling(self, mock_wrapper_class, tmp_path):
        """Test handling of very high worker counts."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

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

        # Test with very high worker count
        result = self.runner.invoke(
            app,
            [
                "process",
                str(structure_file),
                "Fe",
                "--trajectory",
                "--workers",
                "1000",  # Unrealistically high
            ],
        )

        # Should handle gracefully (validation might cap the number)
        assert result.exit_code in [0, 1]  # Either succeed or fail gracefully

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_disk_space_handling(self, mock_wrapper_class, tmp_path):
        """Test handling of potential disk space issues."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Mock disk space error
        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = OSError("No space left on device")
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["process", str(structure_file), "Fe"])

        assert result.exit_code == 1
        assert "Error:" in result.stdout

    # ================== TIMEOUT AND LONG-RUNNING TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_long_running_feff_calculation(self, mock_wrapper_class, tmp_path):
        """Test handling of long-running FEFF calculations."""
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        (feff_dir / "feff.inp").write_text("fake input")

        def slow_feff_run(*args, **kwargs):
            # Simulate long-running FEFF
            time.sleep(0.1)  # Short sleep for testing
            return True

        mock_wrapper = Mock()
        mock_wrapper.run_feff.side_effect = slow_feff_run
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        start_time = time.time()
        result = self.runner.invoke(app, ["run-feff", str(feff_dir)])
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time >= 0.1  # At least the sleep time

    # ================== ERROR RECOVERY TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_partial_failure_recovery(self, mock_wrapper_class, tmp_path):
        """Test recovery from partial failures in processing."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Mock a scenario where some frames fail but others succeed
        def partial_failure(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                progress_callback(0, 5, "Starting...")
                progress_callback(3, 5, "Partial failure occurred")
                progress_callback(5, 5, "Completed with some failures")

            # Return partial result
            return ProcessingResult(
                exafs_group=Mock(),
                plot_paths={"pdf": tmp_path / "plot.pdf"},
                processing_mode="trajectory",
                nframes=3,  # Fewer than expected
            )

        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = partial_failure
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(
            app, ["process", str(structure_file), "Fe", "--trajectory"]
        )

        assert result.exit_code == 0
        assert "Frames processed: 3" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_wrapper_exception_handling(self, mock_wrapper_class, tmp_path):
        """Test handling of various wrapper exceptions."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        exception_types = [
            RuntimeError("Runtime error occurred"),
            ValueError("Invalid value provided"),
            FileNotFoundError("Required file not found"),
            PermissionError("Permission denied"),
            MemoryError("Out of memory"),
            KeyboardInterrupt("User interrupted"),
        ]

        for exception in exception_types:
            mock_wrapper = Mock()
            mock_wrapper.process.side_effect = exception
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            result = self.runner.invoke(app, ["process", str(structure_file), "Fe"])

            # KeyboardInterrupt should return exit code 130 (128 + SIGINT signal 2)
            if isinstance(exception, KeyboardInterrupt):
                assert result.exit_code == 130
                # For KeyboardInterrupt, may not get error message
            else:
                assert result.exit_code == 1
                assert "Error:" in result.stdout

    # ================== CACHE PERFORMANCE TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_large_cache_operations(self, mock_wrapper_class):
        """Test cache operations with large cache sizes."""
        mock_wrapper = Mock()

        # Mock large cache
        mock_wrapper.get_cache_info.return_value = {
            "enabled": True,
            "cache_dir": "/tmp/large_cache",
            "files": 10000,
            "size_mb": 5000.0,
        }
        mock_wrapper_class.return_value = mock_wrapper

        # Test cache info with large cache
        result = self.runner.invoke(app, ["cache", "info"])
        assert result.exit_code == 0
        assert "10000" in result.stdout
        assert "5000.0 MB" in result.stdout

        # Test cache clearing
        result = self.runner.invoke(app, ["cache", "clear"])
        assert result.exit_code == 0

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_cache_hit_miss_ratios(self, mock_wrapper_class, tmp_path):
        """Test display of cache statistics with various hit/miss ratios."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        cache_scenarios = [
            # (hits, misses, expected_percentage)
            (100, 0, 100.0),  # Perfect cache
            (0, 100, 0.0),  # No cache hits
            (50, 50, 50.0),  # 50% hit rate
            (80, 20, 80.0),  # Good hit rate
            (1, 9, 10.0),  # Poor hit rate
        ]

        for hits, misses, expected_pct in cache_scenarios:
            mock_result = ProcessingResult(
                exafs_group=Mock(),
                plot_paths={"pdf": tmp_path / "plot.pdf"},
                processing_mode="trajectory",
                nframes=10,
            )
            mock_result.cache_hits = hits
            mock_result.cache_misses = misses

            mock_wrapper = Mock()
            mock_wrapper.process.return_value = mock_result
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            result = self.runner.invoke(
                app, ["process", str(structure_file), "Fe", "--trajectory"]
            )

            assert result.exit_code == 0
            assert f"{hits} hits" in result.stdout
            assert f"{misses} misses" in result.stdout
            assert f"{expected_pct:.1f}%" in result.stdout

    # ================== PROGRESS TRACKING PERFORMANCE ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_frequent_progress_updates(self, mock_wrapper_class, tmp_path):
        """Test performance with very frequent progress updates."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        def frequent_updates(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            if progress_callback:
                # Very frequent updates
                for i in range(0, 1001, 1):  # Every single frame
                    progress_callback(i, 1000, f"Processing frame {i}/1000...")

            return ProcessingResult(
                exafs_group=Mock(),
                plot_paths={"pdf": tmp_path / "plot.pdf"},
                processing_mode="trajectory",
                nframes=1000,
            )

        mock_wrapper = Mock()
        mock_wrapper.process.side_effect = frequent_updates
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        start_time = time.time()
        result = self.runner.invoke(
            app, ["process", str(structure_file), "Fe", "--trajectory"]
        )
        end_time = time.time()

        assert result.exit_code == 0
        # Should handle frequent updates without significant performance impact
        assert end_time - start_time < 5.0

    # ================== OUTPUT GENERATION STRESS TESTS ==================

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_multiple_output_formats_stress(self, mock_wrapper_class, tmp_path):
        """Test generation of multiple output formats under stress."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Mock many output formats
        many_formats = {
            f"format_{i}": tmp_path / f"plot_{i}.{fmt}"
            for i, fmt in enumerate(["pdf", "png", "svg", "eps", "ps", "tiff", "jpeg"])
        }

        mock_result = ProcessingResult(
            exafs_group=Mock(), plot_paths=many_formats, processing_mode="single_frame"
        )

        mock_wrapper = Mock()
        mock_wrapper.process.return_value = mock_result
        mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
        mock_wrapper.__exit__ = Mock(return_value=None)
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(app, ["process", str(structure_file), "Fe"])

        assert result.exit_code == 0
        # Should list all formats
        assert "Generated plots:" in result.stdout

    # ================== COMMAND LINE LENGTH TESTS ==================

    def test_very_long_command_lines(
        self, mock_generate_workflow, tmp_structure_file, tmp_path
    ):
        """Test handling of very long command lines."""
        # Create files with very long paths
        deep_path = tmp_path
        for i in range(20):  # Create deep directory structure
            deep_path = deep_path / f"very_long_directory_name_level_{i:02d}"
        deep_path.mkdir(parents=True)

        # Copy our proper structure file to the deep path with a long name
        structure_file = deep_path / "structure_with_very_long_filename.cif"
        structure_file.write_text(tmp_structure_file.read_text())

        output_path = deep_path / "output_with_very_long_name"

        result = self.runner.invoke(
            app,
            [
                "generate",
                str(structure_file),
                "Fe",
                "--output",
                str(output_path),
                "--method",
                "larixite",
                "--edge",
                "K",
            ],
        )

        assert result.exit_code == 0
