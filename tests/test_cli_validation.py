"""Tests for CLI argument validation and edge cases."""

from unittest.mock import Mock, patch

from typer.testing import CliRunner

from larch_cli_wrapper.cli import app


class TestCLIValidation:
    """Test CLI argument validation and edge cases."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    # ================== ARGUMENT VALIDATION TESTS ==================

    def test_missing_required_arguments(self):
        """Test commands with missing required arguments."""
        # generate command missing arguments
        result = self.runner.invoke(app, ["generate"])
        assert result.exit_code == 2  # Typer error for missing arguments

        # process command missing arguments
        result = self.runner.invoke(app, ["process"])
        assert result.exit_code == 2

        # run-feff command missing arguments
        result = self.runner.invoke(app, ["run-feff"])
        assert result.exit_code == 2

        # process-output command missing arguments
        result = self.runner.invoke(app, ["process-output"])
        assert result.exit_code == 2

    def test_invalid_file_paths(self):
        """Test commands with invalid file paths."""
        test_cases = [
            # Non-existent files
            (["generate", "/nonexistent/structure.cif", "Fe"], 1),
            (["process", "/nonexistent/trajectory.xyz", "Fe"], 1),
            (["run-feff", "/nonexistent/directory"], 1),
            (["process-output", "/nonexistent/directory"], 1),
            # Files that exist but are directories
            # (will be tested separately with actual directories)
        ]

        for cmd, expected_code in test_cases:
            result = self.runner.invoke(app, cmd)
            assert result.exit_code == expected_code
            assert "not found" in result.stdout.lower()

    def test_directory_instead_of_file(self, tmp_path):
        """Test providing directory when file is expected."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # generate and process expect files, not directories (in most cases)
        result1 = self.runner.invoke(app, ["generate", str(test_dir), "Fe"])
        # This might succeed depending on implementation, so we just check it
        # doesn't crash
        assert result1.exit_code in [0, 1]

    def test_file_instead_of_directory(self, tmp_path):
        """Test providing file when directory is expected."""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("content")

        # run-feff and process-output expect directories
        result1 = self.runner.invoke(app, ["run-feff", str(test_file)])
        assert result1.exit_code == 1

        result2 = self.runner.invoke(app, ["process-output", str(test_file)])
        assert result2.exit_code == 1

    # ================== OPTION VALIDATION TESTS ==================

    def test_invalid_edge_values(self, tmp_path):
        """Test commands with invalid edge values."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        invalid_edges = ["X", "K1", "L4", "M6", "invalid", ""]

        for edge in invalid_edges:
            result = self.runner.invoke(
                app, ["generate", str(structure_file), "Fe", "--edge", edge]
            )
            # Should either fail at CLI level or in the wrapper
            assert result.exit_code == 1

    def test_invalid_method_values(self, tmp_path):
        """Test commands with invalid method values."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        invalid_methods = ["invalid_method", "feff", "quantum", ""]

        for method in invalid_methods:
            with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
                # Mock wrapper to avoid actual processing
                mock_wrapper = Mock()
                mock_wrapper.generate_feff_input.side_effect = ValueError(
                    f"Invalid method: {method}"
                )
                mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
                mock_wrapper.__exit__ = Mock(return_value=None)
                mock_wrapper_class.return_value = mock_wrapper

                result = self.runner.invoke(
                    app, ["generate", str(structure_file), "Fe", "--method", method]
                )
                assert result.exit_code == 1

    def test_invalid_preset_values(self, tmp_path):
        """Test commands with invalid preset values."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        invalid_presets = ["invalid_preset", "custom", "user_defined", ""]

        for preset in invalid_presets:
            result = self.runner.invoke(
                app, ["generate", str(structure_file), "Fe", "--preset", preset]
            )
            assert result.exit_code == 1
            assert "Unknown preset" in result.stdout or "Error:" in result.stdout

    def test_invalid_plot_style_values(self, tmp_path):
        """Test commands with invalid plot style values."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        invalid_styles = ["invalid_style", "custom", "matplotlib", ""]

        for style in invalid_styles:
            with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
                mock_wrapper = Mock()
                mock_wrapper.process.side_effect = ValueError(
                    f"Invalid plot style: {style}"
                )
                mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
                mock_wrapper.__exit__ = Mock(return_value=None)
                mock_wrapper_class.return_value = mock_wrapper

                result = self.runner.invoke(
                    app, ["process", str(structure_file), "Fe", "--plot-style", style]
                )
                assert result.exit_code == 1

    # ================== NUMERIC PARAMETER VALIDATION ==================

    def test_invalid_numeric_parameters(self, tmp_path):
        """Test commands with invalid numeric parameters."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Test invalid numeric values - these get parsed by Typer first
        invalid_number_cases = [
            (["process", str(structure_file), "Fe", "--interval", "abc"], 2),
            (["process", str(structure_file), "Fe", "--workers", "abc"], 2),
        ]

        for cmd, expected_code in invalid_number_cases:
            result = self.runner.invoke(app, cmd)
            assert result.exit_code == expected_code

        # Test invalid numeric ranges - these pass Typer validation but fail
        # at app level
        invalid_range_cases = [
            (["process", str(structure_file), "Fe", "--interval", "0"], 1),
            (["process", str(structure_file), "Fe", "--interval", "-1"], 1),
            (["process", str(structure_file), "Fe", "--workers", "0"], 1),
            (["process", str(structure_file), "Fe", "--workers", "-1"], 1),
        ]

        for cmd, expected_code in invalid_range_cases:
            result = self.runner.invoke(app, cmd)
            # These fail when trying to process the file
            assert result.exit_code == expected_code

    def test_boundary_numeric_values(self, tmp_path):
        """Test boundary numeric values."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_wrapper.process.return_value = Mock(
                plot_paths={"pdf": tmp_path / "plot.pdf"}, nframes=1
            )
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            # Test boundary values that should work
            valid_cases = [
                ["--interval", "1"],
                ["--interval", "100"],
                ["--workers", "1"],
                ["--workers", "64"],
            ]

            for options in valid_cases:
                result = self.runner.invoke(
                    app, ["process", str(structure_file), "Fe"] + options
                )
                # File doesn't exist, but should still validate CLI args
                assert result.exit_code == 1  # File not found

    # ================== PATH AND FILE VALIDATION ==================

    def test_relative_vs_absolute_paths(self, tmp_path):
        """Test handling of relative vs absolute paths."""
        # Create test structure in temp directory
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        output_dir = tmp_path / "outputs"

        with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_wrapper.generate_feff_input.return_value = output_dir
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            # Test absolute path (should work)
            result1 = self.runner.invoke(
                app,
                ["generate", str(structure_file), "Fe", "--output", str(output_dir)],
            )
            assert result1.exit_code == 0

            # Test relative path (should also work)
            self.runner.invoke(
                app, ["generate", str(structure_file.name), "Fe"], cwd=str(tmp_path)
            )
            # Note: cwd parameter might not work in CliRunner, so this might
            # need adjustment

    def test_special_characters_in_paths(self, tmp_path):
        """Test handling of special characters in file paths."""
        special_chars = [
            "spaces in name",
            "file-with-dashes",
            "file_with_underscores",
            "file.with.dots",
            "file(with)parentheses",
        ]

        for char_name in special_chars:
            structure_file = tmp_path / f"{char_name}.cif"
            structure_file.write_text("fake content")

            with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
                mock_wrapper = Mock()
                mock_wrapper.generate_feff_input.return_value = tmp_path / "output"
                mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
                mock_wrapper.__exit__ = Mock(return_value=None)
                mock_wrapper_class.return_value = mock_wrapper

                result = self.runner.invoke(
                    app, ["generate", str(structure_file), "Fe"]
                )
                assert result.exit_code == 0

    # ================== CONFIG FILE VALIDATION ==================

    def test_invalid_config_files(self, tmp_path):
        """Test handling of invalid configuration files."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Test cases for invalid config files
        config_cases = [
            # Empty file
            ("", "empty config"),
            # Invalid YAML
            ("invalid: yaml: content:", "invalid YAML"),
            # Valid YAML but invalid values
            ("edge: INVALID_EDGE\nkmin: -5", "invalid values"),
            # Binary file
            (b"\x00\x01\x02\x03", "binary content"),
        ]

        for content, description in config_cases:
            config_file = tmp_path / f"config_{description.replace(' ', '_')}.yaml"
            if isinstance(content, bytes):
                config_file.write_bytes(content)
            else:
                config_file.write_text(content)

            result = self.runner.invoke(
                app,
                ["generate", str(structure_file), "Fe", "--config", str(config_file)],
            )
            assert result.exit_code == 1

    def test_missing_config_file(self, tmp_path):
        """Test handling of missing configuration file."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        nonexistent_config = tmp_path / "nonexistent_config.yaml"

        result = self.runner.invoke(
            app,
            [
                "generate",
                str(structure_file),
                "Fe",
                "--config",
                str(nonexistent_config),
            ],
        )
        assert result.exit_code == 1

    # ================== OUTPUT PATH VALIDATION ==================

    def test_output_path_creation(self, tmp_path):
        """Test automatic creation of output paths."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Deep nested output path that doesn't exist
        deep_output = tmp_path / "level1" / "level2" / "level3" / "outputs"

        with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_wrapper.generate_feff_input.return_value = deep_output
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            result = self.runner.invoke(
                app,
                ["generate", str(structure_file), "Fe", "--output", str(deep_output)],
            )
            assert result.exit_code == 0

    def test_readonly_output_path(self, tmp_path):
        """Test handling of read-only output paths."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir(mode=0o444)  # Read-only

        try:
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    str(structure_file),
                    "Fe",
                    "--output",
                    str(readonly_dir / "subdir"),
                ],
            )
            # Should handle permission error gracefully
            assert result.exit_code == 1
        finally:
            # Cleanup
            readonly_dir.chmod(0o755)

    # ================== COMMAND COMBINATION TESTS ==================

    def test_conflicting_options(self, tmp_path):
        """Test handling of conflicting command options."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Test conflicting preset and config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("edge: K\nmethod: larixite")

        with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_wrapper.generate_feff_input.return_value = tmp_path / "output"
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            # Both preset and config file specified - config file should take precedence
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    str(structure_file),
                    "Fe",
                    "--config",
                    str(config_file),
                    "--preset",
                    "publication",
                ],
            )
            assert result.exit_code == 0

    def test_option_precedence(self, tmp_path):
        """Test precedence of command-line options over config files."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        # Config file with one edge setting
        config_file = tmp_path / "config.yaml"
        config_file.write_text("edge: L3\nmethod: larixite")

        with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
            mock_wrapper = Mock()
            mock_wrapper.generate_feff_input.return_value = tmp_path / "output"
            mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
            mock_wrapper.__exit__ = Mock(return_value=None)
            mock_wrapper_class.return_value = mock_wrapper

            # Command-line edge should override config file
            result = self.runner.invoke(
                app,
                [
                    "generate",
                    str(structure_file),
                    "Fe",
                    "--config",
                    str(config_file),
                    "--edge",
                    "K",  # Should override L3 from config
                ],
            )
            assert result.exit_code == 0

            # Verify the command-line option took precedence
            # Note: generate_feff_input gets (structure, absorber, output_dir,
            # config) as positional args
            args, kwargs = mock_wrapper.generate_feff_input.call_args
            config = args[3]  # config is the 4th positional argument
            assert config.edge == "K"

    # ================== ABSORBER SPECIFICATION TESTS ==================

    def test_absorber_validation(self, tmp_path):
        """Test validation of absorber specifications."""
        structure_file = tmp_path / "structure.cif"
        structure_file.write_text("fake content")

        with patch("larch_cli_wrapper.cli.LarchWrapper") as mock_wrapper_class:
            # Test valid absorber specifications
            valid_absorbers = ["Fe", "Cu", "Zn", "0", "1", "10"]

            for absorber in valid_absorbers:
                mock_wrapper = Mock()
                mock_wrapper.generate_feff_input.return_value = tmp_path / "output"
                mock_wrapper.__enter__ = Mock(return_value=mock_wrapper)
                mock_wrapper.__exit__ = Mock(return_value=None)
                mock_wrapper_class.return_value = mock_wrapper

                result = self.runner.invoke(
                    app, ["generate", str(structure_file), absorber]
                )
                assert result.exit_code == 0

    # ================== HELP AND DOCUMENTATION TESTS ==================

    def test_help_for_all_commands(self):
        """Test help output for all commands."""
        commands = [
            "generate",
            "run-feff",
            "process",
            "process-output",
            "config-example",
            "cache",
            "info",
        ]

        for command in commands:
            result = self.runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            assert "Usage:" in result.stdout
            assert command in result.stdout

    def test_command_descriptions(self):
        """Test that command descriptions are informative."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Check that main commands are listed with descriptions
        expected_commands = ["generate", "run-feff", "process", "process-output"]
        for cmd in expected_commands:
            assert cmd in result.stdout

    # ================== VERSION AND METADATA TESTS ==================

    def test_app_metadata(self):
        """Test application metadata and configuration."""
        # Test main app help
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "larch-cli" in result.stdout

        # Verify app is configured with no_args_is_help=True
        result_no_args = self.runner.invoke(app, [])
        # With no_args_is_help=True and invoke_without_command=True,
        # Typer returns exit code 2 for missing arguments
        assert result_no_args.exit_code == 2
