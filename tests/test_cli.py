"""Basic tests for the CLI interface."""

from unittest.mock import Mock, patch

from typer.testing import CliRunner

from larch_cli_wrapper.cli import app


class TestCLI:
    """Test cases for CLI interface."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    def test_info_command(self):
        """Test info command."""
        result = self.runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "LARCH WRAPPER DIAGNOSTICS" in result.stdout

    def test_help_command(self):
        """Test main help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CLI wrapper for larch EXAFS processing" in result.stdout

    def test_run_feff_missing_directory(self):
        """Test run-feff with missing directory."""
        result = self.runner.invoke(app, ["run-feff", "nonexistent_dir"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_run_feff_missing_input_file(self, mock_wrapper_class, tmp_path):
        """Test run-feff with missing input file."""
        # Create directory but no input file
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()

        result = self.runner.invoke(app, ["run-feff", str(feff_dir)])

        assert result.exit_code == 1
        assert "feff input file" in result.stdout.lower()

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_run_feff_success(self, mock_wrapper_class, tmp_path):
        """Test successful run-feff command."""
        # Create directory and input file
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()
        input_file = feff_dir / "feff.inp"
        input_file.write_text("fake feff input")

        # Mock successful FEFF run
        mock_wrapper = Mock()
        mock_wrapper.run_feff.return_value = True
        mock_wrapper_class.return_value = mock_wrapper

        result = self.runner.invoke(
            app, ["run-feff", str(feff_dir), "--input", "feff.inp"]
        )

        assert result.exit_code == 0
        assert "completed successfully" in result.stdout
        mock_wrapper.run_feff.assert_called_once()

    def test_process_missing_directory(self):
        """Test process with missing directory."""
        result = self.runner.invoke(app, ["process", "nonexistent_dir"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    @patch("larch_cli_wrapper.cli.LarchWrapper")
    def test_process_missing_chi_file(self, mock_wrapper_class, tmp_path):
        """Test process with missing chi.dat file."""
        # Create directory but no chi.dat
        feff_dir = tmp_path / "feff"
        feff_dir.mkdir()

        result = self.runner.invoke(app, ["process", str(feff_dir)])

        assert result.exit_code == 1
        assert "feff output file" in result.stdout.lower()
