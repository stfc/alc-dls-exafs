"""CLI wiring tests: new pipeline flags must reach ``FeffConfig``.

These assert the argument-plumbing (not the science): the ``pipeline`` command
is invoked with a mocked ``PipelineProcessor`` so we can capture the exact
``FeffConfig`` the CLI builds.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from larch_cli_wrapper.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def _invoke(runner, structure, extra_args):
    """Run the pipeline command with mocks, return (result, captured_config)."""
    captured = {}

    def _factory(*_a, **kw):
        captured["config"] = kw.get("config")
        proc = Mock()
        proc.process_trajectory.return_value = (Mock(), {}, {}, [Mock()])
        return proc

    with (
        patch("larch_cli_wrapper.cli.ase_read") as mock_read,
        patch("larch_cli_wrapper.cli.PipelineProcessor", side_effect=_factory),
        patch("larch_cli_wrapper.cli.plot_exafs_matplotlib"),
        patch("larch_cli_wrapper.cli.parse_absorber_specification") as mock_parse,
    ):
        atoms = Mock()
        atoms.__len__ = Mock(return_value=3)
        mock_read.return_value = atoms
        mock_parse.return_value = {"absorber": 0, "description": "Fe (site 0)"}
        result = runner.invoke(app, ["pipeline", str(structure), "Fe", *extra_args])
    return result, captured.get("config")


def test_defaults_reach_config(runner, tmp_structure_file):
    result, cfg = _invoke(runner, tmp_structure_file, [])
    assert result.exit_code == 0, result.stdout
    assert cfg is not None
    assert cfg.stream_chunk_size == 256
    assert cfg.potential_link_mode == "copy"
    assert cfg.store_path_params is False
    assert cfg.store_min_cw_ratio is None


def test_flags_reach_config(runner, tmp_structure_file):
    result, cfg = _invoke(
        runner,
        tmp_structure_file,
        [
            "--stream-chunk-size",
            "7",
            "--potential-link-mode",
            "symlink",
            "--store-path-params",
            "--min-cw-ratio",
            "12.5",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert cfg.stream_chunk_size == 7
    assert cfg.potential_link_mode == "symlink"
    assert cfg.store_path_params is True
    # Unified knob: --min-cw-ratio drives write-time pruning.
    assert cfg.store_min_cw_ratio == 12.5


def test_stream_chunk_size_zero_reaches_config(runner, tmp_structure_file):
    result, cfg = _invoke(runner, tmp_structure_file, ["--stream-chunk-size", "0"])
    assert result.exit_code == 0, result.stdout
    assert cfg.stream_chunk_size == 0


def test_no_store_path_params_flag(runner, tmp_structure_file):
    result, cfg = _invoke(runner, tmp_structure_file, ["--no-store-path-params"])
    assert result.exit_code == 0, result.stdout
    assert cfg.store_path_params is False


def test_hardlink_mode_reaches_config(runner, tmp_structure_file):
    result, cfg = _invoke(
        runner, tmp_structure_file, ["--potential-link-mode", "hardlink"]
    )
    assert result.exit_code == 0, result.stdout
    assert cfg.potential_link_mode == "hardlink"


def test_invalid_potential_link_mode_exits_nonzero(runner, tmp_structure_file):
    result, _cfg = _invoke(
        runner, tmp_structure_file, ["--potential-link-mode", "bogus"]
    )
    assert result.exit_code != 0
    assert "potential-link-mode" in result.stdout
