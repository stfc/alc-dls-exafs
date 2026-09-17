"""Tests for md_exafs.selection (ADR 0008)."""

import pytest

from md_exafs.selection import (
    parse_frame_slice,
    resolve_frame_absorbers,
    resolve_trajectory_absorbers,
)


def test_resolve_frame_absorbers_element():
    symbols = ["Fe", "O", "Fe", "Ti"]
    assert resolve_frame_absorbers(symbols, "Fe") == [0, 2]
    assert resolve_frame_absorbers(symbols, "Ti") == [3]
    with pytest.raises(ValueError, match="No atoms with element 'Cu'"):
        resolve_frame_absorbers(symbols, "Cu")


def test_resolve_frame_absorbers_int():
    symbols = ["Fe", "O", "Fe"]
    assert resolve_frame_absorbers(symbols, 0) == [0]
    assert resolve_frame_absorbers(symbols, -1) == [2]
    with pytest.raises(ValueError, match="out of range"):
        resolve_frame_absorbers(symbols, 5)


def test_resolve_frame_absorbers_list():
    symbols = ["Fe", "O", "Fe", "Ti"]
    assert resolve_frame_absorbers(symbols, [0, 2]) == [0, 2]
    # Enforces single species
    with pytest.raises(ValueError, match="same element"):
        resolve_frame_absorbers(symbols, [0, 1])


def test_resolve_frame_absorbers_relative():
    symbols = ["Ti", "Fe", "O", "Fe", "Fe"]
    # Fe sites are at indices [1, 3, 4]
    assert resolve_frame_absorbers(symbols, "Fe:0,2") == [1, 4]
    with pytest.raises(ValueError, match="Relative index 5 out of range"):
        resolve_frame_absorbers(symbols, "Fe:5")


def test_resolve_trajectory_absorbers():
    traj = [
        ["Fe", "O", "Fe"],
        ["Fe", "Fe", "O"],  # Permuted
        ["Fe", "O", "Fe"],
    ]
    res = resolve_trajectory_absorbers(traj, "Fe")
    assert res == [[0, 2], [0, 1], [0, 2]]

    # Error if species changes across frames
    traj_inconsistent = [
        ["Fe", "O"],
        ["Cu", "O"],
    ]
    with pytest.raises(ValueError, match="species changed across frames"):
        resolve_trajectory_absorbers(traj_inconsistent, 0)


def test_parse_frame_slice():
    assert parse_frame_slice("all", 10) == list(range(10))
    assert parse_frame_slice(":", 5) == list(range(5))
    assert parse_frame_slice("0:10:2", 15) == [0, 2, 4, 6, 8]
    assert parse_frame_slice("::3", 10) == [0, 3, 6, 9]
    assert parse_frame_slice("1,3,5", 10) == [1, 3, 5]
    assert parse_frame_slice(3, 10) == [3]
    assert parse_frame_slice(-1, 10) == [9]
    with pytest.raises(ValueError, match="out of range"):
        parse_frame_slice(15, 10)
