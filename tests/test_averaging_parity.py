"""Every averaging route must go through ``spectra.average_chi_arrays``.

Four wrappers used to interpolate and combine χ(k) themselves, with three
different conventions for k outside a spectrum's own range (zero-fill,
endpoint clamp, NaN). They now delegate, and these tests pin that down:

1. ``feff_utils.average_chi_spectra``
2. ``exafs_data.create_averaged_group``
3. ``ArchiveReader.chi`` on a batch shard
4. ``execution.merge_shards``

The rest of the module covers the two things the unification has to get
right to be an improvement rather than a swap of one artifact for another:
ragged k-ranges must not damp the mean, and a grid-registration mismatch of
a fraction of a step must not read as missing data.
"""

from pathlib import Path

import numpy as np
import pytest
from larch import Group

from md_exafs.exafs_data import create_averaged_group
from md_exafs.execution import (
    DEFAULT_K_GRID,
    FeffTask,
    merge_shards,
    parse_task_outputs,
    write_batch_shard,
)
from md_exafs.feff_utils import average_chi_spectra
from md_exafs.hdf5 import ArchiveReader, EnsembleWriter
from md_exafs.spectra import average_chi_arrays, resample_chi, xftf_arrays

# ---------------------------------------------------------------------------
# The reference grid must be exactly what it claims
# ---------------------------------------------------------------------------


def test_default_k_grid_endpoints_are_exact():
    """np.arange(0.05, 20.0, 0.05) overshoots 19.95 by 3.6e-15.

    A chi.dat ending at exactly 19.95 then falls outside the grid, resamples
    to NaN at the last point, and takes the whole Fourier transform with it.
    """
    assert DEFAULT_K_GRID[0] == 0.05
    assert DEFAULT_K_GRID[-1] == 19.95
    assert len(DEFAULT_K_GRID) == 399
    np.testing.assert_array_equal(DEFAULT_K_GRID, np.round(DEFAULT_K_GRID, 10))


# ---------------------------------------------------------------------------
# resample_chi: registration artifact vs genuinely absent data
# ---------------------------------------------------------------------------


def test_resample_carries_endpoint_across_sub_step_overshoot():
    """FEFF writes chi.dat with 400 or 401 rows depending on where it starts.

    Both describe the same k-range; only the final sample position differs.
    That must not read as missing data.
    """
    k_src = np.arange(0, 400) / 20.0  # 0.00 .. 19.95, the 400-row case
    chi_src = np.sin(2.0 * k_src)

    out = resample_chi(k_src, chi_src, DEFAULT_K_GRID)

    assert not np.isnan(out).any()
    # Interior points are interpolated exactly; both grids share their nodes.
    np.testing.assert_allclose(out[:-1], np.sin(2.0 * DEFAULT_K_GRID[:-1]), atol=1e-12)


def test_resample_nans_a_genuinely_shorter_spectrum():
    """A run with a lower EXAFS k_max really has no data up there."""
    k_src = np.arange(0, 281) / 20.0  # 0.00 .. 14.00
    chi_src = np.sin(2.0 * k_src)

    out = resample_chi(k_src, chi_src, DEFAULT_K_GRID)

    covered = DEFAULT_K_GRID <= 14.0
    assert not np.isnan(out[covered]).any()
    assert np.isnan(out[~covered]).all()


def test_resample_tolerance_is_half_a_source_step():
    """Half a step is the distance within which the terminal sample is nearest.

    One step out is missing data; a hair under half a step is registration.
    """
    k_src = np.array([1.0, 2.0, 3.0])
    chi_src = np.array([10.0, 20.0, 30.0])

    out = resample_chi(k_src, chi_src, np.array([0.51, 0.49, 3.49, 3.51]))

    np.testing.assert_array_equal(np.isnan(out), [False, True, False, True])
    assert out[0] == 10.0  # clamped to the left endpoint
    assert out[2] == 30.0  # clamped to the right endpoint


# ---------------------------------------------------------------------------
# Ragged ensembles must not be damped towards zero
# ---------------------------------------------------------------------------


def test_short_member_stops_contributing_rather_than_pulling_the_mean_down():
    """Zero-filling would damp the mean exactly where the Debye-Waller
    information lives. The short member must simply drop out."""
    k_ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    long_chi = np.full(5, 2.0)
    short_chi = np.full(3, 4.0)

    avg = average_chi_arrays([k_ref, k_ref[:3]], [long_chi, short_chi], k_grid=k_ref)

    np.testing.assert_array_equal(avg.n_contributors, [2, 2, 2, 1, 1])
    np.testing.assert_allclose(avg.mean, [3.0, 3.0, 3.0, 2.0, 2.0])
    # Zero-filling would have given 3, 3, 3, 1, 1 — a factor-of-two error.


def test_std_is_nan_where_a_single_member_contributes():
    """One sample gives no spread estimate; zero would understate it."""
    k_ref = np.array([1.0, 2.0, 3.0])
    avg = average_chi_arrays(
        [k_ref, k_ref[:2]], [np.ones(3), np.full(2, 3.0)], k_grid=k_ref
    )

    assert np.isfinite(avg.std[:2]).all()
    assert np.isnan(avg.std[2])


def test_equal_weights_reproduce_the_ddof_one_sample_estimator():
    """The weighted spread must reduce to the unweighted one, or the two
    code paths disagree about what 'standard deviation' means."""
    rng = np.random.default_rng(0)
    k = np.linspace(2.0, 12.0, 40)
    chis = [rng.normal(size=k.size) for _ in range(5)]

    plain = average_chi_arrays([k] * 5, chis)
    weighted = average_chi_arrays([k] * 5, chis, weights=[1.0] * 5)

    np.testing.assert_allclose(weighted.mean, plain.mean)
    np.testing.assert_allclose(weighted.std, plain.std)
    np.testing.assert_allclose(plain.std, np.std(chis, axis=0, ddof=1))


def test_weights_shift_the_mean_towards_the_heavier_member():
    k = np.array([1.0, 2.0])
    avg = average_chi_arrays([k, k], [np.zeros(2), np.ones(2)], weights=[1.0, 3.0])
    np.testing.assert_allclose(avg.mean, [0.75, 0.75])


# ---------------------------------------------------------------------------
# The Fourier transform must survive uncovered k
# ---------------------------------------------------------------------------


def test_uncovered_k_outside_the_window_does_not_poison_chi_r():
    """The window is zero out there, but 0 * nan is nan, so a single
    uncovered point at the top of the grid would blank every χ(R) value."""
    k = DEFAULT_K_GRID
    chi = 0.1 * np.sin(2 * k * 2.5) * np.exp(-0.05 * k**2)
    truncated = chi.copy()
    truncated[k > 16.0] = np.nan

    ft_full = xftf_arrays(k, chi, {"kmin": 3.0, "kmax": 15.0})
    ft_trunc = xftf_arrays(k, truncated, {"kmin": 3.0, "kmax": 15.0})

    assert np.isfinite(ft_trunc["chir_mag"]).all()
    # kmax + dk = 16, so the window never reaches the missing data and the
    # transform is unchanged.
    np.testing.assert_allclose(ft_trunc["chir_mag"], ft_full["chir_mag"], atol=1e-12)


def test_uncovered_k_inside_the_window_warns(caplog):
    """Zero-padding inside the window damps |χ(R)|. Silence would hide it."""
    k = DEFAULT_K_GRID
    chi = 0.1 * np.sin(2 * k * 2.5) * np.exp(-0.05 * k**2)
    chi[k > 12.0] = np.nan

    with caplog.at_level("WARNING", logger="md_exafs.spectra"):
        ft = xftf_arrays(k, chi, {"kmin": 3.0, "kmax": 15.0})

    assert np.isfinite(ft["chir_mag"]).all()
    assert "no spectrum has data" in caplog.text


def test_short_ensemble_still_produces_a_finite_transform(tmp_path: Path):
    """End to end: every member stopping at EXAFS k_max = 14 used to yield an
    all-NaN χ(R), because merge_shards fed the masked mean straight to larch."""
    from md_exafs.hdf5 import BatchShardWriter

    k_native = np.arange(0, 281) / 20.0  # 0.00 .. 14.00
    shards = []
    for frame in range(2):
        path = tmp_path / f"shard_{frame}.h5"
        with BatchShardWriter(path, k_grid=DEFAULT_K_GRID) as writer:
            chi = 0.1 * np.sin(2 * k_native * (2.5 + 0.1 * frame))
            writer.add_task_result(
                frame_idx=frame,
                site_idx=0,
                absorber_element="Fe",
                chi=resample_chi(k_native, chi, DEFAULT_K_GRID),
                paths=None,
            )
        shards.append(path)

    ensemble = tmp_path / "ensemble.h5"
    merge_shards(shards, ensemble, fourier_params={"kmin": 3.0, "kmax": 13.0})

    reader = ArchiveReader(ensemble)
    assert np.isfinite(reader.chir_mag).all()
    # χ(k) still records honestly where the ensemble ran out of data.
    assert np.isnan(reader.chi[DEFAULT_K_GRID > 14.0]).all()
    assert (reader.n_contributors[DEFAULT_K_GRID > 14.0] == 0).all()
    assert (reader.n_contributors[DEFAULT_K_GRID <= 14.0] == 2).all()


# ---------------------------------------------------------------------------
# Every wrapper delegates to the canonical implementation
# ---------------------------------------------------------------------------


def test_feff_utils_average_chi_spectra_delegates():
    k1 = np.linspace(2.0, 12.0, 50)
    k2 = np.linspace(2.0, 12.0, 50)
    chi1, chi2 = np.sin(2.0 * k1), np.cos(1.5 * k2)

    canon = average_chi_arrays([k1, k2], [chi1, chi2])
    chi_avg, k_avg = average_chi_spectra([k1, k2], [chi1, chi2])

    np.testing.assert_array_equal(k_avg, canon.k)
    np.testing.assert_array_equal(chi_avg, canon.mean)


def test_feff_utils_average_chi_spectra_delegates_on_ragged_input():
    k1 = np.linspace(2.0, 12.0, 80)
    k2 = np.linspace(4.0, 15.0, 90)
    chi1, chi2 = np.sin(2.0 * k1), np.cos(1.5 * k2)

    canon = average_chi_arrays([k1, k2], [chi1, chi2])
    chi_avg, k_avg = average_chi_spectra([k1, k2], [chi1, chi2])

    np.testing.assert_array_equal(k_avg, canon.k)
    np.testing.assert_array_equal(chi_avg, canon.mean)


def test_create_averaged_group_delegates():
    k = np.linspace(2.0, 12.0, 60)
    chi1, chi2 = np.sin(2.0 * k), np.sin(2.0 * k) * 1.2

    groups = [Group(k=k, chi=chi1), Group(k=k, chi=chi2)]
    averaged = create_averaged_group(groups, {"kmin": 3.0, "kmax": 11.0})
    canon = average_chi_arrays([k, k], [chi1, chi2])

    np.testing.assert_array_equal(averaged.k, canon.k)
    np.testing.assert_array_equal(averaged.chi, canon.mean)


def test_shard_reader_chi_delegates(tmp_path: Path):
    """ArchiveReader.chi on a shard averages the tasks it holds."""
    k_native = np.linspace(0.0, 20.0, 200)
    tasks = []
    for frame, chi in enumerate([np.sin(k_native), np.cos(k_native)]):
        snap = tmp_path / f"snap_{frame}"
        snap.mkdir()
        rows = [f"{ki:.4f} {ci:.6f} 0 0" for ki, ci in zip(k_native, chi, strict=True)]
        (snap / "chi.dat").write_text("# k chi\n" + "\n".join(rows))
        tasks.append(
            FeffTask(frame_idx=frame, site_idx=0, input_dir=snap, absorber_element="Cu")
        )

    shard = tmp_path / "shard.h5"
    write_batch_shard(tasks, shard, store_paths=False)

    stored = []
    for task in tasks:
        k_p, chi_p, _ = parse_task_outputs(task, store_paths=False)
        stored.append(resample_chi(k_p, chi_p, DEFAULT_K_GRID).astype(np.float32))
    canon = average_chi_arrays([DEFAULT_K_GRID] * len(stored), stored)

    np.testing.assert_array_equal(ArchiveReader(shard).chi, canon.mean)


def test_merge_shards_stores_the_canonical_average(tmp_path: Path):
    k_native = np.linspace(0.0, 20.0, 200)
    tasks = []
    for frame, chi in enumerate([np.sin(k_native), np.cos(k_native)]):
        snap = tmp_path / f"snap_{frame}"
        snap.mkdir()
        rows = [f"{ki:.4f} {ci:.6f} 0 0" for ki, ci in zip(k_native, chi, strict=True)]
        (snap / "chi.dat").write_text("# k chi\n" + "\n".join(rows))
        tasks.append(
            FeffTask(frame_idx=frame, site_idx=0, input_dir=snap, absorber_element="Cu")
        )

    shard = tmp_path / "shard.h5"
    write_batch_shard(tasks, shard, store_paths=False)
    ensemble = tmp_path / "ensemble.h5"
    merge_shards([shard], ensemble)

    stored = []
    for task in tasks:
        k_p, chi_p, _ = parse_task_outputs(task, store_paths=False)
        stored.append(resample_chi(k_p, chi_p, DEFAULT_K_GRID).astype(np.float32))
    canon = average_chi_arrays([DEFAULT_K_GRID] * len(stored), stored)

    reader = ArchiveReader(ensemble)
    assert reader.is_ensemble
    np.testing.assert_array_equal(reader.chi, canon.mean)
    np.testing.assert_array_equal(reader.n_contributors, canon.n_contributors)
    assert np.all(reader.n_contributors == 2)


# ---------------------------------------------------------------------------
# Archive compatibility
# ---------------------------------------------------------------------------


def test_v2_archives_remain_readable(tmp_path: Path):
    """Archives written before n_contributors existed must still open."""
    path = tmp_path / "ensemble_v2.h5"
    k = np.linspace(0.05, 15.0, 100)
    chi = np.sin(k)

    with EnsembleWriter(path, k_grid=k) as writer:
        writer._f["meta"].attrs["format_version"] = 2
        writer.set_overall_average(k, chi)

    reader = ArchiveReader(path)
    assert reader.is_ensemble
    np.testing.assert_array_equal(reader.chi, chi)
    assert reader.n_contributors is None


def test_site_and_frame_indices_are_discoverable(tmp_path: Path):
    """Downstream code needs the stored site list without reaching for _open."""
    k_native = np.linspace(0.0, 20.0, 100)
    tasks = []
    for frame in range(2):
        for site in (0, 3):
            snap = tmp_path / f"snap_{frame}_{site}"
            snap.mkdir()
            chi = np.sin(k_native * (1 + site))
            rows = [
                f"{ki:.4f} {ci:.6f} 0 0" for ki, ci in zip(k_native, chi, strict=True)
            ]
            (snap / "chi.dat").write_text("# k chi\n" + "\n".join(rows))
            tasks.append(
                FeffTask(
                    frame_idx=frame,
                    site_idx=site,
                    input_dir=snap,
                    absorber_element="Cu",
                )
            )

    shard = tmp_path / "shard.h5"
    write_batch_shard(tasks, shard, store_paths=False)
    ensemble = tmp_path / "ensemble.h5"
    merge_shards([shard], ensemble)

    reader = ArchiveReader(ensemble)
    assert reader.site_indices == [0, 3]
    assert reader.frame_indices == [0, 1]


def test_averaging_rejects_mismatched_weights():
    k = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="must match number of"):
        average_chi_arrays([k, k], [k, k], weights=[1.0])
