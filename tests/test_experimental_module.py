"""Tests for md_exafs.experimental (ADR 0009)."""

import numpy as np
import pytest
from larch import Group

from md_exafs.constants import HBAR2_OVER_2M_EV_ANGSTROM2
from md_exafs.experimental import (
    read_experimental_spectrum,
    scaled_chi_arrays,
    shifted_k_mask,
)
from md_exafs.paths import path_chi


def test_shifted_k_mask():
    k = np.array([0.0, 1.0, 2.0, 3.0])
    # e0_shift = -10 eV -> k_min_shift = sqrt(10 / 3.81) ≈ 1.62 Å⁻¹
    mask = shifted_k_mask(k, e0_shift=-10.0)
    assert not mask[0]
    assert not mask[1]
    assert mask[2]
    assert mask[3]
    # Positive shift keeps everything
    assert shifted_k_mask(k, e0_shift=10.0).all()


def test_scaled_chi_arrays():
    k = np.linspace(1.0, 10.0, 50)
    chi = np.sin(k)
    s02 = 0.85

    k_low, chi_low = scaled_chi_arrays(k, chi, s02=s02, e0_shift=-10.0)
    mask = shifted_k_mask(k, -10.0)
    assert len(k_low) < len(k)
    assert np.allclose(chi_low, s02 * chi[mask])
    assert np.allclose(k_low**2, k[mask] ** 2 - 10.0 / HBAR2_OVER_2M_EV_ANGSTROM2)

    k_up, chi_up = scaled_chi_arrays(k, chi, s02=s02, e0_shift=5.0)
    assert len(k_up) == len(k)
    assert np.all(k_up > k)
    assert np.allclose(k_up**2, k**2 + 5.0 / HBAR2_OVER_2M_EV_ANGSTROM2)
    assert np.allclose(chi_up, s02 * chi)


@pytest.mark.parametrize("de0", [-4.0, 0.0, 3.0, 7.5])
def test_scaled_chi_arrays_matches_path_chi_sign(de0):
    k_native = np.arange(0.5, 20.01, 0.5)
    feff_data = np.column_stack(
        [
            0.1 * k_native,  # real_phc
            0.5 / (1.0 + 0.1 * k_native),  # mag_feff
            np.sin(0.2 * k_native),  # pha_feff
            np.ones_like(k_native),  # red_fact
            8.0 + 0.0 * k_native,  # lam
            k_native,  # rep
        ]
    )
    k = np.linspace(3.0, 12.0, 91)
    kwargs = {"r_eff": 2.0, "degeneracy": 4.0, "sigma2": 0.005}
    chi_0 = path_chi(k_native, feff_data, k_out=k, e0_shift=0.0, **kwargs)
    k_s, chi_s = scaled_chi_arrays(k, chi_0, s02=0.85, e0_shift=de0)
    expected = path_chi(
        k_native, feff_data, k_out=k_s, s02=0.85, e0_shift=de0, **kwargs
    )
    assert np.allclose(chi_s, expected, rtol=1e-6, atol=1e-9)


def test_read_experimental_spectrum_k_chi_sources():
    k = np.linspace(0.0, 12.0, 241)
    chi = 0.1 * np.sin(2 * 2.0 * k)

    for source in ({"k": k, "chi": chi}, {"k": k, "chi_k": chi}, (k, chi)):
        k_out, chi_out, meta = read_experimental_spectrum(source)
        assert np.array_equal(k_out, k)
        assert np.array_equal(chi_out, chi)
        assert "e0" not in meta

    grp = Group(k=k, chi=chi)
    k_out, chi_out, meta = read_experimental_spectrum(grp)
    assert np.array_equal(chi_out, chi)
    assert "e0" not in meta

    _, _, meta = read_experimental_spectrum({"k": k, "chi": chi, "e0": 8979.0})
    assert meta["e0"] == 8979.0

    with pytest.raises(ValueError, match="kind"):
        read_experimental_spectrum((k, chi), kind="bogus")
    with pytest.raises(ValueError, match="lengths differ"):
        read_experimental_spectrum({"k": k, "chi": chi[:-1]})


def _synthetic_mu(e0: float = 8979.0):
    energy = np.arange(e0 - 150.0, e0 + 1000.0, 1.0)
    kk = np.sqrt(np.clip(energy - e0, 0.0, None) / HBAR2_OVER_2M_EV_ANGSTROM2)
    chi = 0.1 * np.sin(2 * 2.2 * kk) * np.exp(-0.02 * kk**2) * (energy > e0)
    mu = (
        0.5 * (1 + np.tanh((energy - e0) / 2.0)) * (1 + chi)
        + 0.001 * (e0 - energy) / 150.0
    )
    return energy, mu


def test_read_experimental_spectrum_energy_mu_sources():
    energy, mu = _synthetic_mu()

    k_d, chi_d, meta_d = read_experimental_spectrum({"energy": energy, "mu": mu})
    assert k_d.shape == chi_d.shape
    assert k_d.size > 100
    assert abs(meta_d["e0"] - 8979.0) < 10.0

    grp = Group(energy=energy.copy(), mu=mu.copy())
    k_g, chi_g, _ = read_experimental_spectrum(grp)
    assert not hasattr(grp, "chi")
    assert np.allclose(k_g, k_d)
    assert np.allclose(chi_g, chi_d)

    k_t, chi_t, _ = read_experimental_spectrum((energy, mu), kind="energy_mu")
    assert np.allclose(chi_t, chi_d)

    _, _, meta_e = read_experimental_spectrum(
        {"energy": energy, "mu": mu, "e0": 8980.0}
    )
    assert meta_e["e0"] == 8980.0

    with pytest.raises(ValueError, match="autobk"):
        read_experimental_spectrum({"energy": energy, "mu": mu}, autobk=False)


def test_read_experimental_spectrum_file_descriptor_mapping(tmp_path):
    k = np.linspace(1.0, 12.0, 100)
    chi = 0.08 * np.sin(2 * 2.55 * k)
    f = tmp_path / "reversed_chi_k.txt"
    np.savetxt(f, np.column_stack([chi, k]))

    k_out, chi_out, meta = read_experimental_spectrum(
        {"path": str(f), "labels": "chi,k"}
    )
    assert np.allclose(k_out, k)
    assert np.allclose(chi_out, chi)
    assert meta["source_file"] == str(f)

    with pytest.raises(ValueError, match="Unknown experimental descriptor key"):
        read_experimental_spectrum({"path": str(f), "bogus": 123})
