"""Tests for md_exafs.experimental (ADR 0009)."""

import numpy as np

from md_exafs.experimental import scaled_chi_arrays, shifted_k_mask


def test_shifted_k_mask():
    k = np.array([0.0, 1.0, 2.0, 3.0])
    # e0_shift = 10 eV -> k_min_shift = sqrt(10 / 3.81) ≈ 1.62 Å⁻¹
    mask = shifted_k_mask(k, e0_shift=10.0)
    assert not mask[0]
    assert not mask[1]
    assert mask[2]
    assert mask[3]


def test_scaled_chi_arrays():
    k = np.linspace(1.0, 10.0, 50)
    chi = np.sin(k)
    s02 = 0.85
    e0_shift = 5.0
    k_shifted, chi_scaled = scaled_chi_arrays(k, chi, s02=s02, e0_shift=e0_shift)
    assert len(k_shifted) <= len(k)
    assert np.allclose(chi_scaled, s02 * chi[shifted_k_mask(k, e0_shift)])
