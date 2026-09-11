"""Tests for md_exafs.constants."""

import numpy as np

from md_exafs.constants import ETOK, HBAR2_OVER_2M_EV_ANGSTROM2


def test_hbar2_over_2m_value():
    # Value should be ~3.80998 eV·Å²
    assert np.isclose(HBAR2_OVER_2M_EV_ANGSTROM2, 3.80998, atol=1e-4)


def test_etok_value():
    # ETOK = 1 / HBAR2_OVER_2M ≈ 0.262468 Å⁻²·eV⁻¹
    assert np.isclose(ETOK, 1.0 / HBAR2_OVER_2M_EV_ANGSTROM2)
    assert np.isclose(ETOK, 0.262468, atol=1e-4)
