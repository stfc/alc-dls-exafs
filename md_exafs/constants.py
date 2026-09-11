"""Physical constants and unit conventions used across md_exafs.

Units throughout md_exafs are Ångström and electron-volt.
Disorder is always a variance, σ² (Å²), never σ.
"""

from __future__ import annotations

from scipy.constants import eV, hbar, m_e

#: ħ²/2mₑ in eV·Å², i.e. E = HBAR2_OVER_2M · k². ≈ 3.80998 eV·Å².
HBAR2_OVER_2M_EV_ANGSTROM2: float = (hbar**2 / (2 * m_e)) * (1e20 / eV)

#: Energy → wavenumber conversion, k² = ETOK · E. ≈ 0.262468 Å⁻²·eV⁻¹.
#: This is IFEFFIT's and Larch's ``ETOK``.
ETOK: float = 1.0 / HBAR2_OVER_2M_EV_ANGSTROM2

__all__ = [
    "HBAR2_OVER_2M_EV_ANGSTROM2",
    "ETOK",
]
