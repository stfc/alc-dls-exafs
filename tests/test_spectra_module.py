"""Tests for md_exafs.spectra."""

import numpy as np

from md_exafs.spectra import (
    average_chi_arrays,
    format_chi_ascii,
    resolve_ft_params,
)


def test_resolve_ft_params():
    p = resolve_ft_params({"kmin": 2.5, "kweight": 3})
    assert p["kmin"] == 2.5
    assert p["kweight"] == 3
    assert p["kmax"] == 15.0
    assert p["window"] == "kaiser"


def test_average_chi_arrays():
    k = np.linspace(2.0, 12.0, 50)
    chi1 = np.sin(k)
    chi2 = np.sin(k) * 1.1
    k_ref, mean_chi, std_chi = average_chi_arrays([k, k], [chi1, chi2])
    assert np.allclose(k_ref, k)
    assert np.allclose(mean_chi, 1.05 * np.sin(k))
    assert np.all(std_chi >= 0.0)


def test_format_chi_ascii():
    k = np.array([2.0, 3.0])
    chi = np.array([0.01, -0.02])
    text = format_chi_ascii(k, chi, metadata={"element": "Cu"})
    assert "# element: Cu" in text
    assert "#    k           chi" in text
    assert "2.000000" in text
