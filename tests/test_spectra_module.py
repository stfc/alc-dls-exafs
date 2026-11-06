"""Tests for md_exafs.spectra."""

import logging

import numpy as np

from md_exafs.feff_input import WindowType
from md_exafs.spectra import (
    FT_WINDOWS,
    average_chi_arrays,
    format_chi_ascii,
    resolve_ft_params,
    xftf_arrays,
)


def test_ft_windows_are_the_window_type_values_in_order():
    assert FT_WINDOWS == ("hanning", "parzen", "welch", "gaussian", "sine", "kaiser")
    assert FT_WINDOWS == tuple(w.value for w in WindowType)


def test_xftf_warn_uncovered_controls_logging(caplog):
    k = np.linspace(0.05, 15.0, 300)
    chi = np.sin(2 * 2.5 * k) * np.exp(-0.02 * k**2)
    chi[k > 10.0] = np.nan
    params = {"kmin": 3.0, "kmax": 12.0, "dk": 1.0, "window": "hanning"}
    with caplog.at_level(logging.WARNING, logger="md_exafs.spectra"):
        quiet = xftf_arrays(k, chi, params, warn_uncovered=False)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    with caplog.at_level(logging.WARNING, logger="md_exafs.spectra"):
        loud = xftf_arrays(k, chi, params)
    assert any("no spectrum has data" in r.getMessage() for r in caplog.records)
    np.testing.assert_array_equal(quiet["chir_mag"], loud["chir_mag"])


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
    k_ref, mean_chi, std_chi, n_contrib = average_chi_arrays([k, k], [chi1, chi2])
    assert np.allclose(k_ref, k)
    assert np.allclose(mean_chi, 1.05 * np.sin(k))
    assert np.all(std_chi >= 0.0)
    assert np.all(n_contrib == 2)


def test_format_chi_ascii():
    k = np.array([2.0, 3.0])
    chi = np.array([0.01, -0.02])
    text = format_chi_ascii(k, chi, metadata={"element": "Cu"})
    assert "# element: Cu" in text
    assert "#    k           chi" in text
    assert "2.000000" in text
