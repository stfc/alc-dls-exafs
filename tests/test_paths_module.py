"""Tests for md_exafs.paths."""

import numpy as np

from md_exafs.paths import make_path_key, path_chi


def test_make_path_key():
    assert make_path_key("Cu", 2, 2.548) == "Cu_2_2.55"
    assert make_path_key("Cu-Cu", 3, 3.612, angle=120.4) == "Cu-Cu_3_3.60_A120"


def test_path_chi_evaluation():
    # Construct synthetic PathResult with 6 columns
    k_native = np.linspace(0.0, 20.0, 100)
    feff_data = np.zeros((100, 6))
    feff_data[:, 0] = 0.5  # real_phc
    feff_data[:, 1] = 1.0  # mag_feff
    feff_data[:, 2] = 0.2  # pha_feff
    feff_data[:, 3] = 0.9  # red_fact
    feff_data[:, 4] = 10.0  # lam
    feff_data[:, 5] = k_native  # rep

    k_out = np.linspace(2.0, 15.0, 131)
    chi = path_chi(
        k_native=k_native,
        feff_data=feff_data,
        r_eff=2.5,
        degeneracy=12.0,
        k_out=k_out,
        sigma2=0.005,
        s02=0.9,
    )

    assert chi.shape == k_out.shape
    assert np.all(np.isfinite(chi))
    # EXAFS chi should oscillate around 0
    assert np.abs(np.mean(chi)) < np.max(np.abs(chi))
