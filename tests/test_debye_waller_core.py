import numpy as np
from ase import Atoms

from larch_cli_wrapper.debye_waller_core import calculate_grouped_msrd


def test_msrd_uses_ase_consistent_mic_for_nonorthogonal_cell():
    cell = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.6, 3.7, 0.0],
            [0.0, 0.0, 4.2],
        ]
    )
    frames = [
        Atoms(
            "MnO",
            scaled_positions=[[0.03, 0.5, 0.5], [0.97, 0.5, 0.5]],
            cell=cell,
            pbc=True,
        ),
        Atoms(
            "MnO",
            scaled_positions=[[0.04, 0.5, 0.5], [0.96, 0.5, 0.5]],
            cell=cell,
            pbc=True,
        ),
    ]

    dummy_processed_positions = np.zeros((len(frames), len(frames[0]), 3))

    res_2b, res_3b = calculate_grouped_msrd(
        frames,
        dummy_processed_positions,
        central_indices=[0],
        central_label="Mn",
        cutoff=1.0,
        cutoff_3body=0,
    )

    expected_distances = np.array(
        [atoms.get_distance(0, 1, mic=True) for atoms in frames]
    )

    assert len(res_2b) == 1
    assert res_2b[0]["type"] == "Mn-O"
    assert np.isclose(res_2b[0]["reff"], expected_distances.mean())
    assert np.isclose(res_2b[0]["sigma2"], np.var(expected_distances, ddof=1))
    assert res_2b[0]["atom_indices"] == [(0, 1)]
    assert res_3b == []
