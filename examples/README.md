# Worked examples

Each subdirectory holds the inputs needed to reproduce one of the case studies,
from the molecular dynamics run through to the ensemble-averaged EXAFS spectrum.

| Example | System | Demonstrates |
|---|---|---|
| [auag_nanoparticle](auag_nanoparticle/) | 147-atom Au–Ag icosahedral nanoparticle | Finite non-periodic cluster, multiple absorber sites, potential reuse across frames |

The pipeline does not run molecular dynamics. Each example therefore has two
stages: an MD stage using an external engine (LAMMPS with a MACE potential in
all of the case studies here), and an EXAFS stage using `larch-cli`. The MD
stage inputs are provided for reproducibility, but you can substitute any
trajectory in a format ASE can read.

Running the EXAFS stage requires the `feff8l` executable on your `PATH`; it is
distributed with [xraylarch](https://xraypy.github.io/xraylarch/).
