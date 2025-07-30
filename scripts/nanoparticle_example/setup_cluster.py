#!/usr/bin/env python
"""
Utility script for generating or converting nanostructures for LAMMPS input.

Features:
- Builds FCC nanoclusters with user-defined surfaces, layer counts, and lattice constants.
- Assigns atoms to two numeric types, 1 and 2, using:
    (i) a radial cutoff (core vs shell), or
    (ii) a hemisphere split defined by a plane normal vector.
- Optionally maps types 1 and 2 to user-specified element symbols (defaults: Ag, Au)
  in the XYZ output.
- Can also read an existing XYZ file and convert its first two species into types 1 and 2,
  retaining original species labels in the XYZ output.
- Outputs both LAMMPS data format (atomic style, types '1' and '2') and XYZ format.
"""

import numpy as np
import argparse
from ase.io import write, read
from ase.cluster.cubic import FaceCenteredCubic

def setup_cluster(method='cutoff', axis_vec=(1,0,0),
                  cutoff=3.0, surfaces=None, layers=None,
                  latticeconstant=4.09, ratio=0.5):
    atoms = FaceCenteredCubic('Ag', surfaces, layers,
                              latticeconstant=latticeconstant)
    atoms.center(vacuum=10, axis=(0,1,2))
    atoms.set_pbc([True, True, True])
    positions = atoms.positions
    center = atoms.get_center_of_mass()
    if method == 'cutoff':
        dist = np.linalg.norm(positions - center, axis=1)
        shell = np.where(dist >= cutoff)[0]
        core  = np.where(dist <  cutoff)[0]
        atoms.numbers[:]    = 1
        atoms.numbers[core] = 2
        print(f"Method: cutoff   cutoff = {cutoff:.2f} Å")
        print(f"  shell atoms (1): {len(shell)}")
        print(f"  core  atoms (2): {len(core)}")
    elif method == 'hemisphere':
        axis = np.array(axis_vec, float)
        axis /= np.linalg.norm(axis)
        dots = np.dot(positions - center, axis)
        side1 = np.where(dots >= 0)[0]
        side2 = np.where(dots <  0)[0]
        atoms.numbers[:]     = 1
        atoms.numbers[side2] = 2
        print(f"Method: hemisphere along {axis_vec}")
        print(f"  side1 (1): {len(side1)}")
        print(f"  side2 (2): {len(side2)}")
    elif method == 'mixed':
        N = len(atoms)
        indices = np.arange(N)
        np.random.shuffle(indices)
        n2 = int(round(ratio * N))
        atoms.numbers[:]        = 1
        atoms.numbers[indices[:n2]] = 2
        print(f"Method: mixed nanoparticle (ratio={ratio})  "
              f"type 1: {N-n2}, type 2: {n2}")
    else:
        raise ValueError("Invalid method: use 'cutoff' or 'hemisphere'")
    return atoms

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-xyz", help="Path to existing XYZ for conversion")
    p.add_argument("--method", choices=("cutoff","hemisphere", "mixed"),
                   default="cutoff",
                   help="‘cutoff’ (core/shell) or ‘hemisphere’ split")
    p.add_argument("--axis", default="1,0,0",
                   help="Axis for hemisphere split, e.g. 1,0,0")
    p.add_argument("--cutoff", type=float, default=3.0,
                   help="Absolute cutoff radius in Å for shell/core")
    p.add_argument("--latticeconstant", type=float, default=4.09)
    p.add_argument("--surfaces", default="1,0,0;1,1,0;1,1,1")
    p.add_argument("--layers", default="4,6,3")
    p.add_argument("--species1", default="Ag",
                   help="Element symbol for type 1 atoms")
    p.add_argument("--species2", default="Au",
                   help="Element symbol for type 2 atoms")
    p.add_argument("--ratio", type=float, default=0.5,
                   help="Fraction (0–1) of atoms to assign type 2 in mixed mode")

    args = p.parse_args()

    if args.input_xyz:
        atoms = read(args.input_xyz)
    else:
        surfaces = [tuple(int(x) for x in s.split(",")) 
                    for s in args.surfaces.split(";")]
        layers   = [int(x) for x in args.layers.split(",")]
        axis_vec = tuple(float(x) for x in args.axis.split(","))
        atoms = setup_cluster(method=args.method,
                              axis_vec=axis_vec,
                              cutoff=args.cutoff,
                              surfaces=surfaces,
                              layers=layers,
                              latticeconstant=args.latticeconstant,
                              ratio=args.ratio)

    # write LAMMPS data (numeric types only)
    write('Ag_cluster.lmp', atoms, format='lammps-data', atom_style='atomic')
 
    # write XYZ
    if args.input_xyz:
        # just dump original XYZ symbols
        write('Ag_cluster.xyz', atoms)
    else:
        # map types 1/2 → species1/species2
        symbols = [args.species1 if n==1 else args.species2
                   for n in atoms.numbers]
        atoms_xyz = atoms.copy()
        atoms_xyz.set_chemical_symbols(symbols)
        write('Ag_cluster.xyz', atoms_xyz)

    print(f"Written Ag_cluster.lmp and Ag_cluster.xyz ({len(atoms)} atoms)")

