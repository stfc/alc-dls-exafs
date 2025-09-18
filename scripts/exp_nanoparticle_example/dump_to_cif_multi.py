#!/usr/bin/env python3
import argparse
from pathlib import Path
from ase.io import iread, read, write
import numpy as np
import sys

def read_species_from_lammps_input(lammps_input):
    species = []
    with open(lammps_input) as f:
        for line in f:
            if line.strip().startswith("pair_coeff") and "*" in line:
                species = line.split()[4:]
                break
    if not species:
        raise RuntimeError("No species found in pair_coeff line")
    return species

def parse_first_frame_types(dump_file):
    with open(dump_file) as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("No timestep found in dump file")
            if line.startswith("ITEM: ATOMS"):
                header = line.split()[2:]
                if "type" not in header:
                    raise RuntimeError("Dump header lacks 'type' column")
                type_idx = header.index("type")
                id_is_first_col = True
                break
        atom_lines = []
        for line in f:
            if line.startswith("ITEM:"):
                break
            atom_lines.append(line.strip())
    atom_info = []
    for l in atom_lines:
        parts = l.split()
        atom_id = int(parts[0]) if id_is_first_col else int(parts[header.index("id")])
        atom_type = int(parts[type_idx])
        atom_info.append((atom_id, atom_type))
    atom_info.sort(key=lambda x: x[0])
    atom_types = [atype for _, atype in atom_info]
    return atom_types

def sort_by_id(atoms):
    if "id" in atoms.arrays:
        order = np.argsort(atoms.arrays["id"])
        atoms = atoms[order]
    return atoms

def write_cif(atoms, symbols, outpath):
    atoms.set_chemical_symbols(symbols)
    atoms = atoms[[i for i, _ in sorted(enumerate(atoms.get_chemical_symbols()), key=lambda x: x[1])]]
    atoms.set_scaled_positions(np.round(atoms.get_scaled_positions(), 6))
    write(outpath, atoms, format="cif")

def main():
    parser = argparse.ArgumentParser(description="Convert multiple frames from a LAMMPS dump to CIFs")
    parser.add_argument("dump_file", help="LAMMPS dump file (text)")
    parser.add_argument("lammps_input", help="LAMMPS input (for species list)")
    parser.add_argument("-o", "--outdir", default="cifs", help="Output directory [default: cifs]")
    parser.add_argument("-s", "--stride", type=int, default=50, help="Frame stride [default: 50]")
    parser.add_argument("-S", "--start", type=int, default=0, help="First frame index to write [default: 0]")
    parser.add_argument("-m", "--max_frames", type=int, default=None, help="Max number of CIFs to write [default: all]")
    args = parser.parse_args()

    dump_file = args.dump_file
    lammps_input = args.lammps_input
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    species = read_species_from_lammps_input(lammps_input)
    print(f"Species identified: {species}")

    atom_types_first = parse_first_frame_types(dump_file)
    symbols_id_order = [species[t - 1] for t in atom_types_first]

    frames_written = 0
    first_ids = None
    last_ids = None

    for i, atoms in enumerate(iread(dump_file, format="lammps-dump-text")):
        if i < args.start or ((i - args.start) % args.stride != 0):
            continue
        atoms = sort_by_id(atoms)
        if first_ids is None and "id" in atoms.arrays:
            first_ids = np.asarray(atoms.arrays["id"], dtype=int).copy()
        if len(atoms) != len(symbols_id_order):
            print(f"[ABORT] Atom count changed at frame {i} ({len(atoms)} vs {len(symbols_id_order)}).", file=sys.stderr)
            sys.exit(1)
        fout = outdir / f"struct_f{i:06d}.cif"
        write_cif(atoms, symbols_id_order, str(fout))
        frames_written += 1
        if "id" in atoms.arrays:
            last_ids = np.asarray(atoms.arrays["id"], dtype=int)
        if args.max_frames is not None and frames_written >= args.max_frames:
            break

    if frames_written == 0:
        raise RuntimeError("No frames selected. Check start/stride/max_frames.")

    if first_ids is not None and last_ids is not None and not np.array_equal(first_ids, last_ids):
        print("[ABORT] First vs last frame atom IDs differ. Possible relabeling.", file=sys.stderr)
        sys.exit(1)

    print(f"Done. Wrote {frames_written} CIFs to {outdir}")

if __name__ == "__main__":
    main()
