
#
# Uses a combination of ASE and bespoke scripting to read in a LAMMPS dump file and convert the first config to CIF.
#
# - ASE read-in of the dump file omits species information, so we manually parse this before invoking ASE.
# - The dump file lacks species names, so these are read from the LAMMPS input file instead.
# - Atoms in the dump are unordered by species; for a cleaner CIF they are regrouped by species.
#

from ase.io import read, write

dump_file = "dump_production.xyz"
lammps_input = "lammpsIn_constP"
output_cif = "first_structure.cif"

# Read species from LAMMPS input
species = []
with open(lammps_input) as f:
    for line in f:
        if line.strip().startswith("pair_coeff") and "*" in line:
            species = line.split()[4:]  # species list after potential file
            break

if not species:
    raise RuntimeError("No species found in pair_coeff line")

print(f"Species identified: {species}")

# Manually parse first timestep's atom types from dump
atom_types = []
with open(dump_file) as f:
    while True:
        line = f.readline()
        if not line:  # EOF
            raise RuntimeError("No timestep found in dump file")
        if line.startswith("ITEM: ATOMS"):
            header = line.split()[2:]  # extract columns after "ITEM: ATOMS"
            type_idx = header.index("type")  # locate 'type' column index
            break

    # Now read atom lines for this first frame
    atom_lines = []
    for line in f:
        if line.startswith("ITEM:"):  # stop when next section starts
            break
        atom_lines.append(line.strip())

# Parse IDs and types, then sort by ID
atom_info = []
for l in atom_lines:
    parts = l.split()
    atom_id = int(parts[0])
    atom_type = int(parts[type_idx])
    atom_info.append((atom_id, atom_type))

# Sort by ID (ASE default)
atom_info.sort(key=lambda x: x[0])

# Extract types in ID-sorted order
atom_types = [atype for _, atype in atom_info]

# Read first frame into ASE
atoms = read(dump_file, index=0, format="lammps-dump-text")

# Overwrite symbols based on parsed atom_types
symbols = [species[t - 1] for t in atom_types]  # map 1-based types to species
atoms.set_chemical_symbols(symbols)

# Clump species together for the cif
atoms = atoms[[i for i, _ in sorted(enumerate(atoms.get_chemical_symbols()), key=lambda x: x[1])]]

# Write CIF
write(output_cif, atoms, format="cif")
print(f"CIF written: {output_cif}")
