Brief summary of codes/scripts (see script headers for full details):

setup_cluster.py:
Generate or convert nanostructures for LAMMPS input.
- Builds FCC nanoclusters with custom surfaces/layers or converts from XYZ.
- Supports type assignment (core–shell or hemisphere split) and optional element mapping.
- Outputs LAMMPS data and XYZ formats.

lammpsIn_constP:
LAMMPS driver script for constant-pressure MD:
- Generates configurations, radial distribution functions (RDF), coordination numbers (CN), and mean squared displacements (MSD).

post_process_lammps.py:
Post-process LAMMPS outputs:
- Parse simulation details (e.g., temperature, timestep, species).
- Plot RDF (rdfs_plot.png) and CN (coordination_plot.png).
- Read MSDs (total and per species) from msd_final.dat and warn if exceeding threshold.

dump_to_cif.py:
Convert LAMMPS dump file to CIF:
- Reads species info from LAMMPS input file.
- Regroups atoms by species for cleaner CIF output.

run_feff.py:
Convert CIF into input for FEFF to simulate experimental spectra.
