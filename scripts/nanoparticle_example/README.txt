Brief summary of codes/scripts etc (see scripts in question for full details):

setup_cluster.py:
    Setup nanoparticle cluster for subsequent LAMMPS simulation.

lammpsIn_constP:
    LAMMPS driver script for MD simulation (constant P) including:
        - RDF and CN calculation
        - MSD calculation for total and per-species displacements
    Example job submission script included.
    Potential file included (zipped)

post_process_lammps.py:
    Plot RDFs and CNs, report MSDs (total and per species),
    and issue a warning if MSD threshold exceeded (diffusion detected).
    Example RDF plot ('rdfs_plot.png') included.

dump_to_cif.py:
    Convert the first config of 'dump_production.xyz' into a CIF file (only single config included in this dump)
    (species names parsed and reassigned for clean output).
    FEFF output (EXAFS_reference.pdf/.svg)
