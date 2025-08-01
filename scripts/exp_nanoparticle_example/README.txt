Description:

Current workflow for using experimental .xyz starting point to generate equilibrated config/s to be fed into feff to generate experimental spectra.
Includes example output.

Inputs:

Ag55Au92.xyz, lammpsIn_constP, mace-mpa-0-medium.model

Scripts:

dump_to_cif.py, feff, post_process_lammps.py, run_feff.py, setup_cluster.py, script # latter: job submission script

Workflow:

m="lammps-mace"
micromamba activate $m
module purge
module load foss/2025a CMake/3.31.3-GCCcore-14.2.0
python ~/progs/mace_stuff/symmetrix/symmetrix/utilities/create_symmetrix_mace.py mace-mpa-0-medium.model 47 79    # generate potential file (force-field) for just the Ag, Au system
sed -i '2s/^/Lattice="50.0 0.0 0.0 0.0 50.0 0.0 0.0 0.0 50.0" Properties=species:S:1:pos:R:3 pbc="T T T"/' Ag55Au92.xyz    # make sure xyz file has some box defined, eg, use this command
python setup_cluster.py   --input-xyz Ag55Au92.xyz    # generate Ag_cluster.lmp lammps structure input
sbatch script    # run lammps
python post_process_lammps.py    # generate rdf, coordination numbers, test msds
python dump_to_cif.py     # convert config from lammps production run into a cif file (currently first config of production run)
mkdir feff
python run_feff.py
ls rdfs_plot.png EXAFS_reference.pdf EXAFS_reference.svg # outputs

Next step:

Feed multiple configs into feff and average over the resulting spectra (will need feff to output plot as data file to enable averaging)

