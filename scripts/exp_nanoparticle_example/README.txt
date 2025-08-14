Description:

Current workflows for using experimental .xyz starting point to generate equilibrated config/s to be fed into feff to generate experimental spectra.
Includes example output.
Two workflows are included with descriptions below.

Workflow i (single config):
---------------------------

Description:
A single LAMMPS output configuration from the equilibrated MD is used to generate the EXAFS.

Inputs:

Ag55Au92.xyz, lammpsIn_constP_shorter, mace-mpa-0-medium.model

Scripts:

dump_to_cif.py, post_process_lammps.py, run_feff.py, setup_cluster.py, script # latter: lammps job submission script

Workflow:

m="lammps-mace"
micromamba activate $m
module purge
module load foss/2025a CMake/3.31.3-GCCcore-14.2.0
python ~/progs/mace_stuff/symmetrix/symmetrix/utilities/create_symmetrix_mace.py mace-mpa-0-medium.model 47 79    # generate potential file (force-field) for just the Ag, Au system
sed -i '2s/^/Lattice="50.0 0.0 0.0 0.0 50.0 0.0 0.0 0.0 50.0" Properties=species:S:1:pos:R:3 pbc="T T T"/' Ag55Au92.xyz    # make sure xyz file has some box defined, eg, use this command
python setup_cluster.py   --input-xyz Ag55Au92.xyz    # generate Ag_cluster.lmp lammps structure input
cp lammpsIn_constP_shorter lammpsIn_constP
sbatch script    # run lammps
python post_process_lammps.py    # generate rdf, coordination numbers, test msds
python dump_to_cif.py     # convert config from lammps production run into a cif file (currently first config of production run)
mkdir feff
python run_feff.py
ls rdfs_plot.png EXAFS_reference.pdf EXAFS_reference.svg # outputs

Workflow ii (multiple config averaging):
----------------------------------------

Description:
Multiple configs are sampled from the equilibrated MD to generate (complex) spectra which are then averaged. 
Uses longer LAMMPS job so we can test convergence of EXAFS spectrum w.r.t number configs.
LAMMPS job also corrected for rotation (rotation of nanoparticle is prevented)
This workflow uses some outputs from the first (.cif and potential file) so complete the first prior to this.

Inputs:

Ag_Cluster.lmp, lammpsIn_constP_longer, mace-mpa-0-medium-47-79.json

Scripts:

dump_to_cif_multi.py, post_process_lammps.py, run_feff_multi.py, script # lammps job submission script

Workflow:

m="lammps-mace"
micromamba activate $m
module purge
module load foss/2025a CMake/3.31.3-GCCcore-14.2.0
cp lammpsIn_constP_longer lammpsIn_constP
sbatch script    # run lammps
python post_process_lammps.py    # generate rdf, coordination numbers, test msds
rm -rf cifs/ feff/
mkdir feff
python dump_to_cif_multi.py dump_production.xyz lammpsIn_constP_longer --stride 100 # 10,000 configs in production run, so e.g. stride 100 => 101 snapshots (including 0th)
salloc # access an interactive node
python run_feff_multi.py    # currently in serial ( ~ 10 m for 101 snapshots; but ~ 2hrs for 1001 configs)
mv EXAFS_reference.pdf EXAFS_reference_101configs.pdf

