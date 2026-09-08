# Au–Ag nanoparticle

Ensemble-averaged Au $L_3$-edge and Ag $K$-edge EXAFS for a 147-atom
closed-shell icosahedral Au–Ag nanoparticle, in two chemical orderings:

| File | Ordering |
|---|---|
| `Ag55Au92.xyz` | Ag-centred: 55-atom Ag core, 92-atom Au shell |
| `Au55Ag92.xyz` | Au-centred: 55-atom Au core, 92-atom Ag shell |

Both the core and the whole particle sit on the Mackay icosahedral
magic-number sequence.

## Stage 1 — molecular dynamics

Run externally; the pipeline consumes the resulting trajectory. Inputs for the
LAMMPS route used here are in `md/`.

```bash
# Give the cluster a box so LAMMPS has a periodic cell to work in
sed -i '2s/^/Lattice="50.0 0.0 0.0 0.0 50.0 0.0 0.0 0.0 50.0" \
Properties=species:S:1:pos:R:3 pbc="T T T"/' Ag55Au92.xyz

python md/setup_cluster.py --input-xyz Ag55Au92.xyz   # writes Ag_cluster.lmp
lmp -in md/lammpsIn_constP -log output
```

The protocol is 5 ps NVT followed by 5 ps NVE equilibration, then a 20 ps NVE
production run, with a 2 fs timestep at 300 K. Total angular momentum is
removed (`velocity ... rot no`) so the particle does not tumble; without this,
rigid-body rotation inflates the apparent disorder. `lammpsIn_constP` ships
with short step counts for a quick smoke test — raise `NUMPREEQMDSTEPS`,
`NUMPREEQMDSTEPS2` and `NUMMDSTEPS` to 2500, 2500 and 10000 to reproduce the
published run.

Any MD engine works. The pipeline reads any trajectory format ASE supports.

## Stage 2 — ensemble EXAFS

```bash
larch-cli pipeline dump_production.xyz Au \
    --config auag.yaml \
    --output results/au_l3/
```

`auag.yaml` sets every FEFF and pipeline parameter, so the command line stays
short. It selects every tenth production frame, treats all Au atoms as
absorbers, computes the scattering potentials once from the time-averaged
structure and reuses them for all frames, and writes per-site $\chi(k)$ plus
the per-path decomposition to a single HDF5 archive.

For the Ag $K$ edge, override the two parameters that differ:

```bash
larch-cli pipeline dump_production.xyz Ag \
    --config auag.yaml --edge K \
    --output results/ag_k/
```

## Stage 3 — Debye–Waller factors (optional)

MSRDs for every resolved two- and three-body path around the Au absorbers,
without running FEFF at all:

```bash
larch-cli debye-waller dump_production.xyz \
    --site Au --cutoff 8.0 --cutoff-3body 8.0 \
    --prefix results/auag
```

This writes an MSRD table (CSV), an ADP-decorated CIF, and summary plots. The
particle is not periodic, so leave Kabsch alignment enabled — it is what
removes any residual whole-particle rotation and translation before the
displacement statistics are accumulated.
