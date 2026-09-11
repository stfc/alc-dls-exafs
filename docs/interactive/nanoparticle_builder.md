# Nanoparticle cluster builder

This interactive tool generates atomic coordinates for mono- and bimetallic nanoparticles, including cuboctahedral, icosahedral, spherical, and core-shell configurations.

The tool executes client-side via WebAssembly. Generated structures can be downloaded in standard crystallographic and simulation formats (XYZ, CIF, POSCAR).

---

## Capabilities

1. **Geometry selection**: Generate Wulff constructions, spherical cuts, or shell-by-shell regular geometric polyhedra.
2. **Core-shell partitioning**: Define core and shell compositions (e.g. Au core with Ag shell) and adjust radial boundaries.
3. **Lattice adjustment**: Control bulk lattice constants and surface relaxations.
4. **Export**: Export configurations directly for LAMMPS molecular dynamics runs or FEFF EXAFS calculations.

---

## Live application

<div class="app-launcher-card">
  <div class="app-meta">
    <h4>Launch in full window</h4>
    <p>Open the Nanoparticle Builder in a standalone tab with maximum screen area.</p>
  </div>
  <a class="app-launch-btn" href="../apps/nanoparticle_builder/index.html" target="_blank" rel="noopener noreferrer">
    Launch Fullscreen ↗
  </a>
</div>

<div class="marimo-app-container">
  <iframe src="../apps/nanoparticle_builder/index.html" title="Nanoparticle Builder WASM App" loading="lazy"></iframe>
</div>
