# Interactive applications (WebAssembly)

The MD-EXAFS ecosystem includes interactive applications that execute directly in the browser via WebAssembly (Pyodide via Marimo).

Because the Python runtime executes client-side within the browser, these applications require no server infrastructure, no local Python installation, and no cloud allocation. Atomic coordinate files uploaded into the interface remain in local browser memory.

---

## Available applications

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Debye–Waller disorder screening
:link: debye_waller
:link-type: doc

<span class="badge-wasm">WASM App</span>

Calculates Mean Squared Relative Displacements ($\sigma^2$), third/fourth cumulants, and atomic displacement parameters ($B$-factors) from trajectory files. Includes 3D structure visualization.
:::

:::{grid-item-card} Nanoparticle cluster builder
:link: nanoparticle_builder
:link-type: doc

<span class="badge-wasm">WASM App</span>

Generates spherical, icosahedral, and core-shell bimetallic clusters (e.g. Au@Ag) with custom lattice parameters, shell thicknesses, and radial cutoffs. Exports coordinates to standard formats.
:::

::::

---

## How it works

The applications are exported from reactive Marimo notebooks (`notebooks/debye_waller.py` and `notebooks/nanoparticle_builder.py`) using `marimo export html-wasm`.

When a page loads:
1. Pyodide initialises a WebAssembly-compiled CPython interpreter inside a Web Worker.
2. Core dependencies (`numpy`, `ase`, `scipy`) load into browser memory.
3. The reactive cells execute, rendering interactive inputs, calculations, and 3D atomic visualisations.
