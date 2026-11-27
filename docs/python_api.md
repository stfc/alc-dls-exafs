# Python API reference

The foundational library `md_exafs` provides pure domain functions, data structures, and mathematical routines for EXAFS analysis without framework dependencies.

---

## 1. md_exafs.constants

Physical constants and unit conversion factors. Units throughout `md_exafs` are Ångström and electron-volt.

- `HBAR2_OVER_2M_EV_ANGSTROM2` (`float`): $\hbar^2 / (2 m_e)$ in $\text{eV}\cdot\text{\AA}^2 \approx 3.80998$. That is, the kinetic energy relation $E = \frac{\hbar^2}{2m_e} k^2$.
- `ETOK` (`float`): Reciprocal conversion factor, $k^2 = \text{ETOK} \cdot E \approx 0.262468\text{ \AA}^{-2}\cdot\text{eV}^{-1}$.

---

## 2. md_exafs.selection

Absorber selection and trajectory sampling algorithms (ADR 0008).

### `resolve_frame_absorbers`
```python
def resolve_frame_absorbers(
    symbols: list[str],
    spec: str | int | list[int] | tuple[int, ...],
) -> list[int]:
```
Resolves absorber indices for a single configuration. Enforces single-species consistency, raising a `ValueError` if the selection encompasses more than one chemical element.

### `resolve_trajectory_absorbers`
```python
def resolve_trajectory_absorbers(
    trajectory_symbols: list[list[str]],
    spec: str | int | list[int] | tuple[int, ...],
) -> list[list[int]]:
```
Resolves absorber indices per frame across a trajectory. Accommodates variable atom ordering and permuted topologies while asserting that the chemical species of the absorber remains invariant throughout the simulation.

### `parse_frame_slice`
```python
def parse_frame_slice(
    slice_spec: str | int | slice | list[int],
    n_frames: int,
) -> list[int]:
```
Parses slice specifications (e.g. `":100:5"`, `"all"`, or explicit lists) into 0-based integer frame indices bounded by `n_frames`.

---

## 3. md_exafs.debye_waller

Trajectory-based extraction of vibrational disorder parameters.

### `calculate_grouped_msrd`
```python
def calculate_grouped_msrd(
    structures: list[Any],
    site_spec: str = "0",
    max_reff: float = 6.0,
    r_bin: float = 0.15,
) -> dict[str, Any]:
```
Calculates Mean Squared Relative Displacements (MSRD, $\sigma^2$), third and fourth cumulants ($C_3, C_4$), and autocorrelation effective sample sizes ($N_\text{eff}$) for all scattering paths within `max_reff`.

### `compute_adp_results`
```python
def compute_adp_results(structures: list[Any]) -> dict[str, Any]:
```
Calculates atomic displacement parameter (ADP) tensors $U_{ij}$ and isotropic thermal $B$-factors for all sites in the structure.

### `save_cif_with_adp`
```python
def save_cif_with_adp(results: dict[str, Any]) -> str:
```
Exports average atomic positions and anisotropic ADP tensors to a standardised crystallographic CIF string.

---

## 4. md_exafs.feff_input

Card normalisation and `feff.inp` input generation (ADR 0004).

### `FeffConfig`
```python
@dataclass
class FeffConfig:
    spectrum_type: str = "EXAFS"
    edge: str = "K"
    radius: float = 4.0
    exclude_hydrogen: bool = False
    control: Any | None = None
    print: Any | None = "1 0 0 0 0 3"
    s02: Any | None = 1.0
    scf: Any | None = None
    exchange: Any | None = 0
    nleg: Any | None = 6
    kmin: float = 2.0
    kmax: float = 12.0
    kweight: int = 2
    dk: float = 1.0
```
Dataclass holding all card parameters for FEFF8L and Fourier transform post-processing. Can be initialised from bundled presets (`quick`, `nscf`, `publication`) via `FeffConfig.from_preset()`.

### `build_feff_inp`
```python
def build_feff_inp(
    structure: Atoms | Structure,
    config: FeffConfig | None = None,
    absorber_idx: int = 0,
    output_path: Path | str | None = None,
) -> str:
```
Generates the text of `feff.inp` from an ASE `Atoms` or pymatgen `Structure`. Prunes `COREHOLE` cards to ensure compatibility with FEFF8L.

---

## 5. md_exafs.potentials

Precomputation and distribution of self-consistent field potentials (ADR 0006).

### `PotentialsManager`
- `MANIFEST`: Canonical tuple of FEFF8L potential files (`phase.pad`, `pot.pad`, and associated JSON tables).
- `precompute(structure, absorber_idx, config, output_dir, feff_executable="feff") -> Path`: Computes potentials once on a representative structure (`CONTROL 1 1 1 0 0 0`).
- `distribute(potentials_dir, target_dirs, mode="symlink", n_workers=8) -> int`: Links or copies precomputed potentials into worker task directories.

---

## 6. md_exafs.paths

Scattering path extraction and EXAFS equation evaluation (ADR 0003).

### `path_chi`
```python
def path_chi(
    k_native: np.ndarray,
    feff_data: np.ndarray,
    r_eff: float,
    degeneracy: float,
    k_out: np.ndarray,
    *,
    sigma2: float = 0.0,
    s02: float = 1.0,
    e0_shift: float = 0.0,
    deltar: float = 0.0,
) -> np.ndarray:
```
Evaluates a single path's fine-structure oscillation $\chi(k)$ on `k_out` using the complex electron momentum $p = \mathrm{rep} + i/\lambda$ and cubic spline interpolation matching Larch.

### `read_paths_from_dir`
```python
def read_paths_from_dir(
    feff_dir: Path | str,
    max_paths: int | None = None,
    threshold: float = 0.0,
    frame_idx: int = 0,
    site_idx: int = 0,
) -> list[PathResult]:
```
Parses `files.dat` and `feffNNNN.dat` in a calculation directory into a list of `PathResult` objects, filtering out paths below `threshold`.

---

## 7. md_exafs.spectra

Fourier transformation and ensemble averaging.

### `xftf_arrays`
```python
def xftf_arrays(
    k: np.ndarray,
    chi: np.ndarray,
    ft_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
```
Transforms $\chi(k) \to \chi(R)$ using Larch, returning a dictionary containing `r`, magnitude `chir_mag`, real component `chir_re`, imaginary component `chir_im`, and the resolved parameters `ft_params`.

### `average_chi_arrays`
```python
def average_chi_arrays(
    k_list: list[np.ndarray],
    chi_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```
Averages multiple $\chi(k)$ spectra onto a common reference grid with NaN-aware statistics. Returns `(k_ref, mean_chi, std_chi)` where `std_chi` is the sample standard deviation ($N-1$ degrees of freedom).

---

## 8. md_exafs.experimental

Experimental spectrum loading and alignment (ADR 0009).

- `shifted_k_mask(k: np.ndarray, e0_shift: float) -> np.ndarray`: Mask identifying wavenumber points that remain real under a $\Delta E_0$ shift, where $k^2 + \Delta E_0 / (\hbar^2/2m_e) \ge 0$.
- `scaled_chi_arrays(k: np.ndarray, chi: np.ndarray, s02: float = 1.0, e0_shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]`: Applies $S_0^2$ amplitude scaling and $\Delta E_0$ shifts, $k' = \sqrt{k^2 + \Delta E_0 / (\hbar^2/2m_e)}$, dropping points below the shifted threshold. This is the same sign convention as `paths.path_chi` and Larch: a positive $\Delta E_0$ moves features to higher $k$. Earlier releases used the opposite sign.
- `read_experimental_spectrum(source, group=None, autobk=True, labels=None, *, kind=None) -> tuple[np.ndarray, np.ndarray, dict]`: Returns `(k, chi, meta)`. `source` may be a path (Athena `.prj`, ASCII, or any format Larch reads), a Larch `Group`, a dict `{"k", "chi"}` or `{"energy", "mu"}`, or a 2-tuple read as `(k, chi)`, or as `(energy, mu)` with `kind="energy_mu"`. Background removal uses Larch's `pre_edge` and `autobk`.

See [Experimental comparison, fitting, and attribution](analysis.md) for the full workflow.

---

## 9. md_exafs.hdf5

HDF5 archive storage management (ADR 0002, ADR 0003, ADR 0004).

- `BatchShardWriter(output_path, k_grid, threshold=0.0, config=None)`: Writes single-batch calculation shards (`batch_shard.h5`). With `config`, the FEFF settings (including `radius`) are stored in `meta`.
- `EnsembleWriter(output_path, k_grid, fourier_params=None, radius=None)`: Writes consolidated trajectory ensemble archives (`ensemble_results.h5`). `ENSEMBLE_VERSION = 4`. `radius` is the FEFF cluster radius (Å), recorded in `meta`. `add_top_paths(top_paths, n_tasks=None)` stores the top-path groups and the task count.
- `ArchiveReader(path)`: High-level reader for both shard and ensemble archives. It provides `.k`, `.chi`, `.r`, `.chir_mag`, `.n_contributors`, `.iter_paths()`, `.get_site_average()`, and `.get_frame_average()`. It also provides `.get_top_paths()` (a list of dicts with `chi` and `chir_mag`), and the properties `.n_tasks`, `.radius` (`None` if unknown), and `.format_version`. Versions 2 to 4 are readable.

---

## 10. md_exafs.execution

Streaming execution engine.

- `FeffTask(frame_idx, site_idx, input_dir, absorber_element)`: Atomic execution task descriptor.
- `BatchExecutor(tasks, output_h5, chunk_size=256, n_workers=None, ..., config=None)`: Bounded-memory streaming batch runner. `config` (a `FeffConfig`) is recorded in the shard metadata.
- `write_batch_shard(tasks, output_h5, k_grid=None, threshold=5.0, store_paths=True, config=None) -> tuple[Path, int]`: Serialises FEFF runs that have already completed into a shard.
- `merge_shards(shard_paths, ensemble_path, fourier_params=None, top_n_paths=25, radius=None) -> Path`: Consolidates $K$ shard files into an ensemble archive. `radius` is recorded in the archive. If omitted, it is read from the shards' FEFF config.

---

## 11. md_exafs.viz

Headless visualisation builders (ADR 0007).

- `group_path_results(paths, r_bin_width=0.1) -> list[dict]`: Groups scattering paths by $(scatterer, nlegs, r_{bin})$ and averages their properties.
- `build_chi_chart(df, title="χ(k)") -> alt.Chart`: Altair line chart generator for wavenumber spectra.
- `build_chir_chart(df, title="|χ(R)|") -> alt.Chart`: Altair line chart generator for radial Fourier transforms.
- `build_sigma2_chart(df, title="σ² vs R_eff") -> alt.Chart`: Altair scatter chart generator for MSRD disorder curves.
- `calculate_path_vectors(positions, path_atom_indices) -> list[dict]`: Computes 3D arrow vectors for WEAS-widget geometry overlays.
- `series_to_vega_lite(series, *, view="chi_k", width=480, height=260) -> dict`: Vega-Lite spec for one view of an `archive_series` result. Views are `chi_k`, `chi_r`, `sites` and `frames`. When the series holds `experiment`, it is overlaid as a dashed line in `chi_k` and `chi_r`. When it holds `comparison`, the adjusted ensemble curve (S₀² and ΔE₀ applied) is drawn instead of the raw one.

---

## 12. md_exafs.archive_series

Plot-ready, JSON-safe series from an ensemble archive.

- `archive_series(path, *, ft_params=None, max_frames=20, max_sites=50, include=(), experiment=None, s02=1.0, e0_shift=0.0) -> dict`: `ft_params` overrides the Fourier-transform settings and recomputes chi(R) with them. `experiment` (anything `read_experimental_spectrum` accepts) is transformed with the same parameters as the theory. `s02` and `e0_shift` add a `comparison` block. The `overall` curve stays raw. `ft_limits["rmax_valid"]` is `radius - 0.5` Å when the archive records its radius.

See [Experimental comparison, fitting, and attribution](analysis.md) for conventions.

---

## 13. md_exafs.attribution

Attribution of chi(R) peaks to paths, absorber sites, and site groups.

- `group_sites(path, *, ft_params=None, rrange=None, tol=0.15, n_groups=None) -> dict`: Clusters absorber sites by the chi(R) spectrum each produces. Returns the site indices, weight and medoid of each group.
- `peak_attribution(path, *, by="paths", ft_params=None, rrange=None, prominence=0.05, max_components=20, groups=None) -> dict`: For each chi(R) peak, returns the signed projection share and the leave-one-out change for each component, plus an explicit `unassigned` component. `by` is `"paths"` (requires archive format 4), `"sites"` or `"site_groups"`.

See [Experimental comparison, fitting, and attribution](analysis.md) for conventions.

---

## 14. md_exafs.fit and md_exafs.fit_static

Fitting an experiment to an ensemble archive or a static structure.

- `fit_ensemble(path, experiment, *, ft_params=None, rrange=None, kweights=(1, 2, 3), fit_space="r", vary=("s02", "e0_shift"), s02=1.0, e0_shift=0.0) -> dict`: Fit A. Fits S₀² and ΔE₀ of the overall average chi(k). No structural parameters are free.
- `fit_static(structures, experiment, config, absorber, *, structure=None, sites=None, sigma2="md", deltar_mode="isotropic", ft_params=None, rrange=None, kweights=(1, 2, 3), fit_space="r", workdir=None, msrd=None, archive_path=None, r_tol=0.025, angle_tol=10.0, match_r_tol=0.15, force=False) -> dict`: Fit B. Fits a conventional `feffit` to one static structure with MD thermal σ². `structure=None` uses the trajectory average. Single MD frames are rejected.
- `fit_site_group_weights(path, experiment, groups, *, ft_params=None, rrange=None, kweights=(1, 2, 3), s02=None, e0_shift=None) -> dict`: Fits non-negative site-group weights that sum to 1 to the experiment, using `groups` from `group_sites`.
- `md_exafs.fit_static.pool_paths(paths, n_abs, *, r_tol=0.025, angle_tol=10.0, absorber="") -> list[PooledPath]`: Pools FEFF paths from all absorbers into effective paths by an exact complex sum.
- `md_exafs.fit_static.to_feffit_path(pooled, label=None, *, degen=None, s02=1.0, e0=0.0, deltar=0.0, sigma2=0.0)`: Wraps a pooled path as a `larch.xafs.feffdat.FeffPathGroup`.
- `md_exafs.fit_static.static_paths(structure, config, absorber, sites=None, workdir=None) -> tuple[list[PathResult], int, tuple | None]`: Runs FEFF at each absorber of a static structure and returns the paths, the absorber count, and the mean `chi.dat` for the truncation check.

See [Experimental comparison, fitting, and attribution](analysis.md) for conventions.
