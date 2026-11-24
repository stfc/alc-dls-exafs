# Changelog

## Unreleased

### Changed

- Python 3.10 is no longer supported. Requires Python 3.11 or newer.
- Requires h5py 3.5 or newer (for the `locking` argument to `h5py.File`).
- **Breaking:** The ensemble archive format is bumped from version 3 to 4
  (`ENSEMBLE_VERSION = 4`). Version 2 and 3 archives remain readable through
  `ArchiveReader`.
- **Breaking:** `/top_paths/*/chi` in ensemble archives now stores the
  ensemble-weighted path contribution (per-instance `path_chi` summed and
  divided by the per-k `n_contributors`), not `path_chi` evaluated on the
  group-mean parameters. Path contributions therefore keep the disorder damping
  and sum with `unassigned` to the overall average. `/top_paths` groups paths
  with angle-aware clustering (`cluster_path_category`), ranks them by
  ensemble-weighted importance, and stores a per-path `count` and a file-level
  `top_paths.attrs["n_tasks"]`.
- **Breaking:** Removed the duplicate `exafs_data.make_path_key` (`SS_`/`MS3_`
  labels). All callers now use `paths.make_path_key`, which gives labels such as
  `Cu_2_2.55` and `Cu-Cu_3_3.60_A120`. Keys already stored in old archives are
  unchanged. `aiida-feff` carries a separate downstream copy of
  `make_path_key`, which is not updated by this release.
- **Breaking:** The $\Delta E_0$ sign convention is fixed in
  `experimental.shifted_k_mask` and `experimental.scaled_chi_arrays`. Both now
  use $k^2 + \Delta E_0 / (\hbar^2/2m_e)$, matching `paths.path_chi` and Larch's
  `feffdat._calc_chi`; the previous sign was opposite to both. Downstream `aiida-feff` and
  `aiidalab-exafs` code that applies `e0_shift` through these functions must be
  checked and updated, because results for a nonzero `e0_shift` change.

### Added

- `md_exafs.stop`: `STOP`, a `threading.Event` set by SIGTERM or SIGUSR1 once
  `install_sigterm_handler()` has been called, and `RunStoppedError`.
  `md-exafs pipeline` installs the handler. Library code does not.
- `BatchShardWriter.flush()`.
- `EnsembleWriter` stores `radius` in archive metadata. It is threaded from
  `FeffConfig` through `BatchExecutor`, `write_batch_shard` and `merge_shards`.
- `experimental.read_experimental_spectrum` accepts a Larch `Group`, a dict
  (`{"k", "chi"}` or `{"energy", "mu"}`), or a 2-tuple, in addition to file
  paths. A 2-tuple is read as `(k, chi)` unless `kind="energy_mu"` is given.
- `archive_series` accepts `experiment=`, `s02=` and `e0_shift=`. An
  `experiment` entry is added in the same shape as `overall`. When `s02` or
  `e0_shift` is set, a `comparison` block holds
  `{"s02", "e0_shift", "overall_adjusted"}`. `overall` stays raw, so the schema
  version is unchanged.
- `archive_series` reports `ft_limits["rmax_valid"]` (the archive `radius` minus
  0.5 Å) and adds a structured warning when the requested `rmax` exceeds it.
- `viz.series_to_vega_lite` overlays `experiment` as a dashed line in the
  `chi_k` and `chi_r` views, and plots `comparison["overall_adjusted"]` when it
  is present.
- `ArchiveReader` gains `get_top_paths()`, `n_tasks`, `radius` and
  `format_version`.
- `paths.cluster_path_category`, a shared angle-aware clustering helper used by
  `PathAggregator` and archive path grouping.
- `md_exafs.fit.fit_ensemble` (also re-exported as `md_exafs.fit_ensemble`) fits
  global $S_0^2$ and $\Delta E_0$ to an ensemble archive's overall $\chi(k)$
  against experimental data over several $k$-weights (default `(1, 2, 3)`) in
  R-, k-, or q-space. It uses Larch's `TransformGroup`, per-$k$-weight
  `estimate_noise` normalization, and $\chi^2$ and uncertainty rescaling matched
  to Larch's $N_{\mathrm{idp}}$. It returns a versioned, JSON-safe report with
  R-factors and aligned theory and experiment curves.
- `md_exafs.attribution.peak_attribution` and `md_exafs.attribution.group_sites`
  (also re-exported at top-level `md_exafs`): decomposes $\chi(R)$ peaks by
  scattering path (`by="paths"`, requiring archive format version 4), absorber
  site (`by="sites"`), or site group (`by="site_groups"`), reporting signed
  projections that sum to 1 with an explicit `unassigned` component, leave-one-out
  peak deltas, peak windows between neighbouring minima with shoulder detection,
  and hierarchical clustering of sites by their complex $\chi(R)$ spectra.

### Fixed

- HDF5 results survive a killed run. `ExafsHDF5Store.write_site_results_batch`
  flushes after each batch, and `BatchExecutor` flushes once per chunk, before
  removing that chunk's scratch directories. Previously a run killed at a Slurm
  time limit left an unreadable `results.h5`.
- On a stop request, the pipeline and `BatchExecutor` finish the chunk in
  flight and start no new one. `process_trajectory` then raises
  `RunStoppedError` without averaging, and `md-exafs pipeline` exits with
  status 1.
- An existing results file that cannot be opened raises `UnreadableHDF5Error`
  (an `OSError`) naming the file. `ExafsHDF5Store.close()` logs flush errors
  instead of silently discarding them.
- A run killed on a shared filesystem (for example a preempted HPC job) no
  longer leaves `results.h5` locked. HDF5 file locking is now disabled when
  opening results archives, so a stale lock cannot block a resume.

## 0.3.1

### Added

- `md_exafs.archive_series.archive_series`, which reads an ensemble archive
  into a versioned, JSON-safe, plot-ready series: the stored overall average
  (with std and contributor counts) and, when requested with
  `include=("sites", "frames")`, per-site averages and a capped, evenly
  subsampled set of per-frame spectra. Breakdowns that are not requested are not
  read. The series also carries the Fourier-transform parameters, units and
  warnings.
- `viz.series_to_vega_lite` (also `md_exafs.series_to_vega_lite`), which turns a
  series into a Vega-Lite spec (views `chi_k`, `chi_r`, `sites`, `frames`) with
  the existing chart builders. Requires `altair`. The `chi_r` view plots
  |chi(R)|, Re and Im under a neutral `χ(R) (Å⁻ⁿ)` axis.
- `ArchiveReader.fourier_params` and `ArchiveReader.get_overall_average`.
- Fourier-transform parameters can be changed on the fly.
  `archive_series(path, ft_params=...)` recomputes chi(R) with `xftf_arrays`
  from the stored unweighted chi(k), for the overall average and every site and
  frame spectrum read. `ft_params` takes any subset of `kmin`, `kmax`, `dk`,
  `window`, `kweight` (integer 0 to 4) and `rmax`; the rest default to the
  archive's own values. Invalid requests raise `InvalidFtParamsError` (a
  `ValueError`) listing one message per problem. A window that reaches past the
  k with data is reported in `ft_warnings` and in `warnings`, and the series is
  still returned; sites and frames get one `window_beyond_data` entry per
  section with `n_affected` and `n_shown`.
- Series fields: `ft_params` (the resolved parameters used), `ft_params_source`
  (`"archive"` or `"request"`), `ft_defaults` (the archive's stored parameters),
  `ft_limits` (k with data, k-weights, windows and the R_max cap, so a UI can
  build its controls from the series) and `ft_warnings`. Each spectrum gains
  `chi_weighted` = k**kweight * chi, and the overall gains `chi_std_weighted`.
  `chi` stays unweighted. `units` gains `chi_weighted`.
- `spectra.FT_WINDOWS`, the window names the API accepts, derived from
  `feff_input.WindowType`.
- `viz.build_chi_chart` and `viz.build_chir_chart` take `y_field` and `y_title`.
- `spectra.xftf_arrays` takes `warn_uncovered: bool = True`. Pass `False` to
  suppress the log warning when uncovered k fall inside the window; the
  default is unchanged. `archive_series` passes `False`, so a recompute no
  longer logs one warning per spectrum.

### Changed

- The chi(k) views plot `chi_weighted`, and their y-axis states the k-weight
  and units (for example `k²χ(k) (Å⁻²)`). The chi_k view draws dashed rules at
  kmin and kmax, and the chi_r view draws one at rmax. Vega-Lite specs
  carry `ft_params_source`, `ft_defaults`, `ft_limits` and `ft_warnings` in
  `usermeta`. Consumers reading the chi_k `chi` field should read
  `chi_weighted` instead.

### Fixed

- `ArchiveReader.is_ensemble`, `is_shard` and `fourier_params` raised
  `AttributeError` on an HDF5 file with no `/meta` group (an aggregates-only
  archive). They now treat a missing `/meta` as empty metadata, and
  `archive_series` no longer relies on catching that error.

## 0.3.0

Averaging χ(k) had four implementations with three different conventions for
k outside a spectrum's own range. This release collapses them onto one and
fixes the two ways the reference grid could silently corrupt a result.

### Numerical changes

- **Ragged ensembles are masked, not zero-filled.** A member whose `chi.dat`
  stops early now stops contributing above its own k_max instead of
  contributing a zero. Zero-filling dragged the ensemble mean towards zero at
  high k, which is where the Debye-Waller information lives. Affects any
  ensemble whose members do not all share a k-range.
- **`DEFAULT_K_GRID` endpoints are now exact.** It was
  `np.arange(0.05, 20.0, 0.05)`, whose accumulated float error puts the last
  point at 19.950000000000003. A `chi.dat` ending at exactly 19.95 fell
  outside that, resampled to NaN, and took the whole Fourier transform with
  it. FEFF writes 400 or 401 rows depending on whether its grid starts at
  k=0, so both cases occur in practice.
- **The weighted standard deviation is now weighted.** `average_chi_arrays`
  paired a weighted mean with an unweighted spread. It now uses the
  reliability-weighted unbiased estimator, which reduces to the `ddof=1`
  sample estimator when the weights are equal.
- **The spread is NaN, not zero, where one member contributes.** One sample
  gives no spread estimate, and zero understates the uncertainty exactly
  where the ensemble has thinned out.

### Added

- `spectra.resample_chi`, the single place a χ(k) spectrum is put onto another
  k-grid. It separates a grid-registration mismatch of under half a source
  step, where the endpoint value carries across, from a genuinely shorter
  spectrum, which becomes NaN.
- `n_contributors`, the per-k count of spectra behind each averaged point,
  returned by `average_chi_arrays` and stored in ensemble archives under
  `aggregates/{overall_average,frame_averages,site_averages}/n_contributors`.
- `ArchiveReader.site_indices` and `.frame_indices`, so callers can discover
  what an archive holds without reaching for the private `_open`.

### Changed

- `average_chi_arrays` returns a `ChiAverage` named tuple
  `(k, mean, std, n_contributors)`. The `return_counts` flag and the
  `AverageChiResult` tuple subclass are gone; the subclass could not be
  pickled and gave two ways to ask for the same thing.
- `xftf_arrays` zero-fills NaN before transforming, which is what a
  finite-range Fourier transform does beyond its data anyway. Left in place a
  single NaN anywhere on the grid produced an all-NaN χ(R), because the
  window multiplies the whole array and `0 * nan` is `nan`. Uncovered k
  falling inside the window are logged as a warning, since there the zeros
  damp |χ(R)| and the fix is to lower `kmax`.
- `feff_utils.average_chi_spectra`, `exafs_data.create_averaged_group`,
  `ArchiveReader.chi` on a shard, and `execution.merge_shards` all delegate to
  `average_chi_arrays`.
- `ENSEMBLE_VERSION` is 3. `ArchiveReader` still reads version 2 archives and
  reports `n_contributors` as `None` for them.
- `pipeline.load_results_from_hdf5` recomputes the averages from the per-site
  groups it loads rather than returning the store's own `/aggregates`. Both
  go through `average_chi_arrays` and the groups are already in memory, so
  reading the stored copy saved one `nanmean` and cost the guarantee that the
  returned averages match the returned groups.
