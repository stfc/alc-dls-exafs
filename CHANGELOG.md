# Changelog

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
