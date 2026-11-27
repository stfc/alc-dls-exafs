# Comparing an ensemble with experiment

This page takes a format-4 ensemble archive (`ensemble_results.h5` — written automatically by `md-exafs pipeline --hdf5 --keep-paths`, or exported from an existing `results.h5` via `md-exafs export-ensemble results.h5`) and, for the static + MD $\sigma^2$ fit, the MD trajectory it came from, and walks through everything `md_exafs` offers for comparing it with an experimental spectrum and for understanding where the $\chi(R)$ peaks come from. Every code block below runs in order in one Python session.

| Step | Function | Answers |
|---|---|---|
| 1 | `read_experimental_spectrum` | Get $k$ and $\chi(k)$ from a file, a Larch `Group`, a dict or a tuple |
| 2 | `archive_series(..., experiment=...)`, `series_to_vega_lite` | Plot the experiment over the ensemble spectrum |
| 3 | `fit_ensemble` (MD-ensemble alignment) | Does the MD ensemble match experiment after a global $S_0^2$ / $\Delta E_0$ alignment? |
| 4 | `group_sites`, `peak_attribution` | Which paths, sites and site groups build each $\chi(R)$ peak? |
| 5 | `fit_static` (Static + MD $\sigma^2$ fit) | How good a fit does a conventional single-structure `feffit` give, with $\sigma^2$ from the MD? |
| 6 | `fit_site_group_weights` (Site-population reweighting) | Do different weights of the site groups match experiment better than the MD weights? |

All functions return JSON-safe dicts (non-finite numbers become `None`) and never write files. They use the same Fourier-transform parameters for the experiment and the simulation, and echo the parameters they used in `ft_params`.

Read [Conventions and caveats](#conventions-and-caveats) before interpreting numbers. `fit_static` runs FEFF8L (`feff8l` on the `PATH`) at every selected absorber of the static structure, so on a large cell restrict it with `sites=`.

---

## 1. Setup and experiment import

```python
from pathlib import Path

import numpy as np
from ase.io import read

from md_exafs import (
    FeffConfig,
    archive_series,
    fit_ensemble,
    fit_site_group_weights,
    fit_static,
    group_sites,
    peak_attribution,
    series_to_vega_lite,
)
from md_exafs.experimental import read_experimental_spectrum, scaled_chi_arrays

archive = "ensemble_results.h5"
experiment = "experiment.chi"  # path to a .prj, .xdi, .chi, ... file
```

`read_experimental_spectrum(source, group=None, autobk=True, labels=None, *, kind=None)` returns `(k, chi, meta)`. `meta["e0"]` is set when a finite $E_0$ is known. It accepts five kinds of `source`, and every function on this page accepts the same kinds as its `experiment` argument:

```python
# 1. A file path, read with Larch's own readers (or a file-descriptor dict to pass
#    reader options like labels/group/autobk through archive_series, fit_ensemble, etc.).
k, chi, meta = read_experimental_spectrum("experiment.chi")
k, chi, meta = read_experimental_spectrum({"path": "cuexp.txt", "labels": "chi,k"})
print(sorted(meta), k.shape)

# 2. A Larch Group (it is never modified).
from larch.io import read_ascii

group = read_ascii("experiment.chi")
k, chi, meta = read_experimental_spectrum(group)

# 3. A dict, either {"k", "chi"} or {"energy", "mu"} (pre_edge + autobk is run).
k, chi, meta = read_experimental_spectrum({"k": k, "chi": chi})
energy, mu = np.loadtxt("experiment_mu.dat", unpack=True)
k_mu, chi_mu, meta_mu = read_experimental_spectrum({"energy": energy, "mu": mu})
print(meta_mu["e0"], k_mu.shape)

# 4. A 2-tuple, which is (k, chi) unless kind="energy_mu" says otherwise.
k, chi, meta = read_experimental_spectrum((k, chi))
k_mu, chi_mu, meta_mu = read_experimental_spectrum((energy, mu), kind="energy_mu")
```

An untyped tuple is never guessed to be `(energy, mu)`, because energies in keV and wavenumbers in Å⁻¹ overlap.

---

## 2. Overlay the experiment on the ensemble

```python
series = archive_series(
    archive,
    experiment=experiment,
    s02=0.8,
    e0_shift=2.0,
    include=("sites",),
)
print(sorted(series))
print(series["ft_params"])
print("rmax_valid:", series["ft_limits"]["rmax_valid"])
print(sorted(series["experiment"]))
print(series["comparison"]["s02"], series["comparison"]["e0_shift"])
print(sorted(series["comparison"]["overall_adjusted"]))
```

`series["overall"]` stays the raw ensemble average. The $S_0^2$ and $\Delta E_0$ adjusted curve is in the additive `series["comparison"]["overall_adjusted"]` block, which only appears when `s02 != 1.0` or `e0_shift != 0.0`. The experiment is never adjusted. `series["experiment"]` has the same shape as `series["overall"]` and is transformed with the same `ft_params`; if the archive stores no Fourier parameters, `overall` is recomputed with the defaults so both agree.

`series_to_vega_lite` turns the series into a Vega-Lite spec. With a `comparison` block the ensemble curve is the adjusted one, and the experiment is a dashed black line:

```python
import json

spec_k = series_to_vega_lite(series, view="chi_k")
spec_r = series_to_vega_lite(series, view="chi_r")
print(spec_k["$schema"].split("/")[-1], spec_r["$schema"].split("/")[-1])
print(sorted(spec_k["usermeta"]))

Path("chi_k.vl.json").write_text(json.dumps(spec_k))
Path("chi_r.vl.json").write_text(json.dumps(spec_r))
```

In a notebook, `alt.Chart.from_dict(spec_k)` (with `import altair as alt`) renders it. In the `chi_r` view the experiment is drawn as $|\chi(R)|$ only.

---

## 3. MD-ensemble alignment (`fit_ensemble`)

MD-ensemble alignment refines only $S_0^2$ and $\Delta E_0$ of the archive's overall $\chi(k)$. No structural parameter is free; it answers whether the MD ensemble spectrum matches experiment after a global alignment.

```python
fit_a = fit_ensemble(archive, experiment, kweights=(1, 2, 3), fit_space="r")

for name, p in fit_a["params"].items():
    print(f"{name:9s} value={p['value']:.4f} stderr={p['stderr']} "
          f"start={p['start']} vary={p['vary']}")
print("rfactor   (fit space):", fit_a["rfactor"])
print("rfactor_r (R space)  :", fit_a["rfactor_r"])
print("rfactor_k (k space)  :", fit_a["rfactor_k"])
print("n_idp:", fit_a["n_idp"], "nvarys:", fit_a["nvarys"])
print("chi_square:", fit_a["chi_square"], "chi2_reduced:", fit_a["chi2_reduced"])
print("correlations:", fit_a["correlations"])
print("rrange:", fit_a["rrange"], "rmax_valid:", fit_a["rmax_valid"])
for w in fit_a["warnings"]:
    print("warning:", w)
```

The result also holds `schema_version`, `fit_type`, `fit_space`, `ft_params`, `kweights`, `success`, `units`, and `experiment` and `theory` (the best fit) on the archive's $k$ grid, shaped like `series["overall"]`.

- `rfactor` is $\sum \mathrm{resid}^2 / \sum \mathrm{data}^2$ in the fit space, the headline goodness of fit. `rfactor_r` and `rfactor_k` give the same measure in $R$ and $k$ space.
- Every $k$-weight block of the residual is divided by that weight's noise estimate (Larch's `estimate_noise`, echoed in `epsilon_r` and `epsilon_k`), so $k^3$ does not outweigh $k^1$ and $k^2$. When the experiment has no high-$R$ noise (a synthetic spectrum), the RMS of the data in the window is used instead and a warning says that `chi_square` is relative only.
- `chi_square`, `chi2_reduced` and the standard errors are rescaled to `n_idp` exactly as `larch.xafs.feffit` does, so MD-ensemble alignment and the static + MD $\sigma^2$ fit are directly comparable. `n_idp` is $1 + 2\,\Delta R\,\Delta k / \pi$. On high-SNR synchrotron data, $\varepsilon_R$ estimated from $15\text{–}30\text{ \AA}$ is very small, so `chi2_reduced` values of $10^2\text{–}10^3$ are normal even alongside excellent R-factors ($\le 0.01$): systematic theory-vs-experiment differences dominate statistical noise, and parameter uncertainties are already scaled by $\sqrt{\chi^2_\mathrm{red}}$. Use `rfactor` and compare models rather than interpreting `chi2_reduced` as a goodness-of-fit probability.
- `rrange` defaults to `(1.0, min(3.0, rmax_valid))`. A window beyond `rmax_valid` or beyond the data produces an entry in `warnings`.
- `vary=("s02",)` or `vary=()` fixes parameters; `s02` and `e0_shift` give the starting values.

To redraw the overlay with the fitted values:

```python
series_fit = archive_series(
    archive,
    experiment=experiment,
    s02=fit_a["params"]["s02"]["value"],
    e0_shift=fit_a["params"]["e0_shift"]["value"],
)
spec_fit = series_to_vega_lite(series_fit, view="chi_r")
```

---

## 4. Site groups and $\chi(R)$ peak attribution

`group_sites` clusters the absorber sites by the $\chi(R)$ spectrum each produces (average-linkage clustering of Re and Im $\chi(R)$ over `rrange`). No environment labels are needed.

```python
groups = group_sites(archive)  # or group_sites(archive, n_groups=2)
print("n_sites:", groups["n_sites"], "n_groups:", groups["n_groups"], "tol:", groups["tol"])
for g in groups["groups"]:
    print(g["group_id"], g["site_indices"], f"weight={g['weight']:.3f}",
          "medoid:", g["medoid_site"], f"rms={g['within_group_rms']:.4f}",
          f"dist_to_overall={g['distance_to_overall']:.4f}")
print(sorted(groups["groups"][0]["spectrum"]))
for w in groups["warnings"]:
    print("warning:", w)
```

Each group carries `site_indices`, `n_sites`, `weight` (the weights sum to 1), `medoid_site`, `within_group_rms`, `distance_to_overall` and `spectrum` (the group-mean $\chi(k)$ and $\chi(R)$). When the archive stores at least two frame averages, `group_sites` estimates a trajectory sampling-noise floor (`noise_floor`) from the frame-to-frame variance and raises the clustering cut to at least `2.0 * noise_floor` so statistically equivalent sites are not over-split by finite-frame sampling noise (emitting a warning whenever the floor overrides `tol` or when a requested `n_groups` split lies below it). Groups of fewer than two sites also trigger a warning.

`peak_attribution` finds the peaks of $|\chi(R)|$ inside `rrange` and, for each, reports how much every component builds it:

```python
def show(att, peak_index=0, n=5):
    pk = att["peaks"][peak_index]
    print(f"by={att['decomposition']}: peak R={pk['r_peak']:.2f} A, "
          f"height={pk['height']:.4f}, window={[round(x, 2) for x in pk['window']]}, "
          f"shoulder_warning={pk['shoulder_warning']}")
    for s in pk["shares"][:n]:
        r_max = "-" if s["r_max"] is None else f"{s['r_max']:.2f}"
        print(f"  {s['id']:<22} projection={s['projection']:+.3f} "
              f"loo_delta={s['loo_delta']:+.4f} r_max={r_max}")
    u = pk["unassigned"]
    print(f"  {'unassigned':<22} projection={u['projection']:+.3f}")
    total = sum(s["projection"] for s in pk["shares"]) + u["projection"]
    total += pk.get("remainder", {"projection": 0.0})["projection"]
    print(f"  sum of projections = {total:.6f}")


by_paths = peak_attribution(archive, by="paths")
by_sites = peak_attribution(archive, by="sites")
by_groups = peak_attribution(archive, by="site_groups", groups=groups)

for att in (by_paths, by_sites, by_groups):
    show(att)
    print("  unassigned norm_fraction:", round(att["unassigned"]["norm_fraction"], 4))
    for w in att["warnings"]:
        print("  warning:", w)
```

The result has `schema_version`, `units`, `ft_params`, `rrange`, `rmax_valid`, `source`, `decomposition`, `r`, `total`, `components`, `remainder`, `unassigned`, `peaks` and `warnings`. Each entry of `peaks` has `r_peak`, `window`, `height`, `shoulder_warning`, `shares` (`id`, `projection`, `loo_delta`, `r_max`), `unassigned` and, when more than `max_components` components exist, `remainder`. `components` holds the kept components' complex $\chi(R)$ as `chir_mag`, `chir_re` and `chir_im` curves on `r`.

- `by="paths"` uses the paths stored in the archive and needs format 4 or later.
- `by="sites"` uses each site's weighted contribution to the overall average.
- `by="site_groups"` sums the sites of each group. `groups` can be the output of `group_sites` or a plain list of site-index lists, such as `[[0, 1, 2], [3, 4]]`; leave it out to run `group_sites` with its defaults.
- `prominence` (default 0.05) is the minimum peak prominence as a fraction of the largest $|\chi(R)|$ in `rrange`, and `max_components` (default 20) caps the number of components reported per peak.
- Pass `s02=` and `e0_shift=` (e.g. from `fit_ensemble`) to scale and shift `total` and every component identically before transforming; signed projections remain invariant under `s02`, the sum rule still holds exactly, and the returned component curves overlay directly on the aligned MD-ensemble spectrum.

---

## 5. Static + MD $\sigma^2$ fit (`fit_static`)

`fit_static` runs a conventional Larch `feffit` of the experiment using ONE static structure whose thermal $\sigma^2$ comes from the MD trajectory. It answers how good a fit a conventional single-structure model gives. Its free parameters are $S_0^2$, $\Delta E_0$ and one isotropic $\Delta R = \alpha R_\mathrm{eff}$; degeneracies are fixed and $\sigma^2$ is fixed at the MD value.

```python
frames = read("md.traj", index=":")
config = FeffConfig(radius=4.0)
sites = "Cu.0-3"  # FEFF runs once per absorber, so use a few representative ones

# Default: the trajectory-average structure (structure=None), with the
# side-by-side comparison against MD-ensemble alignment from the archive.
fit_b = fit_static(frames, experiment, config, "Cu", sites=sites, archive_path=archive)

for name, p in fit_b["params"].items():
    print(f"{name:6s} value={p['value']:.4f} stderr={p['stderr']} vary={p['vary']}")
print("rfactor:", fit_b["rfactor"], "n_idp:", fit_b["n_idp"], "nvarys:", fit_b["nvarys"])
print("n_absorbers:", fit_b["n_absorbers"], "truncation_fraction:", fit_b["truncation_fraction"])
for p in fit_b["paths"]:
    print(f"  {p['label']:<18} nlegs={p['nlegs']} r_eff={p['r_eff']:.3f} "
          f"N={p['degeneracy']:.2f} sigma2_md={p['sigma2_md']:.4f} deltar={p['deltar']:+.4f}")
print("dropped:", [d["label"] for d in fit_b["dropped_paths"]])
for w in fit_b["warnings"]:
    print("warning:", w)
```

Pass an idealised or external structure, an ASE `Atoms` object or a structure file, to override the trajectory average. The trajectory is still needed for the MD $\sigma^2$:

```python
fit_b_ideal = fit_static(
    frames, experiment, config, "Cu",
    sites=sites,
    structure="ideal.cif",  # or structure=read("ideal.cif")
    archive_path=archive,
)

for label, fit in (("average", fit_b), ("ideal.cif", fit_b_ideal)):
    print(label, "rfactor =", round(fit["rfactor"], 4), "alpha =", fit["params"]["alpha"]["value"])
```

With `archive_path=` the result gains a `comparison` block that reports, on identical data, FT settings and window:

```python
cmp_ = fit_b["comparison"]
for row in ("fit_a", "b0", "fit_b"):
    c = cmp_[row]
    print(f"{row:6s} rfactor={c['rfactor']:.4f} rfactor_r={c['rfactor_r']:.4f} "
          f"rfactor_k={c['rfactor_k']:.4f} nvarys={c['nvarys']} params={c['params']}")
```

- `fit_a` — **MD-ensemble alignment** (`fit_ensemble` on the archive).
- `b0` — **Static baseline (MD $\sigma^2$)**: the static structure with MD $\sigma^2$, $\Delta R = 0$, and $S_0^2$ and $\Delta E_0$ fixed at the MD-ensemble values. Nothing is fitted, so comparing `fit_a` versus `b0` isolates the structural/disorder model. Its spectrum curve is available in `cmp_["b0"]["theory"]` with the same shape as `fit_b["theory"]`.
- `fit_b` — **Static + MD $\sigma^2$ fit** (the refined single-structure result).

Options worth knowing: `sites=` restricts the absorbers for both the FEFF runs and the MSRD (any `parse_site_specification` string such as `"Cu.0-3"`, or a list of indices; without it every absorber of the element is used, which is slow for a large cell); `sigma2="free"` starts $\sigma^2$ at the MD value and fits it; `deltar_mode="shells"` fits one $\Delta R$ per single-scattering shell; `msrd=(res_2b, res_3b)` reuses a precomputed `calculate_grouped_msrd` result; `workdir=` keeps the FEFF runs and reuses finished site directories; `force=True` runs even when more than two thirds of `n_idp` is free.

---

## 6. Site-population reweighting (`fit_site_group_weights`)

`fit_site_group_weights` fits the experiment as $\sum_g w_g \,\mathrm{mean}_g$ with $w_g \ge 0$ and $\sum_g w_g = 1$, starting from the MD weights, over the site groups from `group_sites`. $S_0^2$ and $\Delta E_0$ come from `fit_ensemble` unless you pass them; $\Delta E_0$ is held fixed at that value.

```python
w = fit_site_group_weights(archive, experiment, groups)

print("s02:", w["s02"], "e0_shift:", w["e0_shift"], "held fixed:", w["e0_shift_held_fixed"])
print("n_idp:", w["n_idp"], "n_groups:", w["n_groups"], "nvarys:", w["nvarys"], "success:", w["success"])
for row in w["weights"]:
    print(f"{row['group_id']}  sites={row['site_indices']}  md_weight={row['md_weight']:.3f}  "
          f"fitted_weight={row['fitted_weight']:.3f}  delta={row['delta_weight']:+.3f}")
print("sum md:", round(sum(r["md_weight"] for r in w["weights"]), 6),
      "sum fitted:", round(sum(r["fitted_weight"] for r in w["weights"]), 6))
print("rfactor_md :", w["rfactor_md"], " rfactor_fit:", w["rfactor_fit"])
print(sorted(k for k in w if k.startswith("theory")))
for msg in w["warnings"]:
    print("warning:", msg)
```

`rfactor_md` is the R-factor with the MD weights and `rfactor_fit` the R-factor with the fitted weights (`rfactor_r_*` and `rfactor_k_*` give the other spaces). `theory_md` and `theory_fit` are the two spectra on the archive's $k$ grid. A warning is added when the number of groups exceeds `n_idp / 2`: the weights are then poorly determined by the fit window. A fitted weight that differs a lot from the MD weight says the MD population of that environment disagrees with experiment; it does not by itself say why.

---

## Conventions and caveats

### Signed projection

For a peak with window $W$ (the $R$ points between the neighbouring minima of $|\chi(R)|$, capped at ±0.5 Å from the peak), total complex $\chi(R)$ equal to $T$ and component $C_j$:

$$
s_j = \frac{\sum_W \mathrm{Re}\left(T^{*} C_j\right)}{\sum_W |T|^2}
$$

All spectra are Fourier transformed with the same parameters. The transform is linear, so $T = \sum_j C_j + U$ holds exactly with $U$ = `unassigned`. Summing the numerators over $j$ and $U$ gives $\sum_W \mathrm{Re}(T^* T) = \sum_W |T|^2$, so the projections of all components plus `unassigned` sum to exactly 1. The sign carries the phase: a component in phase with the peak is positive, one in antiphase is negative, one at 90° contributes zero.

A negative projection means destructive interference: the component cancels part of the peak. Worked example, with the window shrunk to one point where $T = 1.0$ (real):

| Component | $C_j$ | $s_j = \mathrm{Re}(T^* C_j)/|T|^2$ |
|---|---|---|
| path A | $+1.0$ | $+1.0$ |
| path B | $-0.3$ | $-0.3$ |
| `unassigned` $= T - C_A - C_B = 1.0 - 1.0 + 0.3$ | $+0.3$ | $+0.3$ |

$+1.0 - 0.3 + 0.3 = 1.0$. Path A alone would build a peak of 1.0, path B removes 0.3 of it, so the two stored paths give a net 0.7 and the remaining 0.3 of the observed peak comes from paths that are not stored. Path A's share is 100 %, not 70 %: shares are not normalised to the visible components. The same arithmetic holds over a real window of many $R$ points.

```python
T = np.array([1.0 + 0j])
C = {"path A": np.array([1.0 + 0j]), "path B": np.array([-0.3 + 0j])}
C["unassigned"] = T - sum(C.values())
for name, c in C.items():
    print(f"{name:<11} projection={np.sum(np.real(np.conj(T) * c)) / np.sum(np.abs(T) ** 2):+.1f}")
for name in ("path A", "path B"):
    print(f"{name} loo_delta={np.abs(T).max() - np.abs(T - C[name]).max():+.1f}")
```

`loo_delta` is the leave-one-out change $\max_W|T| - \max_W|T - C_j|$. It is positive when removing $j$ lowers the peak (j builds it) and negative when removing $j$ raises it. In the example, removing A drops the peak by $+1.0$ and removing B raises it from 1.0 to 1.3, giving $-0.3$. It is reported next to the projection because it is a direct, non-additive statement about the peak height. `r_max` is the $R$ at which $|C_j|$ is largest inside the window, which exposes the phase-shift offset between a path's $R_\mathrm{eff}$ and where it shows up in $\chi(R)$.

No magnitude share is reported. $|C_j| / \sum_k |C_k|$ is never negative, cannot tell a path that builds the peak from one that cancels it, does not sum to 1 together with `unassigned` when components interfere, and would rank a large cancelling path as a major contributor.

### Stored paths are a subset: `unassigned`

`/top_paths` holds only the top $N$ path groups (default `top_n_paths=25`) of those that survived the shard threshold, ranked by ensemble-weighted importance. All other paths contribute to the overall average but are not stored. So in `by="paths"` the `unassigned` component is a real completeness diagnostic, not an error: it is the signal of paths that are not stored, and `unassigned["norm_fraction"]` is its norm relative to the total over `rrange`. A warning is added when it exceeds 10 %. Use `by="sites"` when you need a complete decomposition: every task belongs to exactly one site and the per-$k$ contributor counts are stored, so the weighted site spectra reproduce the overall average and `unassigned` is about zero. The same holds for `by="site_groups"` when all sites are in a group. (On legacy archives without `n_contributors`, sites are weighted equally and a warning says the components may not sum to the overall average.)

### `rmax_valid`

`rmax_valid = radius - 0.5` Å, where `radius` is the FEFF cluster radius recorded in the archive and 0.5 Å is `PHASE_SHIFT_ALLOWANCE_A`. FEFF only includes paths up to the cluster radius, so shells beyond it are truncated; because a path of length $R_\mathrm{eff}$ appears in $\chi(R)$ at roughly $R_\mathrm{eff} - 0.5$ Å, $\chi(R)$ is not reliable above `radius - 0.5`. `archive_series` reports it as `series["ft_limits"]["rmax_valid"]` and adds an `rmax_beyond_valid` entry to `ft_warnings` when the Fourier-transform `rmax` exceeds it. The fits and `group_sites` default to the window `(1.0, min(3.0, rmax_valid))` and warn if you ask for more. `fit_static` takes the radius from its `config`. An archive without a stored radius has no `rmax_valid`: it is `None`, a note is added to `warnings`, and the default window is `(1.0, 3.0)`.

### $S_0^2$ and $\Delta E_0$ are applied to the simulation only

`scaled_chi_arrays(k, chi, s02, e0_shift)` is the one place where the comparison convention lives, and it is always applied to the simulated spectrum, never to the experiment. The shifted wavenumber is

$$
k_\mathrm{shifted}^2 = k^2 + \frac{\Delta E_0}{\hbar^2 / 2 m_e}
$$

so a positive $\Delta E_0$ moves features to higher $k$. This matches `md_exafs.paths.path_chi` and Larch (`feffdat._calc_chi`), where $q = \sqrt{k^2 - \Delta E_0 \cdot \mathrm{ETOK}}$ is the wavenumber the FEFF tables are read at. `fit_ensemble`, the static baseline (`b0`), `fit_static` and `archive_series(..., e0_shift=...)` therefore share one sign, and a $\Delta E_0$ from one is valid in the others. Points below the shifted threshold are dropped. It is an approximation: it rescales a finished $\chi(k)$ and does not re-evaluate the paths. This is accepted because the stored path set is usually only a subset.

```python
k_grid = np.arange(2.0, 12.0, 1.0)
k_adj, chi_adj = scaled_chi_arrays(k_grid, np.ones_like(k_grid), s02=0.8, e0_shift=2.0)
print(np.allclose(k_adj**2, k_grid**2 + 2.0 / 3.80998, atol=1e-4), chi_adj[0])
```

### `average_structure` and single MD snapshots

By default `fit_static` uses `md_exafs.pipeline.average_structure(frames)`. It does not Kabsch-align the frames and it shortens bonds: $R(\langle x \rangle) < \langle R \rangle$ by roughly $\sigma_\perp^2 / (2R)$, where $\sigma_\perp^2$ is the mean-square displacement perpendicular to the bond. A fitted $\Delta R$ can absorb that bias (`fit_static` says so in `warnings`). `md_exafs.debye_waller.kabsch_align` is available if you need aligned frames before averaging.

Single MD snapshots (`structure="first"`, `"last"` or a frame index) are rejected with a `ValueError`. A frozen snapshot already contains one random thermal displacement, which the MD `sigma2_thermal_A2` would count a second time. Use the trajectory average (`None`) or an idealised structure, either an `Atoms` object or a file.

```python
try:
    fit_static(frames, experiment, config, "Cu", sites=sites, structure="first")
except ValueError as exc:
    print("rejected:", str(exc)[:60])
```

### Static-fit pooling and MD $\sigma^2$

FEFF runs at every selected absorber of the static structure, and paths of different absorbers are pooled by an exact complex sum rather than by averaging parameters. Members of a cluster (complete-linkage clustering, `r_tol` = 0.025 Å, so the diameter is bounded) are summed as complex amplitudes at their own lengths and divided by the number of absorbers that ran. That preserves the static dephasing between absorbers, which a static structure has no thermal disorder to hide. Each member keeps its own FEFF degeneracy; there is no factor of 2 for path reversal. Each pooled path carries its weighted $R_\mathrm{eff}$ spread as `r_eff_std`.

The MD value is the thermal-only `sigma2_thermal_A2`, count-weighted over the MD path groups matched to the pooled path. For 4-leg paths with FEFF leg geometry, true collinear rattles (`0 → A → 0 → A → 0`) receive $4\sigma^2_\mathrm{1NN}$, collinear forward-focusing paths (`0 → A → B → A → 0`) receive the 2-body $\sigma^2(0, B)$ at $R_\mathrm{eff}$, collinear absorber-crossing paths (`0 → S_1 → 0 → S_2 → 0`) receive the collinear 3-body $\sigma^2$ at $R_\mathrm{eff}$, and bent 4-leg paths are dropped. For 3-leg paths, near-collinear angles (within 10° of 0° or 180°) are snapped on both sides so thermal bending does not reject collinear multiple-scattering matches. The spread between instances is deliberately excluded, because the pooled static path already carries it. A pooled path without a matching MD group, typically a multiple-scattering path beyond the MSRD, is dropped and listed in `dropped_paths` and in `warnings` instead of getting a guessed $\sigma^2$. `truncation_fraction` compares the kept paths with the static FEFF `chi.dat` over the fit window and warns above 10 %. It describes the static run and is not the same thing as the archive's `unassigned`.

In `fit_site_group_weights`, `e0_shift` and `s02` are not fitted; `e0_shift` is held fixed at the MD-ensemble alignment value (or the one you pass) and `w["e0_shift_held_fixed"]` is `True`.

### Archive format 4

Format 4 changes what `/top_paths` stores:

```text
/meta/
    format_version  = 4
    radius          = FEFF cluster radius in Å (attribute, absent if unknown)
/top_paths/
    attrs: n_tasks                    (snapshots x absorbers behind the averages)
    path_0000/ ... path_NNNN/
        attrs: path_key, scatterer, nlegs, r_eff, degeneracy, cw_ratio, sig2,
               count                  (instances in the path group)
        chi         = float32[N]      ensemble-weighted per-instance chi_g(k)
        chir_mag    = float32[K]
```

`chi_g(k)` is the sum of the per-instance $\chi(k)$ of the group divided by the per-$k$ number of contributing tasks, so the top paths plus `unassigned` add up to the overall average. Note that `/top_paths/*/chi` is stored as `float32`. Version 2 and 3 archives remain readable by `ArchiveReader`, `archive_series`, the fits, `group_sites` and `peak_attribution(by="sites")`; `peak_attribution(by="paths")` raises a `ValueError` on anything older than version 4.

```python
from md_exafs import ArchiveReader

reader = ArchiveReader(archive)
print("format:", reader.format_version, "radius:", reader.radius, "n_tasks:", reader.n_tasks)
top = reader.get_top_paths()
print(len(top), sorted(top[0]))
```
