# Interpretation & Troubleshooting Reference

Consult this reference when interpreting fit diagnostics, warnings, or edge cases produced by `run_comparison.py`.

## Goodness-of-fit: `rfactor` vs `chi2_reduced`
- **`rfactor`** ($\sum \mathrm{resid}^2 / \sum \mathrm{data}^2$ in fit space) is the headline comparison metric across models. `rfactor_r` and `rfactor_k` give the same metric in $R$ and $k$ space.
- **`chi2_reduced` on high-SNR data:** `fit_ensemble` and `fit_static` estimate $\varepsilon_R$ from $15\text{–}30\text{ \AA}$ via Larch's `estimate_noise` (echoed in `epsilon_r` and `epsilon_k`). On low-noise synchrotron data, SNR in the fit window is often $150\text{–}300$, so even a 0.8% R-factor systematic difference between FEFF theory and experiment yields $\chi^2_\mathrm{red} \sim 10^2\text{–}10^3$. Parameter `stderr` values are already inflated by $\sqrt{\chi^2_\mathrm{red}}$ (matching `larch.xafs.feffit`). Compare models by `rfactor` and relative $\chi^2_\mathrm{red}$ rather than treating $\chi^2_\mathrm{red}$ as a p-value.

## Model comparison: MD ensemble vs Static baseline vs Static fit
- **MD-ensemble alignment (`fit_ensemble`, JSON key `comparison["fit_a"]`):** Refines only global $S_0^2$ and $\Delta E_0$ on the MD ensemble average (2 free parameters). Tests whether the MD trajectory reproduces experiment with no structural adjustment.
- **Static baseline (MD $\sigma^2$) (JSON key `comparison["b0"]`):** Static structure + MD $\sigma^2_\mathrm{thermal}$, $\Delta R = 0$, $S_0^2$ and $\Delta E_0$ fixed at the MD-ensemble values (0 free parameters). Comparing MD ensemble vs Static baseline isolates anharmonic / full-ensemble disorder against the harmonic single-structure model.
- **Static + MD $\sigma^2$ fit (`fit_static`, JSON key `comparison["fit_b"]`):** Conventional single-structure `feffit` refining $S_0^2$, $\Delta E_0$, and isotropic $\alpha$ ($\Delta R = \alpha R_\mathrm{eff}$; 3 free parameters). Note that `average_structure` does not Kabsch-align frames and shortens bonds by $\sim \sigma_\perp^2 / (2R)$; a small negative $\alpha$ can absorb both lattice mismatch and that averaging bias.

## Static + MD $\sigma^2$ multiple-scattering paths & `truncation_fraction`
- **3-leg near-collinear paths:** Angles within $10^\circ$ of $0^\circ$ or $180^\circ$ are snapped on both the static and MD sides so thermal bending fold bias does not reject collinear MS matches.
- **4-leg geometry classification:**
  - `rattle` ($0\to A\to 0\to A\to 0$, all $\beta \approx 180^\circ$): receives $4\sigma^2_\mathrm{1NN}$.
  - `collinear_focus` ($0\to A\to B\to A\to 0$, $\beta \approx 180, 0, 180, 0$): receives 2-body $\sigma^2(0, B)$ at $R_\mathrm{eff}$.
  - `collinear_thru_abs` ($0\to S_1\to 0\to S_2\to 0$, $\beta \approx 0, 180, 0, 180$): receives collinear 3-body $\sigma^2$ at $R_\mathrm{eff}$.
  - `bent`: left unmatched and listed in `dropped_paths` (honest truncation rather than guessed $\sigma^2$).
- **`truncation_fraction`** measures the relative $\chi(R)$ difference between kept static paths and static FEFF `chi.dat` over `rrange`. Above 10% triggers a warning and indicates missing high-order MS paths in the static model.

## Peak attribution (`by="paths"`, `"sites"`, `"site_groups"`)
- **Signed projection:** $s_j = \sum_W \mathrm{Re}(T^* C_j) / \sum_W |T|^2$. Positive builds the peak; negative cancels part of it (destructive interference). Projections of all kept components + `remainder` + `unassigned` sum to `1.0`.
- **`unassigned`:** In `by="paths"`, represents signal from paths beyond the stored top $N$ (default 25); warn threshold is `norm_fraction > 0.10`. In `by="sites"` and `by="site_groups"`, `unassigned` is $\approx 0$ on format-4 archives with `n_contributors`.
- **Scaling:** `peak_attribution(..., s02=..., e0_shift=...)` scales `total` and all components identically so overlaid path curves match the MD-ensemble alignment amplitude while preserving signed projections and the sum rule.

## Site grouping (`group_sites`) & `fit_site_group_weights`
- When the archive contains $\ge 2$ frame averages, `group_sites` computes `noise_floor` (expected pairwise RMS distance between two noise-only site averages) and cuts at $\max(\mathrm{tol}\cdot\mathrm{rms\_norm},\; 2.0\cdot\mathrm{noise\_floor})$.
- On homogeneous systems (e.g. bulk FCC Cu), `group_sites` collapses to 1 group and emits a warning when the noise floor overrides `tol`. Forcing `n_groups > 1` below $2\times\text{noise\_floor}$ triggers a statistical-significance warning, and `fit_site_group_weights` on spurious groups fits sampling noise.
- On genuinely heterogeneous systems (multi-site catalysts, surfaces, amorphous/defective hosts) whose inter-site spectral differences exceed $2\times\text{noise\_floor}$, `group_sites` separates environments and `fit_site_group_weights` tests whether reweighting those populations improves agreement with experiment.

## Experimental file quirks
- **Headerless or non-standard columns:** Pass `{"path": "cuexp.txt", "labels": "chi,k"}` to Python functions or `--experimental-kwargs '{"labels": "chi,k"}'` on the CLI.
- **Multi-group Athena `.prj`:** Pass `{"path": "data.prj", "group": "<group_name>"}` (inspect groups with `md_exafs.experimental.list_athena_groups("data.prj")`).
- **Energy/mu data:** Pass `{"path": "mu.dat", "labels": "energy,mu"}` (runs Larch `pre_edge` + `autobk`).
