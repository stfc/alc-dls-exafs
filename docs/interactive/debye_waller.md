# Debye–Waller disorder screening

This application computes EXAFS Debye–Waller parameters ($\sigma^2$), higher cumulants ($C_3, C_4$), and atomic displacement parameters ($B$-factors) directly from molecular dynamics trajectories.

The application runs entirely in your browser using WebAssembly. Uploaded trajectory files are processed client-side.

---

## Scientific background

In EXAFS analysis, thermal and structural disorder enters the fine-structure equation through the Debye–Waller factor $e^{-2 k^2 \sigma^2}$. Here, $\sigma^2$ is the **Mean Squared Relative Displacement** (MSRD) between the absorbing atom and a scattering partner:

$$\sigma^2 = \langle (R - \langle R \rangle)^2 \rangle$$

This variance differs fundamentally from the single-atom Mean Squared Displacement (MSD), as correlated vibrations between neighbouring atoms reduce relative distance fluctuations.

For asymmetric pair distributions (common at elevated temperatures or in disordered systems), the cumulant expansion corrects both the amplitude and phase:
- $C_3$: Measures distribution skewness, introducing a phase shift $\Delta \phi \approx \frac{4}{3} k^3 C_3$ that mimics a contraction of the interatomic distance if neglected.
- $C_4$: Measures kurtosis (distribution peak sharpness), contributing an amplitude damping term $\approx e^{\frac{2}{3} k^4 C_4}$.

---

## Live application

<div class="app-launcher-card">
  <div class="app-meta">
    <h4>Launch in full window</h4>
    <p>Open the Debye–Waller screening workbench in a standalone tab with maximum screen area.</p>
  </div>
  <a class="app-launch-btn" href="../apps/debye_waller/index.html" target="_blank" rel="noopener noreferrer">
    Launch Fullscreen ↗
  </a>
</div>

<div class="marimo-app-container">
  <iframe src="../apps/debye_waller/index.html" title="Debye-Waller Screening WASM App" loading="lazy"></iframe>
</div>
