"""
Step 4 — Random Matrix Theory / Marchenko-Pastur Law
======================================================
Purpose : Compare the empirical eigenvalue distribution of the sample
          covariance matrix against the theoretical Marchenko-Pastur (MP)
          distribution, to identify which eigenvalues carry true signal
          and which are consistent with pure noise.

What it does:
  - Computes the sample covariance matrix of the feature matrix.
  - Computes its eigenvalues.
  - Overlays the theoretical Marchenko-Pastur density:
      p(λ) = (1/2πγσ²) * sqrt((λ_max - λ)(λ - λ_min)) / λ
      where γ = D/N (aspect ratio), σ² = 1 (standardised data)
      λ_min/max = σ²(1 ∓ sqrt(γ))²
  - Eigenvalues above λ_max are above the "noise ceiling" — they represent
    true signal components that cannot be explained by random chance.

Key insight: if most eigenvalues fall inside the MP bulk, the data has
significant noise. Eigenvalues sticking out above λ_max are the signal.

Input  : heart_features.csv
Output : rmt_spectrum.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "heart_features.csv")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(FEATURES_FILE)
X = df.drop(columns=["target"]).values
N, D = X.shape

# sample covariance matrix and its eigenvalues
cov = np.cov(X.T)
eigenvalues = np.linalg.eigvalsh(cov)
eigenvalues = np.sort(eigenvalues)[::-1]

# Marchenko-Pastur parameters
gamma = D / N          # aspect ratio
sigma2 = 1.0           # standardised data
lambda_max = sigma2 * (1 + np.sqrt(gamma)) ** 2
lambda_min = sigma2 * (1 - np.sqrt(gamma)) ** 2

print(f"N={N}, D={D}, γ=D/N={gamma:.3f}")
print(f"MP bulk: [{lambda_min:.3f}, {lambda_max:.3f}]")

n_signal = (eigenvalues > lambda_max).sum()
print(f"Eigenvalues above MP upper edge (signal): {n_signal}")
print(f"Largest eigenvalue: {eigenvalues[0]:.3f}")

# theoretical MP density
lam = np.linspace(lambda_min * 0.5, lambda_max * 1.5, 500)
lam_clipped = np.clip(lam, lambda_min + 1e-9, lambda_max - 1e-9)
mp_density = (1 / (2 * np.pi * gamma * sigma2)) * \
             np.sqrt(np.maximum((lambda_max - lam_clipped) *
                                (lam_clipped - lambda_min), 0)) / lam_clipped
mp_density[(lam < lambda_min) | (lam > lambda_max)] = 0

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(eigenvalues, bins=20, density=True, color="steelblue",
        edgecolor="black", alpha=0.7, label="Empirical eigenvalues")

ax.plot(lam, mp_density, "r-", linewidth=2,
        label="Marchenko-Pastur density (noise)")

ax.axvline(lambda_max, color="red", linestyle="--", linewidth=1.5,
           label=f"MP upper edge  λ_max = {lambda_max:.2f}")

# mark signal eigenvalues
signal_eigs = eigenvalues[eigenvalues > lambda_max]
for ev in signal_eigs:
    ax.axvline(ev, color="orange", linestyle=":", linewidth=1, alpha=0.8)
ax.axvline(signal_eigs[0], color="orange", linestyle=":",
           label=f"{n_signal} signal eigenvalue(s)")

ax.set_xlabel("Eigenvalue  λ")
ax.set_ylabel("Density")
ax.set_title("Eigenvalue Spectrum vs Marchenko-Pastur Law")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "rmt_spectrum.png"), dpi=150)
plt.close()
print("Saved: rmt_spectrum.png")
