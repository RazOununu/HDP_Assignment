"""
Step 6 — Nonlinear Visualisations (t-SNE and UMAP)
====================================================
Purpose : Reveal nonlinear structure in the dataset using manifold learning
          methods that cannot be captured by linear PCA projections.

What it does:
  - t-SNE (perplexity = 10, 30, 50): three runs to verify robustness of
    the embedding — if structure is consistent across perplexity values,
    it reflects real data geometry rather than algorithm artefacts.
  - UMAP (n_neighbors = 15, min_dist = 0.1): a faster alternative that
    better preserves global structure than t-SNE.
  - Both methods are coloured by (a) class label and (b) age quartile.

Input  : heart_features.csv
Output : tsne.png, umap.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_FILE = os.path.join(BASE_DIR, "heart_features.csv")
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(FEATURES_FILE)
X = df.drop(columns=["target"]).values
y = df["target"].values

# age quartile (from original standardized — reconstruct approximate quartile)
age_std = X[:, 0]   # age is first column, already z-scored
age_quartile = pd.qcut(age_std, q=4, labels=["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"])

colors_label = {0: "seagreen", 1: "indianred"}
label_names  = {0: "Healthy", 1: "Disease"}
cmap_age = plt.cm.plasma

# ─────────────────────────────────────────
# 1. t-SNE
# ─────────────────────────────────────────
perplexities = [10, 30, 50]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for col, perp in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X)

    # row 0: colour by label
    for cls in [0, 1]:
        mask = y == cls
        axes[0, col].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                             c=colors_label[cls], label=label_names[cls],
                             s=18, alpha=0.7, edgecolors="k", linewidths=0.2)
    axes[0, col].set_title(f"t-SNE  perplexity={perp}\n(coloured by label)")
    axes[0, col].legend(fontsize=7)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])

    # row 1: colour by age quartile
    for q_idx, q_label in enumerate(["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"]):
        mask = age_quartile == q_label
        axes[1, col].scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                             c=[cmap_age(q_idx / 3)], label=q_label,
                             s=18, alpha=0.7, edgecolors="k", linewidths=0.2)
    axes[1, col].set_title(f"t-SNE  perplexity={perp}\n(coloured by age quartile)")
    if col == 0:
        axes[1, col].legend(fontsize=6)
    axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

plt.suptitle("t-SNE Embeddings", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tsne.png"), dpi=150)
plt.close()
print("Saved: tsne.png")

# ─────────────────────────────────────────
# 2. UMAP
# ─────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# label
for cls in [0, 1]:
    mask = y == cls
    axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                    c=colors_label[cls], label=label_names[cls],
                    s=20, alpha=0.7, edgecolors="k", linewidths=0.2)
axes[0].set_title("UMAP (coloured by label)")
axes[0].legend()
axes[0].set_xticks([]); axes[0].set_yticks([])

# age quartile
for q_idx, q_label in enumerate(["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"]):
    mask = age_quartile == q_label
    axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1],
                    c=[cmap_age(q_idx / 3)], label=q_label,
                    s=20, alpha=0.7, edgecolors="k", linewidths=0.2)
axes[1].set_title("UMAP (coloured by age quartile)")
axes[1].legend(fontsize=8)
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.suptitle("UMAP Embedding", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "umap.png"), dpi=150)
plt.close()
print("Saved: umap.png")
