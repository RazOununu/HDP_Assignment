import pandas as pd
import matplotlib.pyplot as plt

RAW_FILE = "../heart_clean.csv"
STD_FILE = "../heart_standardized.csv"

CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

df_raw = pd.read_csv(RAW_FILE)
df_std = pd.read_csv(STD_FILE)

print(f"Dataset shape: {df_raw.shape[0]} samples, {df_raw.shape[1]} columns")
print(f"  Continuous columns ({len(CONTINUOUS_COLUMNS)}): {CONTINUOUS_COLUMNS}")
print(f"  Categorical/binary columns ({len(CATEGORICAL_COLUMNS)}): {CATEGORICAL_COLUMNS}")
print("  Label column: target")

print("\nTarget distribution:")
print(df_raw["target"].value_counts().rename({0: "healthy (0)", 1: "disease (1)"}))

print("\nContinuous columns - mean/std BEFORE standardization:")
print(df_raw[CONTINUOUS_COLUMNS].agg(["mean", "std"]).T)

print("\nContinuous columns - mean/std AFTER standardization:")
print(df_std[CONTINUOUS_COLUMNS].agg(["mean", "std"]).T)

print("\nCategorical/binary columns - value distribution:")
for col in CATEGORICAL_COLUMNS:
    print(f"  {col}:")
    print(df_raw[col].value_counts().sort_index().to_string())
    print()

# --- Plots ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, CONTINUOUS_COLUMNS):
    ax.hist(df_raw[col], bins=20, color="steelblue", edgecolor="black")
    ax.set_title(f"{col} (before standardization)")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig("continuous_before.png")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, CATEGORICAL_COLUMNS):
    df_raw[col].value_counts().sort_index().plot(kind="bar", ax=ax, color="darkorange", edgecolor="black", rot=0)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("categorical_distribution.png")

plt.figure(figsize=(5, 4))
df_raw["target"].value_counts().sort_index().rename({0: "healthy", 1: "disease"}).plot(
    kind="bar", color=["seagreen", "indianred"], edgecolor="black"
)
plt.title("Target distribution")
plt.tight_layout()
plt.savefig("target_distribution.png")

print("\nSaved plots: continuous_before.png, categorical_distribution.png, target_distribution.png")
