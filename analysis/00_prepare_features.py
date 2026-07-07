"""
Step 0 — Feature Matrix Preparation
=====================================
Purpose : Convert the cleaned, standardized Heart Disease dataset into a fully
          numerical feature matrix ready for all downstream HDP analyses.

What it does:
  - Leaves the 5 continuous columns (already z-score standardized) as-is.
  - One-hot encodes the 8 categorical columns so that distance-based algorithms
    (PCA, Mahalanobis, k-NN) are not misled by arbitrary integer category codes.
  - Saves the result to heart_features.csv (296 samples × 20 features + label).

Input  : heart_standardized.csv
Output : heart_features.csv
"""
import os
import pandas as pd

# The Heart Disease dataset contains categorical features whose integer encodings
# carry no ordinal meaning. One-hot encoding converts them to binary columns so
# that distance-based algorithms (PCA, Mahalanobis, k-NN) operate on a
# geometrically meaningful feature space.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD_FILE = os.path.join(BASE_DIR, "heart_standardized.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "heart_features.csv")

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
LABEL_COL = "target"

df = pd.read_csv(STD_FILE)

df_cont = df[CONTINUOUS_COLS].copy()
df_cat = pd.get_dummies(df[CATEGORICAL_COLS], columns=CATEGORICAL_COLS, drop_first=True).astype(int)
labels = df[LABEL_COL]

df_features = pd.concat([df_cont, df_cat], axis=1)

df_out = pd.concat([df_features, labels], axis=1)
df_out.to_csv(OUTPUT_FILE, index=False)

print(f"Original features : {len(CONTINUOUS_COLS) + len(CATEGORICAL_COLS)}")
print(f"After one-hot     : {df_features.shape[1]}")
print(f"Samples           : {df_features.shape[0]}")
print(f"\nFeature columns ({df_features.shape[1]}):")
for col in df_features.columns:
    print(f"  {col}")
print(f"\nSaved to {OUTPUT_FILE}")
