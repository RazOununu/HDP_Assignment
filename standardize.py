import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "heart.csv"
OUTPUT_FILE = "heart_standardized.csv"

CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

df = pd.read_csv(INPUT_FILE)

scaler = StandardScaler()
df[CONTINUOUS_COLUMNS] = scaler.fit_transform(df[CONTINUOUS_COLUMNS])

print("Standardization summary (mean, std used per column):")
for col, mean, std in zip(CONTINUOUS_COLUMNS, scaler.mean_, scaler.scale_):
    print(f"  {col}: mean={mean:.4f}, std={std:.4f}")

print("\nPreview of standardized columns:")
print(df[CONTINUOUS_COLUMNS].head())

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved standardized dataset to {OUTPUT_FILE}")
