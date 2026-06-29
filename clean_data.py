import pandas as pd

INPUT_FILE = "heart.csv"
OUTPUT_FILE = "heart_clean.csv"

df = pd.read_csv(INPUT_FILE)

faulty_mask = (df["ca"] == 4) | (df["thal"] == 0)
print(f"Dropping {faulty_mask.sum()} faulty rows (ca=4 or thal=0):")
print(df[faulty_mask])

df_clean = df[~faulty_mask].reset_index(drop=True)

print(f"\nRows before: {len(df)}, rows after: {len(df_clean)}")

df_clean.to_csv(OUTPUT_FILE, index=False)
print(f"Saved cleaned dataset to {OUTPUT_FILE}")
