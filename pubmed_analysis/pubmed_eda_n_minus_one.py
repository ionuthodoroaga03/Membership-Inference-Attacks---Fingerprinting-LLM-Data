from __future__ import annotations

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", 80)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "pubmed_training_data"

csv_files = sorted(DATA_DIR.glob("all_without_split_*.csv"))

print(f"Found {len(csv_files)} n-1 combination files\n")
print("=" * 70)

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    name = csv_path.stem

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    print(f"\nRows: {len(df):,}")

    label_col = "label_str" if "label_str" in df.columns else "labels"
    counts = df[label_col].value_counts().sort_index()
    table = pd.DataFrame({
        "Label": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / counts.values.sum() * 100).round(2),
    })
    table["Percentage_Str"] = table["Percentage"].apply(lambda x: f"{x:.2f}%")

    print(f"\n--- Samples per Label ---")
    print(f"Number of unique labels: {len(counts)}")
    print(table[["Label", "Count", "Percentage_Str"]].to_string(index=False))
    print()

# --- Cross-file summary ---
print("\n" + "=" * 70)
print("  CROSS-FILE SUMMARY (n-1 combinations)")
print("=" * 70)

summary_rows = []
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    label_col = "label_str" if "label_str" in df.columns else "labels"
    counts = df[label_col].value_counts().sort_index()
    row = {"Dataset": csv_path.stem, "Total": len(df)}
    for label, count in counts.items():
        row[label] = count
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).fillna(0).astype({
    col: int for col in pd.DataFrame(summary_rows).columns if col not in ["Dataset"]
})

print(f"\n{summary_df.to_string(index=False)}")
print()
