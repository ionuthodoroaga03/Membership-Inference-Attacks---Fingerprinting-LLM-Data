from __future__ import annotations

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", 80)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "pubmed_splits_deduplicated"

csv_files = sorted(DATA_DIR.glob("pubmed_split_*.csv"))

print(f"Found {len(csv_files)} CSV files in {DATA_DIR}\n")
print("=" * 70)

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    split_name = csv_path.stem  # e.g. pubmed_split_0

    print(f"\n{'=' * 70}")
    print(f"  DATASET (deduplicated): {split_name}")
    print(f"{'=' * 70}")

    # Basic info
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes.to_string()}")

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values:\n{missing[missing > 0].to_string()}")
    else:
        print("\nNo missing values.")

    # Verify no duplicates remain
    num_duplicates = df.duplicated(subset=["text"]).sum()
    print(f"Duplicate texts: {num_duplicates}")

    # Split distribution (train/test/val)
    if "_split" in df.columns:
        print(f"\nSplit distribution:")
        split_counts = df["_split"].value_counts()
        for s, c in split_counts.items():
            print(f"  {s}: {c} ({c / len(df) * 100:.1f}%)")

    # --- Label distribution table ---
    label_col = "label_str" if "label_str" in df.columns else "labels"

    counts = df[label_col].value_counts().sort_index()
    table = pd.DataFrame({
        "Label": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / counts.values.sum() * 100).round(2),
    })
    table["Percentage_Str"] = table["Percentage"].apply(lambda x: f"{x:.2f}%")

    print(f"\n--- Samples per Label ({split_name}) ---")
    print(f"Total samples: {len(df)}")
    print(f"Number of unique labels: {len(counts)}")
    print(table[["Label", "Count", "Percentage_Str"]].to_string(index=False))

    # Text length stats
    if "text" in df.columns:
        text_lengths = df["text"].astype(str).str.len()
        print(f"\nText length (chars): "
              f"min={text_lengths.min()}, "
              f"median={text_lengths.median():.0f}, "
              f"mean={text_lengths.mean():.0f}, "
              f"max={text_lengths.max()}")

        word_counts = df["text"].astype(str).str.split().str.len()
        print(f"Word count:          "
              f"min={word_counts.min()}, "
              f"median={word_counts.median():.0f}, "
              f"mean={word_counts.mean():.0f}, "
              f"max={word_counts.max()}")

    print()

# --- Cross-split summary ---
print("\n" + "=" * 70)
print("  CROSS-SPLIT SUMMARY (deduplicated)")
print("=" * 70)

summary_rows = []
for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    label_col = "label_str" if "label_str" in df.columns else "labels"
    counts = df[label_col].value_counts().sort_index()
    row = {"Split": csv_path.stem, "Total": len(df)}
    for label, count in counts.items():
        row[label] = count
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).fillna(0).astype({
    col: int for col in pd.DataFrame(summary_rows).columns if col not in ["Split"]
})

print(f"\n{summary_df.to_string(index=False)}")
print()
