from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "pubmed_dataset_splits"
OUT_DIR = PROJECT_ROOT / "pubmed_splits_deduplicated"

OUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(DATA_DIR.glob("pubmed_split_*.csv"))

print(f"Found {len(csv_files)} CSV files in {DATA_DIR}\n")

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    before = len(df)
    df_dedup = df.drop_duplicates(subset=["text"], keep="first")
    after = len(df_dedup)
    removed = before - after

    out_path = OUT_DIR / csv_path.name
    df_dedup.to_csv(out_path, index=False)

    print(f"{csv_path.stem}: {before} -> {after} (removed {removed} duplicates) -> saved to {out_path.name}")

print("\nDone.")
