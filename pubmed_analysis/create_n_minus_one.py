from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "pubmed_training_data"

csv_files = sorted(DATA_DIR.glob("pubmed_split_*.csv"))
splits = {f.stem: pd.read_csv(f) for f in csv_files}

print(f"Loaded {len(splits)} splits\n")

for exclude in sorted(splits.keys()):
    included = [name for name in sorted(splits.keys()) if name != exclude]
    df = pd.concat([splits[name] for name in included], ignore_index=True)

    exclude_idx = exclude.replace("pubmed_split_", "")
    out_name = f"all_without_split_{exclude_idx}.csv"
    out_path = DATA_DIR / out_name

    df.to_csv(out_path, index=False)
    print(f"{out_name}: {len(df):,} rows (excluded {exclude}: {len(splits[exclude]):,} rows)")

print("\nDone.")
