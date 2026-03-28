from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "pubmed_splits_deduplicated"
OUT_DIR = PROJECT_ROOT / "pubmed_training_data"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FRACTION = 0.12

csv_files = sorted(DATA_DIR.glob("pubmed_split_*.csv"))

print(f"Sampling {SAMPLE_FRACTION*100:.0f}% from each deduplicated split (stratified by label)\n")

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    label_col = "label_str" if "label_str" in df.columns else "labels"

    sampled, _ = train_test_split(
        df,
        train_size=SAMPLE_FRACTION,
        stratify=df[label_col],
        random_state=42,
    )

    out_path = OUT_DIR / csv_path.name
    sampled.to_csv(out_path, index=False)

    # Verify distribution
    orig_dist = df[label_col].value_counts(normalize=True).sort_index()
    samp_dist = sampled[label_col].value_counts(normalize=True).sort_index()

    print(f"{csv_path.stem}: {len(df):,} -> {len(sampled):,} rows")
    print(f"  {'Label':<15} {'Original':>10} {'Sampled':>10}")
    for label in orig_dist.index:
        print(f"  {label:<15} {orig_dist[label]:>9.2%} {samp_dist[label]:>9.2%}")
    print()

print("Done.")
