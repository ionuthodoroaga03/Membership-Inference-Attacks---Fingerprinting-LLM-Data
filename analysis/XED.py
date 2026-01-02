from __future__ import annotations

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)          # auto-detect width
pd.set_option("display.max_colwidth", None)  # don't truncate long strings

# analysis/XED.py -> project root is one level up from analysis/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "XED"

tsvs = sorted(DATA_DIR.glob("en-annotated.tsv"))

dfs = {p.stem: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False) for p in tsvs}
df_all = pd.concat(dfs.values(), ignore_index=True) if dfs else pd.DataFrame()

print(f"Loaded {len(dfs)} files from {DATA_DIR}")
print(df_all.shape)

print(df_all.head(20))

labels = (
    df_all.iloc[:, 1]
    .str.split(",")
    .explode()
    .str.strip()
)

stats_df = labels.value_counts().reset_index()
stats_df.columns = ["ID", "Count"]

total_annotations = stats_df["Count"].sum()
stats_df["Percentage"] = (stats_df["Count"] / total_annotations) * 100

stats_df["Percentage"] = stats_df["Percentage"].apply(lambda x: f"{x:.2f}%")

print(f"Total Unique Sentences (Rows): {len(df_all)}")
print(f"Total Annotations (Labels):    {total_annotations}")
print("-" * 40)
print(stats_df)