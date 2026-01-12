from __future__ import annotations

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "EmoLit"

# Load all TSV files
tsvs = sorted(DATA_DIR.glob("*.tsv"))
dfs = {p.stem: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False) for p in tsvs}
df_all = pd.concat(dfs.values(), ignore_index=True) if dfs else pd.DataFrame()

print(f"Loaded {len(dfs)} files from {DATA_DIR}")
print(f"Shape: {df_all.shape}")
print(df_all.head(10))

# Extract emotion columns (all columns except 'tid' and 'text')
emotion_columns = [col for col in df_all.columns if col not in ['tid', 'text']]

print(f"\nFound {len(emotion_columns)} emotion columns")

# Count labels (sum of positive values in each column)
emotion_counts = {}
for col in emotion_columns:
    # Convert to float and count values > 0.5 (handles both binary 0/1 and probability scores)
    count = (df_all[col].astype(float) > 0.5).sum()
    emotion_counts[col] = count

# Create stats dataframe
stats_df = pd.DataFrame(list(emotion_counts.items()), columns=["Emotion", "Count"])
stats_df = stats_df.sort_values("Count", ascending=False).reset_index(drop=True)

total_annotations = stats_df["Count"].sum()
stats_df["Percentage"] = (stats_df["Count"] / total_annotations) * 100
stats_df["Percentage_Str"] = stats_df["Percentage"].apply(lambda x: f"{x:.2f}%")

print(f"\nTotal Unique Sentences (Rows): {len(df_all)}")
print(f"Total Annotations (Labels):    {total_annotations}")
print("-" * 40)
print(stats_df[["Emotion", "Count", "Percentage_Str"]])

# Create mapping (sorted alphabetically)
unique_emotions = sorted(emotion_columns)

mapping_data = [
    {"ID": i + 1, "Emotion": emotion}
    for i, emotion in enumerate(unique_emotions)
]

df_mapping = pd.DataFrame(mapping_data)

labels_dir = PROJECT_ROOT / "labels"
output_path = labels_dir / "initial_EmoLit_mapping.csv"

labels_dir.mkdir(parents=True, exist_ok=True)

df_mapping.to_csv(output_path, index=False)

print(f"\nMapping saved successfully to: {output_path}")
print(df_mapping)
