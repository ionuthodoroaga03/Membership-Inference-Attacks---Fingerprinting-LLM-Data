from __future__ import annotations

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "GoEmotions" / "data"

# Load the emotions list
emotions_file = DATA_DIR / "emotions.txt"
with open(emotions_file, 'r') as f:
    emotion_names = [line.strip() for line in f.readlines()]

print(f"Loaded {len(emotion_names)} emotions from {emotions_file}")

# Load all TSV files
tsvs = [DATA_DIR / f for f in ["train.tsv", "dev.tsv", "test.tsv"]]
tsvs = [p for p in tsvs if p.exists()]

dfs = []
for tsv_path in tsvs:
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["text", "emotion_ids", "comment_id"], dtype=str, keep_default_na=False)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print(f"Loaded {len(dfs)} files from {DATA_DIR}")
print(f"Shape: {df_all.shape}")
print(df_all.head(10))

# Extract emotion IDs (split by comma, expand, and count)
emotion_ids = (
    df_all["emotion_ids"]
    .str.split(",")
    .explode()
    .str.strip()
    .astype(int)
)

stats_df = emotion_ids.value_counts().reset_index()
stats_df.columns = ["EmotionID", "Count"]
stats_df = stats_df.sort_values("EmotionID").reset_index(drop=True)

# Map emotion IDs to names
stats_df["Emotion"] = stats_df["EmotionID"].apply(lambda x: emotion_names[x] if x < len(emotion_names) else f"Unknown_{x}")

total_annotations = stats_df["Count"].sum()
stats_df["Percentage"] = (stats_df["Count"] / total_annotations) * 100
stats_df["Percentage_Str"] = stats_df["Percentage"].apply(lambda x: f"{x:.2f}%")

print(f"\nTotal Unique Sentences (Rows): {len(df_all)}")
print(f"Total Annotations (Labels):    {total_annotations}")
print("-" * 40)
print(stats_df[["EmotionID", "Emotion", "Count", "Percentage_Str"]])

# Create mapping
mapping_data = [
    {"ID": i + 1, "Emotion": emotion}
    for i, emotion in enumerate(emotion_names)
]

df_mapping = pd.DataFrame(mapping_data)

labels_dir = PROJECT_ROOT / "labels"
output_path = labels_dir / "initial_GoEmotions_mapping.csv"

labels_dir.mkdir(parents=True, exist_ok=True)

df_mapping.to_csv(output_path, index=False)

print(f"\nMapping saved successfully to: {output_path}")
print(df_mapping)
