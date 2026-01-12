from __future__ import annotations

from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "emotion-dataset-20-emotions"
CSV_PATH = DATA_DIR / "emotion_dataset_v5_clean.csv"

try:
    df = pd.read_csv(CSV_PATH, keep_default_na=False)
except FileNotFoundError:
    print(f"Error: File not found at {CSV_PATH}")
    exit(1)

print(f"Loaded file from {CSV_PATH}")
print(f"Shape: {df.shape}")
print(df.head(20))

label_col = 'emotion'
if label_col not in df.columns:
    df.columns = df.columns.str.strip()

if label_col not in df.columns:
    print(f"Error: '{label_col}' column not found in columns: {df.columns.tolist()}")
    exit(1)

labels = df[label_col].astype(str).str.strip()

stats_df = labels.value_counts().reset_index()
stats_df.columns = ["Emotion", "Count"]

total_annotations = stats_df["Count"].sum()
stats_df["Percentage"] = (stats_df["Count"] / total_annotations) * 100
stats_df["Percentage_Str"] = stats_df["Percentage"].apply(lambda x: f"{x:.2f}%")

print(f"\nTotal Unique Sentences (Rows): {len(df)}")
print(f"Total Annotations (Labels):    {total_annotations}")
print("-" * 40)
print(stats_df[["Emotion", "Count", "Percentage_Str"]])

# Create mapping (sorted alphabetically)
unique_emotions = sorted(stats_df["Emotion"].unique())

mapping_data = [
    {"ID": i + 1, "Emotion": emotion}
    for i, emotion in enumerate(unique_emotions)
]

df_mapping = pd.DataFrame(mapping_data)

labels_dir = PROJECT_ROOT / "labels"
output_path = labels_dir / "initial_emotion_dataset_20_mapping.csv"

labels_dir.mkdir(parents=True, exist_ok=True)

df_mapping.to_csv(output_path, index=False)

print(f"\nMapping saved successfully to: {output_path}")
print(df_mapping)
