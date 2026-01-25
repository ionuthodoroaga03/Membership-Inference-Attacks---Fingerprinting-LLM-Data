from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
GOEMOTIONS_DIR = DATA_DIR / "GoEmotions" / "data"

OUTPUT_DIR = PROJECT_ROOT / "mapped_label_data" / "9_labels"
OUTPUT_FILE = OUTPUT_DIR / "GoEmotions_mapped.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING = {
    "admiration": "Happy",
    "amusement": "Happy",
    "anger": "Angry",
    "annoyance": "Angry",
    "approval": "Trust",
    "caring": "Trust",
    "confusion": "Surprise",
    "curiosity": "Anticipation",
    "desire": "Anticipation",
    "disappointment": "Sad",
    "disapproval": "Disgust",
    "disgust": "Disgust",
    "embarrassment": "Sad",
    "excitement": "Happy",
    "fear": "Fear",
    "gratitude": "Happy",
    "grief": "Sad",
    "joy": "Happy",
    "love": "Happy",
    "nervousness": "Fear",
    "optimism": "Anticipation",
    "pride": "Happy",
    "realization": "Surprise",
    "relief": "Happy",
    "remorse": "Sad",
    "sadness": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral"
}

print(f"--- STARTING GOEMOTIONS MAPPING ---")

emotions_file = GOEMOTIONS_DIR / "emotions.txt"
if not emotions_file.exists():
    print(f"Error: emotions.txt not found at {emotions_file}")
    exit(1)

with open(emotions_file, 'r') as f:
    emotion_names = [line.strip() for line in f.readlines()]
print(f"Loaded {len(emotion_names)} emotion labels source list.")

tsvs = [GOEMOTIONS_DIR / f for f in ["train.tsv", "dev.tsv", "test.tsv"]]
tsvs = [p for p in tsvs if p.exists()]

dfs = []
for tsv_path in tsvs:
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["text", "emotion_ids", "comment_id"], dtype=str, keep_default_na=False)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

print(f"Loaded {len(dfs)} files.")
print(f"Original Shape: {df_all.shape}")

if 'text' in df_all.columns:
    df_all['text'] = df_all['text'].astype(str).str.strip()
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)
    after = len(df_all)
    print(f"Removed {before - after} duplicate sentences.")
else:
    print("Error: 'text' column not found!")
    exit(1)

print("Mapping labels (keeping first emotion only)...")

def get_first_emotion_name(id_str):
    try:
        first_id = int(id_str.split(',')[0])
        if 0 <= first_id < len(emotion_names):
            return emotion_names[first_id]
        return None
    except (ValueError, IndexError):
        return None

df_all['original_emotion'] = df_all['emotion_ids'].apply(get_first_emotion_name)

df_all['label'] = df_all['original_emotion'].map(MAPPING)

unmapped = df_all[df_all['label'].isna()]['original_emotion'].unique()
unmapped = [u for u in unmapped if u is not None]
if len(unmapped) > 0:
    print(f"⚠️  Note: The following labels were dropped (no mapping defined): {unmapped}")

df_final = df_all.dropna(subset=['label']).copy()

df_final = df_final[['text', 'label']]

df_final.to_csv(OUTPUT_FILE, index=False)

print("\n" + "="*40)
print("FINAL STATISTICS")
print("="*40)
print(f"Total Rows: {len(df_final)}")
print("-" * 20)
print(df_final['label'].value_counts())
print("="*40)
print(f"Saved to: {OUTPUT_FILE}")