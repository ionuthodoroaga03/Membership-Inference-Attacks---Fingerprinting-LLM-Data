from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EMOLIT_DIR = DATA_DIR / "EmoLit"

OUTPUT_DIR = PROJECT_ROOT / "mapped_label_data" / "9_labels"
OUTPUT_FILE = OUTPUT_DIR / "EmoLit_mapped.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING = {
    "joy": "Happy",
    "amusement": "Happy",
    "excitement": "Happy",
    "gratitude": "Happy",
    "love": "Happy",
    "pride": "Happy",
    "relief": "Happy",
    "admiration": "Happy",
    "nostalgia": "Sad",         # Kept your change
    "sadness": "Sad",
    "despair": "Sad",
    "disappointment": "Sad",
    "grief": "Sad",
    "guilt": "Sad",
    "embarrassment": "Sad",
    "pain": "Sad",
    "anger": "Angry",
    "annoyance": "Angry",
    "frustration": "Angry",
    "envy": "Angry",
    "indifference": "Neutral",
    "boredom": "Neutral",
    "calmness": "Neutral",
    "fear": "Fear",
    "nervousness": "Fear",
    "doubt": "Fear",
    "disgust": "Disgust",
    "disapproval": "Disgust",
    "surprise": "Surprise",
    "curiosity": "Anticipation",
    "desire": "Anticipation",
    "optimism": "Anticipation",
    "greed": "Anticipation",
    "trust": "Trust",
    "faith": "Trust",
    "approval": "Trust",
    "caring": "Trust",
    "courage": "Trust",
}

print(f"--- STARTING EMOLIT MAPPING (MAX VALUE LOGIC) ---")

tsvs = sorted(EMOLIT_DIR.glob("*.tsv"))
dfs = {p.stem: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False) for p in tsvs}
df_all = pd.concat(dfs.values(), ignore_index=True) if dfs else pd.DataFrame()

print(f"Loaded {len(dfs)} files.")
print(f"Original Shape: {df_all.shape}")

# 1. Deduplicate text first
if 'text' in df_all.columns:
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=['text']).reset_index(drop=True)
    after = len(df_all)
    print(f"Removed {before - after} duplicate sentences.")
else:
    print("Error: 'text' column not found!")
    exit(1)

print("Mapping labels...")

# 2. Identify and Clean Emotion Columns
exclude_cols = ['tid', 'text', 'split']
emotion_cols = [c for c in df_all.columns if c not in exclude_cols]

# Convert all emotion columns to numeric
for col in emotion_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)

# 3. Find the Winner (Highest Value)
# idxmax returns the name of the column with the highest score for each row
df_all['winning_original_label'] = df_all[emotion_cols].idxmax(axis=1)

# 4. Map the Winner to 9 Labels
df_all['label'] = df_all['winning_original_label'].map(MAPPING)

# 5. Clean Final DataFrame
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