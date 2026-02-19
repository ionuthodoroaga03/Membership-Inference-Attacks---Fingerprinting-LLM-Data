from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_FILE = DATA_DIR / "emotion-dataset-20-emotions" / "emotion_dataset_v5_clean.csv"

OUTPUT_DIR = PROJECT_ROOT / "mapped_label_data" / "9_labels"
OUTPUT_FILE = OUTPUT_DIR / "EmotionDataset20_mapped.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING = {
    # --- HAPPY Group ---
    "happiness": "Happy",
    "excitement": "Happy",
    "gratitude": "Happy",
    "love": "Happy",
    "pride": "Happy",
    "relief": "Happy",

    # --- SAD Group ---
    "sadness": "Sad",
    "disappointment": "Sad",
    "guilt": "Sad",
    "embarrassment": "Sad",
    "loneliness": "Sad",

    # --- ANGRY Group ---
    "anger": "Angry",
    "frustration": "Angry",
    "jealousy": "Angry",

    # --- TRUST Group ---
    "love": "Trust",       # love implies deep trust
    "gratitude": "Trust",  # gratitude is rooted in trust toward another

    # --- NEUTRAL Group ---
    "confusion": "Neutral",   # confusion is low-arousal, non-valenced

    # --- ANTICIPATION Group ---
    "hope": "Anticipation",
    "anxiety": "Anticipation",  # anxiety is future-directed tension

    # --- DISGUST Group ---
    "disgust": "Disgust",

    # --- FEAR Group ---
    "fear": "Fear",

    # --- SURPRISE Group ---
    "surprise": "Surprise",
}

print(f"--- STARTING 20-EMOTIONS MAPPING ---")
print(f"Reading from: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE, keep_default_na=False)
except FileNotFoundError:
    print(f"Error: File not found at {INPUT_FILE}")
    exit(1)

print(f"Original Shape: {df.shape}")

if 'sentence' in df.columns:
    df = df.rename(columns={'sentence': 'text'})

if 'text' not in df.columns:
    df = df.rename(columns={df.columns[0]: 'text'})

if 'emotion' not in df.columns:
    df = df.rename(columns={df.columns[1]: 'emotion'})

df['text'] = df['text'].astype(str).str.strip()

before = len(df)
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
after = len(df)

print(f"Removed {before - after} duplicate sentences.")

print("Mapping labels...")

df['emotion'] = df['emotion'].astype(str).str.strip().str.lower()

df['label'] = df['emotion'].map(MAPPING)

unmapped = df[df['label'].isna()]['emotion'].unique()
if len(unmapped) > 0:
    print(f"⚠️  Note: The following labels were dropped (no mapping defined): {unmapped}")

df_final = df.dropna(subset=['label']).copy()

df_final = df_final[['text', 'label']]

df_final.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 40)
print("FINAL STATISTICS")
print("=" * 40)
print(f"Total Rows: {len(df_final)}")
print("-" * 20)
print(df_final['label'].value_counts())
print("=" * 40)
print(f"Saved to: {OUTPUT_FILE}")