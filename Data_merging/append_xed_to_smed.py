from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "merged"
OUTPUT_FILE = OUTPUT_DIR / "XED_and_SMED.csv"

XED_PATH = DATA_DIR / "XED" / "en-annotated.tsv"
SMED_PATH = DATA_DIR / "SMED" / "Social Media Emotion Dataset.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"--- STARTING MERGE (PRESERVING ALL LABELS) ---")

print(f"\n1. Loading SMED from {SMED_PATH}...")
try:
    df_smed = pd.read_csv(SMED_PATH, keep_default_na=False)

    if 'label' not in df_smed.columns:
        df_smed = df_smed.rename(columns={df_smed.columns[0]: 'text', df_smed.columns[1]: 'label'})
    else:
        if 'text' not in df_smed.columns:
            df_smed = df_smed.rename(columns={df_smed.columns[0]: 'text'})

    df_smed = df_smed[['text', 'label']].copy()

    df_smed['text'] = df_smed['text'].astype(str).str.strip()
    df_smed['label'] = df_smed['label'].astype(str).str.strip()
    df_smed['label'] = df_smed['label'].str.title()

    print(f"   Loaded {len(df_smed)} rows from SMED.")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load SMED. {e}")
    exit(1)

print(f"\n2. Loading XED from {XED_PATH}...")

xed_map = {
    '1': 'Angry',
    '2': 'Anticipation',
    '3': 'Disgust',
    '4': 'Fear',
    '5': 'Happy',
    '6': 'Sad',
    '7': 'Surprise',
    '8': 'Trust'
}

try:
    df_xed = pd.read_csv(XED_PATH, sep='\t', header=None, names=['text', 'label_ids'], dtype=str, keep_default_na=False)

    df_xed['label_ids'] = df_xed['label_ids'].str.split(',')
    df_xed = df_xed.explode('label_ids')
    df_xed['label_ids'] = df_xed['label_ids'].str.strip()

    df_xed['label'] = df_xed['label_ids'].map(xed_map)

    df_xed = df_xed.dropna(subset=['label'])

    df_xed['text'] = df_xed['text'].astype(str).str.strip()

    df_xed = df_xed[['text', 'label']]
    print(f"   Loaded {len(df_xed)} rows from XED.")

except Exception as e:
    print(f"CRITICAL ERROR: Could not load XED. {e}")
    exit(1)

print(f"\n3. Merging datasets...")

df_merged = pd.concat([df_smed, df_xed], ignore_index=True)
count_before = len(df_merged)

df_merged = df_merged.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
count_after = len(df_merged)

print(f"   Original Total: {count_before}")
print(f"   Unique Total:   {count_after}")
print(f"   Duplicates Removed: {count_before - count_after}")

df_merged.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 40)
print("FINAL LABEL DISTRIBUTION")
print("=" * 40)
print(df_merged['label'].value_counts())
print("=" * 40)
print(f"Saved merged file to: {OUTPUT_FILE}")