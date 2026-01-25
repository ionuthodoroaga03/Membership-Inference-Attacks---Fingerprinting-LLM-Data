from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "mapped_label_data" / "9_labels"
OUTPUT_DIR = PROJECT_ROOT / "TrainingData"
OUTPUT_FILE = OUTPUT_DIR / "all_4_datasets.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"--- CREATING FINAL TRAINING DATASET ---")
print(f"Input Directory: {INPUT_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

csv_files = [
    INPUT_DIR / "EmoLit_mapped.csv",
    INPUT_DIR / "EmotionDataset20_mapped.csv",
    INPUT_DIR / "GoEmotions_mapped.csv",
    INPUT_DIR / "XED_and_SMED.csv"
]

dfs = []
for p in csv_files:
    if p.exists():
        print(f"\nLoading {p.name}...")
        try:
            # Load
            df = pd.read_csv(p, keep_default_na=False)

            # Basic validation
            if 'text' in df.columns and 'label' in df.columns:
                # Add source column for tracking (optional, helpful for debugging)
                df['source_file'] = p.name
                dfs.append(df)
                print(f"   -> Loaded {len(df)} rows.")
            else:
                print(f"   -> SKIPPING: Missing 'text' or 'label' columns.")

        except Exception as e:
            print(f"   -> ERROR: {e}")
    else:
        print(f"\nWarning: File not found: {p.name}")

if not dfs:
    print("No data loaded. Exiting.")
    exit(1)

print(f"\nMerging {len(dfs)} datasets...")
df_master = pd.concat(dfs, ignore_index=True)

print(f"Total Rows (Combined): {len(df_master)}")

# Ensure text is clean
df_master['text'] = df_master['text'].astype(str).str.strip()

print("Removing duplicates across all datasets...")
before = len(df_master)

df_master = df_master.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

after = len(df_master)
print(f"Removed {before - after} cross-dataset duplicates.")

df_final = df_master[['text', 'label']].copy()

df_final.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 40)
print("FINAL DATASET STATISTICS")
print("=" * 40)
print(f"Final Count: {len(df_final)}")
print("-" * 20)
print(df_final['label'].value_counts())
print("-" * 20)
print(f"Saved to: {OUTPUT_FILE}")