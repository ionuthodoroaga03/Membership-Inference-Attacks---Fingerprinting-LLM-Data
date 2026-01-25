from __future__ import annotations
from pathlib import Path
import pandas as pd

# Set display options to see full output
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

print(f"Checking uniqueness across datasets in: {DATA_DIR}\n")

# Store all text data here
# Format: {'dataset_name': pd.Series(texts)}
all_texts = {}

# ==========================================
# 1. Load GoEmotions
# ==========================================
try:
    ge_dir = DATA_DIR / "GoEmotions" / "data"
    ge_files = [ge_dir / f for f in ["train.tsv", "dev.tsv", "test.tsv"]]
    ge_dfs = []
    for p in ge_files:
        if p.exists():
            # GoEmotions has no header; col 0 is text
            ge_dfs.append(pd.read_csv(p, sep="\t", header=None, usecols=[0], names=["text"]))

    if ge_dfs:
        ge_text = pd.concat(ge_dfs, ignore_index=True)["text"]
        all_texts["GoEmotions"] = ge_text
        print(f"Loaded GoEmotions: {len(ge_text)} rows")
    else:
        print("Warning: GoEmotions files not found.")
except Exception as e:
    print(f"Error loading GoEmotions: {e}")

# ==========================================
# 2. Load Emotion Dataset 20 (Twitter)
# ==========================================
try:
    ed20_path = DATA_DIR / "emotion-dataset-20-emotions" / "emotion_dataset_v5_clean.csv"
    if ed20_path.exists():
        df_ed20 = pd.read_csv(ed20_path, keep_default_na=False)
        # Attempt to find text column, otherwise use index 0
        if 'text' in df_ed20.columns:
            ed20_text = df_ed20['text']
        else:
            ed20_text = df_ed20.iloc[:, 0]

        all_texts["EmotionDataset20"] = ed20_text
        print(f"Loaded EmotionDataset20: {len(ed20_text)} rows")
    else:
        print(f"Warning: {ed20_path} not found.")
except Exception as e:
    print(f"Error loading EmotionDataset20: {e}")

# ==========================================
# 3. Load EmoLit
# ==========================================
try:
    el_dir = DATA_DIR / "EmoLit"
    el_files = sorted(el_dir.glob("*.tsv"))
    el_dfs = []
    for p in el_files:
        # EmoLit has a 'text' column
        df = pd.read_csv(p, sep="\t", keep_default_na=False, usecols=['text'])
        el_dfs.append(df)

    if el_dfs:
        el_text = pd.concat(el_dfs, ignore_index=True)["text"]
        all_texts["EmoLit"] = el_text
        print(f"Loaded EmoLit: {len(el_text)} rows")
    else:
        print("Warning: EmoLit files not found.")
except Exception as e:
    print(f"Error loading EmoLit: {e}")

# ==========================================
# 4. Load SMED
# ==========================================
try:
    smed_path = DATA_DIR / "SMED" / "Social Media Emotion Dataset.csv"
    if smed_path.exists():
        df_smed = pd.read_csv(smed_path, keep_default_na=False)
        # SMED usually has text in the first column
        smed_text = df_smed.iloc[:, 0]
        all_texts["SMED"] = smed_text
        print(f"Loaded SMED: {len(smed_text)} rows")
    else:
        print(f"Warning: {smed_path} not found.")
except Exception as e:
    print(f"Error loading SMED: {e}")

# ==========================================
# 5. Load XED
# ==========================================
try:
    xed_path = DATA_DIR / "XED" / "en-annotated.tsv"
    if xed_path.exists():
        # XED usually has text in column 0, labels in column 1
        df_xed = pd.read_csv(xed_path, sep="\t", header=None, usecols=[0], names=["text"])
        all_texts["XED"] = df_xed["text"]
        print(f"Loaded XED: {len(df_xed)} rows")
    else:
        print(f"Warning: {xed_path} not found.")
except Exception as e:
    print(f"Error loading XED: {e}")

# ==========================================
# ANALYSIS
# ==========================================

print("-" * 50)
print("ANALYSIS RESULTS")
print("-" * 50)

# Combine into one DataFrame with source tracking
dfs_with_source = []
for name, series in all_texts.items():
    temp_df = pd.DataFrame({"text": series})
    # STRIP WHITESPACE (Critical for uniqueness)
    temp_df["text"] = temp_df["text"].astype(str).str.strip()
    temp_df["source"] = name
    dfs_with_source.append(temp_df)

if not dfs_with_source:
    print("No data loaded. Check paths.")
    exit()

master_df = pd.concat(dfs_with_source, ignore_index=True)

# 1. Total Rows vs Unique Sentences
total_rows = len(master_df)
unique_sentences = master_df["text"].nunique()
duplicates_count = total_rows - unique_sentences

print(f"Total Combined Rows:      {total_rows:,}")
print(f"Unique Sentences:         {unique_sentences:,}")
print(f"Total Duplicates:         {duplicates_count:,} ({(duplicates_count / total_rows) * 100:.2f}%)")

# 2. Check for Cross-Dataset Duplicates
# We find texts that appear in more than 1 distinct source
duplicates = master_df[master_df.duplicated("text", keep=False)]

if not duplicates.empty:
    # Group by text and list the sources
    cross_overlaps = duplicates.groupby("text")["source"].unique()

    # Filter for texts that have more than 1 source (intersection between datasets)
    intersections = cross_overlaps[cross_overlaps.apply(lambda x: len(x) > 1)]

    print("-" * 30)
    print(f"INTERSECTIONS (Same sentence in multiple datasets): {len(intersections)}")

    if len(intersections) > 0:
        print("Examples of overlaps:")
        print(intersections.head(10))
    else:
        print("Great! No sentences are shared between different datasets.")

    print("-" * 30)
    print("Internal Duplicates (within same dataset):")
    # Check for duplicates where the source is the same
    internal_dupes = duplicates.groupby("text")["source"].apply(list)
    internal_counts = internal_dupes.apply(lambda x: len(x) - len(set(x))).sum()
    print(f"Total internal redundancies found: {internal_counts}")

else:
    print("\n Amazing! Every single sentence across all datasets is unique.")


# This keeps the first occurrence and removes all subsequent duplicates
df_clean = master_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

print("FINAL CLEAN DATASET")
print(f"Original Row Count: {len(master_df)}")
print(f"Clean Row Count:    {len(df_clean)}")
print(f"Removed:            {len(master_df) - len(df_clean)} rows")

# Optional: Save it to a file if you want
df_clean.to_csv(DATA_DIR / "combined_unique_dataset.csv", index=False) #427290 unique sentences