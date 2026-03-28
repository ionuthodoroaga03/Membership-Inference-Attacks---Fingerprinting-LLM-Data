from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", 80)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "pubmed_training_data"

csv_files = sorted(DATA_DIR.glob("pubmed_split_*.csv"))
print(f"Merging {len(csv_files)} splits...\n")

dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

out_path = DATA_DIR / "all_pubmed_training.csv"
df.to_csv(out_path, index=False)
print(f"Saved merged file to {out_path}\n")

# --- EDA ---
print("=" * 70)
print("  EDA: all_pubmed_training.csv")
print("=" * 70)

texts = df["text"].astype(str)

total_chars = texts.str.len().sum()
total_words = texts.str.split().str.len().sum()

print(f"\nRows:       {len(df):,}")
print(f"Characters: {total_chars:,}")
print(f"Words:      {total_words:,}")

# Token count
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
all_text = texts.tolist()
total_tokens = 0
batch_size = 10000
for i in range(0, len(all_text), batch_size):
    batch = all_text[i : i + batch_size]
    encoded = tokenizer(batch, add_special_tokens=False)
    total_tokens += sum(len(ids) for ids in encoded["input_ids"])

print(f"Tokens:     {total_tokens:,} (pythia-14m tokenizer)")

# Class distribution
label_col = "label_str" if "label_str" in df.columns else "labels"
counts = df[label_col].value_counts().sort_index()
table = pd.DataFrame({
    "Label": counts.index,
    "Count": counts.values,
    "Percentage": (counts.values / counts.values.sum() * 100).round(2),
})
table["Percentage_Str"] = table["Percentage"].apply(lambda x: f"{x:.2f}%")

print(f"\n--- Class Distribution ---")
print(f"Number of unique labels: {len(counts)}")
print(table[["Label", "Count", "Percentage_Str"]].to_string(index=False))

# Duplicates
num_duplicates = df.duplicated(subset=["text"]).sum()
print(f"\nDuplicate texts: {num_duplicates}")
