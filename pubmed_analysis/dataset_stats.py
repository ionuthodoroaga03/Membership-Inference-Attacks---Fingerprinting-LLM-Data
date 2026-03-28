from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "pubmed_splits_deduplicated"

csv_files = sorted(DATA_DIR.glob("pubmed_split_*.csv"))

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")

grand_rows = 0
grand_chars = 0
grand_words = 0
grand_tokens = 0

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str)

    n_rows = len(df)
    n_chars = texts.str.len().sum()
    n_words = texts.str.split().str.len().sum()

    n_tokens = 0
    all_text = texts.tolist()
    batch_size = 10000
    for i in range(0, len(all_text), batch_size):
        batch = all_text[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)
        n_tokens += sum(len(ids) for ids in encoded["input_ids"])

    print(f"{csv_path.stem}: rows={n_rows:,}  chars={n_chars:,}  words={n_words:,}  tokens={n_tokens:,}")

    grand_rows += n_rows
    grand_chars += n_chars
    grand_words += n_words
    grand_tokens += n_tokens

print(f"\n{'=' * 70}")
print(f"TOTAL across all 5 splits:")
print(f"  Rows:       {grand_rows:,}")
print(f"  Characters: {grand_chars:,}")
print(f"  Words:      {grand_words:,}")
print(f"  Tokens:     {grand_tokens:,} (pythia-14m tokenizer)")
