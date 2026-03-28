from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "TrainingData" / "all_4_datasets.csv"

df = pd.read_csv(CSV_PATH)
texts = df["text"].astype(str)

total_chars = texts.str.len().sum()
total_words = texts.str.split().str.len().sum()

print(f"Rows:       {len(df)}")
print(f"Characters: {total_chars:,}")
print(f"Words:      {total_words:,}")

# Token count using pythia-14m tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
all_text = texts.tolist()

total_tokens = 0
batch_size = 10000
for i in range(0, len(all_text), batch_size):
    batch = all_text[i : i + batch_size]
    encoded = tokenizer(batch, add_special_tokens=False)
    total_tokens += sum(len(ids) for ids in encoded["input_ids"])

print(f"Tokens:     {total_tokens:,} (pythia-14m tokenizer)")
