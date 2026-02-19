"""
Create leave-one-out (n-1) dataset combinations.

Reads the 4 mapped datasets from mapped_label_data/9_labels/ and creates
4 CSVs in mapped_label_data/n_minus_one/, each combining 3 of the 4 datasets.
"""

import os
import pandas as pd

# Paths relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
INPUT_DIR = os.path.join(PROJECT_ROOT, "mapped_label_data", "9_labels")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "mapped_label_data", "n_minus_one")

# The 4 datasets
DATASETS = {
    "EmoLit": "EmoLit_mapped.csv",
    "EmotionDataset20": "EmotionDataset20_mapped.csv",
    "GoEmotions": "GoEmotions_mapped.csv",
    "XED_SMED": "XED_and_SMED.csv",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all datasets
    all_dfs = {}
    for name, filename in DATASETS.items():
        path = os.path.join(INPUT_DIR, filename)
        df = pd.read_csv(path)
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str)
        all_dfs[name] = df
        print(f"Loaded {name}: {len(df):,} rows")

    # Create leave-one-out combinations
    dataset_names = list(DATASETS.keys())

    for exclude_name in dataset_names:
        include_names = [n for n in dataset_names if n != exclude_name]
        combined = pd.concat(
            [all_dfs[n] for n in include_names], ignore_index=True
        )

        # Drop duplicate texts (keep first occurrence)
        before = len(combined)
        combined = combined.drop_duplicates(subset=["text"], keep="first")
        after = len(combined)

        output_path = os.path.join(OUTPUT_DIR, f"no_{exclude_name}.csv")
        combined.to_csv(output_path, index=False)

        print(
            f"\nCreated: no_{exclude_name}.csv"
            f"\n  Includes: {', '.join(include_names)}"
            f"\n  Rows: {after:,} (dropped {before - after:,} duplicates)"
        )

    print(f"\nAll files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
