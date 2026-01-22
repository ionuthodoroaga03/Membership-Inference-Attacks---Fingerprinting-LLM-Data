"""
EmoLit File Merger
==================
Merges multiple EmoLit TSV files (gold.tsv, tst.tsv, trn.tsv) into a single file.

Usage:
    python merge_emolit_files.py --input_dir /path/to/EmoLit --output /path/to/emolit_merged.tsv
"""

import pandas as pd
import os
import argparse


def merge_emolit_files(input_dir: str, output_path: str, files: list = None) -> pd.DataFrame:
    """
    Merge multiple EmoLit TSV files into one.
    
    Args:
        input_dir: Directory containing the TSV files
        output_path: Path for the merged output file
        files: List of specific files to merge (default: all TSV files)
    
    Returns:
        Merged DataFrame
    """
    
    # Default files if not specified
    if files is None:
        files = ['gold.tsv', 'tst.tsv', 'val.tsv', 'trn.tsv']
    
    all_dfs = []
    
    print("="*60)
    print("MERGING EMOLIT FILES")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_path}")
    print()
    
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep='\t')
            df['split'] = filename.replace('.tsv', '')  # Add split column (gold, tst, trn)
            all_dfs.append(df)
            print(f" {filename}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"  {filename}: FILE NOT FOUND")
    
    if not all_dfs:
        print("\nERROR: No files loaded!")
        return pd.DataFrame()
    
    # Merge all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    print()
    print("="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns: {merged_df.columns.tolist()}")
    print()
    print("Split distribution:")
    print(merged_df['split'].value_counts().to_string())
    
    # Save merged file
    if output_path.endswith('.tsv'):
        merged_df.to_csv(output_path, sep='\t', index=False)
    else:
        merged_df.to_csv(output_path, index=False)
    
    print()
    print(f" Saved to: {output_path}")
    
    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge EmoLit TSV files')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Directory containing EmoLit TSV files')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file path (TSV or CSV)')
    parser.add_argument('--files', '-f', type=str, nargs='+', 
                        default=['gold.tsv', 'tst.tsv', 'val.tsv', 'trn.tsv'],
                        help='Specific files to merge (default: gold.tsv tst.tsv val.tsv)')
    
    args = parser.parse_args()
    
    merge_emolit_files(args.input_dir, args.output, args.files)