"""
GoEmotions File Merger
======================
Merges multiple GoEmotions CSV files into a single file.

Usage:
    python merge_goemotions_files.py --input_dir /path/to/GoEmotions --output /path/to/goemotions_merged.csv
"""

import pandas as pd
import os
import argparse


def merge_goemotions_files(input_dir: str, output_path: str, files: list = None) -> pd.DataFrame:
    """
    Merge multiple GoEmotions CSV files into one.
    
    Args:
        input_dir: Directory containing the CSV files
        output_path: Path for the merged output file
        files: List of specific files to merge (default: goemotions_1.csv, goemotions_2.csv, goemotions_3.csv)
    
    Returns:
        Merged DataFrame
    """
    
    # Default files if not specified
    if files is None:
        files = ['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv']
    
    all_dfs = []
    
    print("="*60)
    print("MERGING GOEMOTIONS FILES")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_path}")
    print()
    
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['_source_file'] = filename  # Track which file it came from
            all_dfs.append(df)
            print(f" {filename}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f" {filename}: FILE NOT FOUND")
    
    if not all_dfs:
        print("\nERROR: No files loaded!")
        return pd.DataFrame()
    
    # Merge all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    
    # # Remove duplicate rows if any (based on text column)
    # if 'text' in merged_df.columns:
    #     original_len = len(merged_df)
    #     merged_df = merged_df.drop_duplicates(subset=['text'], keep='first')
    #     if len(merged_df) < original_len:
    #         print(f"\n  Removed {original_len - len(merged_df)} duplicate rows")
    
    print()
    print("="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Total rows: {len(merged_df)}")
    print(f"Columns ({len(merged_df.columns)}): {merged_df.columns.tolist()[:15]}{'...' if len(merged_df.columns) > 15 else ''}")
    
    # Save merged file
    merged_df.to_csv(output_path, index=False)
    
    print()
    print(f"✓ Saved to: {output_path}")
    
    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge GoEmotions CSV files')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Directory containing GoEmotions CSV files')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file path (CSV)')
    parser.add_argument('--files', '-f', type=str, nargs='+', 
                        default=['goemotions_1.csv', 'goemotions_2.csv', 'goemotions_3.csv'],
                        help='Specific files to merge (default: goemotions_1.csv goemotions_2.csv goemotions_3.csv)')
    
    args = parser.parse_args()
    
    merge_goemotions_files(args.input_dir, args.output, args.files)