"""
Emotion Dataset Merger for Pythia Fine-tuning
==============================================
Merges 5 emotion datasets into a unified single-label format.

Target labels: XED + SMED combined = 
    anger, anticipation, disgust, fear, joy, sadness, surprise, trust, neutral, happy

Author: Created for Andrada's emotion analysis project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# TARGET LABEL SCHEMA (XED + SMED combined)
# =============================================================================

# XED labels 
XED_LABELS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

# SMED labels (normalized to lowercase)
# 'Happy' maps to 'joy', 'Sad' maps to 'sadness', 'Angry' maps to 'anger'
# 'Neutral' and 'Surprise' are kept
# After mapping, SMED adds: 'neutral' 
SMED_UNIQUE_ADDITIONS = ['neutral']  # 'happy' -> 'joy', 'sad' -> 'sadness', 'angry' -> 'anger'

# Final combined target labels
TARGET_LABELS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 
                 'sadness', 'surprise', 'trust', 'neutral']

print("="*70)
print("TARGET LABEL SCHEMA (XED + SMED combined)")
print("="*70)
print(f"Labels: {TARGET_LABELS}")
print(f"Total: {len(TARGET_LABELS)} labels")
print("="*70)


# =============================================================================
# LABEL MAPPINGS: Other datasets -> Target schema
# =============================================================================

def get_emolit_to_target_mapping() -> Dict[str, str]:
    """
    Map EmoLit's 38 labels to target schema (single best match).
    """
    return {
        'admiration': 'joy',
        'amusement': 'joy',
        'anger': 'anger',
        'annoyance': 'anger',
        'approval': 'trust',
        'boredom': 'neutral',
        'calmness': 'neutral',
        'caring': 'trust',
        'courage': 'trust',
        'curiosity': 'anticipation',
        'desire': 'anticipation',
        'despair': 'sadness',
        'disappointment': 'sadness',
        'disapproval': 'disgust',
        'disgust': 'disgust',
        'doubt': 'fear',
        'embarrassment': 'fear',
        'envy': 'anger',
        'excitement': 'joy',
        'faith': 'trust',
        'fear': 'fear',
        'frustration': 'anger',
        'gratitude': 'joy',
        'greed': 'anticipation',
        'grief': 'sadness',
        'guilt': 'sadness',
        'indifference': 'neutral',
        'joy': 'joy',
        'love': 'joy',
        'nervousness': 'fear',
        'nostalgia': 'sadness',
        'optimism': 'anticipation',
        'pain': 'sadness',
        'pride': 'joy',
        'relief': 'joy',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'trust': 'trust',
    }


def get_emotion_dataset_20_to_target_mapping() -> Dict[str, str]:
    """
    Map emotion_dataset_20's 20 labels to target schema.
    """
    return {
        'anger': 'anger',
        'anxiety': 'fear',
        'confusion': 'surprise', 
        'disappointment': 'sadness',
        'disgust': 'disgust',
        'embarrassment': 'fear',
        'excitement': 'joy',
        'fear': 'fear',
        'frustration': 'anger',
        'gratitude': 'joy',
        'guilt': 'sadness',
        'happiness': 'joy',
        'hope': 'anticipation',
        'jealousy': 'anger',
        'loneliness': 'sadness',
        'love': 'joy',
        'pride': 'joy',
        'relief': 'joy',
        'sadness': 'sadness',
        'surprise': 'surprise',
    }


def get_goemotions_to_target_mapping() -> Dict[str, str]:
    """
    Map GoEmotions' 28 labels to target schema.
    """
    return {
        'admiration': 'joy',
        'amusement': 'joy',
        'anger': 'anger',
        'annoyance': 'anger',
        'approval': 'trust',
        'caring': 'trust',
        'confusion': 'surprise',
        'curiosity': 'anticipation',
        'desire': 'anticipation',
        'disappointment': 'sadness',
        'disapproval': 'disgust',
        'disgust': 'disgust',
        'embarrassment': 'fear',
        'excitement': 'joy',
        'fear': 'fear',
        'gratitude': 'joy',
        'grief': 'sadness',
        'joy': 'joy',
        'love': 'joy',
        'nervousness': 'fear',
        'optimism': 'anticipation',
        'pride': 'joy',
        'realization': 'surprise',
        'relief': 'joy',
        'remorse': 'sadness',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'neutral': 'neutral',
    }


def get_smed_to_target_mapping() -> Dict[str, str]:
    """
    Map SMED's 5 labels to target schema.
    """
    return {
        'angry': 'anger',
        'happy': 'joy',
        'neutral': 'neutral',
        'sad': 'sadness',
        'surprise': 'surprise',
    }


def get_xed_to_target_mapping() -> Dict[str, str]:
    """
    Map XED's labels to target schema (mostly 1:1).
    """
    return {
        'anger': 'anger',
        'anticipation': 'anticipation',
        'disgust': 'disgust',
        'fear': 'fear',
        'joy': 'joy',
        'sadness': 'sadness',
        'surprise': 'surprise',
        'trust': 'trust',
    }


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_xed_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load XED dataset from HuggingFace or local file.
    XED is multi-label with labels as list of integers.
    
    Args:
        filepath: Path to local TSV file, or None to load from HuggingFace
    """
    try:
        if filepath:
            df = pd.read_csv(filepath, sep='\t')
            print(f"  Loaded from TSV. Columns: {df.columns.tolist()}")
        else:
            from datasets import load_dataset
            dataset = load_dataset("Helsinki-NLP/xed_en_fi", "en_annotated")
            df = pd.DataFrame(dataset['train'])
        
        # XED label mapping: 1=anger, 2=anticipation, 3=disgust, 4=fear, 
        # 5=joy, 6=sadness, 7=surprise, 8=trust
        xed_idx_to_label = {
            1: 'anger', 2: 'anticipation', 3: 'disgust', 4: 'fear',
            5: 'joy', 6: 'sadness', 7: 'surprise', 8: 'trust'
        }
        
        labels_col = None
        for col in ['labels', 'label', 'emotions', 'emotion']:
            if col in df.columns:
                labels_col = col
                break
        
        # Find the text column
        text_col = None
        for col in ['sentence', 'text', 'content']:
            if col in df.columns:
                text_col = col
                break
        
        if labels_col is None or text_col is None:
            print(f"  Warning: Could not identify columns. Found: {df.columns.tolist()}")
            text_col = df.columns[0]
            labels_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        # Convert label indices to label names
        def convert_labels(label_val):
            # Handle different formats: list, string representation of list, comma-separated, etc.
            if isinstance(label_val, (list, np.ndarray)):
                labels = label_val
            elif isinstance(label_val, str):
                # Try to parse string representation
                label_val = label_val.strip()
                if label_val.startswith('[') and label_val.endswith(']'):
                    # String like "[1, 2, 3]"
                    import ast
                    try:
                        labels = ast.literal_eval(label_val)
                    except:
                        labels = []
                elif ',' in label_val:
                    # Comma-separated: "1,2,3"
                    labels = [int(x.strip()) for x in label_val.split(',') if x.strip().isdigit()]
                else:
                    # Single value
                    labels = [int(label_val)] if label_val.isdigit() else []
            elif isinstance(label_val, (int, float)):
                labels = [int(label_val)]
            else:
                labels = []
            
            return [xed_idx_to_label.get(l, None) for l in labels if l in xed_idx_to_label]
        
        df['labels_list'] = df[labels_col].apply(convert_labels)
        df['text'] = df[text_col]
        df['source'] = 'XED'
        
        return df[['text', 'labels_list', 'source']]
    
    except Exception as e:
        print(f"Error loading XED: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_smed_dataset(filepath: str) -> pd.DataFrame:
    """
    Load SMED dataset from local CSV or TSV file.
    SMED is single-label.
    """
    try:
        
        if filepath.endswith('.tsv'):
            df = pd.read_csv(filepath, sep='\t')
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            try:
                df = pd.read_csv(filepath)
                if len(df.columns) == 1:  
                    df = pd.read_csv(filepath, sep='\t')
            except:
                df = pd.read_csv(filepath, sep='\t')
        
        print(f"  Loaded SMED. Columns: {df.columns.tolist()}")
        
        # Identify text and label columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            if 'text' in col.lower() or 'sentence' in col.lower() or 'content' in col.lower():
                text_col = col
            if 'label' in col.lower() or 'emotion' in col.lower() or 'class' in col.lower():
                label_col = col
        
        if text_col is None:
            text_col = df.columns[0]
        if label_col is None:
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        df['text'] = df[text_col]
        df['original_label'] = df[label_col].str.lower().str.strip()
        df['source'] = 'SMED'
        
        # Map to target labels
        mapping = get_smed_to_target_mapping()
        df['label'] = df['original_label'].map(mapping)
        
        # Drop unmapped labels
        df = df[df['label'].notna()]
        
        return df[['text', 'label', 'original_label', 'source']]
    
    except Exception as e:
        print(f"Error loading SMED: {e}")
        return pd.DataFrame()


def load_emolit_dataset(filepath: str) -> pd.DataFrame:
    """
    Load EmoLit dataset from local file.
    EmoLit is multi-label with emotion scores.
    
    Args:
        filepath: Path to local file (TSV, CSV, parquet, or JSON)
    """
    try:
        # Detect file format and load accordingly
        if filepath.endswith('.tsv'):
            df = pd.read_csv(filepath, sep='\t')
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.endswith('.json') or filepath.endswith('.jsonl'):
            df = pd.read_json(filepath, lines=filepath.endswith('.jsonl'))
        else:
            try:
                df = pd.read_csv(filepath, sep='\t')
                if len(df.columns) == 1: 
                    df = pd.read_csv(filepath)
            except:
                df = pd.read_csv(filepath)
        
        print(f"  Loaded EmoLit. Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
        df['source'] = 'EmoLit'
        return df
    
    except Exception as e:
        print(f"Error loading EmoLit: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_emotion_dataset_20(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load emotion_dataset_20 from HuggingFace or local file.
    This dataset is single-label.
    
    Args:
        filepath: Path to local file (CSV or TSV), or None to load from HuggingFace
    """
    try:
        if filepath:
            if filepath.endswith('.tsv'):
                df = pd.read_csv(filepath, sep='\t')
            else:
                df = pd.read_csv(filepath)
            print(f"  Loaded from file. Columns: {df.columns.tolist()}")
        else:
            from datasets import load_dataset
            dataset = load_dataset("shreyaspullehf/emotion-dataset-20-emotions")
            df = pd.DataFrame(dataset['train'])
        
        if 'sentence' in df.columns:
            df['text'] = df['sentence']
        if 'cleaned_text' in df.columns and 'text' not in df.columns:
            df['text'] = df['cleaned_text']
            
        df['original_label'] = df['emotion'].str.lower().str.strip()
        df['source'] = 'emotion_dataset_20'
        
        mapping = get_emotion_dataset_20_to_target_mapping()
        df['label'] = df['original_label'].map(mapping)
        
        return df[['text', 'label', 'original_label', 'source']]
    
    except Exception as e:
        print(f"Error loading emotion_dataset_20: {e}")
        return pd.DataFrame()


def load_goemotions_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load GoEmotions dataset from HuggingFace or local file.
    GoEmotions is multi-label with binary columns.
    
    Args:
        filepath: Path to local file (CSV or TSV), or None to load from HuggingFace
    """
    try:
        if filepath:
            # Detect format
            if filepath.endswith('.tsv'):
                df = pd.read_csv(filepath, sep='\t')
            else:
                df = pd.read_csv(filepath)
            print(f"  Loaded from file. Columns: {df.columns.tolist()[:10]}...")
        else:
            from datasets import load_dataset
            dataset = load_dataset("google-research-datasets/go_emotions", "raw")
            
            dfs = []
            for split in ['train', 'validation', 'test']:
                if split in dataset:
                    split_df = pd.DataFrame(dataset[split])
                    dfs.append(split_df)
            df = pd.concat(dfs, ignore_index=True)
        
        df['source'] = 'GoEmotions'
        return df
    
    except Exception as e:
        print(f"Error loading GoEmotions: {e}")
        return pd.DataFrame()


# =============================================================================
# MULTI-LABEL TO SINGLE-LABEL CONVERSION
# =============================================================================

def convert_emolit_to_single_label(df: pd.DataFrame, 
                                   emotion_columns: List[str],
                                   text_column: str = 'text') -> pd.DataFrame:
    """
    Convert EmoLit multi-label to single-label by keeping highest score.
    
    Args:
        df: EmoLit DataFrame with emotion score columns
        emotion_columns: List of column names containing emotion scores
        text_column: Name of the text column
    
    Returns:
        DataFrame with single label (highest scoring emotion)
    """
    mapping = get_emolit_to_target_mapping()
    
    results = []
    for idx, row in df.iterrows():
        # Get scores for each emotion
        scores = {col: row[col] for col in emotion_columns if col in row.index}
        
        if scores:
            # Find emotion with highest score
            best_emotion = max(scores, key=scores.get)
            best_score = scores[best_emotion]
            
            # Map to target label
            target_label = mapping.get(best_emotion.lower(), None)
            
            if target_label and best_score > 0:  # Only include if score > 0
                results.append({
                    'text': row[text_column],
                    'label': target_label,
                    'original_label': best_emotion,
                    'confidence': best_score,
                    'source': 'EmoLit'
                })
    
    return pd.DataFrame(results)


def convert_goemotions_to_single_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert GoEmotions multi-label to single-label.
    Keep only samples where exactly one emotion column has value 1.
    
    Args:
        df: GoEmotions DataFrame with binary emotion columns
    
    Returns:
        DataFrame with single label
    """
    # GoEmotions emotion columns
    emotion_cols = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Filter to columns that exist in the dataframe
    existing_cols = [col for col in emotion_cols if col in df.columns]
    
    if not existing_cols:
        print("Warning: No emotion columns found in GoEmotions data")
        return pd.DataFrame()
    
    mapping = get_goemotions_to_target_mapping()
    
    results = []
    for idx, row in df.iterrows():
        # Count how many emotions are labeled 1
        active_emotions = [col for col in existing_cols if row.get(col, 0) == 1]
        
        # Keep only single-label samples
        if len(active_emotions) == 1:
            original_label = active_emotions[0]
            target_label = mapping.get(original_label, None)
            
            if target_label:
                results.append({
                    'text': row['text'],
                    'label': target_label,
                    'original_label': original_label,
                    'source': 'GoEmotions'
                })
    
    return pd.DataFrame(results)


def convert_xed_to_single_label(df: pd.DataFrame, 
                                strategy: str = 'priority') -> pd.DataFrame:
    """
    Convert XED multi-label to single-label.
    
    Strategies:
    - 'priority': Use emotion priority order (stronger emotions first)
    - 'random': Randomly select one label
    - 'first': Take first label
    - 'filter': Keep only single-label samples
    
    Args:
        df: XED DataFrame with 'labels_list' column containing list of emotions
        strategy: Conversion strategy
    
    Returns:
        DataFrame with single label
    """
    mapping = get_xed_to_target_mapping()
    
    # Priority order: negative/strong emotions first, then positive, then neutral
    priority_order = ['anger', 'fear', 'disgust', 'sadness', 'surprise', 
                      'joy', 'anticipation', 'trust']
    
    results = []
    
    for idx, row in df.iterrows():
        labels = row['labels_list']
        
        if not labels or len(labels) == 0:
            continue
        
        selected_label = None
        
        if strategy == 'filter':
            # Only keep single-label samples
            if len(labels) == 1:
                selected_label = labels[0]
            else:
                continue
                
        elif strategy == 'priority':
            # Select based on priority
            for priority_emotion in priority_order:
                if priority_emotion in labels:
                    selected_label = priority_emotion
                    break
            if selected_label is None:
                selected_label = labels[0]
                
        elif strategy == 'random':
            import random
            selected_label = random.choice(labels)
            
        elif strategy == 'first':
            selected_label = labels[0]
        
        if selected_label:
            target_label = mapping.get(selected_label, selected_label)
            results.append({
                'text': row['text'],
                'label': target_label,
                'original_label': selected_label,
                'original_labels_all': labels,
                'source': 'XED'
            })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN MERGE FUNCTION
# =============================================================================

def merge_all_datasets(
    smed_path: str,
    emolit_path: Optional[str] = None,
    emotion_dataset_20_path: Optional[str] = None,
    goemotions_path: Optional[str] = None,
    xed_path: Optional[str] = None,
    xed_strategy: str = 'priority',
    emolit_emotion_columns: Optional[List[str]] = None,
    emolit_text_column: str = 'text',
) -> Tuple[pd.DataFrame, Dict]:
    """
    Merge all emotion datasets into unified single-label format.
    
    Args:
        smed_path: Path to SMED CSV file (required)
        emolit_path: Path to EmoLit file (optional)
        emotion_dataset_20_path: Path to emotion_dataset_20 or None for HuggingFace
        goemotions_path: Path to GoEmotions or None for HuggingFace
        xed_path: Path to XED or None for HuggingFace
        xed_strategy: Strategy for XED multi-label conversion ('priority', 'random', 'first', 'filter')
        emolit_emotion_columns: List of emotion column names in EmoLit
        emolit_text_column: Name of text column in EmoLit
    
    Returns:
        merged_df: Merged DataFrame with columns [text, label, original_label, source]
        stats: Dictionary with merge statistics
    """
    
    all_dfs = []
    stats = {'per_dataset': {}, 'target_labels': TARGET_LABELS}
    
    # 1. Load and process SMED (single-label, straightforward)
    print("\n" + "="*60)
    print("Loading SMED...")
    smed_df = load_smed_dataset(smed_path)
    if not smed_df.empty:
        all_dfs.append(smed_df[['text', 'label', 'original_label', 'source']])
        stats['per_dataset']['SMED'] = len(smed_df)
        print(f"  Loaded {len(smed_df)} samples")
        print(f"  Labels: {smed_df['label'].value_counts().to_dict()}")
    
    # 2. Load and process XED (multi-label -> single-label)
    print("\n" + "="*60)
    print("Loading XED...")
    xed_df = load_xed_dataset(xed_path)
    if not xed_df.empty:
        xed_single = convert_xed_to_single_label(xed_df, strategy=xed_strategy)
        if not xed_single.empty:
            all_dfs.append(xed_single[['text', 'label', 'original_label', 'source']])
            stats['per_dataset']['XED'] = len(xed_single)
            print(f"  Loaded {len(xed_df)} samples, converted to {len(xed_single)} single-label")
            print(f"  Strategy: {xed_strategy}")
            print(f"  Labels: {xed_single['label'].value_counts().to_dict()}")
    
    # 3. Load and process emotion_dataset_20 (already single-label)
    print("\n" + "="*60)
    print("Loading emotion_dataset_20...")
    ed20_df = load_emotion_dataset_20(emotion_dataset_20_path)
    if not ed20_df.empty:
        all_dfs.append(ed20_df[['text', 'label', 'original_label', 'source']])
        stats['per_dataset']['emotion_dataset_20'] = len(ed20_df)
        print(f"   Loaded {len(ed20_df)} samples")
        print(f"  Labels: {ed20_df['label'].value_counts().to_dict()}")
    
    # 4. Load and process GoEmotions (multi-label -> single-label, filter only)
    print("\n" + "="*60)
    print("Loading GoEmotions...")
    ge_df = load_goemotions_dataset(goemotions_path)
    if not ge_df.empty:
        ge_single = convert_goemotions_to_single_label(ge_df)
        if not ge_single.empty:
            all_dfs.append(ge_single[['text', 'label', 'original_label', 'source']])
            stats['per_dataset']['GoEmotions'] = len(ge_single)
            print(f"   Loaded {len(ge_df)} samples, filtered to {len(ge_single)} single-label")
            print(f"  Labels: {ge_single['label'].value_counts().to_dict()}")
    
    # 5. Load and process EmoLit (multi-label -> single-label by highest score)
    if emolit_path:
        print("\n" + "="*60)
        print("Loading EmoLit...")
        emolit_df = load_emolit_dataset(emolit_path)
        if not emolit_df.empty:
            # Auto-detect emotion columns if not provided
            if emolit_emotion_columns is None:
                emolit_mapping = get_emolit_to_target_mapping()
                emolit_emotion_columns = [col for col in emolit_df.columns 
                                          if col.lower() in emolit_mapping]
            
            emolit_single = convert_emolit_to_single_label(
                emolit_df, 
                emolit_emotion_columns,
                emolit_text_column
            )
            if not emolit_single.empty:
                all_dfs.append(emolit_single[['text', 'label', 'original_label', 'source']])
                stats['per_dataset']['EmoLit'] = len(emolit_single)
                print(f"   Loaded {len(emolit_df)} samples, converted to {len(emolit_single)} single-label")
                print(f"  Labels: {emolit_single['label'].value_counts().to_dict()}")
    
    # Merge all DataFrames
    print("\n" + "="*60)
    print("MERGING ALL DATASETS...")
    print("="*60)
    
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        
        # Ensure all labels are in target schema
        merged_df = merged_df[merged_df['label'].isin(TARGET_LABELS)]
        
        stats['total_samples'] = len(merged_df)
        stats['label_distribution'] = merged_df['label'].value_counts().to_dict()
        stats['source_distribution'] = merged_df['source'].value_counts().to_dict()
        
        print(f"\n Total samples: {stats['total_samples']}")
        print(f"\nPer-dataset breakdown:")
        for ds, count in stats['per_dataset'].items():
            print(f"  {ds}: {count}")
        print(f"\nLabel distribution:")
        for label, count in sorted(stats['label_distribution'].items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")
        
        return merged_df, stats
    else:
        print("No data loaded!")
        return pd.DataFrame(), stats


def save_dataset(df: pd.DataFrame, output_path: str, format: str = 'csv'):
    """Save the merged dataset."""
    if format == 'csv':
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif format == 'json':
        df.to_json(output_path, orient='records', lines=True)
    print(f"\n Saved to: {output_path}")


#  XED conversion strategies:
#    - 'priority': Negative emotions first (anger > fear > disgust > sadness > ...)
#    - 'random': Random selection from multi-labels
#    - 'first': Take first label
#    - 'filter': Keep only samples with exactly one label

# EmoLit (38 labels) -> Target:
#   anger, annoyance, envy, frustration -> anger
#   fear, doubt, embarrassment, nervousness -> fear
#   disgust, disapproval -> disgust
#   sadness, despair, disappointment, grief, guilt, nostalgia, pain -> sadness
#   joy, admiration, amusement, excitement, gratitude, love, pride, relief -> joy
#   surprise -> surprise
#   trust, approval, caring, courage, faith -> trust
#   anticipation, curiosity, desire, greed, optimism -> anticipation
#   neutral, boredom, calmness, indifference -> neutral

# emotion_dataset_20 (20 labels) -> Target:
#   anger, frustration, jealousy -> anger
#   fear, anxiety, embarrassment -> fear
#   disgust -> disgust
#   sadness, disappointment, guilt, loneliness -> sadness
#   joy, happiness, excitement, gratitude, love, pride, relief -> joy
#   surprise, confusion -> surprise
#   hope -> anticipation

# GoEmotions (28 labels) -> Target:
#   Similar mapping to EmoLit
#   Only single-label samples are kept (where exactly one emotion = 1)

# SMED (5 labels) -> Target:
#   angry -> anger
#   happy -> joy
#   sad -> sadness
#   surprise -> surprise
#   neutral -> neutral

# XED (8 labels) -> Target:
#   Direct mapping (1:1)
# """)