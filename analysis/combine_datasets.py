from dataset_merge import merge_all_datasets, save_dataset


BASE_PATH = "path/to/datasets/folder"
merged_df, stats = merge_all_datasets(
    smed_path=f"{BASE_PATH}/SMED/Social Media Emotion Dataset.csv",
    xed_path=f"{BASE_PATH}/XED/en-annotated.tsv",
    emolit_path=f"{BASE_PATH}/EmoLit/emolit_all_merged.tsv",
    emotion_dataset_20_path=f"{BASE_PATH}/emotion-dataset-20-emotions/emotion_dataset_v5_clean.csv",
    goemotions_path=f"{BASE_PATH}/GoEmotions/data/full_dataset/goemotions_3.csv",
    xed_strategy='priority'
)

save_dataset(merged_df, 'all_datasets_merged.csv')



