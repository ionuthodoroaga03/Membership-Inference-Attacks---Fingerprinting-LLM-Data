import pandas as pd

# Load the dataset
df = pd.read_csv("hf://datasets/shreyaspulle98/emotion-dataset-20-emotions/emotion_dataset_v5_clean.csv")

# View emotion distribution
print(df['emotion'].value_counts())

df.to_csv("data/emotion-dataset-20-emotions/emotion_dataset_v5_clean.csv", index=False)
