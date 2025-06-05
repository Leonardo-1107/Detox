import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

############################ TODO ############################
language_list = ['en', 'es', 'zh']  

# Map of language codes to their corresponding file paths
splits = {
    'en': 'data/en-00000-of-00001.parquet',
    'ru': 'data/ru-00000-of-00001.parquet',
    'uk': 'data/uk-00000-of-00001.parquet',
    'de': 'data/de-00000-of-00001.parquet',
    'es': 'data/es-00000-of-00001.parquet',
    'am': 'data/am-00000-of-00001.parquet',
    'zh': 'data/zh-00000-of-00001.parquet',
    'ar': 'data/ar-00000-of-00001.parquet',
    'hi': 'data/hi-00000-of-00001.parquet'
}


data_frames = []
for lang in language_list:
    df = pd.read_parquet("hf://datasets/textdetox/multilingual_paradetox/" + splits[lang])
    df['lang'] = lang
    data_frames.append(df)

data_df = pd.concat(data_frames, ignore_index=True)

# Split data into train (70%) and test (30%)
train_df, test_df = train_test_split(data_df, test_size=0.3, random_state=42, shuffle=True)
train_df.to_csv("input_data/train.tsv", sep='\t', index=False)
test_df.to_csv("input_data/test.tsv", sep='\t', index=False)

# Create a sample submission file with duplicate toxic sentences as neutral placeholders
duplicate_df = test_df.copy()
duplicate_df['neutral_sentence'] = test_df['toxic_sentence']
duplicate_df.to_csv("sample_submissions/duplicate.tsv", sep='\t', index=False)