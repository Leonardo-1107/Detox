import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split




# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'en': 'data/en-00000-of-00001.parquet', 'ru': 'data/ru-00000-of-00001.parquet', 'uk': 'data/uk-00000-of-00001.parquet', 'de': 'data/de-00000-of-00001.parquet', 'es': 'data/es-00000-of-00001.parquet', 'am': 'data/am-00000-of-00001.parquet', 'zh': 'data/zh-00000-of-00001.parquet', 'ar': 'data/ar-00000-of-00001.parquet', 'hi': 'data/hi-00000-of-00001.parquet'}
df_en = pd.read_parquet("hf://datasets/textdetox/multilingual_paradetox/" + splits["en"])
df_en['lang'] = 'en'

df_es = pd.read_parquet("hf://datasets/textdetox/multilingual_paradetox/" + splits["es"])
df_es['lang'] = 'es'

df_zh = pd.read_parquet("hf://datasets/textdetox/multilingual_paradetox/" + splits["zh"])
df_zh['lang'] = 'zh'

# Concatenate them
data_df = pd.concat([df_en, df_es, df_zh], ignore_index=True)

# Split data into train (70%) and test (30%)
train_df, test_df = train_test_split(data_df, test_size=0.3, random_state=42, shuffle=True)

# Save to TSV files
train_df.to_csv("input_data/train.tsv", sep='\t', index=False)
test_df.to_csv("input_data/test.tsv", sep='\t', index=False)

duplicte_df = test_df.copy()
duplicte_df['neutral_sentence'] = test_df['toxic_sentence'].copy()
duplicte_df.to_csv("sample_submissions/duplicate.tsv", sep='\t', index=False)
