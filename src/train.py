import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet
import random

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

nltk.download('wordnet')


class DetoxDataset(Dataset):
    def __init__(self, dataframe, tokenizer, lang_prompts, max_length=128, augment=True):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.lang_prompts = lang_prompts
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        toxic = row["toxic_sentence"]
        neutral = row["neutral_sentence"]
        lang = row["lang"]

        if self.augment:
            toxic = self.augment_text(toxic, lang)

        source = self.lang_prompts.get(lang, "Detoxify: ") + toxic
        source_enc = self.tokenizer(source, truncation=True, padding="max_length", max_length=self.max_length)
        target_enc = self.tokenizer(neutral, truncation=True, padding="max_length", max_length=self.max_length)

        return {
            "input_ids": torch.tensor(source_enc["input_ids"]),
            "attention_mask": torch.tensor(source_enc["attention_mask"]),
            "labels": torch.tensor(target_enc["input_ids"]),
        }

    def augment_text(self, text, lang):
        """Basic data augmentation methods: synonym replace + noise."""
        if lang == "en":
            return self.synonym_replacement(text)
        else:
            return self.add_typo_noise(text)

    def synonym_replacement(self, sentence, prob=0.2):
        words = sentence.split()
        new_words = []
        for word in words:
            if random.random() < prob:
                synonyms = wordnet.synsets(word)
                if synonyms:
                    synonym_words = synonyms[0].lemma_names()
                    if synonym_words:
                        new_words.append(random.choice(synonym_words).replace('_', ' '))
                        continue
            new_words.append(word)
        return ' '.join(new_words)

    def add_typo_noise(self, sentence, prob=0.1):
        chars = list(sentence)
        for i in range(len(chars)):
            if random.random() < prob:
                op = random.choice(["swap", "delete"])
                if op == "swap" and i < len(chars) - 1:
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                elif op == "delete":
                    chars[i] = ''
        return ''.join(chars)
    

# build my own seq2seq model
def train_model(train_path, model_name, output_dir, num_train_epochs, batch_size, learning_rate, augmentation):
    df_train = pd.read_csv(train_path, sep="\t")

    if "t5" in model_name:
        df_train = df_train[df_train["lang"] != "zh"].copy()
        print("[INFO] Using T5 models, keep the latin languages only ...")


    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    lang_prompts = {
        "zh": "排毒：",
        "es": "Desintoxicar: ",
        "ru": "Детоксифицируй: ",
        "ar": "إزالة السموم: ",
        "hi": "विषहरण: ",
        "uk": "Детоксифікуй: ",
        "de": "Entgiften: ",
        "am": "መርዝ መርዝ: ",
        "en": "Detoxify: ",
        "it": "Disintossicare: ",
        "ja": "解毒: ",
        "he": "לְסַלֵק רַעַל: ",
        "fr": "Désintoxiquer:",
        "tt": "Токсиннарны чыгару: ",
        "hin": "Detoxify: ",
    }

    train_dataset = DetoxDataset(df_train, tokenizer, lang_prompts, augment=augmentation)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        save_total_limit=2,
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        fp16=torch.cuda.is_available(),    
    )

    if model_name == 'google/mt5-small':
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            save_total_limit=2,
            save_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            fp16=torch.cuda.is_available(),

            # ✅ ADD THIS:
            predict_with_generate=True, 
            generation_max_length=128,
            generation_num_beams=5,
        )


    print(f"[INFO] Model {model_name} loaded.")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    
    os.makedirs("models", exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def main():

    parser = argparse.ArgumentParser(description="Train MT0 detoxification model")
    parser.add_argument("--train_path", type=str, default="input_data/train.tsv", help="Path to training TSV file")
    parser.add_argument("--model_name", type=str, default="s-nlp/mt0-xl-detox-orp", help="Base model name (default: mt0-small)")
    parser.add_argument("--save_model_dir", type=str, default="models/mt5", help="Directory to save fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--augmentation", type=bool, default=True, help="Apply data augmentation")

    args = parser.parse_args()
    print("Training configuration:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")


    train_model(
        train_path=args.train_path,
        model_name=args.model_name,
        output_dir=args.save_model_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        augmentation=args.augmentation,
    )


if __name__ == "__main__":
    main()