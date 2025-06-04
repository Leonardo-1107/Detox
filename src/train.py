import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


class DetoxDataset(Dataset):
    def __init__(self, dataframe, tokenizer, lang_prompts, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.lang_prompts = lang_prompts
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        source = self.lang_prompts.get(row["lang"], "Detoxify: ") + row["toxic_sentence"]
        target = row["neutral_sentence"]

        source_enc = self.tokenizer(source, truncation=True, padding="max_length", max_length=self.max_length)
        target_enc = self.tokenizer(target, truncation=True, padding="max_length", max_length=self.max_length)

        return {
            "input_ids": torch.tensor(source_enc["input_ids"]),
            "attention_mask": torch.tensor(source_enc["attention_mask"]),
            "labels": torch.tensor(target_enc["input_ids"]),
        }

# build my own seq2seq model
def train_model(train_path, model_name, output_dir, num_train_epochs, batch_size, learning_rate):
    df_train = pd.read_csv(train_path, sep="\t")

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

    train_dataset = DetoxDataset(df_train, tokenizer, lang_prompts)

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
    )


if __name__ == "__main__":
    main()