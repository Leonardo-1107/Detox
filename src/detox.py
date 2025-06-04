import argparse
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class MT0Detoxifier:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)

        # Same prompts used during training
        self.lang_prompts = {
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

    def detoxify(self, texts, langs, batch_size=16):
        prompts = [
            self.lang_prompts.get(lang, "Detoxify: ") + text
            for text, lang in zip(texts, langs)
        ]

        results = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Detoxifying"):
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=5,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    num_return_sequences=1,
                    early_stopping=True,
                    decoder_start_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)

        return results


def main():
    parser = argparse.ArgumentParser(description="Detoxify text using fine-tuned MT0")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--input_tsv", type=str, required=True, help="Input file (TSV)")
    parser.add_argument("--output_tsv", type=str, required=True, help="Output file (TSV)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    # Load test data
    df = pd.read_csv(args.input_tsv, sep="\t")
    assert "toxic_sentence" in df.columns and "lang" in df.columns, "Input TSV must have 'toxic_sentence' and 'lang' columns"

    detoxifier = MT0Detoxifier(args.model_path)
    clean_texts = detoxifier.detoxify(df["toxic_sentence"].tolist(), df["lang"].tolist(), batch_size=args.batch_size)

    df["neutral_sentence"] = clean_texts
    df.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"Saved detoxified output to: {args.output_tsv}")


if __name__ == "__main__":
    main()