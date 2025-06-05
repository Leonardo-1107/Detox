# prepare the data (already a demo in input_data)
python get_data.py

# unsupervised baselines -> delete
python src/baseline_delete.py --input_path input_data/test.tsv --output_path sample_submissions/delete.tsv


# supervised state-of-art -> s-nlp detoxify model
pip install transformers -U
python src/detox.py --model_path s-nlp/mt0-xl-detox-orpo --input_tsv input_data/test.tsv --output_tsv sample_submissions/mt0.tsv


# train and use the seq2seq model
python src/train.py --train_path input_data/train.tsv --model_name t5-small --save_model_dir models/t5 --num_train_epochs 10 --augmentation False
python src/detox.py --model_path models/t5 --input_tsv input_data/test.tsv --output_tsv sample_submissions/t5.tsv

python src/train.py --train_path input_data/train.tsv --model_name t5-small --save_model_dir models/t5-ft --num_train_epochs 50 --augmentation True
python src/detox.py --model_path models/t5-ft --input_tsv input_data/test.tsv --output_tsv sample_submissions/t5-ft.tsv

python src/train.py --train_path input_data/train.tsv --model_name facebook/bart-base --save_model_dir models/bart-ft --num_train_epochs 50 --augmentation True
python src/detox.py --model_path models/bart-ft --input_tsv input_data/test.tsv --output_tsv sample_submissions/bart-ft

# evaluation
pip install transformers==4.35.2

python evaluation/evaluate.py --submission sample_submissions/duplicate.tsv --reference sample_submissions/test.tsv --save_name duplicate
python evaluation/evaluate.py --submission sample_submissions/delete.tsv --reference sample_submissions/test.tsv --save_name delete
python evaluation/evaluate.py --submission sample_submissions/mt0.tsv --reference sample_submissions/test.tsv --save_name mt0
python evaluation/evaluate.py --submission sample_submissions/t5.tsv --reference sample_submissions/test.tsv --save_name t5
python evaluation/evaluate.py --submission sample_submissions/t5-ft.tsv --reference sample_submissions/test.tsv --save_name t5-finetuned
python evaluation/evaluate.py --submission sample_submissions/bart-ft --reference sample_submissions/test.tsv --save_name bart-finetuned