

# unsupervised baselines -> delete
python src/baseline_delete.py --input_path input_data/test.tsv --output_path sample_submissions/delete.tsv


# supervised state-of-art -> s-nlp detoxify model
pip install transformers -U
python src/detox.py --model_path s-nlp/mt0-xl-detox-orpo --input_tsv input_data/test.tsv --output_tsv sample_submissions/mt0.tsv


# train and use the seq2seq model
python src/train.py --train_path input_data/train.tsv --model_name t5-small --save_model_dir models/t5 
python src/detox.py --model_path models/t5 --input_tsv input_data/test.tsv --output_tsv sample_submissions/t5.tsv



# evaluation
pip install transformers==4.35.2

python evaluation/evaluate.py --submission sample_submissions/duplicate.tsv --reference sample_submissions/test.tsv --save_name duplicate
python evaluation/evaluate.py --submission sample_submissions/delete.tsv --reference sample_submissions/test.tsv --save_name delete
python evaluation/evaluate.py --submission sample_submissions/mt0.tsv --reference sample_submissions/test.tsv --save_name mt0
python evaluation/evaluate.py --submission sample_submissions/t5.tsv --reference sample_submissions/test.tsv --save_name t5
