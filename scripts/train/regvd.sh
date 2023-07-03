# Usage:
#   ./regvd.sh [dataset] [epochs]

# Get root path
ROOT="${BASE:=../../}"

# Run ReGVD
python ${BASE}/models/regvd/run.py \
	--output_dir=${BASE}/models/regvd/saved_models/$1 \
	--model_type=roberta --tokenizer_name=microsoft/graphcodebert-base \
	--model_name_or_path=microsoft/graphcodebert-base \
	--do_eval \
	--do_test \
	--do_train \
	--train_data_file=${BASE}/data/jsonl/$1/train.jsonl \
	--eval_data_file=${BASE}/data/jsonl/$1/valid.jsonl \
	--test_data_file=${BASE}/data/jsonl/$1/test.jsonl \
	--block_size 400 \
	--train_batch_size 512 \
	--eval_batch_size 512 \
	--max_grad_norm 1.0 \
	--evaluate_during_training \
	--gnn ReGCN \
	--learning_rate 5e-4 \
	--epoch $2 \
	--hidden_size 128 \
	--num_GNN_layers 2 \
	--format uni \
	--window_size 5 \
	--seed 42