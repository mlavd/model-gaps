# Usage:
#   ./codebert.sh [dataset] [epochs]

# Get root path
ROOT="${BASE:=../../}"

# Run CodeBERT
python ${BASE}/models/codebert/run.py \
    --output_dir=${BASE}/models/codebert/saved_models/$1 \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_test \
    --train_data_file=${BASE}/data/jsonl/$1/train.jsonl \
    --eval_data_file=${BASE}/data/jsonl/$1/valid.jsonl \
    --test_data_file=${BASE}/data/jsonl/$1/test.jsonl \
    --epoch $2 \
    --block_size 512 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 42
