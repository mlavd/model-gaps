# Usage:
#   ./cotext.sh [dataset] [epochs]

# Get root path
ROOT="${BASE:=../../}"

# Run CoTeXT
python ${BASE}/models/cotext/model.py \
    --output_dir=${BASE}/models/cotext/saved_models/$1 \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=${BASE}/data/jsonl/$1/train.jsonl \
    --eval_data_file=${BASE}/data/jsonl/$1/valid.jsonl \
    --test_data_file=${BASE}/data/jsonl/$1/test.jsonl \
    --epoch $2 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --seed 42
