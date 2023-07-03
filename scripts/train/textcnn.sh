# Usage:
#   ./textcnn.sh [dataset]

# Get root path
ROOT="${BASE:=../../}"

# Run textcnn
python $ROOT/models/textcnn/run.py \
    --name=$1 \
    --do_train \
    --do_test \
    --train_data=$ROOT/data/jsonl/$1/train.jsonl \
    --eval_data=$ROOT/data/jsonl/$1/valid.jsonl \
    --test_data=$ROOT/data/jsonl/$1/test.jsonl
