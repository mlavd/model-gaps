# Usage:
#   ./xgboost.sh [dataset]

# Get root path
ROOT="${BASE:=../../}"

python $ROOT/models/xgboost/run.py \
    --dataset=$ROOT/data/embeddings/$1 \
    --name=$1

