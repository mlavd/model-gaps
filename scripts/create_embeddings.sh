python $BASE/embeddings_extractor.py \
    --jsonl=$BASE/data/jsonl/$1/train.jsonl \
    --output=$BASE/data/embeddings/$1/train

python $BASE/embeddings_extractor.py \
    --jsonl=$BASE/data/jsonl/$1/valid.jsonl \
    --output=$BASE/data/embeddings/$1/valid

python $BASE/embeddings_extractor.py \
    --jsonl=$BASE/data/jsonl/$1/test.jsonl \
    --output=$BASE/data/embeddings/$1/test
