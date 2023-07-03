cd ../
python jsonl_merger.py --a=d2a/valid.jsonl --b=codexglue/test.jsonl --out=codexglued2a/test.jsonl
cp ./data/jsonl/codexglue/valid.jsonl ./data/jsonl/codexglued2a/valid.jsonl
python jsonl_merger.py --a=d2a/train.jsonl --b=codexglue/train.jsonl --out=codexglued2a/train.jsonl

python jsonl_merger.py --a=draper/test.jsonl --b=codexglued2a/test.jsonl --out=all/test.jsonl
python jsonl_merger.py --a=draper/valid.jsonl --b=codexglued2a/valid.jsonl --out=all/valid.jsonl
python jsonl_merger.py --a=draper/train.jsonl --b=codexglued2a/train.jsonl --out=all/train.jsonl

cd scripts