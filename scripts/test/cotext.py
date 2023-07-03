from pathlib import Path
import argparse
import subprocess
import os

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Training dataset.')
    parser.add_argument('--test', type=str, required=True, help='Testing dataset.')
    args = parser.parse_args()

    # Get base
    base = os.environ.get('BASE', '../..')

    # Get test file (since the name can vary)
    test_file = args.test + '/' + \
        { 'codexglue': 'test.jsonl', 'd2a': 'valid.jsonl', 'draper': 'test.jsonl' }[args.test]

    valid_file = args.test + '/' + \
        { 'codexglue': 'valid.jsonl', 'd2a': 'valid.jsonl', 'draper': 'valid.jsonl' }[args.test]
    
    checkpoint = sorted(list(Path(f'{base}/models/cotext/saved_models/{args.train}').iterdir()))[-1]

    # Run it
    subprocess.run([
        'python', f'{base}/models/cotext/model.py',
        '--output_dir=./',
        f'--model_name_or_path={checkpoint}',
        '--do_test',
        '--train_data_file=None',
        f'--train_data_file={base}/data/jsonl/{args.test}/train.jsonl',
        f'--eval_data_file={base}/data/jsonl/{valid_file}',
        f'--test_data_file={base}/data/jsonl/{test_file}',
        f'--predictions={base}/logs/predictions/cotext_{args.train}_on_{args.test}.txt',
        '--eval_batch_size=32',
        '--seed=42',
    ])
