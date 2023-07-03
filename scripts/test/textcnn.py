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

    checkpoint = sorted(list(Path(f'{base}/models/textcnn/checkpoints/{args.train}').iterdir()))[-1]

    # Get test file (since the name can vary)
    test_file = {
        'codexglue': 'codexglue/test.jsonl',
        'd2a': 'd2a/valid.jsonl',
        'draper': 'draper/test.jsonl',
    }[args.test]

    # Run it
    subprocess.run([
        'python', f'{base}/models/textcnn/run.py',
        f'--name={args.train}',
        f'--checkpoint={checkpoint}',
        '--do_test',
        f'--train_data={base}/data/jsonl/{test_file}',
        f'--eval_data={base}/data/jsonl/{test_file}',
        f'--test_data={base}/data/jsonl/{test_file}',
        f'--predictions={base}/logs/predictions/textcnn_{args.train}_on_{args.test}.txt',
    ])
