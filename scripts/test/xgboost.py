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
    test_file = {
        'codexglue': 'test',
        'd2a': 'valid',
        'draper': 'test',
        'task1': 'test',
        'task2': 'test',
        'task3': 'test',
        'task4': 'test',
        'task5': 'test',
        'task6': 'test',
    }[args.test]

    # Run it
    subprocess.run([
        'python', f'{base}/models/xgboost/run.py',
        f'--predictions={base}/logs/predictions/xgboost_{args.train}_on_{args.test}.txt',
        f'--dataset={base}/data/embeddings/{args.test}',
        f'--split={test_file}',
        f'--name={args.train}',
    ])
