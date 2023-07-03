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
        'codexglue': 'codexglue/test.jsonl',
        'd2a': 'd2a/valid.jsonl',
        'draper': 'draper/test.jsonl',
    }[args.test]

    # Run it
    subprocess.run([
        'python', f'{base}/models/regvd/run.py',
        f'--output_dir={base}/models/regvd/saved_models/{args.train}',
        '--model_type=roberta',
        '--tokenizer_name=microsoft/graphcodebert-base',
        '--model_name_or_path=microsoft/graphcodebert-base',
        '--do_test',
        f'--train_data_file=None',
        f'--test_data_file={base}/data/jsonl/{test_file}',
        f'--predictions={base}/logs/predictions/regvd_{args.train}_on_{args.test}.txt',
        '--block_size=400',
        '--train_batch_size=512',
        '--eval_batch_size=512',
        '--max_grad_norm=1.0',
        '--evaluate_during_training',
        '--gnn=ReGCN',
        '--hidden_size=128',
        '--num_GNN_layers=2',
        '--format=uni',
        '--window_size=5',
        '--seed=42',
    ])
