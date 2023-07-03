from pathlib import Path
import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=Path, required=True, help='JSONL input')
    parser.add_argument('--indices', type=Path, required=True, help='Indices text')
    args = parser.parse_args()
    print(args)

    print('Loading indices')
    indices = np.loadtxt(args.indices).astype(int)

    print('Loading JSONL')
    df = pd.read_json(args.jsonl, lines=True)

    print('Sampling and shuffle.')
    df = df.iloc[indices].copy()
    df = df.sample(frac=1).reset_index(drop=True)

    print('Saving')
    df.to_json(args.jsonl, orient='records', lines=True)

    print('Done')
