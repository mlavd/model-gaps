from pathlib import Path
from tqdm import tqdm as std_tqdm
from transformers import RobertaModel, RobertaTokenizer
import argparse
import numpy as np
import pandas as pd
import torch
import torch

class tqdm(std_tqdm):
    def display(self, msg=None, pos=None):
        print(self.format_meter(**self.format_dict))

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=Path, required=True, help='JSONL input')
    parser.add_argument('--output', type=str, required=True, help='Output stem')
    parser.add_argument('--save_indices', type=bool, default=False)
    args = parser.parse_args()

    # Load tokenizer and model.
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaModel.from_pretrained('microsoft/codebert-base')
    model.to('cuda')

    # Load the data file
    print('Loading data...')
    df = pd.read_json(args.jsonl, lines=True)

    # Save the labels
    print('Saving labels...')
    Path(args.output).parent.mkdir(exist_ok=True)
    np.savetxt(args.output + '_labels.txt', df.target.values, fmt='%d')

    # Useful for codexglue test, which is out of order
    if args.save_indices:
        np.savetxt(args.output + '_indices.txt', df.idx.values, fmt='%d')

    # Extract the embeddings
    print('Generating embeddings...')
    with (torch.no_grad(),
         open(args.output + '.txt', 'w') as out
         ):        
        embeddings = []

        # Iterate over chunks of 128
        for i, batch in tqdm(df.groupby(np.arange(df.shape[0]) // 128)):
            # Tokenize the batch
            tokens = tokenizer(batch.func.tolist(),
                return_tensors='pt', padding=True, truncation=True)

            # Get the embeddings
            output = model(**tokens.to('cuda'))
            embeddings.append(output.pooler_output.cpu().numpy())

            # Save the embeddings periodically.
            if len(embeddings) >= 50:
                embeddings = np.vstack(embeddings)
                np.savetxt(out, embeddings, fmt='%1.5f')
                embeddings = []
    
        # Make sure to save the last set of embeddings.
        if len(embeddings) > 0:
            embeddings = np.vstack(embeddings)
            np.savetxt(out, embeddings, fmt='%1.5f')
                
    print('Done')
