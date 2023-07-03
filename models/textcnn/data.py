from rich.progress import track
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import torch
import lightning.pytorch as pl
import numpy as np
import multiprocessing

class JSONDataset(Dataset):
    def __init__(self, tokenizer, file_path, balance=False):
        self.df = pd.read_json(file_path, lines=True)
        self.tokenizer = tokenizer

        indices_path = Path(file_path).parent.joinpath('indices.txt')

        if balance and indices_path.exists():
            print('Balancing...')
            indices = np.loadtxt(indices_path).astype(int)
            self.df = self.df.iloc[indices].copy()
            print(self.df.target.value_counts())
        elif balance and not indices_path.exists():
            print('Would balance, no index.')
            # g = self.df.groupby('target')
            # g = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
            # self.df = pd.DataFrame(g).reset_index(drop=True).sample(frac=1)

        # self.examples = []

        # for _, row in track(self.df.iterrows(), total=self.df.shape[0]):
        #     self.examples.append((
        #         tokenizer(
        #             row.func,
        #             padding='max_length', max_length=1024, truncation=True,
        #         )['input_ids'],
        #         row.target,
        #     ))

    def __len__(self):
        return self.df.shape[0]
        # return len(self.examples)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        input_ids = self.tokenizer(
            row.func,
            padding='max_length', max_length=1024, truncation=True,
        )['input_ids']
        return torch.tensor(input_ids), torch.tensor(row.target)
        # return torch.tensor(self.examples[i][0]), torch.tensor(self.examples[i][1])

class JSONDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size, train_path, test_path, val_path, balance):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.train = None
        self.balance = balance

    def setup(self, stage):
        if stage == 'fit':
            self.train = JSONDataset(self.tokenizer, self.train_path, balance=self.balance)
        if stage == 'validate' or stage == 'fit':
            self.val = JSONDataset(self.tokenizer, self.val_path)
        if stage == 'test':
            self.test = JSONDataset(self.tokenizer, self.test_path)
    
    def get_class_weights(self):
        if not self.train: self.setup('fit')
        return self.train.df.shape[0] / np.bincount(self.train.df.target)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def predict_dataloader(self, batch_size=None):
        return DataLoader(self.test, batch_size=batch_size or self.batch_size)