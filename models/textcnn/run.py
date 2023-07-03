from data import JSONDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, RichProgressBar
from model import TextCNN
from pathlib import Path
from transformers import AutoTokenizer
import argparse
import lightning.pytorch as pl
import logging
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class LitTextCNN(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters,
                 num_classes, dropout, mode, class_weights):
        super().__init__()
        self.model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            kernel_sizes=kernel_sizes,
            num_filters=num_filters,
            num_classes=num_classes,
            dropout=dropout,
            mode=mode,
        )

        _metrics = lambda: (
                torchmetrics.Accuracy(task='binary', num_classes=num_classes).cuda(),
                torchmetrics.Precision(task='binary', num_classes=num_classes).cuda(),
                torchmetrics.Recall(task='binary', num_classes=num_classes).cuda(),
                torchmetrics.F1Score(task='binary', num_classes=num_classes).cuda())
        
        self.train_accuracy, self.train_precision, self.train_recall, self.train_f1 = _metrics()
        self.test_accuracy, self.test_precision, self.test_recall, self.test_f1 = _metrics()
        self.val_accuracy, self.val_precision, self.val_recall, self.val_f1 = _metrics()

        self.train_metrics = { 'train_acc': self.train_accuracy, 'train_prec': self.train_precision, 'train_rec': self.train_recall, 'train_f1': self.train_f1, }
        self.test_metrics = { 'test_acc': self.test_accuracy, 'test_prec': self.test_precision, 'test_rec': self.test_recall, 'test_f1': self.test_f1, }
        self.val_metrics = { 'val_acc': self.val_accuracy, 'val_prec': self.val_precision, 'val_rec': self.val_recall, 'val_f1': self.val_f1 }

        class_weights = torch.tensor(class_weights).to(torch.float)
        self.train_loss = torch.nn.CrossEntropyLoss(class_weights)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self.model(x)
    
    def common_step(self, batch, metrics, loss_fn=F.cross_entropy):
        x, y = batch
        logits = self.model(x)
        loss = loss_fn(logits, y)

        y_hat = logits[:, 1] > 0.5

        for name, metric in metrics.items():
            metric.update(y_hat, y)
            self.log(name, metric, on_epoch=True, prog_bar=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, self.train_metrics, loss_fn=self.train_loss)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, self.val_metrics)

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, self.test_metrics)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

def main(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    data = JSONDataModule(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        train_path=args.train_data,
        test_path=args.test_data,
        val_path=args.eval_data,
        balance='balance' not in args.train_data,
    )

    class_weights = data.get_class_weights()
    print('class weights:', class_weights)

    if args.checkpoint:
        model = LitTextCNN.load_from_checkpoint(
            args.checkpoint,
            vocab_size=tokenizer.vocab_size,
            embedding_dim=256,
            kernel_sizes=[5, 21, 43],
            num_filters=50,
            num_classes=2,
            dropout=0.5,
            mode='static',
            class_weights=class_weights,
        )
    else:
        model = LitTextCNN(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=256,
            kernel_sizes=[5, 21, 43],
            num_filters=50,
            num_classes=2,
            dropout=0.5,
            mode='static',
            class_weights=class_weights,
        )
    
    if 'draper' in args.train_data or 'all' in args.train_data:
        print('Equal weights, monitoring accuracy.')
        monitor = 'val_acc'
    else:
        print('Different class weights, monitoring F1.')
        monitor = 'val_f1'

    checkpoint_dir = Path(__file__).parent.joinpath('checkpoints').joinpath(args.name)
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[
            ModelCheckpoint(checkpoint_dir, 'checkpoint', monitor=monitor, mode='max'),
            RichProgressBar(leave=True),
            StochasticWeightAveraging(swa_lrs=1e-2),
        ])
    
    if args.do_train:
        trainer.fit(model, data)
    
    if args.do_test:
        data.setup('test')
        test_data = data.predict_dataloader().dataset.df
        indices = test_data.idx
        y_true = test_data.target

        results = trainer.predict(model, data)
        y_hat = (np.vstack(results)[:, 1] > 0.5).astype(int)

        print('-' * 20)
        print(f'Accuracy : {accuracy_score(y_true, y_hat):.4f}')
        print(f'Precision: {precision_score(y_true, y_hat):.4f}')
        print(f'Recall   : {recall_score(y_true, y_hat):.4f}')
        print(f'F1 Score : {f1_score(y_true, y_hat):.4f}')
        print('-' * 20)
        
        if args.predictions:
            with open(args.predictions,'w') as f:
                for idx, yhi in zip(indices, np.vstack(results)[:, 1]):
                    f.write(f'{idx}\t{yhi:.4f}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='textcnn', type=str, help='Model name.')
    parser.add_argument('--tokenizer_name', default='microsoft/codebert-base', type=str,
                        help='HuggingFace tokenizer for tokenization.')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
    parser.add_argument("--train_data", type=str, required=True, help='Training data.')
    parser.add_argument("--eval_data", type=str, required=True, help="Validation data.")
    parser.add_argument("--test_data", type=str, required=True, help="Test data.")
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--predictions', type=str, default=None)
    args = parser.parse_args()

    main(args)
