from pathlib import Path
from rich.progress import track
from sklearn import metrics as skmets
import numpy as np
import pandas as pd

def add_truth(df, dataset, path):
    df[('truth', dataset, dataset)] = pd.read_csv(path)[['target']].rename(columns={'target': 'pred'})

def load_predictions():
    pred_paths = list(Path('../logs/predictions').rglob('*.txt'))
    predictions = {}

    for path in pred_paths:
        model, train, _, test = path.stem.split('_')
        df = pd.read_csv(path, delimiter='\t', names=['pred'], index_col=0)
        predictions[(model, train, test)] = df.sort_index()
        # predictions[path.stem].columns = ['index', 'pred']
    
    return predictions

def load_metrics(predictions):
    metrics = []

    for (model, train, test), df in track(predictions.items()):
        if model == 'truth': continue
        y_true = predictions[('truth', test, test)].pred

        m = get_metrics(y_true, df.pred)
        m.update({ 'model': model, 'train': train, 'test': test })
        metrics.append(m)

    metrics = pd.DataFrame(metrics).round(4)
    metrics.insert(0, 'model', metrics.pop('model'))
    metrics.insert(1, 'train', metrics.pop('train'))
    metrics.insert(2, 'test', metrics.pop('test'))
    metrics = metrics.sort_values(by=['model', 'train'])
    return metrics

def has_value(y_pred):
    return (
        0.05 < y_pred.sum() / y_pred.shape[0] and
        y_pred.sum() / y_pred.shape[0] < 0.95
    )

    
def get_metrics(y_true, y_pred, threshold=0.5):
    y_p = y_pred > 0.5
    # tn, fp, fn, tp = skmets.confusion_matrix(y_true, y_p, labels=[0, 1]).ravel()
    
    return {
        'has_logits': np.unique(y_pred).shape[0] > 2,
        'avg_prec': skmets.average_precision_score(y_true, y_pred),
        'accuracy': skmets.accuracy_score(y_true, y_p),
        'f1': skmets.f1_score(y_true, y_p),
        'precision': skmets.precision_score(y_true, y_p, zero_division=0),
        'recall': skmets.recall_score(y_true, y_p),
        # 'tp': tp,
        # 'fp': fp,
        # 'tn': tn,
        # 'fn': fn,
    }