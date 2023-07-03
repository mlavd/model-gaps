from datetime import datetime
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pathlib import Path
from sklearn import metrics
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

def load_data(base, split):
    x = np.loadtxt(base.joinpath(f'{split}.txt'))
    y = np.loadtxt(base.joinpath(f'{split}_labels.txt')).astype(int)
    return xgb.DMatrix(x, label=y)

def custom_acc(y_hat, d_true):
    return 'acc', metrics.accuracy_score(d_true.get_label(), y_hat > 0.5)

def custom_f1(y_hat, d_true):
    return 'f1', metrics.f1_score(d_true.get_label(), y_hat > 0.5)

def evaluate(model, data, predictions=None, indices=None):
    y_hat = model.predict(data)

    if predictions:
        with open(args.predictions, 'w') as f:
            for i, ri in zip(indices, y_hat):
                f.write(f'{i}\t{ri:.4f}\n')

    y_hat = (y_hat > 0.5).astype(int)

    y_true = data.get_label()
    average_precision = metrics.average_precision_score(y_true, y_hat)
    print('-' * 20)
    print(f'Accuracy : {metrics.accuracy_score(y_true, y_hat):.4f}')
    print(f'Recall   : {metrics.recall_score(y_true, y_hat):.4f}')
    print(f'Precision: {metrics.precision_score(y_true, y_hat, zero_division=0):.4f}')
    print(f'F1 Score : {metrics.f1_score(y_true, y_hat):.4f}')
    print(f'Avg. Prec: {average_precision:.4f}')
    print('-' * 20)
    return average_precision

def objective(space):
    r = ', '.join(
        f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}'
        for k, v in sorted(space.items()) if k != 'data')
    print(f"Trial: {r}")

    d_train, d_valid = space['data']

    model = xgb.train(
        {
            'booster': space['booster'],
            'gamma': space['gamma'],
            'max_depth': int(space['max_depth']),
            'objective': space['objective'],
            'rate_drop': space['rate_drop'],
            'reg_alpha': space['alpha'],
            'reg_lambda': space['lambda'],
            'skip_drop': space['skip_drop'],
            'eval_metric': 'auc',
            'seed': space['seed'],
        },
        dtrain=d_train,
        evals=[(d_train, 'train'), (d_valid, 'valid')],
        custom_metric=custom_acc,
        early_stopping_rounds=space['early_stopping'],
        num_boost_round=space['n_estimators'],
        # verbose=True,
    )

    
    score = evaluate(model, d_valid)
    print(f'----- SCORE: {score:.4f} -----')
    return { 'loss': -score, 'status': STATUS_OK, 'model': model }

def main(args):
    print(args)

    # Load training data
    print('Loading training dataset...')
    d_train = load_data(args.dataset, 'train')
    # print(f'\tcounts: {np.bincount(y_train)}')

    # Load validation data
    print('Loading validation dataset...')
    d_valid = load_data(args.dataset, 'valid')
    # print(f'\tcounts: {np.bincount(y_valid)}')

    # Setup the hyperparameter tuning space
    param_space = {
        'max_depth': hp.quniform('max_depth', 2, 20, 1),
        'gamma': hp.uniform('gamma', 1, 9),
        'alpha': hp.choice('alpha', [ 1, 100, 1_000, 10_000 ]),
        'lambda': hp.uniform('lambda', 0, 1),
        'objective': 'binary:logistic', #hp.choice('objective', [ 'binary:logistic', 'binary:hinge' ]),
        'booster': hp.choice('booster', ['gbtree', 'dart']),
        'rate_drop': hp.uniform('rate_drop', 0.0, 0.95),
        'skip_drop': hp.uniform('skip_drop', 0.0, 0.95),
        'n_estimators': hp.choice('n_estimators', [ 10, 25, 50, 100, 250, 500, 1_000 ]),
        'seed': 42,
        'data': (d_train, d_valid),
        'monitor': args.metric,
        'early_stopping': args.early_stopping,
    }

    trials = Trials()

    best_params = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    # Print the parameters
    print(best_params)

    # Get the best model
    losses = np.array(list(t['result']['loss'] for t in trials)).round(4)
    best_idx = np.argmin(losses)
    best_model = trials.results[best_idx]['model']

    evaluate(best_model, d_valid)

    # Save the model
    output = Path(__file__).parent.joinpath('saved_models')
    output.mkdir(exist_ok=True)
    best_model.save_model(output.joinpath(f'{args.name}.json'))


def predict(args):
    print('Loading model...')
    saved_models = Path(__file__).parent.joinpath('saved_models')
    model = xgb.Booster()
    model.load_model(saved_models.joinpath(f'{args.name}.json'))

    print('Loading testing data...')
    d_test = load_data(args.dataset, args.split)

    indices_path = args.dataset.joinpath(args.split + '_indices.txt')
    if indices_path.exists():
        print('Found indices!')
        indices = np.loadtxt(indices_path).astype(int)
    else:
        indices = np.arange(d_test.get_data().shape[0])

    print('Evaluating...')
    evaluate(model, d_test, predictions=args.predictions, indices=indices)


if __name__ == "__main__":
    # Parse args and run
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True,
                        help='Input dataset folder.')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of file to output model.')
    parser.add_argument('--max_evals', type=int, default=100,
                        help='Number of hyperparameter tuning runs.')
    parser.add_argument('--metric', type=str, default='average_precision',
                        help='Metric to monitoring during tuning.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Number of rounds for early stopping')
    parser.add_argument('--predictions', type=Path, default=None,
                        help='Path to save predictions.')
    parser.add_argument('--split', type=str, default=None,
                        help='Split to use for predictions.')
    args = parser.parse_args()

    if args.predictions:
        predict(args)
    else:
        main(args)