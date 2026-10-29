"""Serves the frozen three-class residual classifier and trains practice copies."""

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.class_selection import prepare_selected_splits
from training.prepare import feature_tensor
from training.train import evaluate, train_epoch


class ResidualMLP(nn.Module):
    """Maps 1,176 distances to three class scores through a residual branch."""

    def __init__(self, dropout=0.8):
        super().__init__()
        self.stem = nn.Linear(1176, 32)
        self.first = nn.Linear(32, 32)
        self.second = nn.Linear(32, 32)
        self.head = nn.Linear(32, 3)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # The second hidden layer feeds a dropout-regularized linear branch.
        x = self.activation(self.stem(x))
        branch = self.second(self.dropout(self.activation(self.first(x))))
        return self.head(self.activation(x + branch))


def baseline_model():
    """Creates the fixed one-hidden-layer comparison model."""
    return nn.Sequential(nn.Linear(1176, 32), nn.ReLU(), nn.Dropout(0), nn.Linear(32, 3))


def digest(path):
    """Returns a file's SHA-256 identity."""
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def verify_freeze(folder):
    """Checks the fixed model, metadata and feature implementation identities."""
    folder = Path(folder)
    freeze = json.loads((folder / 'freeze.json').read_text())
    for name, expected in freeze['artifacts'].items():
        if digest(folder / name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    for name, expected in freeze['implementation'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Feature or model implementation changed: {name}')
    return freeze


def load_final(folder):
    """Loads the verified predictor and its saved feature and label contract."""
    folder = Path(folder)
    verify_freeze(folder)
    metadata = json.loads((folder / 'run.json').read_text())
    model = ResidualMLP(metadata['dropout'])
    model.load_state_dict(torch.load(folder / 'weights.pt', map_location='cpu', weights_only=True))
    model.eval()
    return {'model': model, 'metadata': metadata}


def selected_inputs(data_directory):
    """Builds train and validation tensors from the training-selected three classes."""
    train, val, train_y, val_y, cleaning, names, counts = prepare_selected_splits(data_directory)
    tensors = (feature_tensor(train), torch.tensor(train_y.to_numpy(), dtype=torch.long),
               feature_tensor(val), torch.tensor(val_y.to_numpy(), dtype=torch.long))
    schema = {'cleaning': cleaning, 'loci': cleaning['retained_loci'],
              'feature_names': train.columns.tolist(), 'class_names': names,
              'filter_counts': counts}
    return tensors, schema


def fit_practice(model, train_x, train_y, validation_x, validation_y, settings, on_epoch=None):
    """Fits a separate model and restores the best validation macro-F1 checkpoint."""
    generator = torch.Generator().manual_seed(settings['seed'] + 10000)
    batches = DataLoader(TensorDataset(train_x, train_y), batch_size=settings['batch_size'],
                         shuffle=True, generator=generator)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings['learning_rate'])
    history, best_f1, best_loss, stale = [], -1.0, float('inf'), 0
    # Epoch zero records the initialized weights before any gradient updates.
    for epoch in range(settings['epochs'] + 1):
        if epoch:
            train_epoch(model, batches, nn.CrossEntropyLoss(), optimizer)
        for split, x, y in [('train', train_x, train_y), ('validation', validation_x, validation_y)]:
            history.append({'epoch': epoch, 'split': split, **evaluate(model, x, y)})
        value = history[-1]
        if value['macro_f1'] > best_f1:
            best_f1, best_loss, best_epoch, stale = value['macro_f1'], value['loss'], epoch, 0
            weights = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            # Equal F1 uses lower cross-entropy as the predeclared tie breaker.
            if value['macro_f1'] == best_f1 and value['loss'] < best_loss:
                best_loss, best_epoch = value['loss'], epoch
                weights = copy.deepcopy(model.state_dict())
        if on_epoch:
            on_epoch(epoch, [dict(row) for row in history[-2:]])
        if stale >= settings['early_stopping_patience']:
            break
    model.load_state_dict(weights)
    stopping = {'reason': 'validation_patience_exhausted' if stale >= settings['early_stopping_patience']
                else 'maximum_epochs_reached', 'completed_epochs': epoch,
                'maximum_epochs': settings['epochs'], 'patience': settings['early_stopping_patience'],
                'epochs_without_improvement': stale, 'best_epoch': best_epoch,
                'best_validation_macro_f1': best_f1, 'checkpoint_restored': True,
                'comparison': 'maximum_macro_f1_then_minimum_loss_on_ties'}
    return pd.DataFrame(history), best_epoch, stopping
