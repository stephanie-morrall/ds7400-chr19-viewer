"""Trains fresh residual and shallow models on the recorded three-class task."""

import argparse
import json
from pathlib import Path

import torch

from training.data import core_paths
from training.evaluation import metrics
from training.final_model import ResidualMLP, baseline_model, digest, fit_practice, selected_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2000)
    args = parser.parse_args()
    # Fresh outputs preserve the evaluated submission checkpoints.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tensors, schema = selected_inputs(args.data)
    settings = {'seed': args.seed, 'epochs': args.epochs, 'batch_size': 64,
                'learning_rate': 0.01, 'early_stopping_patience': 200}
    paths = [core_paths(args.data)[rep] for rep in [1, 2, 3]]
    paths += [args.data / 'labels' / name for name in
              ['Cell label.csv', 'cell_type_mapping.csv']]
    source_hashes = {str(p.relative_to(args.data)): digest(p) for p in paths}
    # Both models receive identical examples, seed, SGD settings, and checkpoint rule.
    for name, factory in [('residual', ResidualMLP), ('baseline', baseline_model)]:
        torch.manual_seed(args.seed)
        model = factory()
        folder = args.output / name
        folder.mkdir()
        torch.save(model.state_dict(), folder / 'initial.pt')
        history, epoch, stopping = fit_practice(model, *tensors, settings)
        torch.save(model.state_dict(), folder / 'weights.pt')
        history.to_csv(folder / 'history.csv', index=False)
        measured = {split: metrics(model, x, y, schema['class_names'])
                    for split, x, y in [('training', tensors[0], tensors[1]),
                                        ('validation', tensors[2], tensors[3])]}
        record = {**schema, 'training': settings, 'model': name, 'best_epoch': epoch,
                  'dropout': 0.8 if name == 'residual' else 0.0, 'stopping': stopping,
                  'source_sha256': source_hashes, 'metrics': measured,
                  'test_evaluated': False}
        (folder / 'run.json').write_text(json.dumps(record, indent=2))
        print(name, 'best epoch', epoch, 'validation macro F1',
              measured['validation']['macro_f1'])


if __name__ == '__main__':
    main()
