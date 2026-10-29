"""Evaluate the frozen seed-42 pair once on previously used replicate 4."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import torch

from training.class_selection import apply_class_selection
from training.data import core_paths
from training.final_model import baseline_model, digest, load_final, verify_freeze
from training.evaluation import metrics
from training.prepare import feature_tensor, load_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    marker = args.model / 'test_access_started.json'
    if marker.exists() or (args.model / 'test_metrics.json').exists():
        raise FileExistsError('Replicate 4 evaluation already started. Preserve the existing result.')
    # Validates both fixed checkpoints and writes a one-time record before reading test data.
    freeze = verify_freeze(args.model)
    predictor = load_final(args.model)
    baseline = baseline_model()
    baseline.load_state_dict(torch.load(args.model / 'baseline.pt', weights_only=True,
                                        map_location='cpu'))
    baseline.eval()
    marker.write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
                                  'freeze_sha256': digest(args.model / 'freeze.json'),
                                  'note': freeze['test']}, indent=2))
    metadata = predictor['metadata']
    labels = pd.read_csv(args.data / 'labels/Cell label.csv').set_index('Cell_ID',
                                                                       verify_integrity=True)
    labels = labels.loc[labels['Cell_Type'].isin(range(1, 9))].copy()
    labels['target'] = labels['Cell_Type']
    path = core_paths(args.data)[4]
    features, paper_y, counts = load_cells({4: path}, labels, metadata['cleaning'])
    features, y, selection_counts = apply_class_selection(features, paper_y, metadata['selection'])
    if features.columns.tolist() != metadata['feature_names']:
        raise ValueError('Test feature order differs from the frozen contract.')
    x, truth = feature_tensor(features), torch.tensor(y.to_numpy(), dtype=torch.long)
    result = {name: metrics(model, x, truth, metadata['class_names'])
              for name, model in [('improved', predictor['model']), ('baseline', baseline)]}
    result.update({'improved_has_lower_error': result['improved']['n_errors'] <
                   result['baseline']['n_errors'], 'filter_counts': counts,
                   'class_selection_counts': selection_counts, 'source_sha256': digest(path),
                   'evaluation_note': 'Replicate 4 was used by earlier models. This reused '
                                      'holdout is not an unseen final test.',
                   'freeze_sha256': digest(args.model / 'freeze.json')})
    (args.model / 'test_metrics.json').write_text(json.dumps(result, indent=2))
    # Paired outcomes support an audit without publishing source-row identifiers.
    with torch.inference_mode():
        frame = pd.DataFrame({'target': y.to_numpy(),
                              'improved': predictor['model'](x).argmax(1).numpy(),
                              'baseline': baseline(x).argmax(1).numpy()})
    frame.to_csv(args.model / 'test_predictions.csv', index=False)
    print(json.dumps({k: result[k] for k in ['improved', 'baseline',
                                               'improved_has_lower_error']}, indent=2))


if __name__ == '__main__':
    main()
