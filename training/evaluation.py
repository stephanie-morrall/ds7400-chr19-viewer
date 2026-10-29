"""Measures fixed classifiers on a specified split."""

import torch
from sklearn.metrics import classification_report, balanced_accuracy_score


def metrics(model, x, y, names):
    """Measures one fixed model on a specified labeled tensor."""
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y).item()
    truth, predicted = y.numpy(), logits.argmax(1).numpy()
    report = classification_report(truth, predicted, labels=list(range(len(names))),
                                   target_names=names, output_dict=True, zero_division=0)
    accuracy = float((truth == predicted).mean())
    return {'n_cells': len(truth), 'n_errors': int((truth != predicted).sum()), 'loss': loss,
            'accuracy': accuracy, 'error': 1 - accuracy,
            'balanced_accuracy': float(balanced_accuracy_score(truth, predicted)),
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'], 'per_class': report}
