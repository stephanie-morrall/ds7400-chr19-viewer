"""Selects cell types from quality-passing training cells for a new comparison."""

from pathlib import Path

import pandas as pd

from training.cleaning import fit_cleaning
from training.data import core_paths, read_fofct
from training.prepare import _labeled_features


def fit_class_selection(training_targets, class_names, min_training_cells=300):
    """Fits paper-code-to-target mapping using training cells only.

    Unknown (8) is excluded regardless of its count. All other classes (1–7)
    need at least ``min_training_cells`` quality-passing training cells.
    """
    if not isinstance(min_training_cells, int) or min_training_cells < 1:
        raise ValueError("Minimum training cell count must be a positive integer.")
    counts = training_targets.value_counts()
    retained = [code for code in range(1, 8) if counts.get(code, 0) >= min_training_cells]
    if len(retained) < 2:
        raise ValueError("Class selection must retain at least two classes.")
    return {
        "min_training_cells": min_training_cells,
        "training_counts_by_paper_code": {
            str(code): int(counts.get(code, 0)) for code in range(1, 9)
        },
        "paper_code_to_target": {str(code): target for target, code in enumerate(retained)},
        "class_names": [class_names[code] for code in retained],
        "excluded_paper_codes": [8],
    }


def apply_class_selection(features, paper_targets, selection):
    """Applies a training-fitted mapping and reports excluded cell counts."""
    mapping = {int(code): target for code, target in selection["paper_code_to_target"].items()}
    keep = paper_targets.isin(mapping)
    counts = {
        "input_cells": int(len(paper_targets)),
        "retained_cells": int(keep.sum()),
        "excluded_unknown": int((paper_targets == 8).sum()),
        "excluded_small_class": int((paper_targets.between(1, 7) & ~keep).sum()),
        "counts_by_paper_code": {
            str(code): int((paper_targets == code).sum()) for code in range(1, 9)
        },
    }
    return features.loc[keep], paper_targets.loc[keep].map(mapping).astype("int64"), counts


def prepare_selected_splits(data_directory, min_training_cells=300):
    """Prepares replicates 1–3 with class selection fitted on replicates 1–2.

    Returns the same seven values as ``prepare_splits``. The final counts include
    the fitted mapping and split-specific exclusion reasons under class_selection.
    Replicate 4 is not opened. Reuse this mapping for any future test data.
    """
    paths = core_paths(data_directory)
    folder = Path(data_directory) / "labels"
    names = pd.read_csv(folder / "cell_type_mapping.csv").set_index("paper_code")
    names = names["cell_type_name"].to_dict()
    labels = pd.read_csv(folder / "Cell label.csv").set_index("Cell_ID", verify_integrity=True)
    # Paper codes remain intact until the training-only selection has been fitted.
    labels = labels.loc[labels["Cell_Type"].isin(range(1, 9))].copy()
    labels["target"] = labels["Cell_Type"]
    spots = {replicate: read_fofct(paths[replicate]) for replicate in [1, 2, 3]}
    protocol = fit_cleaning({replicate: spots[replicate] for replicate in [1, 2]})
    features, targets, counts = _labeled_features(spots, labels, protocol)
    is_training = features.index.get_level_values("replicate").isin([1, 2])
    selection = fit_class_selection(targets.loc[is_training], names, min_training_cells)
    train_f, train_y, train_counts = apply_class_selection(
        features.loc[is_training], targets.loc[is_training], selection
    )
    val_f, val_y, val_counts = apply_class_selection(
        features.loc[~is_training], targets.loc[~is_training], selection
    )
    selection["splits"] = {"train": train_counts, "validation": val_counts}
    counts["class_selection"] = selection
    return train_f, val_f, train_y, val_y, protocol, selection["class_names"], counts
