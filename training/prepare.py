"""Builds filtered distance tables using a training-only locus mask."""

import numpy as np
import pandas as pd
import torch

from training.cleaning import fit_cleaning
from training.data import core_paths, read_fofct, read_labels
from training.features import features_for_cells


def _labeled_features(spots_by_replicate, labels, protocol):
    """Pairs cell feature rows with labels and summarizes exclusions by class."""
    tables, reports = {}, {}
    for replicate, spots in spots_by_replicate.items():
        features, counts = features_for_cells(spots, protocol)
        # The inner join keeps supported, coordinate-bearing cells for classification.
        cells = features.join(labels[["target"]], how="inner")
        if cells.empty:
            raise ValueError(
                f"Replicate {replicate} has no labeled cells passing quality filters."
            )
        tables[replicate] = cells
        # Class audits use labels after geometry construction, keeping targets outside inputs.
        for stage in ["raw", "after_filter_b", "after_filter_c", "after_interpolation"]:
            trace_counts = counts[stage].pop("_trace_counts_by_cell")
            per_class = {}
            for cell, traces in trace_counts.items():
                target = (
                    str(int(labels.loc[cell, "target"])) if cell in labels.index else "excluded"
                )
                values = per_class.setdefault(target, {"cells": 0, "traces": 0})
                values["cells"] += 1
                values["traces"] += traces
            counts[stage]["per_class"] = per_class
        counts["labeled_retained"] = {
            "cells": len(cells),
            "traces": sum(
                values["traces"] for target, values in
                counts["after_filter_c"]["per_class"].items() if target != "excluded"
            ),
            "per_class": {str(int(target)): int(count) for target, count in
                          cells["target"].value_counts().sort_index().items()},
        }
        reports[str(replicate)] = counts
    combined = pd.concat(tables, names=["replicate", "Cell_ID"])
    return combined.drop(columns="target"), combined["target"].astype("int64"), reports


def load_cells(paths, labels, protocol):
    """Reads requested replicate files and returns quality-passing labeled cells.

    Parameters
    ----------
    paths : dict[int, pathlib.Path]
        Explicit source files keyed by replicate identity.
    labels : pandas.DataFrame
        Supported target indices keyed by Cell_ID.
    protocol : dict
        Previously fitted training-only cleaning contract.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, dict]
        Ordered distances, integer targets and replicate/class exclusion counts
        used for evaluating the requested fixed split.
    """
    # Only caller-requested coordinate files are read at this stage.
    spots = {replicate: read_fofct(path) for replicate, path in paths.items()}
    return _labeled_features(spots, labels, protocol)


def prepare_splits(data_directory):
    """Prepares training and validation without reading replicate 4 coordinates.

    Parameters
    ----------
    data_directory : pathlib.Path
        Source folder containing raw replicate files and recorded labels.

    Returns
    -------
    tuple
        Train/validation tables and targets, cleaning protocol, class names and
        exclusion counts for model fitting and reporting.
    """
    paths = core_paths(data_directory)
    labels, class_names = read_labels(data_directory)
    # Only raw training tables determine Filter A; validation cannot change the mask.
    spots = {replicate: read_fofct(paths[replicate]) for replicate in [1, 2, 3]}
    protocol = fit_cleaning({replicate: spots[replicate] for replicate in [1, 2]})
    features, targets, counts = _labeled_features(spots, labels, protocol)
    training = features.index.get_level_values("replicate").isin([1, 2])
    return (
        features.loc[training], features.loc[~training],
        targets.loc[training], targets.loc[~training], protocol, class_names, counts,
    )


def feature_tensor(features):
    """Converts complete unscaled distances directly into float32 inputs.

    Parameters
    ----------
    features : pandas.DataFrame
        Quality-filtered, interpolated cell distances in saved feature order.

    Returns
    -------
    torch.Tensor
        Finite cells-by-distances inputs retaining the physical distance scale
        for the network and shallow benchmark.
    """
    values = features.to_numpy(dtype=np.float32)
    if features.empty or not np.isfinite(values).all():
        raise ValueError("Distance inputs must contain finite, quality-passing cell rows.")
    # Conversion preserves numeric distances apart from float32 rounding.
    return torch.tensor(values, dtype=torch.float32)
