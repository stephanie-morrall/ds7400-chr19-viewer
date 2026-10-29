"""Filters chromosome traces and fills their coordinates before distance features."""

import math

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


# These thresholds reproduce the reviewed chromosome-trace quality rules.
THRESHOLDS = {
    "max_locus_missing_rate": 0.50,
    "max_traces_per_cell": 4,
    "max_trace_missing_rate": 0.20,
}
TRACE_KEYS = ["Cell_ID", "Trace_ID"]
XYZ = ["X", "Y", "Z"]


def fit_cleaning(training_spots):
    """Learns the retained locus order from raw training replicates.

    Parameters
    ----------
    training_spots : dict[int, pandas.DataFrame]
        Measured spot tables from training replicates only.

    Returns
    -------
    dict
        JSON-serializable mask and settings reused for all prediction inputs.
    """
    # The first training replicate supplies the full ordered assay axis.
    first = next(iter(training_spots.values()))
    raw_loci = sorted(int(value) for value in first["Chrom_Start"].unique())
    rates = {}
    for replicate, spots in training_spots.items():
        trace_count = len(spots[TRACE_KEYS].drop_duplicates())
        if not trace_count or not spots["Chrom_Start"].isin(raw_loci).all():
            raise ValueError("Training spots are empty or contain an unknown locus.")
        # A trace contributes at most one observation to each locus's coverage.
        observed = spots[TRACE_KEYS + ["Chrom_Start"]].drop_duplicates()
        observed_counts = observed.groupby("Chrom_Start").size().reindex(raw_loci, fill_value=0)
        rates[str(replicate)] = 1.0 - observed_counts / trace_count
    # A locus passes when every training replicate misses it at most half the time.
    worst_rate = pd.DataFrame(rates).max(axis=1)
    retained = worst_rate.index[worst_rate <= THRESHOLDS["max_locus_missing_rate"]]
    if len(retained) < 2:
        raise ValueError("Filter A retained fewer than two loci for distance features.")
    return {
        "raw_loci": raw_loci,
        "retained_loci": [int(value) for value in retained],
        "thresholds": dict(THRESHOLDS),
        "interpolation": "linear",
        "endpoint_policy": "linear_extrapolation",
        "coordinate_axis": "locus_rank",
        "standardize": False,
        "training_locus_missingness": {
            replicate: {str(locus): float(rate) for locus, rate in values.items()}
            for replicate, values in rates.items()
        },
    }


def _stage_counts(spots):
    """Counts cells, traces and coordinate rows for one filtering stage."""
    # The temporary map supplies class-aware audits in prepare.load_cells.
    per_cell = spots[TRACE_KEYS].drop_duplicates().groupby("Cell_ID").size()
    return {
        "spots": len(spots), "traces": int(per_cell.sum()), "cells": len(per_cell),
        "_trace_counts_by_cell": {int(cell): int(count) for cell, count in per_cell.items()},
    }


def _passing_spots(spots, protocol):
    """Applies raw cell-count and measured retained-locus coverage filters."""
    thresholds = protocol["thresholds"]
    counts = {"raw": _stage_counts(spots)}
    # Filter B removes every trace from a cell whose raw trace count exceeds four.
    per_cell = spots[TRACE_KEYS].drop_duplicates().groupby("Cell_ID").size()
    passing_cells = per_cell.index[per_cell <= thresholds["max_traces_per_cell"]]
    after_b = spots.loc[spots["Cell_ID"].isin(passing_cells)]
    counts["after_filter_b"] = _stage_counts(after_b)
    # Filter C counts measured retained loci before any coordinate is filled.
    retained = protocol["retained_loci"]
    minimum = math.ceil((1 - thresholds["max_trace_missing_rate"]) * len(retained))
    measured = after_b.loc[after_b["Chrom_Start"].isin(retained)]
    coverage = measured[TRACE_KEYS + ["Chrom_Start"]].drop_duplicates().groupby(TRACE_KEYS).size()
    passing_traces = coverage.index[coverage >= minimum]
    keep = pd.MultiIndex.from_frame(after_b[TRACE_KEYS]).isin(passing_traces)
    after_c = after_b.loc[keep]
    counts["after_filter_c"] = _stage_counts(after_c)
    counts["minimum_observed_loci"] = minimum
    counts["retained_loci"] = len(retained)
    return after_c, counts


def _fill_trace(xyz):
    """Fills one trace on rank, preserving measured anchors exactly."""
    ranks = np.arange(len(xyz))
    observed = np.isfinite(xyz).all(axis=1)
    # A degree-one spline extends the nearest line at both ends of the rank axis.
    linear = make_interp_spline(ranks[observed], xyz[observed], k=1)
    filled = linear(ranks)
    filled[observed] = xyz[observed]
    return filled


def clean_spots(spots, protocol):
    """Returns complete retained-locus XYZ rows after quality filtering.

    Parameters
    ----------
    spots : pandas.DataFrame
        Measured Cell_ID, Trace_ID, Chrom_Start and finite X/Y/Z columns.
    protocol : dict
        Saved output of fit_cleaning, including the full and retained axes.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Filled coordinate rows and stage counts for distance construction and
        exclusion reporting. No surviving traces yields an empty coordinate table.
    """
    raw_loci = protocol["raw_loci"]
    if not spots["Chrom_Start"].isin(raw_loci).all():
        raise ValueError("Spots contain a locus outside the saved assay axis.")
    if not np.isfinite(spots[XYZ].to_numpy()).all():
        raise ValueError("Measured coordinates must be finite.")
    if spots.duplicated(TRACE_KEYS + ["Chrom_Start"]).any():
        raise ValueError("A trace contains duplicate measurements at the same locus.")
    # Quality decisions inspect measured observations and all raw trace counts.
    passing, counts = _passing_spots(spots, protocol)
    columns = TRACE_KEYS + ["Chrom_Start"] + XYZ + ["is_imputed"]
    retained = protocol["retained_loci"]
    if passing.empty:
        result = pd.DataFrame(columns=columns)
    else:
        # One pivot aligns all traces before SciPy fills each coordinate grid.
        wide = passing.pivot(index=TRACE_KEYS, columns="Chrom_Start", values=XYZ)
        xyz = np.stack([wide[axis].reindex(columns=raw_loci).to_numpy()
                        for axis in XYZ], axis=-1)
        observed = np.isfinite(xyz).all(axis=-1)
        filled = np.stack([_fill_trace(trace) for trace in xyz])
        # Excluded-locus measurements remain anchors until filling is complete.
        keep = np.isin(raw_loci, retained)
        result = pd.DataFrame(filled[:, keep].reshape(-1, 3), columns=XYZ)
        result["Cell_ID"] = np.repeat(wide.index.get_level_values("Cell_ID"), len(retained))
        result["Trace_ID"] = np.repeat(wide.index.get_level_values("Trace_ID"), len(retained))
        result["Chrom_Start"] = np.tile(retained, len(wide))
        result["is_imputed"] = ~observed[:, keep].reshape(-1)
        result = result[columns]
    counts["after_interpolation"] = _stage_counts(result)
    return result, counts
