"""Turns chromosome spots from one replicate into one distance row per cell."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from training.cleaning import clean_spots


def features_for_cells(spots, protocol):
    """Builds one mean-distance row per cell after the saved quality procedure.

    Parameters
    ----------
    spots : pandas.DataFrame
        Measured coordinates for one replicate or application cell.
    protocol : dict
        Training-fitted mask and approved quality/interpolation settings.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Cell rows in saved locus-pair order and filtering counts. Cells without
        surviving traces have no feature row.
    """
    # Shared cleaning makes training and application distances follow the same rules.
    cleaned, counts = clean_spots(spots, protocol)
    loci = protocol["retained_loci"]
    names = [f"{left}:{right}" for left, right in combinations(loci, 2)]
    if cleaned.empty:
        empty = pd.DataFrame(columns=names, index=pd.Index([], name="Cell_ID"), dtype=float)
        return empty, counts
    # Every passing trace has a complete retained-locus coordinate order.
    traces = cleaned.pivot(
        index=["Cell_ID", "Trace_ID"], columns="Chrom_Start", values=["X", "Y", "Z"]
    )
    xyz = np.stack(
        [traces[axis].reindex(columns=loci).to_numpy() for axis in ["X", "Y", "Z"]],
        axis=-1,
    )
    # SciPy produces one distance per unordered locus pair within each trace.
    distances = np.stack([pdist(trace) for trace in xyz])
    pairs = pd.DataFrame(distances, index=traces.index, columns=names)
    # Each cell receives mean distances across its quality-passing traces.
    return pairs.groupby(level="Cell_ID").mean(), counts
