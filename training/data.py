"""Reads the supplied source files and pairs cell features with recorded labels."""

from pathlib import Path

import pandas as pd



def read_fofct(path):
    """Reads a FOF-CT table whose column names are stored in a metadata line."""
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("##columns=("):
                names = line.strip().removeprefix("##columns=(").removesuffix(")")
                columns = names.split(",")
                # pandas handles the CSV values; comment lines contain metadata.
                return pd.read_csv(path, names=columns, header=None, comment="#")
    raise ValueError(f"No FOF-CT column declaration was found in {path}.")


def core_paths(data_directory):
    """Returns the four reviewed coordinate files below the supplied data folder."""
    raw = Path(data_directory) / "raw"
    return {
        1: raw / "biorep01/4DNFIQCHBYZ6.csv",
        2: raw / "biorep02/4DNFIA855FW4.csv",
        3: raw / "biorep03/4DNFIY3XDBW8.csv",
        4: raw / "biorep04/4DNFI593PAKT.csv",
    }


def read_labels(data_directory):
    """Maps the six named source classes to network classes 0 through 5."""
    folder = Path(data_directory) / "labels"
    mapping = pd.read_csv(folder / "cell_type_mapping.csv").set_index("paper_code")
    paper_codes = list(range(2, 8))
    names = mapping.loc[paper_codes, "cell_type_name"].tolist()
    labels = pd.read_csv(folder / "Cell label.csv").set_index("Cell_ID", verify_integrity=True)
    labels = labels.loc[labels["Cell_Type"].isin(paper_codes)].copy()
    # The network's output positions start at zero, unlike the paper's class codes.
    labels["target"] = labels["Cell_Type"].map(dict(zip(paper_codes, range(6))))
    return labels, names


