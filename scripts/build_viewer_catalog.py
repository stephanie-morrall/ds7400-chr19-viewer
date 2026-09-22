"""Builds the cell catalog the HW1 gallery reads.

Reads the gitignored FOF-CT tables for biorep01–04, joins each spot to its
lamina and nucleolus distances, and writes one JSON file per cell plus a catalog
the gallery paginates and filters. The browser loads those files from
web/public/biorep0N/, which is gitignored so the CSVs never go on GitHub.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
LABELS = ROOT / "data" / "labels"
PUBLIC = ROOT / "web" / "public"

# Labeled cells with chr19 spots, used as a write-time check.
EXPECTED_TOTAL = 13306
EXPECTED_BY_REP = {1: 3093, 2: 3252, 3: 3220, 4: 3741}
EXPECTED_BIOREP01_BY_TYPE = {
    "Hepatocyte": 998,
    "Unknown": 638,
    "Erythroblast": 596,
    "Proerythroblast": 435,
    "Macrophage": 151,
    "Other": 122,
    "Endothelial": 97,
    "Megakaryocyte": 56,
}


def skip_fofct_comments(handle):
    """Yields data rows after FOF-CT metadata lines that start with #."""
    for line in handle:
        if line.startswith("#"):
            continue
        yield line


def table_namespace(path):
    """Reads the 4DN table class from the FOF-CT header comments."""
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.startswith("##Table_namespace="):
                return line.strip().split("=", 1)[1]
    return None


def find_table(folder, namespace):
    """Finds the CSV in a biorep folder whose header matches a 4DN namespace."""
    for path in folder.glob("*.csv"):
        if table_namespace(path) == namespace:
            return path
    raise FileNotFoundError(f"{folder} has no {namespace} table")


def load_type_names():
    """Maps paper_code (as a string) to the cell-type name, in file order."""
    order = []
    names = {}
    with (LABELS / "cell_type_mapping.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            code = row["paper_code"]
            names[code] = row["cell_type_name"]
            order.append(code)
    return names, order


def load_cell_types(names):
    """Maps each Cell_ID to its cell-type name via the paper code in the label table."""
    types = {}
    with (LABELS / "Cell label.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            types[row["Cell_ID"]] = names[row["Cell_Type"]]
    return types


def um(value):
    """Rounds a micron coordinate so each cell JSON stays compact."""
    return round(float(value), 6)


def load_distances(bio_table):
    """Maps (Spot_ID, Trace_ID) to (distance to lamina, distance to nucleolus)."""
    distances = {}
    with bio_table.open(encoding="utf-8-sig", newline="") as handle:
        for line in skip_fofct_comments(handle):
            spot_id, trace_id, lamina, nucleolus = line.rstrip().split(",")[:4]
            distances[(spot_id, trace_id)] = (float(lamina), float(nucleolus))
    return distances


def load_spots(core_table, distances, cell_types):
    """Groups joined xyz+distance rows by Cell_ID."""
    by_cell = defaultdict(list)
    with core_table.open(encoding="utf-8-sig", newline="") as handle:
        for line in skip_fofct_comments(handle):
            cols = line.rstrip().split(",")
            cell_id = cols[8]
            lamina, nucleolus = distances[(cols[0], cols[1])]
            # Reads Chrom_Start and Chrom_End, the 100 kb probe window on chr19.
            by_cell[cell_id].append(
                {
                    "Spot_ID": cols[0],
                    "Trace_ID": cols[1],
                    "X": um(cols[2]),
                    "Y": um(cols[3]),
                    "Z": um(cols[4]),
                    "Chrom_Start": int(float(cols[6])),
                    "Chrom_End": int(float(cols[7])),
                    "Distance_To_Lamina": um(lamina),
                    "Distance_To_Nucleolus": um(nucleolus),
                    "cellType": cell_types[cell_id],
                }
            )
    return by_cell


def write_cells(by_cell, cells_dir, replicate, grouped):
    """Writes one cells/{Cell_ID}.json per cell and appends catalog rows."""
    cells_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for cell_id, spots in by_cell.items():
        cell_type = spots[0]["cellType"]
        traces = {spot["Trace_ID"] for spot in spots}
        path = cells_dir / f"{cell_id}.json"
        if path.exists():
            skipped += 1
        else:
            payload = {
                "cellId": cell_id,
                "cellType": cell_type,
                "replicate": replicate,
                "spots": [
                    {key: spot[key] for key in (
                        "Spot_ID",
                        "Trace_ID",
                        "X",
                        "Y",
                        "Z",
                        "Chrom_Start",
                        "Chrom_End",
                        "Distance_To_Lamina",
                        "Distance_To_Nucleolus",
                    )}
                    for spot in spots
                ],
            }
            path.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
            written += 1
            if written % 500 == 0:
                print(f"  wrote {written} new cells…")
        grouped[cell_type].append(
            {
                "id": cell_id,
                "nSpots": len(spots),
                "nTraces": len(traces),
                "replicate": replicate,
            }
        )
    return written, skipped


def check_counts(catalog):
    """Raises if the catalog does not match the known labeled-cell counts."""
    if catalog["cellCount"] != EXPECTED_TOTAL:
        raise SystemExit(
            f"Expected {EXPECTED_TOTAL} cells, wrote {catalog['cellCount']}"
        )
    by_rep = defaultdict(int)
    by_rep_type = defaultdict(lambda: defaultdict(int))
    for entry in catalog["types"]:
        for cell in entry["cells"]:
            by_rep[cell["replicate"]] += 1
            by_rep_type[cell["replicate"]][entry["name"]] += 1
    for rep, expected in EXPECTED_BY_REP.items():
        got = by_rep.get(rep, 0)
        if got != expected:
            raise SystemExit(
                f"replicate {rep}: expected {expected} cells, wrote {got}"
            )
    for name, expected in EXPECTED_BIOREP01_BY_TYPE.items():
        got = by_rep_type[1].get(name, 0)
        if got != expected:
            raise SystemExit(
                f"biorep01 {name}: expected {expected} cells, wrote {got}"
            )


def main():
    """Writes per-replicate cell JSON and one combined catalog.json."""
    names, type_order = load_type_names()
    cell_types = load_cell_types(names)
    grouped = defaultdict(list)
    new_files = 0
    for replicate in (1, 2, 3, 4):
        folder = RAW_ROOT / f"biorep{replicate:02d}"
        print(f"biorep{replicate:02d}")
        core = find_table(folder, "4dn_FOF-CT_core")
        bio = find_table(folder, "4dn_FOF-CT_bio")
        distances = load_distances(bio)
        by_cell = load_spots(core, distances, cell_types)
        out_dir = PUBLIC / f"biorep{replicate:02d}" / "cells"
        written, skipped = write_cells(
            by_cell, out_dir, replicate, grouped,
        )
        new_files += written
        print(f"  {len(by_cell)} cells ({written} new, {skipped} already present)")

    types = []
    # Groups cells by the eight names in cell_type_mapping.csv, Unknown and Other included.
    for code in type_order:
        name = names[code]
        cells = sorted(
            grouped[name],
            key=lambda row: (row["replicate"], int(row["id"])),
        )
        types.append({"name": name, "code": code, "cells": cells})

    catalog = {
        "cellCount": sum(len(entry["cells"]) for entry in types),
        "replicates": [1, 2, 3, 4],
        "types": types,
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "catalog.json").write_text(
        json.dumps(catalog, separators=(",", ":")),
        encoding="utf-8",
    )
    check_counts(catalog)
    print(f"{catalog['cellCount']} cells in catalog.json")
    print(f"{new_files} new cell JSON files")
    for entry in catalog["types"]:
        n = len(entry["cells"])
        pages = (n + 99) // 100
        print(f"  {entry['name']}: {n} cells, {pages} page(s)")


if __name__ == "__main__":
    if "--span-only" in sys.argv:
        raise SystemExit(
            "previewSpanUm is no longer written; gallery tiles fit each cell"
        )
    main()
