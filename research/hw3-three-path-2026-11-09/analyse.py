"""Checks six HW3 fits and writes aggregate results without publishing cell records."""

# Standard-library records keep settings and completed artifact identities explicit.
import argparse
import hashlib
import json
from pathlib import Path

# Independent NumPy and sklearn calculations check saved predictions and losses.
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]


def digest(path):
    """Returns the byte identity of an input, source or completed artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_predictions(path, split, observed, classes):
    """Recomputes metrics while checking the exact original cohort and class order."""
    with np.load(path, allow_pickle=False) as saved:
        for name, suffix in (("target", "y"), ("cell", "cell"), ("replicate", "replicate")):
            if not np.array_equal(saved[name], observed[f"{split}_{suffix}"]):
                raise ValueError(f"Prediction cohort differs: {path.name}, {name}")
        target = saved["target"]
        logits = saved["logits"].astype(np.float64)
        predicted = logits.argmax(1)
        if not np.array_equal(predicted, saved["prediction"]):
            raise ValueError("Saved classes differ from the saved logits.")
    if logits.shape != (len(target), 3) or not np.isfinite(logits).all():
        raise ValueError("Expected finite three-class logits.")
    shifted = logits - logits.max(1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(target)), target]
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, labels=[0, 1, 2], zero_division=0,
    )
    values = {
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "loss": float(loss.mean()),
    }
    per_class = [
        {"class": name, "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i])}
        for i, name in enumerate(classes)
    ]
    return values, per_class


def analyse(results, output):
    """Verifies the declared six full-budget fits before exporting aggregate tables."""
    config = json.loads((FOLDER / "training_config.json").read_text())
    if config["conditions"] != ["attention", "no_attention"] or config["seeds"] != [42, 43, 44]:
        raise ValueError("This analysis requires the declared two-arm, three-seed panel.")
    for identity in config["inputs"].values():
        if digest(ROOT / identity["path"]) != identity["sha256"]:
            raise ValueError("An authorized input cache differs from its recorded identity.")
    rows, class_rows, histories = [], [], []
    with np.load(ROOT / config["inputs"]["observed"]["path"], allow_pickle=False) as observed:
        for condition in config["conditions"]:
            for seed in config["seeds"]:
                directory = results / condition / f"seed{seed}"
                complete = json.loads((directory / "completion.json").read_text())
                request = complete["request"]
                if (not complete["complete"] or request["scope"] != "full_budget"
                        or request["condition"] != condition or request["seed"] != seed):
                    raise ValueError("A complete, correctly identified full fit is required.")
                if request["config_sha256"] != digest(FOLDER / "training_config.json"):
                    raise ValueError("The fit used a different configuration.")
                for name, identity in complete["artifacts_sha256"].items():
                    if digest(directory / name) != identity:
                        raise ValueError(f"Completed artifact changed: {name}")
                for name, identity in request["source_sha256"].items():
                    if digest(ROOT / name) != identity:
                        raise ValueError(f"Fitted source changed: {name}")
                if json.loads((directory / "config.json").read_text()) != config:
                    raise ValueError("Saved configuration values differ.")
                metrics = json.loads((directory / "metrics.json").read_text())
                stop = metrics["stopping"]
                history = pd.read_csv(directory / "history.csv")
                validation = history[history.split.eq("validation")]
                selected = validation.sort_values(
                    ["macro_f1", "loss", "epoch"], ascending=[False, True, True],
                ).iloc[0]
                if int(selected.epoch) != stop["best_epoch"]:
                    raise ValueError("Selected checkpoint differs from validation history.")
                if stop["maximum_epochs"] != config["maximum_epochs"]:
                    raise ValueError("The full training cap differs.")
                if stop["patience"] != config["early_stopping_patience"]:
                    raise ValueError("The stopping patience differs.")
                if not stop["checkpoint_restored"]:
                    raise ValueError("Saved predictions must use the selected checkpoint.")
                best, stale = -1.0, 0
                for row in validation.itertuples():
                    if row.macro_f1 > best:
                        best, stale = row.macro_f1, 0
                    else:
                        stale += 1
                    if stale >= stop["patience"] and row.epoch != stop["completed_epochs"]:
                        raise ValueError("A fit continued after its stopping rule.")
                if stale != stop["epochs_without_f1_improvement"]:
                    raise ValueError("The saved patience counter differs.")
                row = {"condition": condition, "seed": seed,
                       "selected_epoch": stop["best_epoch"],
                       "completed_epochs": stop["completed_epochs"],
                       "wall_seconds": metrics["wall_seconds"]}
                for split in ("train", "validation"):
                    split_history = history[history.split.eq(split)]
                    if split_history.epoch.tolist() != list(range(stop["completed_epochs"] + 1)):
                        raise ValueError("Training history has missing or repeated epochs.")
                    checked, classes = check_predictions(
                        directory / f"predictions_{split}.npz", split, observed,
                        config["class_names"],
                    )
                    for name, value in checked.items():
                        tolerance = 2e-6 if name == "loss" else 1e-12
                        if not np.isclose(value, metrics["metrics"][split][name],
                                          atol=tolerance, rtol=tolerance):
                            raise ValueError(f"Recomputed {split} {name} differs.")
                        row[f"{split}_{name}"] = value
                    class_rows.extend({"condition": condition, "seed": seed,
                                       "split": split, **item} for item in classes)
                rows.append(row)
                histories.append((condition, seed, history, stop["best_epoch"]))

    # Only aggregate metrics leave this analysis; source cell identifiers stay local.
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per-seed.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per-class.csv", index=False)
    summary = []
    for condition, group in frame.groupby("condition", sort=False):
        item = {"condition": condition, "n_seeds": len(group)}
        for name in ("validation_macro_f1", "validation_balanced_accuracy",
                     "validation_loss", "train_loss", "wall_seconds"):
            item[f"{name}_mean"] = float(group[name].mean())
            item[f"{name}_sample_sd"] = float(group[name].std(ddof=1))
        summary.append(item)
    pd.DataFrame(summary).to_csv(output / "condition-summary.csv", index=False)
    paired = frame.pivot(index="seed", columns="condition", values="validation_macro_f1")
    paired["attention_minus_bypass"] = paired.attention - paired.no_attention
    paired.to_csv(output / "paired-attention-deltas.csv")
    baseline_path = ROOT / config["hw2_frozen_folder"] / "validation_metrics.json"
    baseline = json.loads(baseline_path.read_text())["validation"]["improved"]
    result = {"status": "COMPLETE_VERIFIED", "unit": "cell", "fits": len(rows),
              "training_protocol": config["protocol"], "class_names": config["class_names"],
              "cohort_counts": config["expected_cells"], "conditions": summary,
              "paired_attention_minus_bypass_mean": float(paired.attention_minus_bypass.mean()),
              "seed_variation": "Sample SD across seeds on the same validation cells.",
              "original_hw2_validation_macro_f1": baseline["macro_f1"],
              "limits": ["Private input caches are required for independent regeneration.",
                         "Original HW2 uses different preprocessing and training settings."]}
    (output / "regenerated-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=FOLDER / "results")
    parser.add_argument("--output", type=Path, default=FOLDER / "report")
    arguments = parser.parse_args()
    analyse(arguments.results, arguments.output)
