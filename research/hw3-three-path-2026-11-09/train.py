"""Fits one declared seed and condition while saving reproducible cell-level evidence."""

# Standard-library files preserve the protocol, source identities and training history.
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

# NumPy retains aligned cells, and Torch supplies the shared two-forward training loop.
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# The frozen HW2 implementation is imported directly, preserving its actual architecture.
FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
sys.path.insert(0, str(ROOT))
from training.final_model import verify_freeze
from model import ObservedGeometryCNN


def digest(path):
    """Returns the SHA-256 identity of one source or artifact file."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    """Saves readable strict JSON, rejecting non-finite result values."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_inputs(config):
    """Loads frozen aligned cohorts without constructing or imputing new coordinates.

    Returns
    -------
    tuple[dict, np.ndarray, dict]
        Train/validation tensors, 49 genomic starts, and verified data/HW2 provenance.
    """
    # Pinned bytes prevent a stale or rebuilt cache from silently changing this experiment.
    paths = {}
    for name, identity in config["inputs"].items():
        path = ROOT / identity["path"]
        if digest(path) != identity["sha256"]:
            raise ValueError(f"Input identity changed: {path}")
        paths[name] = path
    frozen = ROOT / config["hw2_frozen_folder"]
    verify_freeze(frozen)
    metadata = json.loads((frozen / "run.json").read_text())
    saved_metrics = json.loads((frozen / "validation_metrics.json").read_text())
    if metadata["class_names"] != config["class_names"]:
        raise ValueError("Class order differs from the frozen HW2 predictor.")

    # The dense reference is exactly the selected original HW2 input, including its history.
    splits = {}
    with (np.load(paths["observed"]) as observed, np.load(paths["hw2_dense"]) as dense,
          np.load(paths["hw2_original"]) as original):
        starts = observed["locus_start"].copy()
        if not np.array_equal(starts, metadata["loci"]):
            raise ValueError("Observed locus order differs from the HW2 schema.")
        for split in ("train", "validation"):
            y = observed[f"{split}_y"]
            keep = np.isin(original[f"{split}_y"], [1, 4, 5])
            original_y = np.searchsorted([1, 4, 5], original[f"{split}_y"][keep])
            if not np.array_equal(y, original_y):
                raise ValueError(f"Original HW2 class order differs for {split}.")
            for key in ("y", "cell", "replicate", "mask"):
                if not np.array_equal(observed[f"{split}_{key}"], dense[f"{split}_{key}"]):
                    raise ValueError(f"Dense/observed {split} {key} are misaligned.")
            for key in ("cell", "replicate"):
                if not np.array_equal(observed[f"{split}_{key}"],
                                      original[f"{split}_{key}"][keep]):
                    raise ValueError(f"Original HW2 {split} {key} are misaligned.")
            mean = dense[f"{split}_mean"]
            if not np.array_equal(mean, original[f"{split}_x"][keep]):
                raise ValueError(f"The {split} reference is not the exact original HW2 tensor.")

            # Missing storage remains behind observation masks; only measured XYZ must be finite.
            measured = observed[f"{split}_locus_observed"]
            xyz = observed[f"{split}_xyz_observed"]
            traces = observed[f"{split}_mask"]
            area = observed[f"{split}_cell_area"]
            replicates = observed[f"{split}_replicate"]
            if len(y) != config["expected_cells"][split]:
                raise ValueError(f"Unexpected {split} cell count.")
            if sorted(np.unique(replicates).tolist()) != config["expected_replicates"][split]:
                raise ValueError(f"Unexpected {split} biological replicates.")
            if not np.isfinite(xyz[measured]).all() or not np.isfinite(mean).all():
                raise ValueError(f"Non-finite measured input in {split}.")
            if np.isinf(area).any():
                raise ValueError(f"Infinite measured cell area in {split}.")
            if (measured & ~traces[..., None]).any() or not traces.any(1).all():
                raise ValueError(f"Invalid trace support in {split}.")
            splits[split] = {
                "batch": {"xyz": torch.from_numpy(xyz.copy()).float(),
                          "observed": torch.from_numpy(measured.copy()),
                          "traces": torch.from_numpy(traces.copy()),
                          "area": torch.from_numpy(area.copy()).float()},
                "y": torch.from_numpy(y.copy()).long(),
                "cell": observed[f"{split}_cell"].copy(),
                "replicate": replicates.copy(),
            }
    provenance = {
        "inputs": config["inputs"], "hw2_freeze_sha256": digest(frozen / "freeze.json"),
        "dense_original_equality": True, "cell_label_order_equality": True,
        "historical_hw2": {
            "scope": config["historical_hw2_scope"], "seed": metadata["training"]["seed"],
            "validation": saved_metrics["validation"]["improved"],
            "weights_sha256": digest(frozen / "weights.pt"),
        },
    }
    return splits, starts, provenance


def sample_rotations(count, generator):
    """Returns count Haar-distributed proper 3D rotation matrices on CPU."""
    # QR of Gaussian matrices with corrected column signs samples the orthogonal group.
    gaussian = torch.randn(count, 3, 3, generator=generator)
    rotation, triangular = torch.linalg.qr(gaussian)
    signs = triangular.diagonal(dim1=-2, dim2=-1).sign()
    signs = torch.where(signs == 0, 1.0, signs)
    rotation = rotation * signs[:, None]

    # A final column flip restricts the draw to rotations with determinant positive one.
    rotation[:, :, -1] *= torch.linalg.det(rotation).sign()[:, None]
    return rotation


def batch_for(split, indices, device):
    """Selects whole cells and transfers their original measurement masks together."""
    return {key: value[indices].to(device) for key, value in split["batch"].items()}


def predict(model, split, indices, condition, device, rotation=None):
    """Returns scores with an explicit arm and optional centered-XYZ training view."""
    batch = batch_for(split, indices, device)
    if rotation is not None:
        batch["xyz_rotation"] = rotation.to(device)
    return model(batch, use_attention=condition == "attention")


def metrics_from_logits(logits, target, class_names):
    """Returns whole-split loss, confusion counts and classwise classification metrics."""
    # A fixed three-class confusion matrix keeps absent predictions from dropping a class.
    predicted = logits.argmax(1)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(matrix, (target.numpy(), predicted.numpy()), 1)
    tp = np.diag(matrix).astype(float)
    support, called = matrix.sum(1), matrix.sum(0)
    precision = np.divide(tp, called, out=np.zeros_like(tp), where=called > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, support + called, out=np.zeros_like(tp), where=support + called > 0)
    return {
        "n_cells": len(target), "loss": nn.functional.cross_entropy(logits, target).item(),
        "accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "weighted_f1": float(np.sum(f1 * support) / support.sum()),
        "confusion_matrix": matrix.tolist(), "confusion_rows": "true class",
        "confusion_columns": "predicted class", "class_names": class_names,
        "per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(class_names)
        },
    }


def evaluate(model, split, condition, device, batch_size, class_names):
    """Evaluates every original cell with dropout disabled and no training rotation."""
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(split["y"]), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(split["y"])))
            pieces.append(predict(model, split, indices, condition, device).cpu())
    logits = torch.cat(pieces)
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite whole-split evaluation logits.")
    metrics = metrics_from_logits(logits, split["y"], class_names)
    if not np.isfinite(metrics["loss"]):
        raise ValueError("Non-finite whole-split evaluation loss.")
    return metrics, logits


def two_view_loss(model, split, indices, condition, device, rotation_generator):
    """Averages original and second-view CE with two independent dropout forwards."""
    # Every arm pays for two forwards and receives one average-gradient SGD update.
    target = split["y"][indices].to(device)
    original = predict(model, split, indices, condition, device)
    rotation = sample_rotations(len(indices), rotation_generator)
    second = predict(model, split, indices, condition, device, rotation)
    return (nn.functional.cross_entropy(original, target)
            + nn.functional.cross_entropy(second, target)) / 2


def fit(model, splits, condition, seed, config, output, device, maximum_epochs):
    """Fits the common protocol and restores the selected validation checkpoint."""
    # Separate random streams make cell order and geometry draws independent of dropout.
    shuffle = torch.Generator().manual_seed(seed + config["shuffle_seed_offset"])
    rotations = torch.Generator().manual_seed(seed + config["rotation_seed_offset"])
    batches = DataLoader(torch.arange(len(splits["train"]["y"])),
                         batch_size=config["batch_size"], shuffle=True, generator=shuffle)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    best_f1, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    history, training_seconds = [], []
    columns = ["epoch", "split", "loss", "accuracy", "macro_f1", "balanced_accuracy",
               "weighted_f1", "optimization_loss", "training_seconds", "elapsed_seconds"]
    started = time.perf_counter()
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for epoch in range(maximum_epochs + 1):
            # Epoch zero is evaluated and eligible before the first optimizer update.
            epoch_start, total_loss = time.perf_counter(), 0.0
            if epoch:
                model.train()
                for indices in batches:
                    optimizer.zero_grad(set_to_none=True)
                    loss = two_view_loss(model, splits["train"], indices, condition,
                                         device, rotations)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite training loss in epoch {epoch}.")
                    loss.backward()
                    valid_gradients = [torch.isfinite(p.grad).all() for p in model.parameters()
                                       if p.grad is not None]
                    if not torch.stack(valid_gradients).all():
                        raise ValueError(f"Non-finite gradient in epoch {epoch}.")
                    optimizer.step()
                    total_loss += loss.item() * len(indices)
            duration = time.perf_counter() - epoch_start
            training_seconds.append(duration)

            # Whole-split curves compare identical unaugmented evaluation conditions.
            for split in ("train", "validation"):
                data = splits[split]
                value, _ = evaluate(model, data, condition, device,
                                    config["evaluation_batch_size"], config["class_names"])
                row = {key: value[key] for key in columns if key in value}
                row.update(epoch=epoch, split=split, optimization_loss=(
                    total_loss / len(splits["train"]["y"]) if epoch else None),
                           training_seconds=duration,
                           elapsed_seconds=time.perf_counter() - started)
                writer.writerow(row)
                history.append(row)
            handle.flush()

            # Only strictly higher macro F1 resets patience, matching the HW2 trainer.
            selected = value["macro_f1"] > best_f1
            if selected:
                best_f1, best_loss = value["macro_f1"], value["loss"]
                best_epoch, stale = epoch, 0
            else:
                stale += 1
                selected = value["macro_f1"] == best_f1 and value["loss"] < best_loss
                if selected:
                    best_loss, best_epoch = value["loss"], epoch
            if selected:
                weights = {key: tensor.detach().cpu().clone()
                           for key, tensor in model.state_dict().items()}
                torch.save(weights, output / "weights.pt")
            if epoch % 20 == 0 or selected:
                print(json.dumps({"epoch": epoch, "validation_macro_f1": value["macro_f1"],
                                  "best_epoch": best_epoch, "stale": stale}), flush=True)
            if stale >= config["early_stopping_patience"]:
                break

    # The returned model uses exactly the saved selected weights for final predictions.
    model.load_state_dict(weights)
    stopping = {
        "completed_epochs": epoch, "maximum_epochs": maximum_epochs,
        "declared_full_budget_epochs": config["maximum_epochs"],
        "patience": config["early_stopping_patience"], "epochs_without_f1_improvement": stale,
        "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
        "reason": "patience_exhausted" if stale >= config["early_stopping_patience"]
                  else "maximum_epochs_reached",
        "checkpoint_restored": True, "selection": config["selection"],
        "training_seconds": training_seconds,
    }
    return history, stopping


def run(condition, seed, output, config_path, threads=1, device="cpu", smoke_epochs=None):
    """Runs one independently schedulable fit and saves all reviewable evidence."""
    started = time.perf_counter()
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if condition not in config["conditions"] or seed not in config["seeds"]:
        raise ValueError("Condition and seed must belong to the declared experiment.")
    if config["optimizer"] != "SGD" or not config["epoch_zero_eligible"]:
        raise ValueError("The declared SGD and epoch-zero selection protocol is required.")
    if threads < 1 or (smoke_epochs is not None and smoke_epochs < 0):
        raise ValueError("Threads must be positive; smoke epochs must be nonnegative.")
    if device not in ("cpu", "cuda"):
        raise ValueError("Device must be explicitly cpu or cuda.")
    if device == "cuda":
        # CUDA's deterministic matrix multiplication requires this declared workspace setting.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config["cublas_workspace_config"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; no device fallback is applied.")
    splits, starts, provenance = load_inputs(config)
    source_paths = [Path(__file__), FOLDER / "model.py", ROOT / "training/final_model.py"]
    request = {
        "condition": condition, "seed": seed, "threads": threads, "device": device,
        "scope": "smoke_only" if smoke_epochs is not None else "full_budget",
        "smoke_epochs": smoke_epochs, "config_sha256": digest(config_path),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        "hw2_freeze_sha256": provenance["hw2_freeze_sha256"],
    }

    # A complete identical run is reused only after its saved artifact identities match.
    completion_path = output / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion["request"] != request:
            raise ValueError("Output belongs to another protocol; choose a fresh destination.")
        for name, identity in completion["artifacts_sha256"].items():
            if digest(output / name) != identity:
                raise ValueError(f"Completed artifact changed: {name}")
        print(json.dumps({"reused": True, "output": str(output)}), flush=True)
        return json.loads((output / "metrics.json").read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete or unrelated output exists; choose a fresh destination.")
    output.mkdir(parents=True, exist_ok=True)

    # Fixed seeds and explicit device/thread settings are retained with the fitted state.
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    area = splits["train"]["batch"]["area"].double()
    area = area[torch.isfinite(area)]
    if len(area) < 2:
        raise ValueError("Training-only area scaling requires at least two measured areas.")
    model = ObservedGeometryCNN(area.mean().item(), area.std(correction=0).item(), starts)
    if model.head[0].p != config["new_model_dropout"]:
        raise ValueError("Configured dropout differs from the selected CNN.")
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    inactive = (sum(p.numel() for p in model.attention.parameters())
                if condition == "no_attention" else 0)
    context_weights = 0
    maximum_epochs = config["maximum_epochs"] if smoke_epochs is None else smoke_epochs
    plan = {
        **request, "config": config, "data": provenance,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform()},
        "parameters": parameters, "inactive_attention_parameters": inactive,
        "constant_zero_context_input_weights": context_weights,
        "parameters_with_training_signal": parameters - inactive - context_weights,
        "cohort_counts": {name: len(split["y"]) for name, split in splits.items()},
        "architecture": model.architecture_spec(),
    }
    write_json(output / "plan.json", plan)
    write_json(output / "config.json", config)
    torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
               output / "initial.pt")
    _, stopping = fit(model, splits, condition, seed, config, output, device, maximum_epochs)

    # Selected predictions retain original cell IDs for independent metric verification.
    final = {}
    for name, split in splits.items():
        final[name], logits = evaluate(model, split, condition, device,
                                        config["evaluation_batch_size"], config["class_names"])
        np.savez_compressed(output / f"predictions_{name}.npz", cell=split["cell"],
                            replicate=split["replicate"], target=split["y"].numpy(),
                            logits=logits.numpy(), prediction=logits.argmax(1).numpy())
    result = {
        "condition": condition, "seed": seed, "scope": request["scope"],
        "parameters": parameters, "inactive_attention_parameters": inactive,
        "constant_zero_context_input_weights": context_weights,
        "metrics": final, "stopping": stopping, "wall_seconds": time.perf_counter() - started,
        "historical_hw2": provenance["historical_hw2"],
        "comparison_limit": config["comparison_limit"],
    }
    write_json(output / "metrics.json", result)
    artifacts = ["plan.json", "config.json", "initial.pt", "weights.pt", "history.csv",
                 "metrics.json", "predictions_train.npz", "predictions_validation.npz"]
    write_json(completion_path, {
        "request": request, "complete": True,
        "artifacts_sha256": {name: digest(output / name) for name in artifacts},
    })
    return result


def main():
    """Exposes one condition and seed per process for native or Rivanna parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True,
                        choices=["attention", "no_attention"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FOLDER / "training_config.json")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke-epochs", type=int,
                        help="Explicit short integration check; never a full-budget result.")
    arguments = parser.parse_args()
    result = run(arguments.condition, arguments.seed, arguments.output, arguments.config,
                 arguments.threads, arguments.device, arguments.smoke_epochs)
    print(json.dumps({"condition": result["condition"], "scope": result["scope"],
                      "wall_seconds": result["wall_seconds"],
                      "validation_macro_f1": result["metrics"]["validation"]["macro_f1"]}),
          flush=True)


if __name__ == "__main__":
    main()
