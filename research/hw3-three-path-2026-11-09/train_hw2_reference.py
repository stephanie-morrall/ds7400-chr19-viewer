"""Reproduces the training-budget-matched HW2 residual-MLP diagnostic.

The three authorized input caches establish the original dense-distance cohort.
Each run writes its selected weights, history, cell predictions and completion hashes.
"""

# Standard-library records retain the exact settings, runtime and artifact identities.
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

# NumPy and Torch supply the frozen dense inputs and deterministic two-forward SGD.
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Existing public helpers verify cohort equality and calculate the same metrics.
FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
sys.path.insert(0, str(ROOT))
from training.final_model import ResidualMLP
from train import digest, load_inputs, metrics_from_logits, write_json


def predict(model, split, indices, condition, device, rotation=None):
    """Returns HW2 scores from the exact original dense-distance tensor."""
    return model(split["dense"][indices].to(device))


def two_view_loss(model, split, indices, condition, device, rotation_generator):
    """Averages two identical dense views with independent dropout draws."""
    # Two forwards preserve the diagnostic's original gradient and dropout budget.
    target = split["y"][indices].to(device)
    first = model(split["dense"][indices].to(device))
    second = model(split["dense"][indices].to(device))
    return (nn.functional.cross_entropy(first, target)
            + nn.functional.cross_entropy(second, target)) / 2


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
    splits, _, provenance = load_inputs(config)
    # The shared loader has already verified these dense values against original HW2.
    with np.load(ROOT / config["inputs"]["hw2_dense"]["path"]) as dense:
        for name, split in splits.items():
            split["dense"] = torch.from_numpy(dense[f"{name}_mean"].copy())
    source_paths = [Path(__file__), FOLDER / "train.py", FOLDER / "model.py",
                    ROOT / "training/final_model.py"]
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
    model = ResidualMLP(dropout=config["hw2_dropout"])
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    inactive, context_weights = 0, 0
    maximum_epochs = config["maximum_epochs"] if smoke_epochs is None else smoke_epochs
    plan = {
        **request, "config": config, "data": provenance,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "platform": platform.platform()},
        "parameters": parameters, "inactive_attention_parameters": inactive,
        "constant_zero_context_input_weights": context_weights,
        "parameters_with_training_signal": parameters - inactive - context_weights,
        "cohort_counts": {name: len(split["y"]) for name, split in splits.items()},
        "architecture": {"class": "training.final_model.ResidualMLP",
                         "input_features": 1176, "dropout": model.dropout.p,
                         "parameters": parameters},
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
    """Exposes one diagnostic seed per process for independent parallel runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FOLDER / "hw2_reference_config.json")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke-epochs", type=int,
                        help="Explicit short integration check, not a full-budget result.")
    arguments = parser.parse_args()
    result = run("hw2_reference", arguments.seed, arguments.output, arguments.config,
                 arguments.threads, arguments.device, arguments.smoke_epochs)
    print(json.dumps({"condition": result["condition"], "scope": result["scope"],
                      "wall_seconds": result["wall_seconds"],
                      "validation_macro_f1": result["metrics"]["validation"]["macro_f1"]}),
          flush=True)


if __name__ == "__main__":
    main()

