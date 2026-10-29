"""Runs one local practice model at a time while preserving the frozen predictor."""

import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
from pathlib import Path
from threading import Lock
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
import torch

from training.selected_model import make_selected
from training.final_model import selected_inputs, fit_practice

router = APIRouter()
ACTIVE_STATES = {"preparing", "running"}


class TrainingBody(BaseModel):
    """Exposes the approved epoch-count and dropout controls for separate practice runs."""

    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(default=20, ge=1, le=200, strict=True)
    dropout: FiniteFloat = Field(default=0.0, ge=0.0, le=1.0, strict=True)


def frozen_metrics(folder, metadata):
    """Reads the selected model and the plain ReLU benchmark on the same saved splits."""
    folder = Path(folder)
    selection = json.loads((folder / "validation_metrics.json").read_text())
    test = json.loads((folder / "test_metrics.json").read_text())
    split_metrics = {name: selection[key]["improved"]
                     for name, key in [("train", "training"), ("validation", "validation")]}
    split_metrics["test"] = test["improved"]
    fields = ["n_cells", "n_errors", "loss", "accuracy", "error", "balanced_accuracy",
              "macro_f1", "weighted_f1"]
    # The model identity travels with its dimensions and comparison role.
    architecture = {
        "inputs": metadata["n_features"], "hidden_widths": metadata["hidden_widths"],
        "outputs": len(metadata["class_names"]), "activation": metadata["activation"],
        "negative_slope": metadata["negative_slope"], "dropout": metadata["dropout"],
        "family": metadata["family"], "name": metadata["architecture_name"],
        "graph": metadata.get("graph"),
        "residual_blocks": metadata.get("residual_blocks"),
        "skip_operation": metadata.get("skip_operation"),
        "stem_dropout": metadata.get("stem_dropout"),
        "block_dropout": metadata.get("block_dropout"),
    }
    history = pd.read_csv(folder / "history.csv")
    history_fields = ["epoch", "split", "loss", "accuracy", "macro_f1"]
    return {
        "history": history[history_fields].to_dict(orient="records"),
        "validation_exploration": json.loads((folder / "validation_exploration.json").read_text()),
        "model_version": metadata["model_version"], "best_epoch": metadata["best_epoch"],
        "architecture": architecture, "training": metadata["training"],
        "stopping": metadata["stopping"],
        "comparison_candidates": metadata["comparison_candidates"],
        "comparison_scope": metadata["comparison_scope"],
        "splits": {name: {key: values[key] for key in fields}
                   for name, values in split_metrics.items()},
        "baseline": {"name": "Simple ReLU MLP", "activation": "relu", "dropout": 0.0,
                     "hidden_widths": [32], **metadata["baseline_metadata"],
                     "test": {key: test["baseline"][key] for key in fields}},
        "improved_has_lower_error": test["improved_has_lower_error"],
        "cleaning": metadata["cleaning"], "filter_counts": metadata["filter_counts"],
        "test_filter_counts": test["filter_counts"],
        "evaluation_note": test["evaluation_note"],
        "zero_recall_classes": [name for name in metadata["class_names"]
                                if test["improved"]["per_class"][name]["recall"] == 0],
        "class_names": metadata["class_names"],
    }


class TrainingService:
    """Owns the cache, one worker and locked snapshots for this single API process."""

    def __init__(self, data_directory, output_directory, model_directory, metadata):
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory)
        model_directory = Path(model_directory).resolve()
        if self.output_directory.resolve().is_relative_to(model_directory):
            raise ValueError("Practice outputs must be outside the frozen model folder.")
        self.metadata = metadata
        self.settings = dict(metadata["training"])
        self.metrics = frozen_metrics(model_directory, metadata)
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="practice-training")
        self.jobs, self.current_id, self.input_cache = {}, None, None

    def available(self):
        """Checks the configured training sources without reading their measurements."""
        return all((self.data_directory / name).is_file()
                   for name in self.metadata["source_sha256"])

    def snapshot(self, job_id=None):
        """Copies state under the same lock used by progress updates."""
        with self.lock:
            key = self.current_id if job_id is None else job_id
            if key is None:
                return None
            if key not in self.jobs:
                raise KeyError(key)
            result = copy.deepcopy(self.jobs[key])
            started = result.pop("_started_monotonic")
            if result["status"] in ACTIVE_STATES:
                result["elapsed_seconds"] = round(time.monotonic() - started, 1)
            return result

    def start(self, epochs, dropout=0.0):
        """Admits a job atomically before scheduling background work."""
        with self.lock:
            if self.current_id and self.jobs[self.current_id]["status"] in ACTIVE_STATES:
                raise RuntimeError("A practice run is already active. Wait for it to finish.")
            job_id = uuid4().hex
            self.jobs[job_id] = {
                "id": job_id, "status": "preparing", "epoch": 0, "epochs": epochs,
                "settings": {**self.settings, "epochs": epochs, "dropout": dropout},
                "architecture_name": self.metadata["architecture_name"], "history": [],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "_started_monotonic": time.monotonic(), "elapsed_seconds": 0,
                "served_model_version": self.metadata["model_version"],
                "best_epoch": None, "error": None, "result_folder": None,
            }
            self.current_id = job_id
        self.executor.submit(self._run, job_id)
        return self.snapshot(job_id)

    def _prepare_inputs(self):
        """Reuses prepared tensors only while the frozen run's source identities match."""
        for name, expected in self.metadata["source_sha256"].items():
            with (self.data_directory / name).open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Training source changed: {name}. Restart with reviewed inputs.")
        if self.input_cache is None:
            inputs = selected_inputs(self.data_directory)
            schema = inputs[1]
            for key in ["cleaning", "feature_names", "class_names"]:
                if schema[key] != self.metadata[key]:
                    raise ValueError(f"Practice {key} differs from the frozen predictor.")
            self.input_cache = inputs
        return self.input_cache

    def _record_epoch(self, job_id, epoch, records):
        """Publishes the two fixed-weight split measurements for the completed epoch."""
        with self.lock:
            job = self.jobs[job_id]
            job["status"], job["epoch"] = "running", epoch
            job["history"].extend(records)

    def _save_run(self, folder, model, schema, history, best_epoch, settings, stopping):
        """Stores this practice run without replacing any evaluated artifact."""
        torch.save(model.state_dict(), folder / "weights.pt")
        history.to_csv(folder / "history.csv", index=False)
        record = {
            "model_version": f"practice-{folder.name}", "settings": settings,
            "best_epoch": best_epoch, **schema, "splits": {"train": [1, 2], "validation": [3]},
            "family": self.metadata["family"],
            "n_features": self.metadata["n_features"],
            "n_classes": self.metadata["n_classes"],
            "activation": self.metadata["activation"],
            "negative_slope": self.metadata["negative_slope"],
            "architecture_name": self.metadata["architecture_name"],
            "hidden_widths": self.metadata["hidden_widths"],
            "graph": self.metadata.get("graph"), "stopping": stopping,
            "residual_blocks": self.metadata.get("residual_blocks"),
            "skip_operation": self.metadata.get("skip_operation"),
            "stem_dropout": self.metadata.get("stem_dropout"),
            "block_dropout": (settings["dropout"] if self.metadata.get("residual_blocks")
                              else None),
            "dropout": settings["dropout"],
            "test_evaluated": False, "source_sha256": self.metadata["source_sha256"],
        }
        (folder / "run.json").write_text(json.dumps(record, indent=2))

    def _finish(self, job_id, status, **values):
        """Publishes a terminal state with the actual elapsed time."""
        with self.lock:
            job = self.jobs[job_id]
            job.update(status=status, ended_at=datetime.now(timezone.utc).isoformat(), **values)
            job["elapsed_seconds"] = round(time.monotonic() - job["_started_monotonic"], 1)

    def _run(self, job_id):
        """Prepares data and fits an independent model on the single background worker."""
        try:
            folder = self.output_directory / job_id
            folder.mkdir(parents=True, exist_ok=False)
            tensors, schema = self._prepare_inputs()
            settings = self.snapshot(job_id)["settings"]
            torch.manual_seed(settings["seed"])
            # Each practice run initializes the selected architecture with the requested dropout.
            model = make_selected({**self.metadata, "dropout": settings["dropout"]})
            observer = lambda epoch, rows: self._record_epoch(job_id, epoch, rows)
            history, best, stopping = fit_practice(
                model, *tensors, settings, on_epoch=observer
            )
            self._save_run(folder, model, schema, history, best, settings, stopping)
            self._finish(job_id, "completed", best_epoch=best, stopping=stopping,
                         result_folder=str(folder))
        except Exception as error:
            self._finish(job_id, "failed", error=str(error))

    def close(self):
        """Allows the bounded active run to finish before this process shuts down."""
        self.executor.shutdown(wait=True, cancel_futures=True)


@router.get("/api/training/metrics")
def get_metrics(request: Request):
    return request.app.state.training.metrics


@router.get("/api/training/jobs")
def get_current_job(request: Request):
    service = request.app.state.training
    return {"job": service.snapshot(), "available": service.available()}


@router.post("/api/training/jobs", status_code=202)
def start_job(body: TrainingBody, request: Request):
    try:
        return request.app.state.training.start(body.epochs, dropout=body.dropout)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/api/training/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    try:
        return request.app.state.training.snapshot(job_id)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail="This run is not in this API process."
        ) from error
