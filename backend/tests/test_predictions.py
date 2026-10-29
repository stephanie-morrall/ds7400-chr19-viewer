"""Checks serving parity, invalid measurements and annotation preservation."""

import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from backend.predict import PredictBody, predict_cell, spot_table


def real_payload(cell_id=328):
    """Uses a committed sample cell with four raw chromosome traces."""
    table = pd.read_csv("frontend/public/sample/cells.csv")
    spots = table.loc[table["Cell_ID"].eq(cell_id)].to_dict("records")
    return {"spots": [
        {"trace_id": int(spot["Trace_ID"]), "chrom_start": int(spot["Chrom_Start"]),
         "x": spot["X"], "y": spot["Y"], "z": spot["Z"]}
        for spot in spots
    ]}


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="hw2-predict-test-")
        cls.environment = patch.dict(os.environ, {
            "SQLITE_PATH": str(Path(cls.directory.name) / "annotations.db"),
        })
        cls.environment.start()
        os.environ.pop("MODEL_DIR", None)
        cls.main = importlib.import_module("backend.main")
        cls.database = patch.object(
            cls.main, "DB_PATH", str(Path(cls.directory.name) / "annotations.db")
        )
        cls.database.start()
        cls.loader = patch.object(cls.main, "load_selected", wraps=cls.main.load_selected)
        cls.load_calls = cls.loader.start()
        cls.client = TestClient(cls.main.app)
        cls.client.__enter__()
        cls.payload = real_payload()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.loader.stop()
        cls.database.stop()
        cls.environment.stop()
        cls.directory.cleanup()

    def test_actual_cell_matches_direct_predictor(self):
        direct = predict_cell(
            spot_table(PredictBody.model_validate(self.payload)), self.main.app.state.predictor
        )
        response = self.client.post("/api/predict", json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), direct)
        self.assertEqual(len(self.payload["spots"]), 165)
        self.assertEqual(len({row["trace_id"] for row in self.payload["spots"]}), 4)
        self.assertEqual(len(direct["class_names"]), 3)
        self.assertIn(direct["predicted_type"], direct["class_names"])

    def test_invalid_measurements_and_extra_labels_are_rejected(self):
        variants = [{"spots": []}, {"spots": self.payload["spots"], "label": "Hepatocyte"}]
        for field, value in [("x", None), ("chrom_start", 1), ("recorded_type", "Macrophage")]:
            changed = json.loads(json.dumps(self.payload))
            changed["spots"][0][field] = value
            variants.append(changed)
        variants.append({"spots": [self.payload["spots"][0]] * 2})
        for body in variants:
            with self.subTest(body=body["spots"][:1]):
                self.assertEqual(self.client.post("/api/predict", json=body).status_code, 422)

    def test_non_finite_and_low_coverage_traces_are_rejected(self):
        changed = json.loads(json.dumps(self.payload))
        changed["spots"][0]["x"] = float("inf")
        response = self.client.post(
            "/api/predict", content=json.dumps(changed),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)
        # Four one-spot traces pass the cell trace-count rule but fail locus coverage.
        singletons = [dict(spot, trace_id=index)
                      for index, spot in enumerate(self.payload["spots"][:4])]
        response = self.client.post("/api/predict", json={"spots": singletons})
        self.assertEqual(response.status_code, 422)

    def test_cell_411_is_rejected_by_the_raw_trace_count_filter(self):
        payload = real_payload(411)
        self.assertEqual(len(payload["spots"]), 228)
        self.assertEqual(len({row["trace_id"] for row in payload["spots"]}), 5)
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("quality filters", response.json()["detail"])
        self.assertIn("at most 4 traces", response.json()["detail"])

    def test_prediction_preserves_annotation_storage(self):
        note = {"text": "Prediction must preserve this annotation."}
        self.assertEqual(self.client.put("/api/notebook", json=note).status_code, 200)
        self.assertEqual(self.client.post("/api/predict", json=self.payload).status_code, 200)
        self.assertEqual(self.client.get("/api/notebook").json(), note)
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_model_is_loaded_once_for_multiple_requests(self):
        for _ in range(2):
            self.assertEqual(self.client.post("/api/predict", json=self.payload).status_code, 200)
        self.assertEqual(self.load_calls.call_count, 1)
        self.assertEqual(self.load_calls.call_args.args[0],
                         Path("training/results/three_class_residual_f1_v1"))
        metrics = self.client.get("/api/training/metrics").json()
        self.assertEqual(metrics["model_version"],
                         self.main.app.state.predictor["metadata"]["model_version"])
        self.assertEqual(metrics["baseline"]["activation"], "relu")
        self.assertEqual(metrics["baseline"]["dropout"], 0.0)
        self.assertEqual(metrics["comparison_candidates"], 44)


if __name__ == "__main__":
    unittest.main()
