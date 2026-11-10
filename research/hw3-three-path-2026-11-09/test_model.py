"""Checks the approved three-path model's masks, geometry, attention and repeated layers."""

# Real frozen cells test the new implementation without fitting a classifier on CPU.
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from model import ObservedGeometryCNN


class ThreePathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        root = Path(__file__).resolve().parents[2]
        source = root / "research/hw3-observed-geometry-2026-11-09/observed.npz"
        with np.load(source) as cache:
            area = cache["train_cell_area"]
            self.model = ObservedGeometryCNN(area.mean(), area.std(), cache["locus_start"]).eval()
            self.batch = {
                "xyz": torch.from_numpy(cache["train_xyz_observed"][:4].copy()).float(),
                "observed": torch.from_numpy(cache["train_locus_observed"][:4].copy()),
                "traces": torch.from_numpy(cache["train_mask"][:4].copy()),
                "area": torch.from_numpy(area[:4].copy()).float(),
            }

    def test_shapes_and_approved_repeated_components(self):
        # Each independent branch has the same nine-convolution structure and its own filters.
        scores, info = self.model(self.batch, return_attention=True)
        self.assertEqual(scores.shape, (4, 3))
        self.assertEqual(info["weights"].shape, (4, 4, 4))
        self.assertEqual(info["trace_tokens"].shape, (4, 4, 32))
        self.assertEqual(info["metadata_inputs"].shape, (4, 4))
        for encoder in (self.model.xyz_encoder, self.model.distance_encoder):
            layers = [layer for layer in encoder.modules() if isinstance(layer, nn.Conv1d)]
            self.assertEqual(len(layers), 9)
            self.assertTrue(all(layer.kernel_size == (3,) for layer in layers))
        self.assertEqual(self.model.trace_projection.in_features, 3136)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 161075)
        self.assertIsNot(self.model.xyz_encoder, self.model.distance_encoder)

    def test_masked_coordinate_storage_has_no_effect(self):
        # Missing coordinates may have arbitrary storage values without becoming observations.
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["xyz"][~changed["observed"]] = 1e8
        with torch.no_grad():
            expected, actual = self.model(self.batch), self.model(changed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_trace_permutation_preserves_cell_prediction(self):
        # Entire trace values and masks move together when arbitrary slot numbers are changed.
        changed = {key: value.clone() for key, value in self.batch.items()}
        for key in ("xyz", "observed", "traces"):
            changed[key] = changed[key][:, [3, 1, 0, 2]]
        with torch.no_grad():
            expected, actual = self.model(self.batch), self.model(changed)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    def test_rotation_preserves_distance_and_metadata_inputs(self):
        # All traces share a proper rotation while original-frame distances remain exactly equal.
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        changed = {**self.batch, "xyz_rotation": rotation.expand(4, -1, -1)}
        original, rotated = self.model.compose_inputs(self.batch), self.model.compose_inputs(changed)
        self.assertFalse(torch.equal(original[0], rotated[0]))
        for index in (1, 2, 3, 4, 5, 6):
            torch.testing.assert_close(original[index], rotated[index], rtol=0, atol=0)
        self.assertEqual(torch.linalg.det(rotation).item(), 1)

    def test_attention_masks_probability_sums_and_finite_gradients(self):
        # Valid queries use every valid trace key, including their own; padding receives zero weight.
        scores, info = self.model(self.batch, return_attention=True)
        valid = info["valid"]
        allowed = valid[:, :, None] & valid[:, None, :]
        self.assertEqual(torch.count_nonzero(info["weights"][~allowed]).item(), 0)
        sums = info["weights"].sum(-1)
        torch.testing.assert_close(sums[valid], torch.ones_like(sums[valid]))
        self.assertEqual(torch.count_nonzero(sums[~valid]).item(), 0)
        scores.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                            if p.grad is not None))

    def test_one_trace_can_attend_to_itself(self):
        # Single-trace cells keep a defined self-attention path without an invented second trace.
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["traces"][:, 1:] = False
        scores, info = self.model(changed, return_attention=True)
        self.assertTrue(torch.isfinite(scores).all())
        torch.testing.assert_close(info["weights"][:, 0, 0], torch.ones(4))
        self.assertEqual(torch.count_nonzero(info["weights"][:, 1:]).item(), 0)

    def test_zero_valid_traces_and_missing_area_remain_finite(self):
        # An empty geometry set produces zero attention and explicit missing-metadata support.
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["observed"][:] = False
        changed["area"][:] = torch.nan
        scores, info = self.model(changed, return_attention=True)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertFalse(info["valid"].any())
        self.assertEqual(torch.count_nonzero(info["weights"]).item(), 0)
        self.assertEqual(torch.count_nonzero(info["update"]).item(), 0)
        torch.testing.assert_close(info["metadata_inputs"][:, :3], torch.zeros(4, 3))
        torch.testing.assert_close(info["metadata_inputs"][:, 3], torch.ones(4))
        scores.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()
                            if p.grad is not None))

    def test_ablation_retains_identical_metadata_and_parameter_registration(self):
        # Attention bypass changes only the update, leaving both CNNs and metadata intact.
        _, full = self.model(self.batch, return_attention=True)
        scores, bypass = self.model(self.batch, return_attention=True, use_attention=False)
        torch.testing.assert_close(full["metadata_inputs"], bypass["metadata_inputs"])
        torch.testing.assert_close(full["trace_tokens"], bypass["trace_tokens"])
        self.assertEqual(torch.count_nonzero(bypass["update"]).item(), 0)
        scores.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.attention.parameters()))
        self.assertEqual(sum(p.numel() for p in self.model.attention.parameters()), 4224)


if __name__ == "__main__":
    unittest.main()
