"""Classifies cells with separate XYZ, distance and metadata paths plus trace attention."""

# Torch supplies the repeated local encoders and the cell-level classification head.
import torch
from torch import nn
from torch.nn import functional as F


class MaskedConv(nn.Module):
    """Normalizes a convolution by the available feature support in each window."""

    def __init__(self, incoming, outgoing=32):
        super().__init__()
        self.conv = nn.Conv1d(incoming, outgoing, 3, padding=1)
        self.register_buffer("counter", torch.ones(1, incoming, 3))
        self.full = incoming * 3

    def forward(self, values, mask):
        # Unavailable numeric storage contributes neither values nor observed support.
        safe = torch.where(mask, values, 0.0)
        count = F.conv1d(mask.float(), self.counter, padding=1)
        support = count > 0
        bias = self.conv.bias[None, :, None]
        response = (self.conv(safe) - bias) * self.full / count.clamp_min(1) + bias
        return torch.where(support, response, 0.0), support.expand_as(response)


class ResidualBlock(nn.Module):
    """Adds a two-convolution correction to an unchanged 32-channel identity path."""

    def __init__(self):
        super().__init__()
        self.first = MaskedConv(32)
        self.second = MaskedConv(32)

    def forward(self, values, mask):
        # The repeated block is convolution, ReLU, convolution, identity addition and ReLU.
        correction, support = self.first(values, mask)
        correction, support = self.second(F.relu(correction), support)
        return F.relu(values + correction), mask | support


class TraceEncoder(nn.Module):
    """Applies one width-three stem and four distinct residual blocks to each trace."""

    def __init__(self, incoming):
        super().__init__()
        self.stem = MaskedConv(incoming)
        self.blocks = nn.ModuleList([ResidualBlock() for _ in range(4)])

    def forward(self, values, mask, observed):
        # All nine convolutions preserve the 49 named genomic slots with same padding.
        features, support = self.stem(values, mask)
        features = F.relu(features)
        for block in self.blocks:
            features, support = block(features, support)
        return features * observed[:, None]


class TraceSelfAttention(nn.Module):
    """Returns single-head context among all valid trace tokens, including their own keys."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(32, 32)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.output = nn.Linear(32, 32)

    def forward(self, features, valid):
        # Each of at most four tokens represents one complete ordered chromosome trace.
        query, key, value = self.query(features), self.key(features), self.value(features)
        allowed = valid[:, :, None] & valid[:, None, :]
        available = allowed.any(-1)
        scores = (query @ key.transpose(-1, -2)) / query.shape[-1] ** .5
        scores = scores.masked_fill(~allowed, -torch.inf)

        # Empty rows are finite before softmax and produce an exact zero projected update.
        scores = torch.where(available[..., None], scores, 0.0)
        weights = torch.softmax(scores, -1) * allowed
        update = self.output(weights @ value)
        update = torch.where(available[..., None], update, 0.0)
        return update, weights


class ObservedGeometryCNN(nn.Module):
    """Combines two ordered geometry encoders with one measured-metadata path.

    Parameters
    ----------
    area_mean, area_std : float
        Training-only finite-area population statistics, saved with fitted weights.
    locus_start : array-like
        The 49 increasing genomic positions that fix the sequence and readout order.
    """

    def __init__(self, area_mean, area_std, locus_start):
        super().__init__()
        if not area_std > 0:
            raise ValueError("Training cell-area standard deviation must be positive.")
        starts = torch.as_tensor(locus_start, dtype=torch.int64)
        if starts.shape != (49,) or not torch.all(starts[1:] > starts[:-1]):
            raise ValueError("The model requires 49 increasing genomic positions.")
        self.register_buffer("locus_start", starts.clone())
        self.register_buffer("area_mean", torch.tensor(float(area_mean)))
        self.register_buffer("area_std", torch.tensor(float(area_std)))

        # Each encoder shares weights across traces while keeping its own input-specific filters.
        self.xyz_encoder = TraceEncoder(3)
        self.distance_encoder = TraceEncoder(49)
        self.trace_projection = nn.Linear(49 * 64, 32)
        self.attention = TraceSelfAttention()
        self.metadata = nn.Sequential(nn.Linear(4, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 3))

    def compose_inputs(self, batch):
        """Returns observed geometry, endpoint masks, valid traces and four metadata inputs."""
        xyz = batch["xyz"]
        observed = batch["observed"] & batch["traces"][..., None]
        valid = batch["traces"] & observed.any(-1)
        safe = torch.where(observed[..., None], xyz, 0.0)

        # A common observed-point center preserves relative placement between the cell's traces.
        count = observed.sum((1, 2)).clamp_min(1)
        center = (safe.double().sum((1, 2)) / count.double()[:, None]).to(xyz.dtype)
        centered = torch.where(observed[..., None], safe - center[:, None, None], 0.0)
        points = observed.flatten(0, 1)
        distances = torch.cdist(centered.flatten(0, 1), centered.flatten(0, 1))
        endpoints = points[:, :, None] & points[:, None, :]

        # One proper cell rotation changes only the XYZ view while distance inputs stay identical.
        if "xyz_rotation" in batch:
            centered = torch.einsum("btli,bij->btlj", centered, batch["xyz_rotation"])
        xyz_input = centered.flatten(0, 1).transpose(1, 2)
        xyz_mask = points[:, None].expand_as(xyz_input)

        # Area missingness has an explicit flag; unavailable area uses a neutral scaled placeholder.
        area = batch["area"]
        area_observed = torch.isfinite(area)
        scaled_area = torch.where(area_observed, (area - self.area_mean) / self.area_std, 0.0)
        trace_count = valid.sum(1).to(xyz.dtype)
        fraction = observed.sum((1, 2)) / (trace_count * 49).clamp_min(1)
        metadata = torch.stack((scaled_area, trace_count / 4, fraction,
                                (~area_observed).to(xyz.dtype)), -1)
        return xyz_input, xyz_mask, distances.transpose(1, 2), endpoints, points, valid, metadata

    def forward(self, batch, return_attention=False, use_attention=True):
        """Returns three class logits and optional trace-level attention diagnostics."""
        x, xm, distances, dm, observed, valid, metadata = self.compose_inputs(batch)
        batch_size, traces = valid.shape

        # Independent CNNs read XYZ and unaveraged distance profiles in the same fixed locus order.
        xyz_features = self.xyz_encoder(x, xm, observed)
        distance_features = self.distance_encoder(distances, dm, observed)
        combined = torch.cat((xyz_features, distance_features), 1)
        ordered = combined.transpose(1, 2).reshape(batch_size * traces, 49 * 64)
        tokens = F.relu(self.trace_projection(ordered)).reshape(batch_size, traces, 32)
        tokens = tokens * valid[..., None]

        # One trace-token attention update joins the identity path before the valid-trace mean.
        if use_attention:
            update, weights = self.attention(tokens, valid)
        else:
            update = torch.zeros_like(tokens)
            weights = tokens.new_zeros(batch_size, traces, traces)
        contextual = (tokens + update) * valid[..., None]
        geometry = contextual.sum(1) / valid.sum(1).clamp_min(1)[:, None]

        # The measured-metadata path supplies sixteen features beside the 32 geometry features.
        meta_features = self.metadata(metadata)
        scores = self.head(torch.cat((geometry, meta_features), -1))
        if return_attention:
            return scores, {"weights": weights, "update": update, "valid": valid,
                            "trace_tokens": tokens, "metadata_inputs": metadata}
        return scores

    def architecture_spec(self):
        """Returns actual module dimensions for diagrams, training records and verification."""
        return {
            "version": "three-path-residual-cnn-trace-attention-v1",
            "parameters": sum(p.numel() for p in self.parameters()),
            "xyz_channels": self.xyz_encoder.stem.conv.in_channels,
            "distance_channels": self.distance_encoder.stem.conv.in_channels,
            "encoder_width": self.xyz_encoder.stem.conv.out_channels,
            "stem_kernel": self.xyz_encoder.stem.conv.kernel_size[0],
            "residual_blocks_per_encoder": len(self.xyz_encoder.blocks),
            "convolutions_per_encoder": 1 + 2 * len(self.xyz_encoder.blocks),
            "readout": [self.trace_projection.in_features, self.trace_projection.out_features],
            "readout_order": "locus first, then XYZ32 followed by distance32 channels",
            "attention_heads": 1, "attention_width": self.attention.query.out_features,
            "attention_mask": "valid query and key traces, including the query's own trace",
            "attention_empty_context": "exact zero update and weights",
            "attention_normalization": "scaled dot product and masked softmax; no LayerNorm",
            "metadata": ["train-standardized area", "valid trace count / 4",
                         "observed fraction among valid-trace loci", "area missing flag"],
            "metadata_dimensions": [self.metadata[0].in_features, self.metadata[0].out_features],
            "head": [self.head[1].in_features, self.head[1].out_features,
                     self.head[3].out_features],
            "dropout": self.head[0].p, "activation": "ReLU", "coordinate_imputation": "none",
            "rotation": "common cell SO(3), XYZ only; pair distances remain unchanged",
            "physical_units": "microns", "supplied_center_xy_used": False,
        }
