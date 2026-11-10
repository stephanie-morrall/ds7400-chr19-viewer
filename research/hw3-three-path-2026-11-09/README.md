# HW3 three-path chromosome classifier

This experiment classifies Hepatocyte, Erythroblast and Proerythroblast cells using
chromosome-19 geometry. The fixed cohort has 2,439 training cells from replicates
1–2 and 1,347 validation cells from replicate 3. Each cell has one to four traces
at 49 ordered genomic loci. Replicate 4 is outside this experiment.

## Model and motivation

XYZ and within-trace distance profiles have separate Conv1D encoders. Each has
one width-three stem and four two-convolution residual blocks: nine convolution
layers per encoder. Shared weights process each trace separately. Genomic order
is retained through an ordered 3,136-to-32 dense readout. A single masked
self-attention layer exchanges information between the trace summaries, followed
by an identity skip and a mean over valid traces.

Four metadata inputs are training-standardized cell area, valid-trace count/4,
observed-locus fraction and an area-missing flag. A dense layer creates 16 metadata
features. Concatenating them with 32 geometry features gives 48 inputs for
dropout0.2 and a 48→32→3 classifier. There are **161,075 registered parameters**.
All cohort areas are measured, so the area-missing flag contributes no variation.

Masks exclude missing measurements and padded traces. Convolutions normalize by
available support; these operations do not reconstruct missing physical points.
Training averages cross-entropy from original coordinates and a second view with
one proper rotation shared by all traces in each cell. Distances and metadata
remain unchanged. Validation uses original coordinates with dropout disabled.

The hypothesis was that learning traces separately would retain information lost
by early distance averaging. The new model did not improve the original HW2
validation result. More information and more layers did not guarantee better
generalization. The web API continues to serve the frozen HW2 predictor.

## Recorded results

| Model | Validation macro F1 | Mean GPU fit time |
|---|---:|---:|
| Original frozen HW2, one saved seed | 0.479058 | Not remeasured |
| Three-path CNN, attention bypassed | 0.442044 ± 0.004856 | 249.51 s |
| Three-path CNN with attention | 0.438952 ± 0.005147 | 290.96 s |

CNN rows are mean±sample SD across seeds42,43,44 on the same validation cells.
The paired attention differences were negative for all three seeds. They do not
establish a general harmful effect. Most final training losses were small while
validation losses increased. One bypass fit lost its training fit late. Saved
predictions use the earlier selected checkpoints, not those final epochs.

A previously completed, training-matched HW2 diagnostic averaged
0.206410±0.042939. Its failed fit is a secondary reference; the original0.479058
result remains primary. Different input representations and the original HW2
training regime prevent an architecture-only causal comparison.

Aggregate results and figures are in [report/](report/). The committed summary
contains aggregate evidence rather than private per-cell predictions.

## Data and runtime prerequisites

Training requires the three authorized private NPZ caches whose relative paths
and exact SHA256 identities are recorded in [training_config.json](training_config.json).
They provide observed coordinates/masks and the original HW2 cohort-alignment
checks. The caches are excluded from this public repository. A portable raw-table
to-cache rebuild is not supplied. Viewing and HW2 prediction use the committed
sample and frozen model without these training caches.

The recorded GPU runtime was Python3.12.3, Torch2.12.0a0, NumPy2.1.0 and
scikit-learn1.8.0 on an NVIDIA RTX A6000. The actual Torch build string is in
the aggregate summary. Local model checks used Torch2.11.0. Configure a Python
environment with Torch, NumPy, pandas, SciPy and scikit-learn before fitting.

## Training and parallel execution

From the repository root with the authorized caches:

```sh
python research/hw3-three-path-2026-11-09/train.py \
  --condition attention --seed 42 --device cuda --threads 1 \
  --output research/hw3-three-path-2026-11-09/results/attention/seed42
```

Use `no_attention` for the matched bypass and seeds43,44 for the remaining fits.
The output directory must be fresh or contain a complete identical run. A short
integration check can explicitly add `--smoke-epochs 1`; it is not a full result.

The [SLURM recipe](native/run-panel.slurm) runs all six condition/seed pairs as
independent array jobs, with one A6000 GPU, two allocated CPUs, one numerical
thread, 8GiB and a three-hour allowance per fit. Activate the recorded environment
and submit from the repository root, selecting your authorized allocation:

```sh
sbatch --account=YOUR_ACCOUNT research/hw3-three-path-2026-11-09/native/run-panel.slurm
```

Both arms use SGD lr0.01, momentum0.9, batch64, maximum1,000 epochs and patience100.
Only a strict validation macro-F1 improvement resets patience. Lower validation
loss breaks F1 ties, and the selected checkpoint is restored. These settings are
recorded in the configuration rather than inferred from the observed fit length.

## Verification and analysis

The eight model tests require the authorized observed cache:

```sh
python -m unittest discover -s research/hw3-three-path-2026-11-09 -p test_model.py -v
```

After all six full fits complete, recompute aggregate metrics and check saved
artifact/source identities, cohort alignment, losses, checkpoint selection and
the stopping rule:

```sh
python research/hw3-three-path-2026-11-09/analyse.py \
  --results research/hw3-three-path-2026-11-09/results \
  --output research/hw3-three-path-2026-11-09/regenerated-report
```

This writes aggregate CSVs and `regenerated-summary.json`; cell-level arrays
remain local. The supplied figures retain the recorded six-fit results. Exact
GPU saved-weight replay was separately checked for all six checkpoints and both
splits, with zero logit differences. A CPU run can have small numerical differences.

Validation is reused for selection and comparison. Seed variation measures
optimization variability, not variation between biological replicates. Full-fit
peak memory was not measured. The next controlled experiment is a shallower
encoder with the inputs and optimization settings held fixed.
