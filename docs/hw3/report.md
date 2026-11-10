# Testing three input paths before trace attention

## Baseline and task

I classify Hepatocyte, Erythroblast and Proerythroblast cells from mouse chromosome
19 spatial measurements. The fixed cohort contains 2,439 training cells from
biological replicates 1–2 and 1,347 validation cells from replicate 3. Each cell has
one to four traces at 49 retained genomic loci. Replicate-4 coordinates are outside
this experiment.

The final HW2 residual MLP receives 1,176 within-trace pairwise distances averaged
across each cell's traces. Its 1,176→32 ReLU stem feeds a 32→32→32 residual branch
with dropout 0.8, then a 32→3 classifier. Its 39,875 parameters produced frozen
validation macro F1 **0.479058**, seed 42, under the original training protocol.
This saved result is the primary baseline.

## Original hypothesis and design rationale

**Before observing the new results**, I hypothesized that learning each trace's
ordered geometry before combining traces would retain patterns that HW2's early
averaging removes. The design separates shared spatial layout, folding distances
and measured cell context, then tests whether trace summaries benefit from attention.

All traces use one observed-point center, preserving their relative placement.
Training rotates the whole cell's XYZ together. A separate branch receives each
trace's 49-by-49 pairwise distances in microns. Distances expose rotation-invariant
folding directly, although they overlap mathematically with XYZ. Missing values
remain masked, with no coordinate interpolation.

Each branch uses a width-three Conv1D stem and four residual blocks containing two
width-three convolutions apiece, with 32 channels and ReLU. Stacking small kernels
composes local patterns along genomic order. Residual additions preserve earlier
features and support optimization. The two encoders share weights across traces,
while retaining separate XYZ and distance weights. Increased depth adds capacity
and can worsen overfitting. Unequal genomic gaps remain unequal physical intervals.

Corresponding outputs concatenate into 64 features at each locus. Ordered flattening
and a 3,136→32 dense projection create one summary per trace. One masked self-attention
layer lets the one to four trace summaries exchange context. A residual preserves
the original summaries, then a masked mean combines valid traces. This reduces
attention to trace summaries, at the cost of compressing individual-locus evidence first.

The metadata path maps training-standardized area, valid-trace count divided by
four, observed-point fraction and an area-missing flag to 16 learned features.
These join the 32 geometry features by concatenation, producing 48 entries.
Dropout 0.2 precedes a 48→32 ReLU→3 classifier. Metadata can encode segmentation
or assay coverage as well as biology. The model has **161,075** parameters, compared with 39,875 in HW2.
The design adapts [residual learning](https://arxiv.org/abs/1512.03385) and
[attention over sets](https://proceedings.mlr.press/v97/lee19d.html) to ordered,
partly observed chromosome traces.

## Experimental design

The new panel contains six fits: seeds 42, 43 and 44 for the complete model and a
matched attention-bypass ablation. Both arms keep the same encoders, metadata,
cohort, split and training settings, starting from identical weights within each seed. Bypass removes the attention update while
retaining its identity path. The comparison tests the contribution of trace attention.

Both arms use SGD learning rate 0.01, momentum 0.9, batch size 64, cross-entropy,
maximum 1,000 epochs and patience 100. Two loss forwards are averaged per batch:
one original view and one whole-cell rotation of XYZ. Distances and metadata remain
unchanged. Validation uses the unaugmented view. Only strict validation macro-F1
gains reset patience. Lower loss breaks equal-F1 checkpoint ties, with epoch zero
eligible. Macro F1 gives each class equal evaluation weight. The training loss is
unweighted cross-entropy.

The original HW2 used zero momentum, one loss forward, maximum 2,000 epochs and
patience 200, with interpolated distance inputs. I reduced the cap and patience
to limit overtraining and added momentum to improve optimization. Its independent
effect on generalization remains unmeasured. The baseline comparison evaluates a
package of representation, architecture and training changes. The matched ablation
supports a narrower interpretation about attention within the new model.

A secondary reference reuses three completed HW2 fits with the same cohort,
optimizer, batch size, 1,000-epoch cap, patience 100 and two-forward loss as the new
arms. Its original dense-distance representation and dropout 0.8 remain intact.
This supplies a controlled training-budget comparison while preserving the original
HW2 result as the primary baseline. Representation differences still prevent an
architecture-only interpretation.

## Observations after running the experiments

| Model | Validation macro F1 | Mean GPU fit time |
|---|---:|---:|
| Original frozen HW2, seed 42 | 0.479058 | Not remeasured |
| Three-path CNN, attention bypassed | 0.4420 ± 0.0049 | 249.5 s |
| Three-path CNN with trace attention | 0.4390 ± 0.0051 | 291.0 s |

CNN values are means ± sample SD over three seeds on the same validation cells.
All six fits completed and passed prediction, checkpoint and source checks. GPU
wall times exclude the queue. Seed SD measures training variation on one reused
validation replicate. Biological-replicate uncertainty and full-fit peak memory
remain unmeasured. The new curves and per-class results are in the experiment report.

Attention's paired F1 changes were −0.00396, −0.00048 and −0.00484. All were negative,
with mean −0.00309. This supports no benefit in these runs, with no significance
claim. Mean Proerythroblast F1 fell from 0.3842 to 0.3663. Hepatocyte and Erythroblast
means rose slightly, so the class effects differ.
The secondary budget-matched HW2 reference averaged macro F1 **0.2064±0.0429** and
**37.8 seconds** per fit. It failed to learn useful classification under this regime.
Two seeds selected epoch zero, and final training loss remained near 1.08. Its weak
score cannot establish improvement over the original HW2 model. The optimizer's
role remains a hypothesis requiring a separate controlled test.

## Interpretation and next experiment

The hypothesis was unsupported. Attention selected epochs 11–16 and completed
111–116 epochs under patience 100. Its final training F1 was 0.993–1.000, while
validation cross-entropy reached 3.47–7.95. Earlier selected checkpoints avoid those
extreme final states, but the widening gap shows poor generalization. The bypass
also showed excess fitting. Seed 42 additionally became unstable: training loss
rose from 0.0049 at epoch 50 to about 1.07 by epoch 56 and remained high. The cause
requires a controlled test.

The larger 161,075-parameter model can fit more variation than the 39,875-parameter
HW2 network. Coordinate noise, weaker dropout and replicate shift are plausible
competing explanations for weaker validation performance. The current experiment
cannot assign the failure to one of them. Area was complete in both splits, so
its missing flag supplied no variation. Attention weights and metadata contributions
have no independent biological interpretation here.

The next test would reduce encoder depth while holding inputs, attention and
training fixed. If the training–validation gap shrank without losing validation
F1, that would support excess capacity as an explanation. A metadata ablation could
then test reliance on assay coverage. Genomic order and masks retain observed
structure, while bad measured coordinates and DNA paths between loci remain unresolved.
