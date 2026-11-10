# HW3 explanation and component rationale

## 1. The problem or limitation — approximately 0:00–0:25

**Show:** the original HW2 diagram and its frozen validation score.

My HW2 model averages distances across chromosome traces to classify three cell
types. Averaging before learning can hide differences among traces, and within-trace
distances discard their relative placement. I wanted to retain those patterns.

## 2. The proposed design and why it might work — approximately 0:25–1:25

**Show:** the implemented three-path diagram and repeated residual block.

I used three paths: XYZ, pairwise distances and metadata. Rotating all traces
together preserves their relative geometry. Distances provide orientation-invariant
folding information.

Two CNNs follow genomic order, each with four repeated residual blocks. Stacking
small kernels builds complex patterns. Skip connections provide a direct path for
earlier features.

Their features join in order, creating one vector per trace. Attention compares
all trace vectors before pooling, with another skip preserving each trace's own
features. A small metadata layer adds area and measurement coverage. Concatenation
places geometry and metadata side by side. Dropout masks features during training,
then the classifier produces three scores. Masks keep missing measurements unavailable.

## 3. The experiments performed — approximately 1:25–1:55

**Show:** the frozen HW2 baseline, the matched-training reference and two new conditions.

I kept the original HW2 result and a budget-matched, three-seed HW2 reference.
I trained three seeds with and without attention, sharing data, split, optimizer
and budget: momentum SGD, maximum one thousand epochs and patience one hundred.
Macro F1 weights classes equally. Preprocessing still differs from HW2.

## 4. What happened, including failures or unexpected results — approximately 1:55–2:35

**Show:** recorded scores, class results and actual learning curves.

Original HW2 scored point four seven nine macro F1. The CNN averaged point four
four two without attention, and point four three nine with it. Attention was
slightly worse in every seed and took two hundred ninety-one seconds per fit,
versus two hundred fifty.

Attention overfit: training F1 approached one while validation loss rose.
Proerythroblast F1 fell. One bypass run also became unstable. The budget-matched
HW2 reference failed, so its weak score cannot establish an overall improvement.

## 5. What was learned — approximately 2:35–2:55

**Show:** the supported conclusion and one controlled next test.

The hypothesis was unsupported. More information and depth increased capacity
without improving generalization. My next test would reduce encoder depth while
holding inputs and training fixed, then check the training–validation gap.

## Component rationale for rehearsal and questions

The main idea is to keep three complementary representations separate until each
has learned a useful encoding, then compare traces before the cell-level summary.
The following explanations are rehearsal notes, outside the spoken narration.

| Component | Operation | Rationale and possible failure |
|---|---|---|
| Shared cell center | Subtracts one observed-point center from every trace | Removes arbitrary imaging origin and preserves trace placement. The traced-point center differs from a measured nuclear center. |
| Whole-cell rotation | Applies the same random 3D rotation to all traces during training | Encourages orientation robustness while preserving geometry. It assumes imaging orientation is incidental to the cell label. It does not guarantee invariance or noise robustness. |
| XYZ branch | Reads three coordinates at each of 49 ordered loci | Keeps relative trace arrangement available. Absolute orientation may be a shortcut unless augmentation succeeds. |
| Distance branch | Reads 49 distances to named reference loci at each locus | Exposes rotation-invariant folding without early trace averaging. Distances overlap with XYZ information and inherit coordinate noise. |
| Observation and endpoint masks | Exclude unavailable values and invalid distance endpoints | Preserve measurement availability without preprocessing coordinate imputation. Bad observed measurements remain possible. |
| Two separate Conv1D encoders | Each has a width-three stem and four two-convolution residual blocks | Separates coordinate and distance processing. Shared weights across traces limit growth with trace count. Branch-specific weights still add capacity. |
| Repeated small kernels | Compose local operations along genomic order | Connects VGG-style stacking to local folding patterns. The retained genomic gaps have unequal physical lengths. |
| Residual additions | Add the input to the learned block correction | Preserve an information and gradient path, following ResNet. Skips help optimization and do not guarantee less overfitting. |
| ReLU | Sets negative responses to zero | Gives the repeated layers nonlinear feature extraction. The activation is a declared choice rather than an isolated tested contribution. |
| Per-locus concatenation | Appends XYZ 32 and distance 32 features into 64 entries | Keeps both encodings before the ordered readout. This can retain redundant or noisy information. |
| Ordered readout | Flattens 49×64 entries, then projects 3,136→32 | Lets named loci receive different weights. The 32-value summary deliberately compresses information before attention. |
| One trace-level attention layer | Receives T trace vectors and returns T updated vectors | Models relationships among the one to four trace summaries with a simple, small attention set. Individual-locus relationships may already have been lost. |
| Attention residual skip | Adds the original trace vectors to their context updates | Retains each trace's own evidence. A single-trace cell has no other trace from which to obtain context. |
| Valid-trace mean | Combines the attention-updated valid vectors | Handles unequal trace counts and arbitrary trace numbering. Padded traces contribute nothing. |
| Metadata path | Maps scaled area, trace count, observed fraction and missing-area flag to 16 features | Supplies measured cell context and acquisition coverage. The model may learn technical missingness or segmentation effects. Cell IDs, labels and replicate IDs are excluded. |
| Training-only scaling | Fits area statistics using training cells | Keeps validation information outside fitted preprocessing. Missing flags retain their meaning. |
| Geometry/metadata concatenation | Appends geometry 32 and metadata 16 into 48 entries | Keeps both sources available for the dense classifier to combine. The entries are learned features rather than 48 measured variables. |
| Dropout 0.2 | Masks and rescales feature entries during training, preserving width 48 | Discourages co-adaptation. Evaluation passes all features through. The chosen probability is unoptimized. |
| Dense classifier | Maps 48→32, ReLU, then 32→3 logits | Learns combinations of geometry and metadata for the three classes. Prediction softmax converts scores to probabilities. |
| Matched attention bypass | Bypasses the attention update with all other branches unchanged | Tests attention within this architecture. It does not isolate the individual benefits of XYZ, distances, metadata or depth. |
| Secondary HW2 training reference | Reuses three completed HW2 fits with matched optimizer, budget and two-forward loss | Addresses training-budget consistency while the frozen original remains the primary baseline. Its interpolated-distance representation still differs, and its failure to learn prevents an overall improvement claim. |
| Momentum SGD | Uses learning rate 0.01 and momentum 0.9 | Carries recent gradient direction into later updates. Momentum can also overshoot. Its generalization effect requires a separate optimizer comparison. |
| Checkpoint and stopping rule | Maximum 1,000 epochs, patience 100 on strict macro-F1 improvement | Limits plateau training and retains the best validation checkpoint. Repeated validation use still limits generalization claims. |

