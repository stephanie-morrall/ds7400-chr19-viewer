# Training and evaluation

The deployed model predicts Hepatocyte, Erythroblast and Proerythroblast, in that order.
Its frozen weights and measured results are in `results/three_class_residual_f1_v1/`.

## Data and features

Replicates 1 and 2 supply 2,439 training cells. Replicate 3 supplies 1,347 validation
cells. Unknown is excluded. A named type is included when it has at least 300
quality-passing training cells, yielding paper codes 3, 6 and 7. This rule is fitted
on training cells and applied unchanged to validation and test cells.

Filter A retains loci missing from at most half the traces in each training replicate.
Filter B keeps cells with at most four raw traces. Filter C retains traces with at
least 40 of the 49 retained loci measured. `cleaning.py` fills missing coordinates on
the original locus-rank axis, with endpoint extrapolation. `features.py` calculates
all 1,176 within-trace locus-pair distances and averages those distances across each
cell's surviving traces. `prepare.py` passes unscaled float32 distances to the model.
Labels, notes and source cell identifiers are excluded from model inputs.

## Architectures and training steps

`final_model.py` defines both models. The selected network has three 32-unit hidden
layers, ReLU and one residual addition. Its branch has dropout 0.8 after the first
branch activation. The benchmark has one 32-unit ReLU hidden layer and no dropout.
All parameters start from fresh random weights. Neither uses pretrained weights.

The seven assignment steps map to these functions:

| Step | Implementation |
| --- | --- |
| Initialize parameters | `ResidualMLP()` or `baseline_model()` in `final_model.py` |
| Forward propagation | `ResidualMLP.forward()` or the benchmark's sequential layers |
| Calculate loss | Cross-entropy in `train.py:train_epoch()` |
| Backpropagation | `loss.backward()` in `train_epoch()` |
| Calculate gradients | PyTorch automatic differentiation populates parameter gradients |
| Update parameters | SGD `optimizer.step()` in `train_epoch()` |
| Repeat until stopping | `final_model.py:fit_practice()` |

Both use learning rate 0.01, batch size 64 and a separately seeded shuffle generator.
Epoch zero is evaluated. Maximum validation macro F1 selects the checkpoint, with
lower validation loss breaking exact ties. Training stops after 200 epochs without
a strict F1 increase, up to 2,000 epochs. The frozen pair uses seed 42. The selected
residual checkpoint is epoch 135. Its recorded history includes train/validation
loss, accuracy and F1 for every measured epoch.

## Reproduce training

Install `requirements.txt` in a Python environment and place the original source
files at the paths listed in the root README. From the repository root:

```sh
python3 -m training.train_final --data data --output training/runs/reproduction
```

This trains fresh residual and benchmark models, preserves initial and selected
weights, records both loss curves and F1, and writes source hashes. The output must
be a new folder. It uses replicates 1–3 and leaves the supplied frozen results intact.
Numerical reproducibility can depend on PyTorch version, platform and CPU operations.
The provided training environment used Python 3.14.2, CPU PyTorch 2.11.0 and one thread.

`evaluate_final.py` documents the recorded replicate-4 evaluation of the frozen pair.
It verifies hashes and writes an access marker before opening test coordinates. It
refuses to overwrite an existing test evaluation. The supplied results already have
this marker. The application displays those saved results, and practice training
never reads replicate 4 or replaces the frozen model.

## Comparison and limitations

`results/comparison.csv` contains the full exploratory comparison: 44 configurations,
three seeds each. It includes 23 newly fitted configurations and 21 verified reused
configurations. All used the same class rule and F1 checkpoint selection. Methods
included depth/width and dropout grids, Leaky ReLU, residual additions, SGD momentum,
AdamW, class weighting, a distance CNN, genomic-neighbor message passing, mean-plus-SD
distance inputs and coordinate models. Mean validation macro F1 was 0.486815 for the
selected residual and 0.477357 for the shallow model. The deployed seed-42 validation
F1 is 0.479058. These means describe separate training seeds, not an ensemble or
biological replication.

The exploratory paired randomization calculation gave p = 0.1173. The same validation
replicate selected architectures and checkpoints, so this calculation does not adjust
for selection, training uncertainty or biological sampling. Cell dependence may also
matter. The observed difference does not establish a reliable general advantage.

The frozen pair was evaluated on the same 1,458 eligible replicate-4 cells. This
replicate was used by earlier models, making it a reused holdout. The residual made
782 errors and the benchmark made 802. Macro F1 was 0.442729 and 0.435776, respectively.
`test_metrics.json` records aggregate and per-class results. `test_predictions.csv`
retains paired target and predicted classes without source cell identifiers. Both
models improve some classes and worsen others: the residual's Hepatocyte F1 is 0.3584,
versus 0.4097 for the benchmark. Erythroblast F1 is 0.5626 versus 0.5235, and
Proerythroblast F1 is 0.4071 versus 0.3741.

## Model references

- He et al. (2016), [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
  The residual addition inspired this small fully connected model.
- Srivastava et al. (2014), [Dropout](https://jmlr.org/papers/v15/srivastava14a.html).
  Dropout regularizes the residual branch during training.
- Yeh (2000), [More accurate tests for the statistical significance of result differences](https://aclanthology.org/C00-2137/).
  Paired randomization motivates the exploratory F-score comparison.
- Cawley and Talbot (2010), [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://jmlr.org/papers/v11/cawley10a.html).
  Model selection limits interpretation of the validation differences.

The root README contains the implementation attribution required by assignment section 4.
