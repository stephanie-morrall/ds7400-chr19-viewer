# FOF-CT Chr19 Viewer

A web application for viewing fluorescence-in-situ chromosome tracing (FOF-CT)
spots on mouse chromosome 19, with a trained neural network that predicts three cell
types from chromosome distances. The Viewer tab is a gallery of cells. Click a
tile to open one cell as a 3D Plotly scene. The Notebook tab and each cell
page store text notes, per-trace notes, per-locus TAD notes,
and an optional pinned locus in SQLite through FastAPI.

## Type of data being visualized

Imaged loci on chromosome 19. Each spot has `Spot_ID`, `Trace_ID`, `X`, `Y`,
`Z` (microns), genomic start and end, and `Cell_ID`. Spots that share a
`Trace_ID` are one trace. When the local catalog is present, the
gallery covers 13,306 labeled cells across four biological replicates
(biorep01–04). A clone without that catalog loads the committed sample cells
411, 328, and 530. The replicate 1 coordinates are the 4DN file
[4DNFIQCHBYZ6](https://data.4dnucleome.org/files-processed/4DNFIQCHBYZ6/),
reported in Liu et al.,
[Multiplexed imaging of nucleome architectures in single cells of mammalian tissue](https://pubmed.ncbi.nlm.nih.gov/32518300/)
(Nature Communications, 2020).

## Project organization

- `frontend/`: React application and browser sample files.
- `backend/`: FastAPI application and local annotation database.
- `training/results/three_class_residual_f1_v1/`: frozen model weights, cleaning protocol, settings and evaluation.
- `data/`: local source tables, excluded from Git.
- `training/`: shared model, features, training, evaluation and recorded results.
- `scripts/`: catalog building and sample-shape fitting utilities.

The model is connected to the Viewer and the **Train & monitor** tab.
[training/README.md](training/README.md) maps the seven training steps to code and checks.
[Training details](training/README.md) describe the comparison and reproduction commands.

## Instructions for running the application

The verified API runs in Docker with Python 3.12 and CPU PyTorch 2.11.0.
The React app needs Node.js 20.19+ in the 20.x line, or 22.12+ (`node -v`). Run the following
from a copy containing the entire `training/results/three_class_residual_f1_v1/` folder.
The Docker image needs its weights, baseline weights, metadata, requirements, metrics,
history, validation-exploration summary and freeze record. Raw source data stay outside it.

1. Start Docker Desktop.
2. In the repository root: `docker compose up --build`
3. In another terminal: `cd frontend`
4. `npm install`
5. `npm run dev`
6. Open `http://localhost:5173/`

Vite proxies `/api` to `http://127.0.0.1:8000` (the FastAPI container). Leave
both running. The sample under `frontend/public/sample/` is enough for a clone to
show cells and request predictions for quality-passing cells, with the saved model files above.
Open **Viewer**, select a cell and click **Predict cell type**. The recorded label,
predicted type, uncalibrated model score and model version are shown separately.
Prediction uses all measured spots, including traces hidden in the plot.

The full catalog is gitignored. To rebuild it from local tables:

`python3 scripts/build_viewer_catalog.py`

That writes `frontend/public/catalog.json` and per-cell JSON under
`frontend/public/biorep01/` through `biorep04/`. Do not commit those catalog files,
`data/`, or `backend/data/` (the SQLite file).

### Local data for training and the complete gallery

The committed sample contains cells 411, 328 and 530. Interactive training requires
these original local files at exactly these paths:

| File | Use |
| --- | --- |
| `data/raw/biorep01/4DNFIQCHBYZ6.csv` | Training coordinates, replicate 1. |
| `data/raw/biorep02/4DNFIA855FW4.csv` | Training coordinates, replicate 2. |
| `data/raw/biorep03/4DNFIY3XDBW8.csv` | Validation coordinates, replicate 3. |
| `data/labels/Cell label.csv` | Recorded cell labels, selected by replicate and cell. |
| `data/labels/cell_type_mapping.csv` | Named cell-type mapping. |

The frozen model and benchmark comparison additionally used
`data/raw/biorep04/4DNFI593PAKT.csv` for its held-out test. Rebuilding the full gallery
uses all four coordinate files. The practice trainer reads coordinate replicates 1–3
and never reevaluates replicate 4. Missing or changed source files produce a visible
error. There is no substitute training dataset.

Docker Compose mounts `./data` read-only at `/app/data`. Annotation storage is
`backend/data/annotations.db`. Separate practice outputs go to `backend/training-runs/`.
The source tables, database and generated practice runs stay outside Git. The supplied
three-cell sample supports viewing and prediction without the full source tables.

### Train & monitor

Open **Train & monitor**, choose an epoch count and dropout probability, and start
a separate practice model with the selected architecture. The control lets you try
other probabilities without replacing the evaluated model or its test records.
One practice run is admitted at a time. Each epoch publishes real train/validation
loss and macro F1 to the live charts. The frozen results table includes macro F1
and accuracy for train/validation/test.

Dropout randomly zeros hidden values during training and rescales retained values.
Evaluation disables it. A practice choice is an experiment, not evidence of a better
frozen model. Practice output folders preserve chosen settings and their own results.
The **Frozen evaluated model** table separately shows recorded train/validation/test
metrics. The selected-cell prediction continues to use the frozen predictor.
Live job state resets with the API. Keep its configured single worker.

## Major libraries / frameworks

- React 19
- Vite 8
- Plotly (`react-plotly.js` and `plotly.js-dist-min`)
- FastAPI and Pydantic for request validation, served by Uvicorn
- SQLite
- Docker Compose
- PyTorch 2.11 for layers, automatic differentiation and SGD
- pandas and NumPy for source tables and arrays
- SciPy for chromosome distances
- scikit-learn for classification metrics

Python package versions are pinned in [training/requirements.txt](training/requirements.txt),
[backend/requirements.txt](backend/requirements.txt) and the frozen model
[requirements-model.txt](training/results/three_class_residual_f1_v1/requirements-model.txt). Frontend
versions are in `frontend/package-lock.json`. Docker installs the official CPU wheel.

## Implemented features

- Gallery of square stills, 30 cells per page, three columns; the whole tile
  opens the cell
- Filters: Replicate, Cell type, Traces, each with an Any option
- Jump-to-page number field (Enter or leave the field); Previous and Next
- One-cell Plotly 3D view: rotate, pan, zoom
- Per-trace checkboxes and a color key that matches the 3D plot; click a
  trace name in the key to open a note box for that trace
- Click-to-pin of Spot_ID, genomic start, and recorded distances to the
  lamina and nucleolus (pin saves to SQLite)
- Double-click a 3D spot to open a TAD note titled with that locus's TAD
  number (1-based rank of unique Chrom_Start values in the cell) and its
  genomic location
- Per-cell text note (Save note writes to SQLite; Delete removes the row)
- Notebook tab: one pad for the dataset, then a card per cell that has
  notes, with nested boxes for that cell's trace and TAD notes; click the
  card or the note text to jump to that cell; Delete on each box removes
  that note
- On the cell page, note boxes sit to the left of the 3D cube
- About tab with Homework 1 steps 1–5 and the current Homework 2 prediction workflow
- Nuclear-envelope and nucleolus toggles on sample cells 411, 328, and 530
  only
- **Predict cell type** on the selected cell, with an uncalibrated score, model
  version, supported classes, loading state and request errors
- **Models**: architecture diagrams for the selected network and plain ReLU benchmark,
  their shared inputs, recorded results and evaluation limits
- **Train & monitor**: real separate training runs with editable epochs and dropout,
  live training/validation plots, frozen train/validation/test metrics and predictions
- FastAPI + SQLite in Docker, preserving the original annotation routes

## Annotation

Yes. A Notebook tab holds one pad for the dataset, then a card for each
cell that has a cell note, trace note, or TAD note. Nested boxes on that
card hold the trace and TAD notes. Click a card or its note text to open
that cell. Each cell page puts the note boxes to the left of the 3D cube,
with a color key of traces (click the name to write a trace note), a cell
note, optional TAD notes from a double-clicked locus, and an optional
pinned locus. Notes, labels, TADs, and pins save through FastAPI. Delete
on a note box removes that row from SQLite.

## Backend / database

Yes. FastAPI + SQLite running in Docker Compose. The database stores the
notebook pad, per-cell notes, per-trace labels, per-cell TAD notes,
pins, and the original unused `predictions` table. Prediction requests run the trusted
frozen PyTorch model in memory and return a result directly. The service loads the model
once at startup. Training and serving use the same saved cleaning protocol and feature order.

## Model and measured performance

The predictor classifies Hepatocyte, Erythroblast and Proerythroblast cells. Unknown
labels are excluded. A class enters the comparison when at least 300 quality-passing
training cells have that label. Training uses 2,439 cells from replicates 1 and 2;
validation uses 1,347 cells from replicate 3. The same three classes define the test.

The residual MLP takes 1,176 pairwise-distance features, uses three 32-unit hidden
layers with ReLU, and produces three scores. One skip connection adds the first
hidden activation to the residual branch. Dropout 0.8 acts inside that branch during
training. The skip and input projection do not have dropout. The model has 39,875
parameters and uses SGD. It is a small residual MLP inspired by residual connections,
not a pretrained ImageNet ResNet. The benchmark has one 32-unit ReLU hidden layer
and no dropout. Macro F1 selects each checkpoint, with lower validation loss breaking
ties. The served pair uses seed 42 and the selected residual checkpoint is epoch 135.

Filter A keeps loci missing from at most 50% of traces in each training replicate.
Filter B keeps cells with at most four raw traces. Filter C keeps traces with at
least 40 of 49 retained locations measured. Missing coordinates are filled on the
original locus-rank axis with linear interpolation and endpoint extrapolation. The
1,176 pairwise distances are averaged across each cell's usable traces. Labels and
annotations stay outside model inputs.

The 44-configuration exploratory comparison used validation macro F1, averaged over
three separately trained seeds. The residual model averaged 0.4868 and the fixed
shallow benchmark averaged 0.4774. The paired exploratory randomization test gave
p = 0.1173. The same validation replicate selected the architectures and checkpoints,
so that comparison does not account for selection or biological sampling uncertainty.
The [complete comparison table](training/results/comparison.csv) contains all 44 configurations.

The final seed-42 comparison uses the same 1,458 cells from replicate 4 for both
models. This replicate was used in previous experiments, so it is a **reused holdout**.

| Model | Errors | Error rate | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| One-hidden-layer ReLU benchmark | 802 | 55.01% | 44.99% | 0.4358 |
| Residual MLP | 782 | 53.64% | 46.36% | 0.4427 |

The residual model made 20 fewer errors on this reused holdout. Performance varies by
class. This single comparison does not establish a reliable general advantage on new
biological replicates. The saved metrics and predictions are in
`training/results/three_class_residual_f1_v1/`.

## Methods explored

The work began with the basic MLP, then compared depth and width, ordinary ReLU and
Leaky ReLU, dropout, SGD with and without momentum, AdamW, and residual connections.
The assignment names ResNet among the architectures discussed in class. He et al.'s
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
explains residual connections. Srivastava et al.'s
[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
describes dropout. The project model is a small fully connected network using a skip
connection and was trained from scratch.

The experiments also tried a distance-based CNN, genomic-neighbor message passing,
class weighting and coordinate representations. These are explored methods; the
assignment lists message passing as an example of a more recent architecture. The [44-configuration table](training/results/comparison.csv) records all tested settings,
three-seed validation scores. The residual was essentially tied
with AdamW at learning rate 0.0001 (mean macro F1 0.486790). Mean distances plus trace
standard deviation scored 0.4842, compared with 0.4853 for an equal-size zero-input
control. The tested CNN scored 0.4796, message passing 0.4555, and the best coordinate
model 0.4193. These particular comparisons cannot rule out the broader model families.
Coordinates and distances also used different model capacities.

## Implementation attribution — assignment section 4

I designed and selected the model architecture. I used OpenAI Codex to help implement and debug the neural network and connect it to the web app. I also use OpenAI Codex to refine writing and documentation.

The code reuses
PyTorch layers, automatic differentiation and SGD, SciPy distances, pandas CSV reading,
SciPy interpolation and scikit-learn metrics. Relevant documentation and
training details are described in [training/README.md](training/README.md).
