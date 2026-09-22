# FOF-CT Chr19 Viewer

- **A brief description of your application:** A web application for viewing
  fluorescence-in-situ chromosome tracing (FOF-CT) spots on mouse chromosome
  19. The Viewer tab is a gallery of cells. Click a tile to open one cell as a
  3D Plotly scene. The Notebook tab and each cell page store text notes,
  per-fiber trace notes, per-locus TAD notes, and an optional pinned locus in
  SQLite through FastAPI.

- **The type of data being visualized:** Imaged loci on chromosome 19. Each spot
  has `Spot_ID`, `Trace_ID`, `X`, `Y`, `Z` (microns), genomic start and end,
  and `Cell_ID`. Spots that share a `Trace_ID` are one traced fiber. The
  replicate 1 coordinates are the 4DN DNA-spot/trace core table
  [4DNFIQCHBYZ6](https://data.4dnucleome.org/files-processed/4DNFIQCHBYZ6/)
  from the Siyuan Wang lab. When a local catalog built from the private tables
  is present, the gallery covers 13,306 labeled cells across four biological
  replicates (biorep01 through biorep04). A clone without that catalog loads
  the committed sample cells 411, 328, and 530.

- **Instructions for running the application:**
  1. Start Docker Desktop.
  2. In the repository root, run `docker compose up --build`.
  3. In another terminal, run `cd web`.
  4. Run `npm install`.
  5. Run `npm run dev`.
  6. Open `http://localhost:5173/`.

  Vite proxies `/api` to `http://127.0.0.1:8000` (the FastAPI container). Leave
  both running. The sample under `web/public/sample/` is enough for a clone to
  show cells. The full catalog is gitignored. To rebuild it from local tables,
  run `python3 scripts/build_viewer_catalog.py`. That writes
  `web/public/catalog.json` and per-cell JSON under `web/public/biorep01/`
  through `biorep04/`. Do not commit those catalog files, `data/`, or
  `api/data/` (the SQLite file).

- **Major libraries/frameworks used:** React 19, Vite 8, Plotly
  (`react-plotly.js` and `plotly.js-dist-min`), FastAPI, SQLite, and Docker
  Compose.

- **A brief description of implemented features:**
  - Gallery of square stills, 30 cells per page, three columns; the whole tile
    opens the cell
  - Filters: Replicate, Cell type, Traces, each with an Any option
  - Jump-to-page number field (Enter or leave the field); Previous and Next
  - One-cell Plotly 3D view: rotate, pan, zoom
  - Per-trace checkboxes and a color key that matches the 3D plot; click a
    trace name in the key to open a note box for that fiber
  - Click-to-pin of Spot_ID, genomic start, and recorded distances to the
    lamina and nucleolus (pin saves to SQLite)
  - Double-click a 3D spot to open a TAD note titled with that locus's TAD
    number (1-based rank of unique Chrom_Start values in the cell) and its
    genomic location
  - Per-cell text note (Save note writes to SQLite; Delete removes the row)
  - Notebook tab: one pad for the dataset, then a card per cell that has
    notes, with nested boxes for that cell's fiber and TAD notes; click the
    card or the note text to jump to that cell; Delete on each box removes
    that note
  - On the cell page, note boxes sit to the left of the 3D cube
  - About tab with Homework 1 steps 1–5, including the later classifier
    placement on the one-cell page and gallery tiles
  - Nuclear-envelope and nucleolus toggles on sample cells 411, 328, and 530
    only
  - FastAPI + SQLite in Docker, including an empty `predictions` table for a
    later cell-type classifier

- **Whether you implemented annotation:** Yes. A Notebook tab holds one pad for
  the dataset, then a card for each cell that has a cell note, fiber note, or
  TAD note. Nested boxes on that card hold the fiber and TAD notes. Click a
  card or its note text to open that cell. Each cell page puts the note boxes
  to the left of the 3D cube, with a color key of traces (click the name to
  write a fiber note), a cell note, optional TAD notes from a double-clicked
  locus, and an optional pinned locus. Notes, labels, TADs, and pins save
  through FastAPI. Delete on a note box removes that row from SQLite.

- **Whether you implemented a backend/database:** Yes. FastAPI + SQLite running in
  Docker Compose. The database stores the notebook pad, per-cell notes,
  per-fiber trace labels, per-cell TAD notes, pins, and an empty `predictions`
  scaffold.

## Three-minute demo

Record a live run of `http://localhost:5173/` with Docker Compose already up.
Stay under three minutes.

1. Say the data: FOF-CT spots on mouse chromosome 19, X Y Z in microns.
2. Show the web app (URL bar and the three tabs).
3. On Viewer: gallery, one filter or jump-to-page, click a square preview to
   open a cell, rotate the 3D plot.
4. Type a cell note, click Save. Click a trace name in the color key, type a
   fiber note, and Save. Click a spot once to pin it. Double-click a spot to
   open a TAD note, type, and Save. Delete removes a note.
5. Open Notebook, type a line on the dataset pad, click Save. Show that cell's
   card with nested fiber and TAD notes. Click the card or the note text to
   jump to the cell.
6. Refresh the browser. Open the same cell and Notebook and show the note,
   fiber note, TAD note, and pin are still there (that is the database).

The Homework 1 snapshot is the GitHub Release tagged `hw1`. Later homework
snapshots can be added as new releases on this same repository.
