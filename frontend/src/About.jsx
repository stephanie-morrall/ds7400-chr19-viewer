// Maps the HW1 steps onto the viewer and describes the current HW2 additions.

function About() {
  return (
    <div className="about">
      <section>
        <h2>Homework 1 steps</h2>
        <p>
          This page is the map from the DS7400 Homework 1 PDF (steps 1–5) onto
          this application. Viewer provides data exploration. HW2 adds cell-type
          prediction, a model summary, and interactive training.
        </p>
      </section>

      <section>
        <h2>Step 1 — Get the web application running</h2>
        <p>
          The app is a React 19 site created with Vite 8 in the <code>frontend/</code>
          folder. <code>npm run dev</code> in that folder serves
          <code>http://localhost:5173/</code>. The tabs are Viewer, About,
          Notebook, Models, and Train &amp; monitor. Notes and pinned spots save through FastAPI to SQLite
          when Docker Compose is running.
        </p>
      </section>

      <section>
        <h2>Step 2 — Display your data</h2>
        <p>
          Each spot is one imaged locus on mouse chromosome 19 from
          fluorescence-in-situ chromosome tracing (FOF-CT). Columns include
          Spot_ID, Trace_ID, X, Y, Z in microns, genomic start and end, and
          Cell_ID. Spots that share a Trace_ID are one trace, drawn in
          genomic order.
        </p>
        <p>
          When the local catalog is present, the gallery holds 13,306 labeled
          cells across four biological replicates (3,093 + 3,252 + 3,220 +
          3,741) from the Wang lab tables on the 4D Nucleome portal. Cell-type
          names come from the label table: Erythroblast (3,315), Hepatocyte
          (3,209), Unknown (2,918), Proerythroblast (2,061), Macrophage (731),
          Other (501), Endothelial (400), and Megakaryocyte (171). A clone
          without that catalog loads sample cells 411, 328, and 530 from
          <code>frontend/public/sample/</code>.
        </p>
        <p>
          Rebuild the catalog with
          <code>python3 scripts/build_viewer_catalog.py</code>. That writes
          gitignored <code>frontend/public/catalog.json</code> and per-cell JSON
          under <code>frontend/public/biorep01/</code> through
          <code>biorep04/</code>.
        </p>
      </section>

      <section>
        <h2>Step 3 — Add basic interaction</h2>
        <p>
          The Viewer tab is a three-column grid of square stills, 30 cells per
          page. Replicate, Cell type, and Traces each have an Any option. Type
          a page number next to Previous and Next to jump without stepping page
          by page. Click the square drawing to open that cell. The detail view
          is one
          Plotly 3D scene: rotate, pan, and zoom. Back returns to the same
          gallery page.
        </p>
      </section>

      <section>
        <h2>Step 4 — Make the viewer useful</h2>
        <p>
          The research application this viewer is built for is cell-type
          classification from chromosome-19 spatial structure: whether the 3D
          arrangement of chr19 traces in a cell is enough to recover
          the cell-type label.
        </p>
        <p>
          The Cell type filter is those labels, so a reader can page through
          hepatocytes or macrophages as grouped examples, then compare a selected
          cell with its prediction. The Replicate filter checks whether a pattern holds across
          the four biological replicates. The Traces filter finds cells with a
          chosen number of traces. Jump-to-page makes a 13,306-cell
          catalog searchable by position. Each tile is a canvas snapshot so
          thirty previews can stay on one page.
        </p>
        <p>
          On one cell, a checkbox per Trace_ID hides or shows each trace. A
          color key next to those checkboxes uses
          the same colors as the 3D plot. Click a trace name in the key to
          open a note box for that trace. Click a spot once to pin genomic
          start, Spot_ID, and the recorded distances to the lamina and
          nucleolus; that pin is stored in SQLite. Double-click a spot to
          open a TAD note titled with that locus's TAD number (the 1-based
          rank of unique Chrom_Start values in this cell) and its genomic
          location. Note boxes sit to the left of the 3D cube. A text note on
          the same page is a per-cell label. The
          Notebook tab holds one pad for the whole dataset, then a card for
          each cell that has notes. Trace and TAD notes for that cell sit in
          nested boxes on the same card. Click a card or its note text to open
          that cell.
          Nuclear-envelope and nucleolus wireframes are toggles on sample cells
          411, 328, and
          530, where those shapes were fitted offline. Those controls are the
          inspection tools beside the cell-type predictor: the structure
          the model sees and the recorded label it is asked to match.
        </p>
      </section>

      <section>
        <h2>Step 5 — Think ahead</h2>
        <p>
          HW2 implements the next step: a neural network uses distances between
          chromosome-19 loci to predict Hepatocyte, Erythroblast, or Proerythroblast. The
          recorded type on the detail toolbar stays separate from the prediction.
        </p>
        <p>
          On the one-cell page, Predict cell type sends the selected measurements
          to FastAPI and displays the saved network's prediction and uncalibrated
          model score. Models compares its selected architecture and results with the initial
          plain ReLU MLP benchmark. Train &amp; monitor displays the saved results and runs separate
          practice models with editable epochs and dropout.
        </p>
      </section>
    </div>
  )
}

export default About
