import { useEffect, useState } from 'react'
import PredictionPanel from './PredictionPanel'
import {
  fetchCurrentTraining, fetchTrainingJob, fetchTrainingMetrics, startTraining,
} from './lib/trainingApi'

const ACTIVE = new Set(['preparing', 'running'])

function historyTraces(history, metric) {
  return ['train', 'validation'].map((split, index) => {
    const rows = history.filter((row) => row.split === split)
    return {
      name: split, type: 'scatter', mode: 'lines',
      x: rows.map((row) => row.epoch),
      y: rows.map((row) => metric === 'accuracy' ? 100 * row[metric] : row[metric]),
      line: { dash: index ? 'dash' : 'solid', width: 2 },
    }
  })
}

function HistoryCharts({ Plot, job, dark }) {
  const fontColor = dark ? '#f3f4f6' : '#08060d'
  return (
    <div className="training-charts">
      {['macro_f1', 'loss'].map((metric) => (
        <Plot key={metric} data={historyTraces(job.history, metric)}
          layout={{
            title: { text: metric === 'loss' ? 'Training / validation loss' : 'Training / validation macro F1' },
            xaxis: { title: { text: 'Epoch' }, rangemode: 'tozero' },
            yaxis: { title: { text: metric === 'loss' ? 'Cross-entropy loss' : 'Macro F1' } },
            font: { color: fontColor, size: 13 }, uirevision: job.id,
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 64, r: 20, t: 55, b: 55 },
            legend: { orientation: 'h', y: -0.25 },
          }}
          config={{ responsive: true, displaylogo: false, displayModeBar: false }}
          style={{ width: '100%', height: '310px' }} useResizeHandler />
      ))}
    </div>
  )
}

function FrozenMetrics({ metrics }) {
  // Missing historical F1 measurements remain unavailable until a new report exists.
  const f1 = (value) => Number.isFinite(value) ? value.toFixed(3) : 'Unavailable'
  const activation = {
    relu: 'ReLU', leaky_relu: 'Leaky ReLU',
  }[metrics.architecture.activation] ?? metrics.architecture.activation ?? 'activation unrecorded'

  return (
    <section className="training-section">
      <h2>Frozen evaluated model</h2>
      <p>Model {metrics.model_version}, selected epoch {metrics.best_epoch}.
        These recorded measurements belong to the saved model used for predictions.</p>
      <h3>Selected model by split</h3>
      <div className="training-table-wrap"><table className="training-table">
        <thead><tr><th>Split</th><th>Cells</th><th>Macro F1</th><th>Weighted F1</th>
          <th>Loss</th><th>Accuracy</th><th>Balanced accuracy</th></tr></thead>
        <tbody>{Object.entries(metrics.splits).map(([split, values]) => (
          <tr key={split}><th>{split}</th><td>{values.n_cells}</td>
            <td>{f1(values.macro_f1)}</td><td>{f1(values.weighted_f1)}</td>
            <td>{values.loss.toFixed(4)}</td><td>{(100 * values.accuracy).toFixed(2)}%</td>
            <td>{(100 * values.balanced_accuracy).toFixed(2)}%</td></tr>
        ))}</tbody>
      </table></div>
      <h3>Test comparison on the same cells</h3>
      <div className="training-table-wrap"><table className="training-table">
        <thead><tr><th>Model</th><th>Macro F1</th><th>Error rate</th>
          <th>Errors</th><th>Cells</th></tr></thead>
        <tbody>
          <tr><th>Baseline: single hidden layer MLP<br />
            <span className="training-description">
              {metrics.baseline.hidden_widths.join(' → ')} ReLU neurons · no dropout
            </span></th>
            <td>{f1(metrics.baseline.test.macro_f1)}</td>
            <td>{(100 * metrics.baseline.test.error).toFixed(2)}%</td>
            <td>{metrics.baseline.test.n_errors}</td><td>{metrics.baseline.test.n_cells}</td>
          </tr>
          <tr><th>Selected: {metrics.architecture.name}<br />
            <span className="training-description">
              {metrics.architecture.hidden_widths.join(' → ')} {activation} neurons
              {' '}· dropout {metrics.architecture.dropout}
            </span></th>
            <td>{f1(metrics.splits.test.macro_f1)}</td>
            <td>{(100 * metrics.splits.test.error).toFixed(2)}%</td>
            <td>{metrics.splits.test.n_errors}</td><td>{metrics.splits.test.n_cells}</td>
          </tr>
        </tbody>
      </table></div>
      <p className="training-limit">{metrics.evaluation_note}</p>
      <details>
        <summary>Architecture and checkpoint details</summary>
      <p>{metrics.architecture.name}: {metrics.architecture.family === 'message_passing'
        ? `${metrics.cleaning.retained_loci.length} genomic locations, 2 message-passing layers
          (${metrics.architecture.hidden_widths.join(' → ')} features per location),
          mean pooling and ${metrics.architecture.outputs} class scores.`
        : `${metrics.architecture.inputs.toLocaleString()} inputs →
          ${metrics.architecture.hidden_widths.join(' → ')} hidden neurons (${activation}) →
          ${metrics.architecture.outputs} class scores.`}</p>
      <p>Training stopped after {metrics.stopping.completed_epochs} epochs:
        {' '}{metrics.stopping.completed_epochs < metrics.training.epochs
          ? `${metrics.stopping.patience} epochs passed without a strict increase in validation macro F1.`
          : 'the declared epoch limit was reached.'}</p>
      <p>The checkpoint with the highest validation macro F1 was selected. Lower validation
        loss breaks exact F1 ties, and ties do not reset the stopping patience.</p>
      <p>Frozen model dropout rate: {metrics.architecture.dropout}.
        Dropout is inactive during predictions.</p>
      <p>The first hidden layer is followed by a two-layer residual branch. The branch
        adds its input back before the final ReLU. Dropout acts after the first branch
        activation during training.</p>
      </details>
      <p>Macro F1 averages each type’s precision–recall F1 score equally.
        Weighted F1 weights those scores by recorded cell counts.
        Balanced accuracy averages recall only. F1 ranges from 0 to 1.</p>
      {metrics.zero_recall_classes.length > 0 && <p className="training-limit">
        On test cells, this model correctly identified none of these types:
        {' '}{metrics.zero_recall_classes.join(', ')}.</p>}
    </section>
  )
}

export default function TrainingPanel({ Plot, dark, cellId, replicate, loadSpots }) {
  const [epochs, setEpochs] = useState(20)
  const [useDropout, setUseDropout] = useState(true)
  const [dropoutRate, setDropoutRate] = useState(0.8)
  const [job, setJob] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [available, setAvailable] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')
  const active = Boolean(job && ACTIVE.has(job.status))
  const jobId = job?.id
  const epochNumber = Number(epochs)
  const validEpochs = Number.isInteger(epochNumber) && epochNumber >= 1 && epochNumber <= 200
  // Switching dropout off preserves the editable rate while submitting zero for the next run.
  const dropoutNumber = useDropout ? Number(dropoutRate) : 0
  const validDropout = !useDropout || (dropoutRate !== '' && Number.isFinite(dropoutNumber)
    && dropoutNumber >= 0 && dropoutNumber <= 1)

  useEffect(() => {
    let ignore = false
    const controller = new AbortController()
    Promise.all([fetchTrainingMetrics(controller.signal), fetchCurrentTraining(controller.signal)])
      .then(([frozen, current]) => {
        if (!ignore) {
          setMetrics(frozen); setJob(current.job); setAvailable(current.available)
        }
      }).catch((reason) => { if (!ignore) setError(reason.message) })
    return () => { ignore = true; controller.abort() }
  }, [])

  useEffect(() => {
    if (!jobId || !active) return undefined
    let ignore = false
    let timer
    const controller = new AbortController()
    async function poll() {
      try {
        const current = await fetchTrainingJob(jobId, controller.signal)
        if (ignore) return
        setJob(current)
        if (ACTIVE.has(current.status)) timer = setTimeout(poll, 500)
      } catch (reason) { if (!ignore) setError(reason.message) }
    }
    poll()
    return () => { ignore = true; clearTimeout(timer); controller.abort() }
  }, [jobId, active])

  async function onStart() {
    setStarting(true); setError('')
    try { setJob(await startTraining(epochNumber, dropoutNumber)) }
    catch (reason) { setError(reason.message) }
    finally { setStarting(false) }
  }

  return (
    <div className="training-panel">
      <section className="training-section">
        <h2>Interactive training</h2>
        <p>Train a fresh {metrics?.architecture.name ?? 'selected model'} and watch its
          measurements after each epoch.</p>
        <div className="training-controls">
          <label>Epochs <input aria-label="Training epochs" type="number" min="1" max="200"
            step="1" value={epochs} onChange={(event) => setEpochs(event.target.value)} /></label>
          <label><input type="checkbox" checked={useDropout}
            onChange={(event) => setUseDropout(event.target.checked)} /> Use dropout</label>
          <label>Dropout rate <input aria-label="Dropout rate" type="number" min="0" max="1"
            step="any" value={dropoutRate} disabled={!useDropout}
            aria-describedby="dropout-help"
            onChange={(event) => setDropoutRate(event.target.value)} /></label>
          <button className="page-btn" type="button" onClick={onStart}
            disabled={active || starting || !available || !validEpochs || !validDropout}>
            {starting ? 'Starting…' : active ? 'Training in progress…' : 'Start practice training'}
          </button>
        </div>
        {!available && <p>The full local training data is required for this control.</p>}
        <p id="dropout-help" className="training-description">The rate is the fraction of
          hidden values randomly dropped during training: 0 drops none, 1 drops all.
          {validDropout ? ` Selected for the next run: ${dropoutNumber}
          (${(100 * dropoutNumber).toFixed(1)}% on average).` : ' Enter a rate from 0 to 1.'}</p>
        {metrics && <p className="training-description">
          {metrics.architecture.family === 'message_passing'
            ? '2 message-passing layers, 32 features per location'
            : `${metrics.architecture.hidden_widths.join(' → ')} hidden neurons`}
          {' '}· ReLU · SGD learning rate
          {' '}{metrics.training.learning_rate} · batch {metrics.training.batch_size}.
          Training uses replicates 1–2 and validation uses replicate 3.</p>}
        <p className="training-description">Predictions and the recorded test results continue
          to use the frozen evaluated model. Practice runs do not evaluate the final test set.</p>
        {error && <p role="alert">{error}</p>}
        {job && <div aria-live="polite" className="training-status">
          <p><strong>Practice run {job.id.slice(0, 8)}</strong>: {job.status} ·
            epoch {job.epoch} of {job.epochs} · {job.elapsed_seconds.toFixed(1)} seconds</p>
          <p>Run settings: {job.settings.epochs} epochs · dropout rate {job.settings.dropout}.</p>
          {job.error && <p role="alert">{job.error}</p>}
          {job.status === 'completed' && <p>Completed. This practice run saved its
            highest-validation-macro-F1 checkpoint from epoch {job.best_epoch}. Lower validation
            loss breaks exact F1 ties.</p>}
        </div>}
        {job?.history.length > 0 && <HistoryCharts Plot={Plot} job={job} dark={dark} />}
        <p className="training-description">Restarting the app clears the live run list.
          Each completed practice run is saved separately.</p>
      </section>
      {metrics && <FrozenMetrics metrics={metrics} />}
      <section className="training-section">
        <h2>Frozen model prediction</h2>
        {cellId ? <>
          <p>Selected cell {cellId}, replicate {replicate}.</p>
          <PredictionPanel key={`${replicate}:${cellId}`} cellId={cellId}
            replicate={replicate} loadSpots={loadSpots} />
        </> : <p>Open a cell in Viewer, then return here to request its prediction.</p>}
      </section>
    </div>
  )
}
