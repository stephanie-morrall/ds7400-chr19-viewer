import { useState } from 'react'
import { predictSpots } from './lib/predictionsApi'

function PredictionResult({ result }) {
  return (
    <div>
      <p><strong>Predicted type:</strong> {result.predicted_type}</p>
      <p>Model score: {(100 * result.model_score).toFixed(1)}% (uncalibrated).</p>
      <details className="prediction-detail">
        <summary>About this prediction</summary>
        <p>The score compares supported types; it is not a measured probability of correctness.</p>
        <p>Supported types: {result.class_names.join(', ')}.</p>
        <p>Model: {result.model_version}</p>
      </details>
    </div>
  )
}

export default function PredictionPanel({ cellId, replicate, loadSpots }) {
  const [state, setState] = useState({ status: 'idle' })

  async function onPredict() {
    setState({ status: 'loading' })
    try {
      // Loads all selected-cell spots, including traces hidden in the plot.
      const spots = await loadSpots(cellId, replicate)
      const result = await predictSpots(spots)
      setState({ status: 'ready', result })
    } catch (error) {
      setState({ status: 'error', message: error.message })
    }
  }

  return (
    <section className="prediction-panel" aria-label="Cell-type prediction">
      <button type="button" className="page-btn" onClick={onPredict}
        disabled={state.status === 'loading'}>
        {state.status === 'loading' ? 'Predicting…' : 'Predict cell type'}
      </button>
      <div aria-live="polite">
        {state.status === 'error' && <p role="alert">{state.message}</p>}
        {state.status === 'ready' && <PredictionResult result={state.result} />}
      </div>
    </section>
  )
}
