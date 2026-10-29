import { useEffect, useState } from 'react'
import { fetchTrainingMetrics } from './lib/trainingApi'
import './Models.css'

const approaches = [
  ['One-hidden-layer benchmark', '0.4774'], ['Residual MLP, ReLU, dropout 0.8', '0.4868'],
  ['Residual MLP, Leaky ReLU', '0.4842'], ['Residual MLP, AdamW', '0.4868'],
  ['Residual MLP, momentum SGD', '0.4865'], ['Class-weighted shallow MLP', '0.4718'],
  ['Class-weighted residual MLP', '0.4798'], ['Mean distances plus trace variation', '0.4842'],
  ['Ordered-distance CNN', '0.4796'], ['Genomic-neighbor message passing', '0.4555'],
  ['Aligned coordinate model', '0.4193'],
]

function Architecture({ label, steps }) {
  return <ol className="models-flow" aria-label={label}>{steps.map((step) => (
    <li key={step.title}><span>{step.title}</span><strong>{step.value}</strong>
      <small>{step.detail}</small></li>
  ))}</ol>
}

export default function Models({ onTrain }) {
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    fetchTrainingMetrics(controller.signal).then(setMetrics)
      .catch((reason) => { if (reason.name !== 'AbortError') setError(reason.message) })
    return () => controller.abort()
  }, [])
  if (error) return <p role="alert">Could not load the saved models: {error}</p>
  if (!metrics) return <p role="status">Loading saved model details…</p>

  const { architecture, splits, baseline } = metrics
  const input = { title: 'Input', value: architecture.inputs.toLocaleString(), detail: 'mean distances per cell' }
  const output = { title: 'Output', value: `${architecture.outputs} scores`, detail: 'one per cell type' }
  const compared = [
    { name: baseline.name, metric: baseline.test },
    { name: architecture.name, metric: splits.test },
  ]
  return (
    <section className="models-page" aria-labelledby="models-title">
      <div className="models-heading"><div><h2 id="models-title">Models</h2>
        <p>Predicting cell type from chromosome 19 distances.</p></div>
        <button className="page-btn" type="button" onClick={onTrain}>Try training</button></div>
      <p className="models-inputs"><strong>Shared measurements:</strong> 49 locations give
        1,176 pairwise distances in microns. Each distance is averaged across the usable
        traces within a cell. Cells with one to four usable traces are included.</p>
      <p>Unknown labels were excluded. Types with at least 300 quality-passing training cells
        were retained: Hepatocyte, Erythroblast, and Proerythroblast. Replicates 1–2 supply
        2,439 training cells, replicate 3 supplies 1,347 validation cells, and replicate 4
        supplies the test cells.</p>
      <div className="models-cards">
        <article className="models-card"><p className="models-kind">Simple benchmark</p>
          <h3>{baseline.name}</h3><Architecture label="Simple benchmark architecture" steps={[
            input, { title: 'Hidden layer', value: '32 neurons', detail: 'ReLU · no dropout' }, output,
          ]} /><p>ReLU sets negative hidden values to zero.</p></article>
        <article className="models-card models-card-network">
          <p className="models-kind">Selected architecture · saved predictor</p>
          <h3>{architecture.name}</h3><Architecture label="Selected model architecture" steps={[
            input, { title: 'Hidden layers', value: '32 → 32 → 32', detail: 'ReLU · one residual branch' }, output,
          ]} /><p>The first layer maps distances to 32 values. A two-layer residual branch
            transforms those values and adds its input back before the final ReLU. Dropout
            0.8 acts after the first activation in the branch during training and is off for predictions.</p>
        </article>
      </div>
      <section className="models-results"><h3>Validation exploration</h3>
        <p>Across three training seeds, the selected residual model averaged macro F1 0.4868,
          compared with 0.4774 for the fixed benchmark. It ranked first among 44 tested
          configurations. The exploratory paired test gave p = 0.1173, so the difference was
          not statistically significant at 0.05. The same validation cells were used for model
          selection, and this calculation does not account for selection or biological sampling uncertainty.</p>
        <details><summary>View tested approaches</summary>
          <p className="models-detail">Each row is the best tested setting in that group,
            averaged over three seeds. These are validation results, separate from the saved
            seed 42 model and its test results below.</p>
          <div className="models-table-wrap"><table><thead><tr><th>Approach</th>
            <th>Mean validation macro F1</th></tr></thead><tbody>{approaches.map(([name, f1]) => (
              <tr key={name}><th>{name}</th><td>{f1}</td></tr>
            ))}</tbody></table></div>
          <p className="models-detail">The residual model had the highest tested mean.
            The AdamW setting was nearly tied. More complex tested approaches did not improve
            the mean, supporting the selected model as a compact choice for this comparison.</p>
        </details>
      </section>
      <section className="models-results" aria-labelledby="models-results-title">
        <h3 id="models-results-title">Saved seed 42 model · test results ({splits.test.n_cells.toLocaleString()} cells)</h3>
        <div className="models-table-wrap"><table><thead><tr><th>Model</th><th>Macro F1</th>
          <th>Weighted F1</th><th>Accuracy</th><th>Errors</th><th>Error rate</th></tr></thead>
          <tbody>{compared.map(({ name, metric }) => <tr key={name}><th>{name}</th>
            <td>{metric.macro_f1.toFixed(3)}</td><td>{metric.weighted_f1.toFixed(3)}</td>
            <td>{(100 * metric.accuracy).toFixed(2)}%</td><td>{metric.n_errors}</td>
            <td>{(100 * metric.error).toFixed(2)}%</td></tr>)}</tbody></table></div>
        <p className="models-detail">{metrics.evaluation_note}</p>
      </section>
      <p className="models-practice"><strong>Practice training:</strong> train a fresh copy
        with editable epochs and dropout. Predictions continue to use the saved model.</p>
    </section>
  )
}
