import { useEffect, useRef, useState } from 'react'
import { fetchTrainingMetrics } from './lib/trainingApi'
import hw3Overview from '../../docs/hw3/architecture/proposal-three-path-attention/overview.svg?url'
import hw3Detail from '../../docs/hw3/architecture/proposal-three-path-attention/detail.svg?url'
import hw2Diagram from '../../docs/hw3/architecture/hw2-architecture.svg?url'
import hw3Results from '../../docs/hw3/viewer-results.json'
import hw3PrimaryFigure from '../../docs/hw3/architecture/video-primary-comparison.png?url'
import hw3LossFigure from '../../docs/hw3/architecture/video-overfitting.png?url'
import hw3AllCurves from '../../research/hw3-three-path-2026-11-09/report/learning-curves-ce.png?url'
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

const architectureComparison = [
  { name: 'HW2: final residual MLP', count: '39,875 parameters', image: hw2Diagram,
    width: 640, detail: 'Average trace distances before learning.',
    alt: 'The original HW2 residual MLP: averaged distances, three hidden dense layers, a dropout branch, identity addition and three class scores.' },
  { name: 'HW3: three-path CNN + trace attention', count: '161,075 parameters · XYZ + distances + metadata',
    image: hw3Overview, width: 1600,
    detail: 'Encode ordered loci, exchange trace summaries, then join cell metadata.',
    alt: 'Three-path design: separate XYZ and distance residual CNNs, ordered trace readout, attention over all trace tokens, trace mean, metadata concatenation, dropout and classifier.' },
]

function ComparisonCards({ fullSize = false }) {
  return <div className={`hw3-comparison-grid${fullSize ? ' hw3-comparison-full' : ''}`}>
    {architectureComparison.map(({ name, count, image, detail, alt, width }) => (
      <figure key={name} className="hw3-comparison-card">
        <figcaption><strong>{name}</strong><span>{count}</span><p>{detail}</p></figcaption>
        <a href={image} target="_blank" rel="noreferrer" aria-label={`Zoom ${name}`}>
          <img src={image} alt={alt} style={fullSize ? { width } : undefined} />
        </a>
      </figure>
    ))}
  </div>
}

function HW3ArchitectureComparison() {
  const dialog = useRef(null)
  return <details id="hw3-architecture-comparison" className="hw3-comparison">
    <summary>Compare the final HW2 baseline and new HW3 architecture</summary>
    <div className="hw3-comparison-heading"><h4>Same cell prediction, different representations</h4>
      <button className="page-btn" type="button"
        onClick={() => dialog.current.showModal()}>Compare at full size</button></div>
    <ComparisonCards />
    <dialog ref={dialog} className="hw3-comparison-dialog"
      aria-labelledby="hw3-full-comparison-title">
      <div className="hw3-comparison-heading">
        <h3 id="hw3-full-comparison-title">HW2 and HW3 architectures</h3>
        <form method="dialog"><button className="page-btn">Close</button></form>
      </div>
      <ComparisonCards fullSize />
    </dialog>
  </details>
}

function HW3Experiments() {
  const verified = hw3Results.status === 'verified'
  const protocol = hw3Results.training
  return <>
    <section className="models-results hw3-experiment-section" id="hw3-experiment-settings">
      <h4>Test whether communication between traces helps</h4>
      <p>Compare the complete three-path model with the same model bypassing attention,
        using seeds 42, 43 and 44. The original saved HW2 model remains the assignment baseline.</p>
      <p>Replicates 1–2 supply 2,439 training cells. Replicate 3 supplies 1,347 validation
        cells. Coordinate rotation uses one rotation for all traces in a cell during training.
        Distances and metadata keep their original values.</p>
      {protocol && <p className="models-detail">SGD: momentum {protocol.momentum}, learning rate{' '}
        {protocol.learning_rate}, batch size {protocol.batch_size}. Maximum{' '}
        {protocol.maximum_epochs} epochs, patience {protocol.early_stopping_patience}
        {' '}validation checks. Select the best validation macro-F1 checkpoint.
        Validation uses original inputs with dropout disabled.</p>}
    </section>
    <section className="models-results hw3-experiment-section" id="hw3-experiment-results">
      <h4>{verified ? 'The redesign did not improve validation F1' : 'New three-path results are pending'}</h4>
      <p>{verified ? 'Six new fits were checked against their saved predictions and histories.'
        : 'The earlier CNN scores belong to a different architecture. They do not evaluate this redesign.'}</p>
      <div className="models-table-wrap"><table><thead><tr><th>Model</th>
        <th>Validation macro F1</th><th>Seconds per new fit</th></tr></thead>
        <tbody>{hw3Results.rows.map((row) => <tr key={row.condition}>
          <th>{row.name}</th><td>{row.macro_f1.toFixed(3)}
            {row.sample_sd === null ? ' · saved model' : ` ± ${row.sample_sd.toFixed(3)}`}</td>
          <td>{row.seconds_per_fit === null ? '—' : row.seconds_per_fit.toFixed(0)}</td>
        </tr>)}</tbody></table></div>
      <p className="models-detail">{hw3Results.uncertainty} Seed SD describes training
        variability on the same validation cells, rather than biological sampling uncertainty.
        The original HW2 training rules differ, so its comparison includes a package of changes.</p>
      {verified && <p>The attention ablation shares the new inputs and training rules.
        Mean attention-minus-bypass F1: {hw3Results.attention_minus_bypass.mean.toFixed(4)}.
        This is a descriptive comparison across three seeds. Each seed favored bypassing
        attention slightly; three seeds do not establish that attention generally harms performance.</p>}
      {verified && <><a href={hw3PrimaryFigure} target="_blank" rel="noreferrer">
        <img className="hw3-result-figure" src={hw3PrimaryFigure}
          alt="Original saved HW2 F1 0.479; three-path CNN without attention 0.442 plus or minus 0.005; with attention 0.439 plus or minus 0.005. New values summarize three seeds under different training rules from HW2." /></a>
        <details className="hw3-architecture-details"><summary>Inspect training and validation loss</summary>
          <a href={hw3LossFigure} target="_blank" rel="noreferrer">
            <img className="hw3-result-figure" src={hw3LossFigure}
              alt="Actual seed-42 training and validation cross-entropy with selected checkpoint epochs16 and18 marked; training loss falls while validation loss increases, followed by a late collapse in the bypassed model." /></a>
          <a href={hw3AllCurves} target="_blank" rel="noreferrer">Open all six training curves</a>
        </details></>}
    </section>
    <section className="models-results hw3-experiment-section" id="hw3-experiment-lessons">
      <h4>{verified ? 'The main failure was generalization' : 'Why these components belong together'}</h4>
      {verified && <><p>More capacity did not translate into better validation predictions.
        Training and validation loss diverged, and the best checkpoints occurred early.
        The run continued for 100 checks without a strict F1 gain, then restored the selected weights.</p>
        <p>One explanation is excess capacity for 2,439 training cells. Replicate differences,
          orientation-sensitive XYZ features, or metadata that reflects measurement quality could
          also contribute. These are hypotheses; this experiment does not isolate their effects.</p>
        <p>The next experiment would reduce encoder depth while holding XYZ, distances, metadata
          and optimizer settings fixed. Input-path ablations could follow once that depth comparison
          shows whether the extra convolutional blocks help.</p></>}
      <p>Ordered convolution learns local patterns along the chromosome. Repeated residual
        blocks allow deeper feature extraction while preserving an identity path. Separate
        XYZ and distance encoders retain complementary coordinate and rotation-invariant views.</p>
      <p>Each trace becomes one ordered summary. Attention updates all trace summaries before
        the valid-trace mean. Metadata enters afterward so cell size and observation quality
        can complement geometry without pretending to be genomic positions.</p>
      <p className="models-detail">Masks preserve which measurements exist; they do not recover
        unobserved DNA. Rotation encourages robustness to orientation. Dropout, a validation-selected
        checkpoint and monitored training–validation curves address overfitting, without guaranteeing
        generalization to another biological replicate.</p>
    </section>
    <details className="hw3-architecture-details" id="hw3-matched-reference">
      <summary>Additional HW2 reference with matched training settings</summary>
      <p>Three previously completed HW2 refits used the same optimizer, training budget,
        cells and selection rule as the new CNN panel. Their mean macro F1 was{' '}
        {hw3Results.matched_diagnostic.macro_f1_mean.toFixed(3)}. They failed to learn useful
        classification under that regime. Their input representation still differs,
        so this diagnostic does not isolate an architecture effect or establish an
        improvement over the original saved HW2 model.</p>
    </details>
  </>
}

function HW3Architecture() {
  return <section className="hw3-architecture" aria-labelledby="hw3-architecture-title">
    <div className="hw3-architecture-heading">
      <h3 id="hw3-architecture-title">HW3: XYZ + distances + metadata</h3>
      <span className="hw3-model-status">{hw3Results.status === 'verified'
        ? 'New fits complete · analysis checked' : 'Implemented · new results pending'}</span>
    </div>
    <p>Two repeated CNN encoders read the same named loci in genomic order. Attention lets
      whole-trace summaries exchange information. A small metadata path joins the cell geometry
      before the classifier.</p>
    <p className="models-detail">Four metadata inputs are cell area standardized using training
      cells, valid trace count divided by four, the fraction of observed loci among valid
      traces, and an area-missing flag. Their 16 learned features join 32 geometry features.</p>
    <figure className="hw3-architecture-figure">
      <a href={hw3Overview} target="_blank" rel="noreferrer"
        aria-label="Open the final HW3 architecture at full size">
        <img src={hw3Overview} alt="Three input paths: XYZ and distance residual CNN encoders form one vector per trace; attention updates all vectors before trace averaging; metadata is concatenated before dropout and a three-class classifier." />
      </a><figcaption>Open at full size to inspect shapes and skips; swipe sideways on narrow screens.</figcaption>
    </figure>
    <details className="hw3-architecture-details">
      <summary>Explain one residual block, attention, concatenation and dropout</summary>
      <a href={hw3Detail} target="_blank" rel="noreferrer">
        <img src={hw3Detail} alt="A residual block repeated four times per encoder; the meaning of a whole-trace vector; distinct examples of concatenation, addition and dropout." />
      </a>
    </details>
    <HW3ArchitectureComparison />
    <HW3Experiments />
  </section>
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
  if (error) return <><p role="alert">Could not load the saved models: {error}</p>
    <HW3Architecture /></>
  if (!metrics) return <><p role="status">Loading saved model details…</p>
    <HW3Architecture /></>

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
      <p className="models-inputs"><strong>Saved HW2 predictor inputs:</strong> 49 locations give
        1,176 pairwise distances in microns. Each distance is averaged across the usable
        traces within a cell. Cells with one to four usable traces are included.</p>
      <p>Unknown labels were excluded. Types with at least 300 quality-passing training cells
        were retained: Hepatocyte, Erythroblast, and Proerythroblast. Replicates 1–2 supply
        2,439 training cells, replicate 3 supplies 1,347 validation cells, and replicate 4
        supplies the test cells.</p>
      <div className="models-cards">
        <article className="models-card"><p className="models-kind">Earlier HW2 comparator</p>
          <h3>{baseline.name}</h3><Architecture label="Simple benchmark architecture" steps={[
            input, { title: 'Hidden layer', value: '32 neurons', detail: 'ReLU · no dropout' }, output,
          ]} /><p>ReLU sets negative hidden values to zero.</p></article>
        <article className="models-card models-card-network">
          <p className="models-kind">Final HW2 model · HW3 baseline · saved predictor</p>
          <h3>{architecture.name}</h3><Architecture label="Selected model architecture" steps={[
            input, { title: 'Hidden layers', value: '32 → 32 → 32', detail: 'ReLU · one residual branch' }, output,
          ]} /><p>The first layer maps distances to 32 values. A two-layer residual branch
            transforms those values and adds its input back before the final ReLU. Dropout
            0.8 acts after the first activation in the branch during training and is off for predictions.</p>
        </article>
      </div>
      <HW3Architecture />
      <section className="models-results"><h3>HW2 validation exploration</h3>
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
        <h3 id="models-results-title">Saved HW2 seed 42 model · test results ({splits.test.n_cells.toLocaleString()} cells)</h3>
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
