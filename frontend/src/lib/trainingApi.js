// Reads job snapshots through the same local API as prediction and annotations.
async function trainingJson(url, options = {}) {
  const response = await fetch(url, options)
  const body = await response.json()
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : 'Check the training settings.'
    throw new Error(detail)
  }
  return body
}

export function fetchTrainingMetrics(signal) {
  return trainingJson('/api/training/metrics', { signal })
}

export function fetchCurrentTraining(signal) {
  return trainingJson('/api/training/jobs', { signal })
}

export function fetchTrainingJob(id, signal) {
  return trainingJson(`/api/training/jobs/${encodeURIComponent(id)}`, { signal })
}

export function startTraining(epochs, dropout = 0) {
  return trainingJson('/api/training/jobs', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ epochs, dropout }),
  })
}
