// The request contains measurements, with recorded labels and annotations excluded.
export function predictionSpots(spots) {
  return spots.map((spot) => ({
    trace_id: Number(spot.Trace_ID),
    chrom_start: Number(spot.Chrom_Start),
    x: Number(spot.X),
    y: Number(spot.Y),
    z: Number(spot.Z),
  }))
}

export async function predictSpots(spots) {
  const response = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ spots: predictionSpots(spots) }),
  })
  const body = await response.json()
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : 'Check the spot fields.'
    throw new Error(`Prediction failed (${response.status}): ${detail}`)
  }
  return body
}
