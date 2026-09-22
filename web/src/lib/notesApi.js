// Talks to FastAPI through Vite's `/api` proxy (see vite.config.js).

async function sendJson(url, method, body) {
  const options = { method }
  if (body !== undefined) {
    options.headers = { 'Content-Type': 'application/json' }
    options.body = JSON.stringify(body)
  }
  const res = await fetch(url, options)
  if (!res.ok) {
    throw new Error(`${method} ${url} failed (${res.status})`)
  }
  return res.json()
}

export async function fetchHealth() {
  const res = await fetch('/api/health')
  return res.ok
}

export async function fetchNotebook() {
  const res = await fetch('/api/notebook')
  if (!res.ok) {
    throw new Error('Could not load notebook')
  }
  return res.json()
}

export function saveNotebook(text) {
  return sendJson('/api/notebook', 'PUT', { text })
}

export function deleteNotebook() {
  return sendJson('/api/notebook', 'DELETE')
}

export async function fetchCellNotes() {
  const res = await fetch('/api/cell-notes')
  if (!res.ok) {
    throw new Error('Could not load cell notes')
  }
  return res.json()
}

export async function fetchCellNote(replicate, cellId) {
  const res = await fetch(`/api/cells/${replicate}/${cellId}/note`)
  if (!res.ok) {
    return { text: '' }
  }
  return res.json()
}

export function saveCellNote(replicate, cellId, text) {
  return sendJson(`/api/cells/${replicate}/${cellId}/note`, 'PUT', { text })
}

export function deleteCellNote(replicate, cellId) {
  return sendJson(`/api/cells/${replicate}/${cellId}/note`, 'DELETE')
}

export function pinFromApi(body) {
  const row = body?.pin
  if (!row) {
    return null
  }
  return {
    spotId: row.spot_id,
    traceId: row.trace_id,
    chromStart: row.chrom_start,
    lamina: row.lamina,
    nucleolus: row.nucleolus,
  }
}

export function pinToApi(pin) {
  if (!pin) {
    return { pin: null }
  }
  return {
    pin: {
      spot_id: pin.spotId ?? null,
      trace_id: pin.traceId == null ? null : String(pin.traceId),
      chrom_start: pin.chromStart,
      lamina: pin.lamina,
      nucleolus: pin.nucleolus,
    },
  }
}

export async function fetchPin(replicate, cellId) {
  const res = await fetch(`/api/cells/${replicate}/${cellId}/pin`)
  if (!res.ok) {
    return null
  }
  return pinFromApi(await res.json())
}

export function savePin(replicate, cellId, pin) {
  return sendJson(
    `/api/cells/${replicate}/${cellId}/pin`,
    'PUT',
    pinToApi(pin),
  )
}

export async function fetchCellTraces(replicate, cellId) {
  const res = await fetch(`/api/cells/${replicate}/${cellId}/traces`)
  if (!res.ok) {
    return { traces: [] }
  }
  return res.json()
}

export function saveTraceLabel(replicate, cellId, traceId, text) {
  return sendJson(
    `/api/cells/${replicate}/${cellId}/traces/${encodeURIComponent(traceId)}`,
    'PUT',
    { text },
  )
}

export function deleteTraceLabel(replicate, cellId, traceId) {
  return sendJson(
    `/api/cells/${replicate}/${cellId}/traces/${encodeURIComponent(traceId)}`,
    'DELETE',
  )
}

export async function fetchTraceLabels() {
  const res = await fetch('/api/trace-labels')
  if (!res.ok) {
    throw new Error('Could not load trace labels')
  }
  return res.json()
}

export async function fetchCellTadNotes(replicate, cellId) {
  const res = await fetch(`/api/cells/${replicate}/${cellId}/tad-notes`)
  if (!res.ok) {
    return { tads: [] }
  }
  return res.json()
}

export function saveTadNote(
  replicate,
  cellId,
  chromStart,
  text,
  position,
  chromEnd,
) {
  return sendJson(
    `/api/cells/${replicate}/${cellId}/tad-notes/${chromStart}`,
    'PUT',
    { text, position, chrom_end: chromEnd },
  )
}

export function deleteTadNote(replicate, cellId, chromStart) {
  return sendJson(
    `/api/cells/${replicate}/${cellId}/tad-notes/${chromStart}`,
    'DELETE',
  )
}

export async function fetchTadNotes() {
  const res = await fetch('/api/tad-notes')
  if (!res.ok) {
    throw new Error('Could not load TAD notes')
  }
  return res.json()
}
