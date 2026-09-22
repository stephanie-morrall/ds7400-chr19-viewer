import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import createPlotlyComponent from 'react-plotly.js/factory'
import Plotly from 'plotly.js-dist-min'
import About from './About'
import Gallery from './Gallery'
import Notebook from './Notebook'
import {
  deleteCellNote,
  deleteTadNote,
  deleteTraceLabel,
  fetchCellNote,
  fetchCellTadNotes,
  fetchCellTraces,
  fetchHealth,
  fetchPin,
  saveCellNote,
  savePin,
  saveTadNote,
  saveTraceLabel,
} from './lib/notesApi'
import { envelopeTraces, nucleolusTrace } from './lib/nuclearShapes'
import {
  TRACE_COLORS,
  buildCellView,
  chromStartsFromFibers,
  locusFromPlotPoint,
  parseSpotCsv,
  previewSpanUmFromCells,
  spotTraces,
  tadRank,
} from './lib/spots'
import './App.css'

// Builds a React <Plot> that talks to the Plotly drawing library.
const Plot = createPlotlyComponent(Plotly)

const CATALOG_URL = '/catalog.json'
const LEGACY_CATALOG_URL = '/biorep01/catalog.json'
const CELL_URL = (replicate, cellId) => (
  `/biorep${String(replicate).padStart(2, '0')}/cells/${cellId}.json`
)
const SPOTS_URL = '/sample/cells.csv'
const SHAPES_URL = '/sample/shapes.json'

// Returns catalog JSON, or null when the file is missing.
// Vite answers a missing /catalog.json with index.html and status 200.
async function catalogJson(res) {
  if (!res.ok) return null
  const type = res.headers.get('content-type') || ''
  if (!type.includes('application/json')) return null
  return res.json()
}

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  displayModeBar: false,
  doubleClick: false,
}
const DBLCLICK_MS = 400
const SAME_CLICK_MS = 80

function locusHeading(start, end) {
  const a = Number(start).toLocaleString()
  const b = Number(end).toLocaleString()
  return `chr19:${a}–${b}`
}

// Types for the committed three-cell sample, used when the biorep01 catalog is absent.
const SAMPLE_CELL_TYPES = {
  411: { name: 'Proerythroblast', code: '7' },
  328: { name: 'Macrophage', code: '4' },
  530: { name: 'Unknown', code: '8' },
}

function catalogFromSampleRows(rows) {
  const grouped = new Map()
  for (const [cellId, meta] of Object.entries(SAMPLE_CELL_TYPES)) {
    if (!grouped.has(meta.name)) {
      grouped.set(meta.name, { name: meta.name, code: meta.code, cells: [] })
    }
    const spots = rows.filter((row) => row.Cell_ID === cellId)
    const traces = new Set(spots.map((spot) => spot.Trace_ID))
    grouped.get(meta.name).cells.push({
      id: cellId,
      nSpots: spots.length,
      nTraces: traces.size,
      replicate: 1,
    })
  }
  const types = [...grouped.values()]
  const cellCount = types.reduce((n, entry) => n + entry.cells.length, 0)
  const lists = Object.keys(SAMPLE_CELL_TYPES).map((cellId) => (
    rows.filter((row) => row.Cell_ID === cellId)
  ))
  return {
    cellCount,
    replicates: [1],
    types,
    previewSpanUm: previewSpanUmFromCells(lists),
  }
}

function annotateCatalog(catalog) {
  return {
    ...catalog,
    replicates: catalog.replicates ?? [1],
    types: catalog.types.map((entry) => ({
      ...entry,
      cells: entry.cells.map((cell) => ({
        ...cell,
        replicate: cell.replicate ?? 1,
      })),
    })),
  }
}

function cellRecord(catalog, cellId, replicate) {
  for (const entry of catalog.types) {
    const cell = entry.cells.find((row) => {
      if (String(row.id) !== String(cellId)) {
        return false
      }
      if (replicate == null) {
        return true
      }
      return Number(row.replicate ?? 1) === Number(replicate)
    })
    if (cell) {
      return { ...cell, typeName: entry.name }
    }
  }
  return null
}

function App() {
  const [tab, setTab] = useState('viewer')
  const [catalog, setCatalog] = useState(null)
  const [shapes, setShapes] = useState({})
  const [source, setSource] = useState('loading')
  const [sampleRows, setSampleRows] = useState(null)
  const [typeName, setTypeName] = useState('all')
  const [page, setPage] = useState(0)
  const [traceCount, setTraceCount] = useState('all')
  const [replicate, setReplicate] = useState('all')
  const [cellId, setCellId] = useState('')
  const [cellReplicate, setCellReplicate] = useState(null)
  const [view, setView] = useState(null)
  const [status, setStatus] = useState('loading')
  const [showEnvelope, setShowEnvelope] = useState(true)
  const [showNucleolus, setShowNucleolus] = useState(true)
  const [hiddenTraceIds, setHiddenTraceIds] = useState([])
  const [, setPin] = useState(null)
  const [cellNote, setCellNote] = useState('')
  const [noteStatus, setNoteStatus] = useState('')
  const [traceLabels, setTraceLabels] = useState({})
  const [traceSaveStatus, setTraceSaveStatus] = useState({})
  const [openTraceId, setOpenTraceId] = useState(null)
  const [openTad, setOpenTad] = useState(null)
  const [tadNotes, setTadNotes] = useState({})
  const [tadSaveStatus, setTadSaveStatus] = useState('')
  const [apiOk, setApiOk] = useState(null)
  // Holds parsed spots or an in-flight Promise, keyed by replicate:cellId.
  const cacheRef = useRef(new Map())
  const abortMapRef = useRef(new Map())
  const lastHoverRef = useRef(null)
  const lastClickRef = useRef({ key: null, at: 0 })
  const viewRef = useRef(null)
  const handlePlotSelectRef = useRef(null)
  const openCellKeyRef = useRef(null)
  const plotGdRef = useRef(null)
  const dark = useMemo(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches,
    [],
  )

  // Loads the four-replicate catalog when present, otherwise the three-cell sample.
  useEffect(() => {
    fetch(SHAPES_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`Could not load ${SHAPES_URL}`)
        return res.json()
      })
      .then((shapeTable) => {
        setShapes(shapeTable)
        return fetch(CATALOG_URL)
      })
      .then(async (res) => {
        let catalogBody = await catalogJson(res)
        if (!catalogBody) {
          catalogBody = await catalogJson(await fetch(LEGACY_CATALOG_URL))
        }
        if (catalogBody) {
          const next = annotateCatalog(catalogBody)
          setCatalog(next)
          setTypeName('all')
          setReplicate('all')
          setSource('catalog')
          setStatus('ready')
          return
        }
        const csvRes = await fetch(SPOTS_URL)
        if (!csvRes.ok) throw new Error(`Could not load ${SPOTS_URL}`)
        const rows = parseSpotCsv(await csvRes.text())
        const next = annotateCatalog(catalogFromSampleRows(rows))
        setSampleRows(rows)
        setCatalog(next)
        setTypeName('all')
        setReplicate('all')
        setSource('sample')
        setStatus('ready')
      })
      .catch((err) => {
        console.error(err)
        setStatus('error')
      })
  }, [])

  useEffect(() => {
    fetchHealth()
      .then(setApiOk)
      .catch(() => setApiOk(false))
  }, [])

  const loadSpots = useCallback(async (id, replicateHint, urgent = false) => {
    const record = catalog ? cellRecord(catalog, id, replicateHint) : null
    const replicate = record?.replicate ?? replicateHint ?? 1
    const cacheKey = `${replicate}:${id}`
    if (urgent) {
      for (const [key, controller] of abortMapRef.current) {
        if (key === cacheKey) {
          continue
        }
        controller.abort()
        abortMapRef.current.delete(key)
        const entry = cacheRef.current.get(key)
        if (entry && !Array.isArray(entry)) {
          cacheRef.current.delete(key)
        }
      }
    }
    const cached = cacheRef.current.get(cacheKey)
    // Reuses a download already in flight so a tile click does not fetch twice.
    if (Array.isArray(cached)) {
      return cached
    }
    if (cached) {
      return cached
    }
    const controller = new AbortController()
    abortMapRef.current.set(cacheKey, controller)
    const pending = (async () => {
      let spots
      if (source === 'sample') {
        spots = (sampleRows ?? []).filter((row) => row.Cell_ID === id)
      } else {
        const res = await fetch(CELL_URL(replicate, id), {
          signal: controller.signal,
          priority: urgent ? 'high' : 'low',
        })
        if (!res.ok) {
          throw new Error(`Could not load ${CELL_URL(replicate, id)}`)
        }
        const body = await res.json()
        spots = body.spots
      }
      cacheRef.current.set(cacheKey, spots)
      abortMapRef.current.delete(cacheKey)
      return spots
    })()
    cacheRef.current.set(cacheKey, pending)
    return pending.catch((err) => {
      abortMapRef.current.delete(cacheKey)
      if (cacheRef.current.get(cacheKey) === pending) {
        cacheRef.current.delete(cacheKey)
      }
      throw err
    })
  }, [source, sampleRows, catalog])

  // Opens the cell from cache when the gallery tile already fetched spots,
  // then fills the note, pin, and trace labels from SQLite. Runs again when
  // returning to Viewer so a Notebook edit is visible on the cell page.
  // Keeps the Plotly view on Back so WebGL does not remount on the next open.
  useEffect(() => {
    if (tab !== 'viewer' || !cellId || source === 'loading') {
      return
    }
    const record = catalog ? cellRecord(catalog, cellId, cellReplicate) : null
    const replicate = record?.replicate ?? cellReplicate ?? 1
    const cacheKey = `${replicate}:${cellId}`
    const cached = cacheRef.current.get(cacheKey)
    if (Array.isArray(cached)) {
      setView(buildCellView(cached, shapes, cellId))
      setHiddenTraceIds([])
    }
    const cellChanged = openCellKeyRef.current !== cacheKey
    if (cellChanged) {
      openCellKeyRef.current = cacheKey
      setPin(null)
      setCellNote('')
      setNoteStatus('')
      setTraceLabels({})
      setTraceSaveStatus({})
      setOpenTraceId(null)
      setOpenTad(null)
      setTadNotes({})
      setTadSaveStatus('')
      lastHoverRef.current = null
      lastClickRef.current = { key: null, at: 0 }
    }
    let cancelled = false
    // Draws the 3D cell as soon as spots are in memory; notes fill in after.
    loadSpots(cellId, replicate, true)
      .then((spots) => {
        if (cancelled) {
          return
        }
        setView(buildCellView(spots, shapes, cellId))
        setHiddenTraceIds([])
      })
      .catch((err) => {
        console.error(err)
        if (!cancelled) setStatus('error')
      })
    Promise.all([
      fetchCellNote(replicate, cellId).catch(() => ({ text: '' })),
      fetchPin(replicate, cellId).catch(() => null),
      fetchCellTraces(replicate, cellId).catch(() => ({ traces: [] })),
      fetchCellTadNotes(replicate, cellId).catch(() => ({ tads: [] })),
    ])
      .then(([noteBody, savedPin, tracesBody, tadsBody]) => {
        if (cancelled) {
          return
        }
        setCellNote(noteBody?.text ?? '')
        setPin(savedPin)
        const labels = {}
        for (const row of tracesBody?.traces ?? []) {
          labels[String(row.trace_id)] = row.text ?? ''
        }
        setTraceLabels(labels)
        const notes = {}
        for (const row of tadsBody?.tads ?? []) {
          notes[String(row.chrom_start)] = {
            text: row.text ?? '',
            position: row.position,
            chromEnd: row.chrom_end,
          }
        }
        setTadNotes(notes)
      })
      .catch((err) => {
        console.error(err)
      })
    return () => {
      cancelled = true
    }
  }, [cellId, cellReplicate, loadSpots, shapes, source, catalog, tab])

  viewRef.current = view

  useEffect(() => {
    if (tab !== 'viewer' || !cellId || !plotGdRef.current) {
      return
    }
    const gd = plotGdRef.current
    const frame = requestAnimationFrame(() => Plotly.Plots.resize(gd))
    return () => cancelAnimationFrame(frame)
  }, [tab, cellId])

  const axisColor = dark ? '#2e303a' : '#e5e4e7'
  const bg = dark ? '#16171d' : '#fff'

  const visibleFibers = useMemo(() => {
    if (!view) {
      return []
    }
    return view.fibers.filter((fiber) => !hiddenTraceIds.includes(fiber.traceId))
  }, [view, hiddenTraceIds])

  // Assembles the traces for the current cell under the surface and fiber toggles.
  // A hidden marker keeps the WebGL context alive on the gallery so the first
  // cell does not wait for a new 3D context.
  const data = useMemo(() => {
    if (!view) {
      return [{
        type: 'scatter3d',
        mode: 'markers',
        x: [0],
        y: [0],
        z: [0],
        marker: { size: 1, opacity: 0 },
        hoverinfo: 'skip',
        showlegend: false,
      }]
    }
    const traces = spotTraces(visibleFibers)
    if (showEnvelope && view.envelope) {
      traces.unshift(...envelopeTraces(view.envelope))
    }
    if (showNucleolus && view.nucleolus) {
      traces.unshift(nucleolusTrace(view.nucleolus))
    }
    return traces
  }, [view, visibleFibers, showEnvelope, showNucleolus])

  // Layout stays a cube of equal µm on X, Y, and Z; only the cube center moves.
  const layout = useMemo(() => {
    const range = view?.axisRange
    const axis = (title, bounds) => ({
      title: { text: title },
      range: bounds,
      backgroundcolor: bg,
      gridcolor: axisColor,
    })
    return {
      autosize: true,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { color: dark ? '#f3f4f6' : '#08060d' },
      showlegend: false,
      clickmode: 'event',
      hovermode: 'closest',
      hoverlabel: { font: { size: 13 } },
      uirevision: 'xyz-cube',
      scene: {
        aspectmode: 'cube',
        hovermode: 'closest',
        bgcolor: 'rgba(0,0,0,0)',
        camera: {
          eye: { x: 1.15, y: 1.15, z: 1.22 },
          center: { x: 0, y: 0, z: -0.06 },
        },
        xaxis: axis('X (µm)', range?.x ?? [0, 1]),
        yaxis: axis('Y (µm)', range?.y ?? [0, 1]),
        zaxis: axis('Z (µm)', range?.z ?? [0, 1]),
      },
    }
  }, [view, dark, bg, axisColor])

  const selectedCell = catalog && cellId
    ? cellRecord(catalog, cellId, cellReplicate)
    : null
  const cellTypeLabel = selectedCell?.typeName

  const hasShape = Boolean(view?.envelope)

  const applyPin = useCallback((spot) => {
    if (!spot || !cellId) {
      return
    }
    const next = {
      lamina: spot.lamina,
      nucleolus: spot.nucleolus,
      chromStart: spot.chromStart,
      traceId: spot.traceId,
      spotId: spot.spotId,
    }
    setPin(next)
    const replicate = selectedCell?.replicate ?? 1
    savePin(replicate, cellId, next).catch((err) => {
      console.error(err)
      setNoteStatus('error')
    })
  }, [cellId, selectedCell])

  const openTadNote = useCallback((spot) => {
    const fibers = viewRef.current?.fibers ?? []
    const starts = chromStartsFromFibers(fibers)
    const position = tadRank(starts, spot.chromStart) ?? 1
    setOpenTad({
      chromStart: Number(spot.chromStart),
      chromEnd: Number(spot.chromEnd),
      position,
    })
    setTadSaveStatus('')
  }, [])

  const handlePlotSelect = useCallback((pt) => {
    const spot = locusFromPlotPoint(pt)
    if (!spot || !cellId) {
      return
    }
    const key = `${spot.traceId}:${spot.spotId}:${spot.chromStart}`
    const now = Date.now()
    const prev = lastClickRef.current
    // Ignores the extra Plotly onClick/mouseup pair from one mouse press.
    if (prev.key === key && now - prev.at < SAME_CLICK_MS) {
      return
    }
    // Treats a second click on the same locus within 400 ms as a TAD note.
    if (prev.key === key && now - prev.at < DBLCLICK_MS) {
      lastClickRef.current = { key, at: now }
      openTadNote(spot)
      return
    }
    lastClickRef.current = { key, at: now }
    applyPin(spot)
  }, [applyPin, cellId, openTadNote])

  handlePlotSelectRef.current = handlePlotSelect

  const onSaveCellNote = useCallback(async () => {
    if (!cellId) {
      return
    }
    setNoteStatus('saving')
    try {
      const replicate = selectedCell?.replicate ?? 1
      await saveCellNote(replicate, cellId, cellNote)
      setNoteStatus('saved')
    } catch (err) {
      console.error(err)
      setNoteStatus('error')
    }
  }, [cellId, cellNote, selectedCell])

  const onSaveTraceLabel = useCallback(async (traceId) => {
    if (!cellId) {
      return
    }
    const key = String(traceId)
    setTraceSaveStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      const replicate = selectedCell?.replicate ?? 1
      await saveTraceLabel(replicate, cellId, key, traceLabels[key] ?? '')
      setTraceSaveStatus((current) => ({ ...current, [key]: 'saved' }))
    } catch (err) {
      console.error(err)
      setTraceSaveStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }, [cellId, selectedCell, traceLabels])

  const onSaveTadNote = useCallback(async () => {
    if (!cellId || !openTad) {
      return
    }
    setTadSaveStatus('saving')
    try {
      const replicate = selectedCell?.replicate ?? 1
      const key = String(openTad.chromStart)
      const text = tadNotes[key]?.text ?? ''
      await saveTadNote(
        replicate,
        cellId,
        openTad.chromStart,
        text,
        openTad.position,
        openTad.chromEnd,
      )
      setTadSaveStatus('saved')
    } catch (err) {
      console.error(err)
      setTadSaveStatus('error')
    }
  }, [cellId, openTad, selectedCell, tadNotes])

  const onDeleteCellNote = useCallback(async () => {
    if (!cellId) {
      return
    }
    setNoteStatus('saving')
    try {
      const replicate = selectedCell?.replicate ?? 1
      await deleteCellNote(replicate, cellId)
      setCellNote('')
      setNoteStatus('saved')
    } catch (err) {
      console.error(err)
      setNoteStatus('error')
    }
  }, [cellId, selectedCell])

  const onDeleteTraceLabel = useCallback(async (traceId) => {
    if (!cellId) {
      return
    }
    const key = String(traceId)
    setTraceSaveStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      const replicate = selectedCell?.replicate ?? 1
      await deleteTraceLabel(replicate, cellId, key)
      setTraceLabels((current) => {
        const next = { ...current }
        delete next[key]
        return next
      })
      setOpenTraceId(null)
    } catch (err) {
      console.error(err)
      setTraceSaveStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }, [cellId, selectedCell])

  const onDeleteTadNote = useCallback(async () => {
    if (!cellId || !openTad) {
      return
    }
    setTadSaveStatus('saving')
    try {
      const replicate = selectedCell?.replicate ?? 1
      await deleteTadNote(replicate, cellId, openTad.chromStart)
      const key = String(openTad.chromStart)
      setTadNotes((current) => {
        const next = { ...current }
        delete next[key]
        return next
      })
      setOpenTad(null)
      setTadSaveStatus('')
    } catch (err) {
      console.error(err)
      setTadSaveStatus('error')
    }
  }, [cellId, openTad, selectedCell])

  // Applies cached spots in the same click as setCellId so the first paint
  // is the 3D view rather than "Loading cell".
  const openCell = useCallback((id, replicateHint) => {
    const rec = catalog ? cellRecord(catalog, id, replicateHint) : null
    const r = rec?.replicate ?? replicateHint ?? 1
    const cached = cacheRef.current.get(`${r}:${id}`)
    loadSpots(id, r, true)
    if (Array.isArray(cached)) {
      setView(buildCellView(cached, shapes, id))
      setHiddenTraceIds([])
    }
    setCellReplicate(r)
    setCellId(String(id))
    setTab('viewer')
  }, [catalog, shapes, loadSpots])

  return (
    <main
      className={
        tab === 'viewer' && cellId ? 'viewer viewer-cell' : 'viewer'
      }
    >
      <header className="viewer-header">
        <h1>FOF-CT Chr19 Viewer</h1>
        <p>
          Chromosome tracing spots on mouse chromosome 19, one cell at a time. Each
          spot is an imaged locus with X, Y, and Z in microns, joined in genomic
          order along each traced fiber.
        </p>
        <nav className="viewer-tabs">
          <button
            type="button"
            className={tab === 'viewer' ? 'active' : ''}
            onClick={() => setTab('viewer')}
          >
            Viewer
          </button>
          <button
            type="button"
            className={tab === 'about' ? 'active' : ''}
            onClick={() => setTab('about')}
          >
            About
          </button>
          <button
            type="button"
            className={tab === 'notebook' ? 'active' : ''}
            onClick={() => setTab('notebook')}
          >
            Notebook
          </button>
        </nav>
      </header>

      {apiOk === false && (
        <p className="api-banner">
          Annotation API is not running. In the repository root run
          {' '}
          <code>docker compose up --build</code>
          , then refresh.
        </p>
      )}

      {tab === 'about' && <About />}
      {tab === 'notebook' && (
        <Notebook onOpenCell={openCell} loadSpots={loadSpots} />
      )}

      {tab === 'viewer' && (
        <>
          {status === 'loading' && <p>Loading catalog…</p>}
          {status === 'error' && (
            <p>Could not load the catalog or sample spots.</p>
          )}

          {status === 'ready' && catalog && !cellId && (
            <Gallery
              types={catalog.types}
              typeName={typeName}
              replicate={replicate}
              page={page}
              traceCount={traceCount}
              loadSpots={loadSpots}
              onTypeChange={(name) => {
                setTypeName(name)
                setPage(0)
              }}
              onReplicateChange={(value) => {
                setReplicate(value)
                setPage(0)
              }}
              onTraceCountChange={(value) => {
                setTraceCount(value)
                setPage(0)
              }}
              onPageChange={setPage}
              onOpenCell={openCell}
            />
          )}

          {status === 'ready' && cellId && !view && (
            <p>Loading cell {cellId}…</p>
          )}

          {status === 'ready' && view && cellId && (
            <>
              <div className="viewer-toolbar">
                <button
                  type="button"
                  className="page-btn"
                  onClick={() => setCellId('')}
                >
                  Back
                </button>
                {/* Later homework: Classify control goes here, next to Back. */}
                <dl className="cell-meta">
                  <div>
                    <dt>Cell</dt>
                    <dd>{cellId}</dd>
                  </div>
                  <div>
                    <dt>Replicate</dt>
                    <dd>{selectedCell?.replicate ?? '—'}</dd>
                  </div>
                  <div>
                    <dt>Type</dt>
                    <dd>{cellTypeLabel ?? '—'}</dd>
                  </div>
                  <div>
                    <dt>Spots</dt>
                    <dd>{view.count}</dd>
                  </div>
                  {view.envelope?.atBound ? (
                    <div className="cell-meta-note">
                      <dt>Envelope</dt>
                      <dd>
                        size undetermined, chr19 hugs a flat patch of wall
                      </dd>
                    </div>
                  ) : null}
                  {view.nucleolus || !hasShape ? null : (
                    <div className="cell-meta-note">
                      <dt>Nucleolus</dt>
                      <dd>
                        hidden, nearest spot{' '}
                        {view.nucleolusNote?.closestSpot}
                        {' '}µm away
                      </dd>
                    </div>
                  )}
                </dl>

                {hasShape && (
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={showEnvelope}
                      onChange={(event) => setShowEnvelope(event.target.checked)}
                    />
                    Nuclear envelope
                  </label>
                )}

                {hasShape && (
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={showNucleolus}
                      onChange={(event) => setShowNucleolus(event.target.checked)}
                      disabled={!view.nucleolus}
                    />
                    Nucleolus
                  </label>
                )}
              </div>

              <div className="trace-key">
                {view.fibers.map((fiber, index) => {
                  const color = TRACE_COLORS[index % TRACE_COLORS.length]
                  return (
                    <div key={fiber.traceId} className="trace-key-row">
                      <label className="toggle">
                        <input
                          type="checkbox"
                          checked={!hiddenTraceIds.includes(fiber.traceId)}
                          onChange={(event) => {
                            const fiberId = fiber.traceId
                            setHiddenTraceIds((current) => (
                              event.target.checked
                                ? current.filter((value) => value !== fiberId)
                                : [...current, fiberId]
                            ))
                          }}
                        />
                        <span
                          className="trace-swatch"
                          style={{ background: `rgb(${color})` }}
                        />
                      </label>
                      <button
                        type="button"
                        className="trace-key-name"
                        onClick={() => {
                          setOpenTraceId(String(fiber.traceId))
                          setTraceSaveStatus((current) => {
                            const next = { ...current }
                            delete next[String(fiber.traceId)]
                            return next
                          })
                        }}
                      >
                        Trace {fiber.traceId}
                      </button>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </>
      )}

      {/* Notes sit to the left of the cube; the plot fills the leftover row.
          The Plot stays mounted on the gallery (parked) so WebGL is already warm. */}
      {status === 'ready' && (
        <div
          className={
            tab === 'viewer' && cellId
              ? 'viewer-stage'
              : 'viewer-stage viewer-stage-parked'
          }
          aria-hidden={!(tab === 'viewer' && cellId)}
          inert={!(tab === 'viewer' && cellId)}
        >
          {tab === 'viewer' && cellId && (
            <aside className="viewer-notes">
              <div className="cell-note">
                <label htmlFor="cell-note-text">Cell note</label>
                <textarea
                  id="cell-note-text"
                  className="note-box"
                  rows={5}
                  value={cellNote}
                  onChange={(event) => {
                    setCellNote(event.target.value)
                    if (noteStatus === 'saved') {
                      setNoteStatus('')
                    }
                  }}
                />
                <div className="note-actions">
                  <button
                    type="button"
                    className="page-btn"
                    onClick={onSaveCellNote}
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    className="page-btn"
                    onClick={onDeleteCellNote}
                  >
                    Delete
                  </button>
                  {noteStatus === 'saved' && (
                    <span className="note-status">Saved.</span>
                  )}
                  {noteStatus === 'saving' && (
                    <span className="note-status">Saving…</span>
                  )}
                  {noteStatus === 'error' && (
                    <span className="note-status">
                      Could not save. Is Docker Compose up?
                    </span>
                  )}
                </div>
              </div>

              {openTraceId != null && (
                <div className="cell-note nested-note nested-note-trace">
                  <label htmlFor="trace-note-text">
                    Trace {openTraceId}
                  </label>
                  <textarea
                    id="trace-note-text"
                    className="note-box"
                    rows={4}
                    value={traceLabels[openTraceId] ?? ''}
                    onChange={(event) => {
                      const nextText = event.target.value
                      const id = openTraceId
                      setTraceLabels((current) => ({
                        ...current,
                        [id]: nextText,
                      }))
                      setTraceSaveStatus((current) => {
                        if (current[id] !== 'saved') {
                          return current
                        }
                        const next = { ...current }
                        delete next[id]
                        return next
                      })
                    }}
                  />
                  <div className="note-actions">
                    <button
                      type="button"
                      className="page-btn"
                      onClick={() => onSaveTraceLabel(openTraceId)}
                    >
                      Save
                    </button>
                    <button
                      type="button"
                      className="page-btn"
                      onClick={() => onDeleteTraceLabel(openTraceId)}
                    >
                      Delete
                    </button>
                    {traceSaveStatus[openTraceId] === 'saved' && (
                      <span className="note-status">Saved.</span>
                    )}
                    {traceSaveStatus[openTraceId] === 'saving' && (
                      <span className="note-status">Saving…</span>
                    )}
                    {traceSaveStatus[openTraceId] === 'error' && (
                      <span className="note-status">
                        Could not save. Is Docker Compose up?
                      </span>
                    )}
                  </div>
                </div>
              )}

              {openTad && (
                <div className="cell-note nested-note nested-note-tad">
                  <p className="tad-note-title">
                    TAD {openTad.position}
                  </p>
                  <p className="tad-note-location">
                    {locusHeading(openTad.chromStart, openTad.chromEnd)}
                  </p>
                  <label htmlFor="tad-note-text">TAD note</label>
                  <textarea
                    id="tad-note-text"
                    className="note-box"
                    rows={4}
                    value={tadNotes[String(openTad.chromStart)]?.text ?? ''}
                    onChange={(event) => {
                      const nextText = event.target.value
                      const key = String(openTad.chromStart)
                      setTadNotes((current) => ({
                        ...current,
                        [key]: {
                          text: nextText,
                          position: openTad.position,
                          chromEnd: openTad.chromEnd,
                        },
                      }))
                      if (tadSaveStatus === 'saved') {
                        setTadSaveStatus('')
                      }
                    }}
                  />
                  <div className="note-actions">
                    <button
                      type="button"
                      className="page-btn"
                      onClick={onSaveTadNote}
                    >
                      Save
                    </button>
                    <button
                      type="button"
                      className="page-btn"
                      onClick={onDeleteTadNote}
                    >
                      Delete
                    </button>
                    {tadSaveStatus === 'saved' && (
                      <span className="note-status">Saved.</span>
                    )}
                    {tadSaveStatus === 'saving' && (
                      <span className="note-status">Saving…</span>
                    )}
                    {tadSaveStatus === 'error' && (
                      <span className="note-status">
                        Could not save. Is Docker Compose up?
                      </span>
                    )}
                  </div>
                </div>
              )}
            </aside>
          )}

          <div className="viewer-plot-slot">
            <Plot
              className="viewer-plot"
              data={data}
              layout={layout}
              config={PLOT_CONFIG}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
              onHover={(event) => {
                lastHoverRef.current = event?.points?.[0] ?? null
              }}
              onClick={(event) => {
                handlePlotSelect(event?.points?.[0] ?? lastHoverRef.current)
              }}
              onInitialized={(_, gd) => {
                plotGdRef.current = gd
                Plotly.Plots.resize(gd)
                let down = null
                gd.addEventListener('mousedown', (event) => {
                  down = { x: event.clientX, y: event.clientY }
                })
                gd.addEventListener('mouseup', (event) => {
                  if (!down) {
                    return
                  }
                  const dist = Math.hypot(
                    event.clientX - down.x,
                    event.clientY - down.y,
                  )
                  down = null
                  if (dist < 6) {
                    handlePlotSelectRef.current?.(lastHoverRef.current)
                  }
                })
              }}
            />
          </div>
        </div>
      )}
    </main>
  )
}

export default App
