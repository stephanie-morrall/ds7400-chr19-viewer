import { useEffect, useMemo, useRef, useState } from 'react'
import { drawCellPreview } from './lib/previewCanvas'

export const PAGE_SIZE = 30

function CellTile({ cell, loadSpots, dark, onOpen }) {
  const tileRef = useRef(null)
  const canvasRef = useRef(null)
  const spotsRef = useRef(null)
  const [visible, setVisible] = useState(false)

  // Paints the canvas only when the square is near the viewport.

  useEffect(() => {
    const tile = tileRef.current
    if (!tile) {
      return
    }
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { rootMargin: '0px', threshold: 0.01 },
    )
    observer.observe(tile)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!visible) {
      return
    }
    let cancelled = false
    spotsRef.current = null
    const canvas = canvasRef.current
    const paint = () => {
      if (cancelled || !spotsRef.current) {
        return
      }
      drawCellPreview(canvasRef.current, spotsRef.current, dark)
    }
    const fillPreview = () => {
      if (cancelled) {
        return
      }
      loadSpots(cell.id, cell.replicate)
        .then((spots) => {
          if (cancelled) {
            return
          }
          spotsRef.current = spots
          paint()
        })
        .catch((err) => {
          if (err?.name === 'AbortError') {
            return
          }
          console.error(err)
        })
    }
    // Waits for idle so a click is not queued behind other tile JSON files.
    let idleId = null
    let timeoutId = null
    if (typeof requestIdleCallback === 'function') {
      idleId = requestIdleCallback(fillPreview, { timeout: 800 })
    } else {
      timeoutId = window.setTimeout(fillPreview, 250)
    }
    const observer = new ResizeObserver(paint)
    if (canvas) {
      observer.observe(canvas)
    }
    return () => {
      cancelled = true
      if (idleId != null) {
        cancelIdleCallback(idleId)
      }
      if (timeoutId != null) {
        window.clearTimeout(timeoutId)
      }
      observer.disconnect()
    }
  }, [visible, cell.id, cell.replicate, loadSpots, dark])

  // The whole square is the click target, so the drawing opens the cell.
  return (
    <button
      type="button"
      ref={tileRef}
      className="cell-tile"
      onPointerEnter={() => {
        loadSpots(cell.id, cell.replicate).catch((err) => {
          if (err?.name === 'AbortError') {
            return
          }
          console.error(err)
        })
      }}
      onClick={() => onOpen(cell.id, cell.replicate)}
    >
      <canvas ref={canvasRef} className="cell-tile-plot" />
      <span className="cell-tile-label">
        Cell {cell.id} · {cell.nTraces}{' '}
        {cell.nTraces === 1 ? 'trace' : 'traces'}
      </span>
    </button>
  )
}

export default function Gallery({
  types,
  typeName,
  replicate,
  page,
  traceCount,
  loadSpots,
  onTypeChange,
  onReplicateChange,
  onTraceCountChange,
  onPageChange,
  onOpenCell,
}) {
  const allCells = useMemo(
    () => types.flatMap((entry) => (
      entry.cells.map((cell) => ({
        ...cell,
        typeName: entry.name,
        replicate: cell.replicate ?? 1,
      }))
    )),
    [types],
  )
  const replicates = useMemo(() => {
    const values = new Set(allCells.map((cell) => cell.replicate))
    return [...values].sort((a, b) => a - b)
  }, [allCells])
  const afterReplicate = useMemo(() => {
    if (replicate === 'all') {
      return allCells
    }
    const n = Number(replicate)
    return allCells.filter((cell) => cell.replicate === n)
  }, [allCells, replicate])
  const afterType = useMemo(() => {
    if (typeName === 'all') {
      return afterReplicate
    }
    return afterReplicate.filter((cell) => cell.typeName === typeName)
  }, [afterReplicate, typeName])
  const traceCounts = useMemo(() => {
    const counts = new Set(afterType.map((cell) => cell.nTraces))
    return [...counts].sort((a, b) => a - b)
  }, [afterType])
  const filtered = useMemo(() => {
    if (traceCount === 'all') {
      return afterType
    }
    const n = Number(traceCount)
    return afterType.filter((cell) => cell.nTraces === n)
  }, [afterType, traceCount])
  const typeCounts = useMemo(() => {
    const counts = new Map()
    for (const cell of afterReplicate) {
      counts.set(cell.typeName, (counts.get(cell.typeName) ?? 0) + 1)
    }
    return counts
  }, [afterReplicate])
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const pageIndex = Math.min(page, pageCount - 1)
  const pageCells = filtered.slice(
    pageIndex * PAGE_SIZE,
    pageIndex * PAGE_SIZE + PAGE_SIZE,
  )

  // Draft is the digits in the field while typing; pageIndex is what the grid shows.
  const [pageDraft, setPageDraft] = useState(String(pageIndex + 1))

  // Copies the visible page into the field when Previous, Next, or a filter changes it.
  useEffect(() => {
    setPageDraft(String(pageIndex + 1))
  }, [pageIndex])

  // Reads the field as a 1-based page, clamps to 1..pageCount, and tells App the 0-based index.
  function commitPage(raw) {
    const trimmed = String(raw).trim()
    if (!/^\d+$/.test(trimmed)) {
      setPageDraft(String(pageIndex + 1))
      return
    }
    const requested = Number.parseInt(trimmed, 10)
    const next = Math.min(pageCount, Math.max(1, requested))
    setPageDraft(String(next))
    onPageChange(next - 1)
  }

  const dark = useMemo(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches,
    [],
  )

  return (
    <section className="gallery">
      <div className="viewer-toolbar">
        <label>
          Replicate
          <select
            value={replicate}
            onChange={(event) => onReplicateChange(event.target.value)}
          >
            <option value="all">Any</option>
            {replicates.map((n) => (
              <option key={n} value={String(n)}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label>
          Cell type
          <select
            value={typeName}
            onChange={(event) => onTypeChange(event.target.value)}
          >
            <option value="all">Any</option>
            {types.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name} ({typeCounts.get(entry.name) ?? 0})
              </option>
            ))}
          </select>
        </label>
        <label>
          Traces
          <select
            value={traceCount}
            onChange={(event) => onTraceCountChange(event.target.value)}
          >
            <option value="all">Any</option>
            {traceCounts.map((n) => (
              <option key={n} value={String(n)}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <span className="viewer-note">
          {filtered.length} cells
          {traceCount === 'all' ? '' : ` with ${traceCount} traces`}
        </span>
        <label className="page-jump">
          Page
          <input
            type="number"
            min={1}
            max={pageCount}
            value={pageDraft}
            aria-label="Page number"
            onChange={(event) => setPageDraft(event.target.value)}
            onBlur={(event) => commitPage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                commitPage(event.target.value)
              }
            }}
          />
          of {pageCount}
        </label>
        <button
          type="button"
          className="page-btn"
          disabled={pageIndex === 0}
          onClick={() => onPageChange(pageIndex - 1)}
        >
          Previous
        </button>
        <button
          type="button"
          className="page-btn"
          disabled={pageIndex >= pageCount - 1}
          onClick={() => onPageChange(pageIndex + 1)}
        >
          Next
        </button>
      </div>

      <div className="gallery-grid">
        {pageCells.map((cell) => (
          <CellTile
            key={`${cell.replicate}-${cell.id}`}
            cell={cell}
            loadSpots={loadSpots}
            dark={dark}
            onOpen={onOpenCell}
          />
        ))}
      </div>
    </section>
  )
}
