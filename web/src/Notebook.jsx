import { useEffect, useMemo, useState } from 'react'
import {
  deleteCellNote,
  deleteNotebook,
  deleteTadNote,
  deleteTraceLabel,
  fetchCellNotes,
  fetchNotebook,
  fetchTadNotes,
  fetchTraceLabels,
  saveCellNote,
  saveNotebook,
  saveTadNote,
  saveTraceLabel,
} from './lib/notesApi'

function cellKey(replicate, cellId) {
  return `${replicate}:${cellId}`
}

function traceKey(row) {
  return `${row.replicate}:${row.cell_id}:${row.trace_id}`
}

function tadKey(row) {
  return `${row.replicate}:${row.cell_id}:${row.chrom_start}`
}

function locusHeading(start, end) {
  const a = Number(start).toLocaleString()
  const fallbackEnd = Number(start) + 100000
  const b = Number(end ?? fallbackEnd).toLocaleString()
  return `chr19:${a}–${b}`
}

function compareCells(a, b) {
  if (a.replicate !== b.replicate) {
    return a.replicate - b.replicate
  }
  const na = Number(a.cell_id)
  const nb = Number(b.cell_id)
  if (Number.isFinite(na) && Number.isFinite(nb) && na !== nb) {
    return na - nb
  }
  return String(a.cell_id).localeCompare(String(b.cell_id))
}

function mergeCellCards(cellNotes, traces, tads) {
  const cards = new Map()
  const ensure = (replicate, cellId) => {
    const key = cellKey(replicate, cellId)
    if (!cards.has(key)) {
      cards.set(key, {
        replicate,
        cell_id: cellId,
        text: '',
        traces: [],
        tads: [],
      })
    }
    return cards.get(key)
  }
  for (const note of cellNotes) {
    const card = ensure(note.replicate, note.cell_id)
    card.text = note.text ?? ''
  }
  for (const row of traces) {
    ensure(row.replicate, row.cell_id).traces.push(row)
  }
  for (const row of tads) {
    ensure(row.replicate, row.cell_id).tads.push(row)
  }
  return [...cards.values()].sort(compareCells)
}

// Dataset pad, then one card per cell that has a cell, trace, or TAD note.
function Notebook({ onOpenCell, loadSpots }) {
  const [text, setText] = useState('')
  const [status, setStatus] = useState('loading')
  const [cellNotes, setCellNotes] = useState([])
  const [traces, setTraces] = useState([])
  const [tads, setTads] = useState([])
  const [listStatus, setListStatus] = useState('loading')
  const [rowStatus, setRowStatus] = useState({})
  const [traceRowStatus, setTraceRowStatus] = useState({})
  const [tadRowStatus, setTadRowStatus] = useState({})

  useEffect(() => {
    fetchNotebook()
      .then((body) => {
        setText(body.text ?? '')
        setStatus('ready')
      })
      .catch((err) => {
        console.error(err)
        setStatus('error')
      })
  }, [])

  useEffect(() => {
    Promise.all([
      fetchCellNotes(),
      fetchTraceLabels(),
      fetchTadNotes(),
    ])
      .then(([noteBody, traceBody, tadBody]) => {
        setCellNotes(noteBody.notes ?? [])
        setTraces(traceBody.traces ?? [])
        setTads(tadBody.tads ?? [])
        setListStatus('ready')
      })
      .catch((err) => {
        console.error(err)
        setListStatus('error')
      })
  }, [])

  const cards = useMemo(
    () => mergeCellCards(cellNotes, traces, tads),
    [cellNotes, traces, tads],
  )

  useEffect(() => {
    if (!loadSpots) {
      return
    }
    for (const card of cards) {
      loadSpots(card.cell_id, card.replicate).catch((err) => {
        if (err?.name === 'AbortError') {
          return
        }
        console.error(err)
      })
    }
  }, [cards, loadSpots])

  async function onSave() {
    setStatus('saving')
    try {
      await saveNotebook(text)
      setStatus('saved')
    } catch (err) {
      console.error(err)
      setStatus('error')
    }
  }

  async function onDeletePad() {
    setStatus('saving')
    try {
      await deleteNotebook()
      setText('')
      setStatus('saved')
    } catch (err) {
      console.error(err)
      setStatus('error')
    }
  }

  function updateNoteText(key, nextText) {
    setCellNotes((current) => {
      const found = current.some((note) => cellKey(note.replicate, note.cell_id) === key)
      if (found) {
        return current.map((note) => (
          cellKey(note.replicate, note.cell_id) === key
            ? { ...note, text: nextText }
            : note
        ))
      }
      const [replicate, cellId] = key.split(':')
      return [
        ...current,
        { replicate: Number(replicate), cell_id: cellId, text: nextText },
      ]
    })
    setRowStatus((current) => {
      if (current[key] !== 'saved') {
        return current
      }
      const next = { ...current }
      delete next[key]
      return next
    })
  }

  async function onSaveCellRow(card) {
    const key = cellKey(card.replicate, card.cell_id)
    setRowStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      await saveCellNote(card.replicate, card.cell_id, card.text)
      if (card.text.trim() === '') {
        setCellNotes((current) => (
          current.filter((row) => (
            cellKey(row.replicate, row.cell_id) !== key
          ))
        ))
      }
      setRowStatus((current) => ({ ...current, [key]: 'saved' }))
    } catch (err) {
      console.error(err)
      setRowStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }

  async function onDeleteCellRow(card) {
    const key = cellKey(card.replicate, card.cell_id)
    setRowStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      await deleteCellNote(card.replicate, card.cell_id)
      setCellNotes((current) => (
        current.filter((row) => (
          cellKey(row.replicate, row.cell_id) !== key
        ))
      ))
      setRowStatus((current) => ({ ...current, [key]: 'saved' }))
    } catch (err) {
      console.error(err)
      setRowStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }

  function updateTraceText(key, nextText) {
    setTraces((current) => (
      current.map((row) => (
        traceKey(row) === key ? { ...row, text: nextText } : row
      ))
    ))
    setTraceRowStatus((current) => {
      if (current[key] !== 'saved') {
        return current
      }
      const next = { ...current }
      delete next[key]
      return next
    })
  }

  async function onSaveTraceRow(row) {
    const key = traceKey(row)
    setTraceRowStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      await saveTraceLabel(row.replicate, row.cell_id, row.trace_id, row.text)
      if (row.text.trim() === '') {
        setTraces((current) => (
          current.filter((item) => traceKey(item) !== key)
        ))
      }
      setTraceRowStatus((current) => ({ ...current, [key]: 'saved' }))
    } catch (err) {
      console.error(err)
      setTraceRowStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }

  async function onDeleteTraceRow(row) {
    const key = traceKey(row)
    setTraceRowStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      await deleteTraceLabel(row.replicate, row.cell_id, row.trace_id)
      setTraces((current) => (
        current.filter((item) => traceKey(item) !== key)
      ))
    } catch (err) {
      console.error(err)
      setTraceRowStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }

  function updateTadText(key, nextText) {
    setTads((current) => (
      current.map((row) => (
        tadKey(row) === key ? { ...row, text: nextText } : row
      ))
    ))
    setTadRowStatus((current) => {
      if (current[key] !== 'saved') {
        return current
      }
      const next = { ...current }
      delete next[key]
      return next
    })
  }

  async function onSaveTadRow(row) {
    const key = tadKey(row)
    setTadRowStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      await saveTadNote(
        row.replicate,
        row.cell_id,
        row.chrom_start,
        row.text,
        row.position,
        row.chrom_end,
      )
      if (row.text.trim() === '') {
        setTads((current) => (
          current.filter((item) => tadKey(item) !== key)
        ))
      }
      setTadRowStatus((current) => ({ ...current, [key]: 'saved' }))
    } catch (err) {
      console.error(err)
      setTadRowStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }

  async function onDeleteTadRow(row) {
    const key = tadKey(row)
    setTadRowStatus((current) => ({ ...current, [key]: 'saving' }))
    try {
      await deleteTadNote(row.replicate, row.cell_id, row.chrom_start)
      setTads((current) => (
        current.filter((item) => tadKey(item) !== key)
      ))
    } catch (err) {
      console.error(err)
      setTadRowStatus((current) => ({ ...current, [key]: 'error' }))
    }
  }

  return (
    <div className="notebook">
      <h2>Notebook</h2>
      <p>
        One pad for this dataset. Save stores the text in SQLite through the
        FastAPI service. Refresh the page to confirm it is still there.
      </p>
      {status === 'error' && (
        <p>
          Could not reach the API. In the repository root run
          {' '}
          <code>docker compose up --build</code>
          , then try Save again.
        </p>
      )}
      <textarea
        className="note-box"
        value={text}
        onChange={(event) => {
          setText(event.target.value)
          if (status === 'saved') {
            setStatus('ready')
          }
        }}
        disabled={status === 'loading'}
        rows={12}
      />
      <div className="note-actions">
        <button type="button" className="page-btn" onClick={onSave}>
          Save
        </button>
        <button type="button" className="page-btn" onClick={onDeletePad}>
          Delete
        </button>
        {status === 'saved' && <span className="note-status">Saved.</span>}
        {status === 'saving' && <span className="note-status">Saving…</span>}
      </div>

      <h2 className="notebook-cells-heading">Cell notes</h2>
      <p>
        Every cell with a saved cell note, fiber note, or TAD note appears
        here. Click the card title or the note text to open that cell. Save
        and Delete stay on the buttons. Trace and TAD notes sit in a nested
        box on the same card.
      </p>
      {listStatus === 'loading' && <p>Loading cell notes…</p>}
      {listStatus === 'error' && (
        <p>
          Could not load notes. In the repository root run
          {' '}
          <code>docker compose up --build</code>
          , then open Notebook again.
        </p>
      )}
      {listStatus === 'ready' && cards.length === 0 && (
        <p>
          No cell notes yet. Save a cell note, click a trace name in the key,
          or double-click a 3D locus for a TAD note.
        </p>
      )}
      {cards.map((card) => {
        const key = cellKey(card.replicate, card.cell_id)
        const mark = rowStatus[key]
        return (
          <article
            key={key}
            className="cell-note-card"
            onClick={(event) => {
              const node = event.target
              const el = node instanceof Element ? node : node.parentElement
              if (el?.closest('button')) {
                return
              }
              onOpenCell(card.cell_id, card.replicate)
            }}
          >
            <div className="cell-note-card-head">
              <button
                type="button"
                className="cell-note-card-title"
                onClick={() => onOpenCell(card.cell_id, card.replicate)}
              >
                Cell {card.cell_id}
                {' · replicate '}
                {card.replicate}
              </button>
              <button
                type="button"
                className="page-btn"
                onClick={() => onOpenCell(card.cell_id, card.replicate)}
              >
                Open
              </button>
            </div>
            <textarea
              className="note-box"
              rows={4}
              value={card.text}
              onChange={(event) => updateNoteText(key, event.target.value)}
            />
            <div className="note-actions">
              <button
                type="button"
                className="page-btn"
                onClick={() => onSaveCellRow(card)}
              >
                Save
              </button>
              <button
                type="button"
                className="page-btn"
                onClick={() => onDeleteCellRow(card)}
              >
                Delete
              </button>
              {mark === 'saved' && <span className="note-status">Saved.</span>}
              {mark === 'saving' && <span className="note-status">Saving…</span>}
              {mark === 'error' && (
                <span className="note-status">
                  Could not save. Is Docker Compose up?
                </span>
              )}
            </div>
            {card.traces.map((row) => {
              const tKey = traceKey(row)
              const tMark = traceRowStatus[tKey]
              return (
                <div
                  key={tKey}
                  className="nested-note nested-note-trace"
                >
                  <p className="nested-note-heading">
                    Trace {row.trace_id}
                  </p>
                  <textarea
                    className="note-box"
                    rows={3}
                    value={row.text}
                    onChange={(event) => (
                      updateTraceText(tKey, event.target.value)
                    )}
                  />
                  <div className="note-actions">
                    <button
                      type="button"
                      className="page-btn"
                      onClick={() => onSaveTraceRow(row)}
                    >
                      Save
                    </button>
                    <button
                      type="button"
                      className="page-btn"
                      onClick={() => onDeleteTraceRow(row)}
                    >
                      Delete
                    </button>
                    {tMark === 'saved' && (
                      <span className="note-status">Saved.</span>
                    )}
                    {tMark === 'saving' && (
                      <span className="note-status">Saving…</span>
                    )}
                    {tMark === 'error' && (
                      <span className="note-status">Could not save.</span>
                    )}
                  </div>
                </div>
              )
            })}
            {card.tads.map((row) => {
              const tKey = tadKey(row)
              const tMark = tadRowStatus[tKey]
              return (
                <div
                  key={tKey}
                  className="nested-note nested-note-tad"
                >
                  <p className="nested-note-heading">
                    TAD {row.position}
                  </p>
                  <p className="tad-note-location">
                    {locusHeading(row.chrom_start, row.chrom_end)}
                  </p>
                  <textarea
                    className="note-box"
                    rows={3}
                    value={row.text}
                    onChange={(event) => (
                      updateTadText(tKey, event.target.value)
                    )}
                  />
                  <div className="note-actions">
                    <button
                      type="button"
                      className="page-btn"
                      onClick={() => onSaveTadRow(row)}
                    >
                      Save
                    </button>
                    <button
                      type="button"
                      className="page-btn"
                      onClick={() => onDeleteTadRow(row)}
                    >
                      Delete
                    </button>
                    {tMark === 'saved' && (
                      <span className="note-status">Saved.</span>
                    )}
                    {tMark === 'saving' && (
                      <span className="note-status">Saving…</span>
                    )}
                    {tMark === 'error' && (
                      <span className="note-status">Could not save.</span>
                    )}
                  </div>
                </div>
              )
            })}
          </article>
        )
      })}
    </div>
  )
}

export default Notebook
