// Shared spot helpers for the gallery thumbnails and the one-cell 3D plot.

// Plotly qualitative palette as R,G,B so a line can reuse its marker colour with alpha.
export const TRACE_COLORS = [
  '99,110,250',
  '239,85,59',
  '0,204,150',
  '171,99,250',
  '255,161,90',
  '25,211,243',
]

// Camera for gallery previews: same direction as Plotly's default eye (1.4, 1.4, 1.4).
const INV_SQRT3 = 1 / Math.sqrt(3)
const PREVIEW_FORWARD = [INV_SQRT3, INV_SQRT3, INV_SQRT3]
const PREVIEW_RIGHT = (() => {
  // right = forward × world-Z, so screen-x follows the Plotly scene.
  const rx = -PREVIEW_FORWARD[1]
  const ry = PREVIEW_FORWARD[0]
  const len = Math.hypot(rx, ry)
  return [rx / len, ry / len, 0]
})()
const PREVIEW_UP = [
  PREVIEW_FORWARD[1] * PREVIEW_RIGHT[2] - PREVIEW_FORWARD[2] * PREVIEW_RIGHT[1],
  PREVIEW_FORWARD[2] * PREVIEW_RIGHT[0] - PREVIEW_FORWARD[0] * PREVIEW_RIGHT[2],
  PREVIEW_FORWARD[0] * PREVIEW_RIGHT[1] - PREVIEW_FORWARD[1] * PREVIEW_RIGHT[0],
]

export function projectPreview(dx, dy, dz) {
  return [
    dx * PREVIEW_RIGHT[0] + dy * PREVIEW_RIGHT[1] + dz * PREVIEW_RIGHT[2],
    dx * PREVIEW_UP[0] + dy * PREVIEW_UP[1] + dz * PREVIEW_UP[2],
  ]
}

export function previewDepth(dx, dy, dz) {
  return (
    dx * PREVIEW_FORWARD[0]
    + dy * PREVIEW_FORWARD[1]
    + dz * PREVIEW_FORWARD[2]
  )
}

export function cellCentroid(spots) {
  const n = spots.length
  return {
    x: spots.reduce((s, spot) => s + spot.X, 0) / n,
    y: spots.reduce((s, spot) => s + spot.Y, 0) / n,
    z: spots.reduce((s, spot) => s + spot.Z, 0) / n,
  }
}

export function cellAxisSpan(spots) {
  if (spots.length === 0) {
    return 0
  }
  const xs = spots.map((spot) => spot.X)
  const ys = spots.map((spot) => spot.Y)
  const zs = spots.map((spot) => spot.Z)
  return Math.max(
    Math.max(...xs) - Math.min(...xs),
    Math.max(...ys) - Math.min(...ys),
    Math.max(...zs) - Math.min(...zs),
    1e-6,
  )
}

export function previewSpanUmFromCells(cellSpotLists) {
  let span = 0
  for (const spots of cellSpotLists) {
    span = Math.max(span, cellAxisSpan(spots))
  }
  return span * 1.08
}

export function parseSpotCsv(text) {
  const lines = text.trim().split('\n')
  const header = lines[0].split(',')
  return lines.slice(1).map((line) => {
    const cols = line.split(',')
    const row = Object.fromEntries(header.map((name, i) => [name, cols[i]]))
    return {
      Spot_ID: row.Spot_ID,
      Trace_ID: row.Trace_ID,
      X: Number(row.X),
      Y: Number(row.Y),
      Z: Number(row.Z),
      Cell_ID: row.Cell_ID,
      Chrom_Start: Number(row.Chrom_Start),
      Chrom_End: Number(row.Chrom_End),
      Distance_To_Lamina: Number(row.Distance_To_Lamina),
      Distance_To_Nucleolus: Number(row.Distance_To_Nucleolus),
    }
  })
}

// Same micron span on X, Y, and Z, centered on each axis's data, so the scene is a
// cube and a shape that is round in the cell looks round on screen.
export function equalAxisRange(spots, envelope) {
  const xs = spots.map((spot) => spot.X)
  const ys = spots.map((spot) => spot.Y)
  const zs = spots.map((spot) => spot.Z)
  // The envelope usually reaches past the spots, so its extent sets the cube instead.
  if (envelope) {
    const reach = Math.max(...envelope.semi)
    for (let axis = 0; axis < 3; axis += 1) {
      const target = [xs, ys, zs][axis]
      target.push(envelope.center[axis] - reach, envelope.center[axis] + reach)
    }
  }
  const spans = [xs, ys, zs].map((v) => Math.max(...v) - Math.min(...v))
  const half = Math.max(...spans, 1) / 2 * 1.08
  const range = (values) => {
    const mid = (Math.max(...values) + Math.min(...values)) / 2
    return [mid - half, mid + half]
  }
  return { x: range(xs), y: range(ys), z: range(zs) }
}

// Probe windows in this FOF-CT table are 100 kb. Catalog JSON omits Chrom_End,
// so a TAD click uses the start plus 100000 bp when the end is missing.
function locusChromEnd(spot) {
  if (spot.Chrom_End == null || spot.Chrom_End === '') {
    return Number(spot.Chrom_Start) + 100000
  }
  const end = Number(spot.Chrom_End)
  if (Number.isFinite(end)) {
    return end
  }
  return Number(spot.Chrom_Start) + 100000
}

export function tadRank(starts, chromStart) {
  const unique = [...new Set(starts.map(Number))].sort((a, b) => a - b)
  const idx = unique.indexOf(Number(chromStart))
  if (idx < 0) {
    return null
  }
  return idx + 1
}

export function chromStartsFromFibers(fibers) {
  const starts = []
  for (const fiber of fibers) {
    for (const spot of fiber.points) {
      starts.push(spot.Chrom_Start)
    }
  }
  return starts
}

export function locusFromPlotPoint(pt) {
  if (!pt) {
    return null
  }
  const idx = Array.isArray(pt.pointNumber) ? pt.pointNumber[0] : pt.pointNumber
  const custom = pt.customdata ?? pt.data?.customdata?.[idx]
  if (!Array.isArray(custom) || custom.length < 5) {
    return null
  }
  const chromStart = Number(custom[2])
  const chromEndRaw = custom.length > 5 ? custom[5] : null
  const parsedEnd = Number(chromEndRaw)
  const hasEnd = (
    chromEndRaw != null
    && chromEndRaw !== ''
    && Number.isFinite(parsedEnd)
  )
  return {
    lamina: custom[0],
    nucleolus: custom[1],
    chromStart,
    traceId: custom[3],
    spotId: custom[4],
    chromEnd: hasEnd ? parsedEnd : chromStart + 100000,
  }
}

// Splits a cell's spots into one ordered polyline per traced chr19 fiber.
export function tracesInCell(spots) {
  const byTrace = new Map()
  for (const spot of spots) {
    if (!byTrace.has(spot.Trace_ID)) {
      byTrace.set(spot.Trace_ID, [])
    }
    byTrace.get(spot.Trace_ID).push(spot)
  }
  return [...byTrace.keys()]
    .sort((a, b) => Number(a) - Number(b))
    .map((traceId) => {
      const points = byTrace.get(traceId)
      // Orders loci along chr19 so the line follows the fiber, not the file order.
      points.sort((a, b) => a.Chrom_Start - b.Chrom_Start)
      return { traceId, points }
    })
}

// Builds one scatter trace per traced fiber, each in its own solid colour.
//
// These are labelled by Trace_ID rather than as chromosome copies on purpose. Cell 411
// carries five traces of 41 to 49 spots each, and a diploid mouse cell has two copies
// of chr19, so calling them copies would assert something the data does not show.
export function spotTraces(fibers, { markerSize = 4 } = {}) {
  const starts = chromStartsFromFibers(fibers)
  return fibers.map(({ traceId, points }, index) => {
    const color = TRACE_COLORS[index % TRACE_COLORS.length]
    return {
      type: 'scatter3d',
      mode: 'lines+markers',
      name: `Trace ${traceId}`,
      x: points.map((spot) => spot.X),
      y: points.map((spot) => spot.Y),
      z: points.map((spot) => spot.Z),
      marker: { size: markerSize, color: `rgb(${color})` },
      line: { width: 1, color: `rgba(${color}, 0.35)` },
      customdata: points.map((spot) => [
        spot.Distance_To_Lamina,
        spot.Distance_To_Nucleolus,
        spot.Chrom_Start,
        spot.Trace_ID,
        spot.Spot_ID,
        locusChromEnd(spot),
        tadRank(starts, spot.Chrom_Start),
      ]),
      // Hover names the TAD and the two nuclear distances stored on the locus.
      hovertemplate:
        '%{fullData.name}<br>' +
        'TAD %{customdata[6]}<br>' +
        'Distance to lamina: %{customdata[0]:.2f} µm<br>' +
        'Distance to nucleolus: %{customdata[1]:.2f} µm' +
        '<extra></extra>',
    }
  })
}

export function buildCellView(spots, shapes, cellId) {
  const shape = shapes?.[cellId]
  return {
    count: spots.length,
    fibers: tracesInCell(spots),
    envelope: shape?.envelope ?? null,
    // A nucleolus the data could not pin down is withheld rather than drawn faintly.
    nucleolus: shape?.nucleolus?.trusted ? shape.nucleolus : null,
    nucleolusNote: shape?.nucleolus ?? null,
    axisRange: equalAxisRange(spots, shape?.envelope),
  }
}
