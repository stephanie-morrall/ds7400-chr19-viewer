// Still gallery thumbnail that reads as a snapshot of the Plotly detail cube:
// same camera, grid only on the three back walls, traces in front, no near-side
// box edges. Canvas 2D avoids the browser WebGL-context limit.

import {
  equalAxisRange,
  previewDepth,
  projectPreview,
  TRACE_COLORS,
  tracesInCell,
} from './spots'

const GRID = 5

function lerp(a, b, t) {
  return a + (b - a) * t
}

function cubeCenter(ranges) {
  return [
    (ranges.x[0] + ranges.x[1]) / 2,
    (ranges.y[0] + ranges.y[1]) / 2,
    (ranges.z[0] + ranges.z[1]) / 2,
  ]
}

function projectPoint(x, y, z, mid) {
  return projectPreview(x - mid[0], y - mid[1], z - mid[2])
}

// Maps the back-wall bounds onto the full square so the grid reaches the tile edges.
function screenMapper(corners, width, height) {
  const us = corners.map((point) => point[0])
  const vs = corners.map((point) => point[1])
  const minU = Math.min(...us)
  const maxU = Math.max(...us)
  const minV = Math.min(...vs)
  const maxV = Math.max(...vs)
  const spanU = Math.max(maxU - minU, 1e-6)
  const spanV = Math.max(maxV - minV, 1e-6)
  const scaleX = width / spanU
  const scaleY = height / spanV
  const midU = (minU + maxU) / 2
  const midV = (minV + maxV) / 2
  return (u, v) => [
    width / 2 + (u - midU) * scaleX,
    height / 2 - (v - midV) * scaleY,
  ]
}

function pointOnFace(axis, value, u, v) {
  const p = [0, 0, 0]
  const uAxis = (axis + 1) % 3
  const vAxis = (axis + 2) % 3
  p[axis] = value
  p[uAxis] = u
  p[vAxis] = v
  return p
}

function screenOf(toScreen, mid, x, y, z) {
  const [u, v] = projectPoint(x, y, z, mid)
  return toScreen(u, v)
}

function fillBackFace(ctx, toScreen, mid, face, fill) {
  const { axis, value, u0, u1, v0, v1 } = face
  const quad = [
    pointOnFace(axis, value, u0, v0),
    pointOnFace(axis, value, u1, v0),
    pointOnFace(axis, value, u1, v1),
    pointOnFace(axis, value, u0, v1),
  ]
  ctx.beginPath()
  quad.forEach((p, i) => {
    const [sx, sy] = screenOf(toScreen, mid, p[0], p[1], p[2])
    if (i === 0) {
      ctx.moveTo(sx, sy)
    } else {
      ctx.lineTo(sx, sy)
    }
  })
  ctx.closePath()
  ctx.fillStyle = fill
  ctx.fill()
}

function strokeBackGrid(ctx, toScreen, mid, face, stroke) {
  const { axis, value, u0, u1, v0, v1 } = face
  ctx.strokeStyle = stroke
  ctx.lineWidth = 1
  for (let i = 0; i <= GRID; i += 1) {
    const t = i / GRID
    const u = lerp(u0, u1, t)
    const v = lerp(v0, v1, t)
    const a = pointOnFace(axis, value, u, v0)
    const b = pointOnFace(axis, value, u, v1)
    const c = pointOnFace(axis, value, u0, v)
    const d = pointOnFace(axis, value, u1, v)
    const pa = screenOf(toScreen, mid, a[0], a[1], a[2])
    const pb = screenOf(toScreen, mid, b[0], b[1], b[2])
    const pc = screenOf(toScreen, mid, c[0], c[1], c[2])
    const pd = screenOf(toScreen, mid, d[0], d[1], d[2])
    ctx.beginPath()
    ctx.moveTo(pa[0], pa[1])
    ctx.lineTo(pb[0], pb[1])
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(pc[0], pc[1])
    ctx.lineTo(pd[0], pd[1])
    ctx.stroke()
  }
}

export function drawCellPreview(canvas, spots, dark) {
  if (!canvas || !spots || spots.length === 0) {
    return
  }
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  if (width < 2 || height < 2) {
    return
  }
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.round(width * dpr)
  canvas.height = Math.round(height * dpr)
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  const bg = dark ? '#16171d' : '#fff'
  const axisColor = dark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)'
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, width, height)

  const ranges = equalAxisRange(spots)
  const mid = cubeCenter(ranges)
  const faces = [
    {
      axis: 0,
      value: ranges.x[0],
      u0: ranges.y[0],
      u1: ranges.y[1],
      v0: ranges.z[0],
      v1: ranges.z[1],
    },
    {
      axis: 1,
      value: ranges.y[0],
      u0: ranges.z[0],
      u1: ranges.z[1],
      v0: ranges.x[0],
      v1: ranges.x[1],
    },
    {
      axis: 2,
      value: ranges.z[0],
      u0: ranges.x[0],
      u1: ranges.x[1],
      v0: ranges.y[0],
      v1: ranges.y[1],
    },
  ]
  // Bounds hug the back walls and the traces, not the empty near corners of the cube.
  const corners = []
  faces.forEach((face) => {
    const { axis, value, u0, u1, v0, v1 } = face
    ;[
      [u0, v0],
      [u1, v0],
      [u1, v1],
      [u0, v1],
    ].forEach(([u, v]) => {
      const p = pointOnFace(axis, value, u, v)
      corners.push(projectPoint(p[0], p[1], p[2], mid))
    })
  })
  spots.forEach((spot) => {
    corners.push(projectPoint(spot.X, spot.Y, spot.Z, mid))
  })
  const toScreen = screenMapper(corners, width, height)
  faces.forEach((face) => fillBackFace(ctx, toScreen, mid, face, bg))
  faces.forEach((face) => strokeBackGrid(ctx, toScreen, mid, face, axisColor))

  const fibers = tracesInCell(spots)
  fibers.forEach((fiber, index) => {
    const color = TRACE_COLORS[index % TRACE_COLORS.length]
    ctx.strokeStyle = `rgba(${color}, 0.35)`
    ctx.lineWidth = 1
    ctx.beginPath()
    fiber.points.forEach((spot, i) => {
      const [sx, sy] = screenOf(toScreen, mid, spot.X, spot.Y, spot.Z)
      if (i === 0) {
        ctx.moveTo(sx, sy)
      } else {
        ctx.lineTo(sx, sy)
      }
    })
    ctx.stroke()
  })

  // Draws closer spots last so they sit on top of farther ones.
  const markers = []
  fibers.forEach((fiber, index) => {
    const color = TRACE_COLORS[index % TRACE_COLORS.length]
    fiber.points.forEach((spot) => {
      markers.push({
        spot,
        color,
        depth: previewDepth(spot.X - mid[0], spot.Y - mid[1], spot.Z - mid[2]),
      })
    })
  })
  markers.sort((a, b) => a.depth - b.depth)
  markers.forEach(({ spot, color }) => {
    const [sx, sy] = screenOf(toScreen, mid, spot.X, spot.Y, spot.Z)
    ctx.beginPath()
    ctx.fillStyle = `rgb(${color})`
    ctx.arc(sx, sy, 2, 0, Math.PI * 2)
    ctx.fill()
  })
}
