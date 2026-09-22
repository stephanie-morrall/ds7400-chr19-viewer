// Turns the fitted shapes in public/sample/shapes.json into Plotly traces.
//
// scripts/fit_nuclear_shapes.py does the fitting offline and stores each structure
// as a handful of numbers: a centre, semi-axes, and a rotation. This module rebuilds
// the surfaces from those numbers so the browser never has to run an optimiser.

// Rings around the ellipsoid. Enough to read as a surface, few enough to stay sheer.
const LATITUDE_RINGS = 6
const LONGITUDE_RINGS = 8
const RING_STEPS = 48

// Longitude rings all meet at the poles, so they stop short of them to avoid a knot.
const POLE_GAP = 0.45

// Resolution of the nucleolus sphere passed to Plotly's surface trace.
const SPHERE_STEPS = 24

// Places a point given in the ellipsoid's own frame back into microns of image space.
function toWorld(local, center, rotation) {
  return [
    center[0] + rotation[0][0] * local[0] + rotation[0][1] * local[1] + rotation[0][2] * local[2],
    center[1] + rotation[1][0] * local[0] + rotation[1][1] * local[1] + rotation[1][2] * local[2],
    center[2] + rotation[2][0] * local[0] + rotation[2][1] * local[1] + rotation[2][2] * local[2],
  ]
}

// Surface point of an ellipsoid at polar angle v and azimuth u, before rotation.
function ellipsoidPoint(semi, u, v) {
  return [
    semi[0] * Math.cos(u) * Math.sin(v),
    semi[1] * Math.sin(u) * Math.sin(v),
    semi[2] * Math.cos(v),
  ]
}

// Draws the nuclear envelope as a wireframe of latitude and longitude rings.
//
// A wireframe rather than a solid shell on purpose: the fit is an approximation, and
// roughly one spot in ten sits just outside it because real nuclei are irregular
// while an ellipsoid is smooth. A sheer cage reads as an approximate boundary and
// does not hide the spots behind it.
export function envelopeTraces(envelope) {
  const { center, semi, rotation } = envelope
  const rings = []

  // Plotly's 3D lines do not reliably break on a null, so joining the rings into one
  // trace draws spurious chords straight through the cell. Each ring is its own trace.
  const addRing = (pointAt) => {
    const x = []
    const y = []
    const z = []
    for (let i = 0; i <= RING_STEPS; i += 1) {
      const [px, py, pz] = pointAt((i / RING_STEPS) * 2 * Math.PI)
      x.push(px)
      y.push(py)
      z.push(pz)
    }
    rings.push({ x, y, z })
  }

  // Latitude rings: fixed polar angle, sweeping all the way around.
  for (let i = 1; i < LATITUDE_RINGS; i += 1) {
    const v = (i / LATITUDE_RINGS) * Math.PI
    addRing((u) => toWorld(ellipsoidPoint(semi, u, v), center, rotation))
  }
  // Longitude rings: fixed azimuth, running down one side and back up the other.
  for (let i = 0; i < LONGITUDE_RINGS; i += 1) {
    const u = (i / LONGITUDE_RINGS) * Math.PI
    addRing((angle) => {
      // Remaps the sweep into a band that stops short of both poles, so the rings
      // never all pile into the same two points.
      const v = POLE_GAP + (angle / (2 * Math.PI)) * 2 * (Math.PI - 2 * POLE_GAP)
      const past = v > Math.PI - POLE_GAP
      const polar = past ? 2 * (Math.PI - POLE_GAP) - v : v
      return toWorld(ellipsoidPoint(semi, past ? u + Math.PI : u, polar), center, rotation)
    })
  }

  // A dotted cage marks a cell where an axis rested on its size bound, so the shape
  // is one the distances permit rather than one they actually pinned down.
  const line = {
    width: 1,
    color: 'rgba(148,163,184,0.22)',
    dash: envelope.atBound ? 'dot' : 'solid',
  }
  return rings.map((ring, index) => ({
    type: 'scatter3d',
    mode: 'lines',
    name: 'Nuclear envelope (fitted)',
    ...ring,
    line,
    hoverinfo: 'skip',
    // One legend entry toggles the whole cage instead of fourteen.
    legendgroup: 'envelope',
    showlegend: index === 0,
  }))
}

// Draws the nucleolus as a translucent sphere at its fitted centre and radius.
export function nucleolusTrace(nucleolus) {
  const { center, radius } = nucleolus
  const x = []
  const y = []
  const z = []
  // Plotly's surface trace wants three 2D grids over the sphere's parameter domain.
  for (let i = 0; i <= SPHERE_STEPS; i += 1) {
    const v = (i / SPHERE_STEPS) * Math.PI
    const rowX = []
    const rowY = []
    const rowZ = []
    for (let j = 0; j <= SPHERE_STEPS; j += 1) {
      const u = (j / SPHERE_STEPS) * 2 * Math.PI
      rowX.push(center[0] + radius * Math.cos(u) * Math.sin(v))
      rowY.push(center[1] + radius * Math.sin(u) * Math.sin(v))
      rowZ.push(center[2] + radius * Math.cos(v))
    }
    x.push(rowX)
    y.push(rowY)
    z.push(rowZ)
  }
  return {
    type: 'surface',
    name: 'Nucleolus (fitted)',
    x,
    y,
    z,
    opacity: 0.35,
    showscale: false,
    colorscale: [[0, '#a78bfa'], [1, '#a78bfa']],
    hoverinfo: 'name',
    showlegend: true,
    contours: { x: { highlight: false }, y: { highlight: false }, z: { highlight: false } },
  }
}
