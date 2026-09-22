"""Fit approximate nuclear envelope and nucleolus shapes for the sample cells.

The FOF-CT tables give each chr19 spot a distance to the nuclear lamina and a
distance to the nucleolus, but they never give the shape of either structure. This
script recovers approximate shapes by treating those distances as constraints.

Fitting a free surface does not work: chromosome 19 sits in one territory and only
samples two to four of the eight octants around the cell, so the curvature of the
nuclear wall is unidentifiable and a least-squares sphere comes out with a negative
radius on about half the cells. Instead we start from a canonical shape anchored to
the cell table's independently measured centroid and area, and let the distances
translate, tilt, and mildly rescale it.

Distances enter as *signed* depths, so a spot must sit inside the envelope and
outside the nucleolus to score well. That is what keeps the fit physically legal.

Writes web/public/sample/shapes.json, which the viewer draws as toggleable surfaces.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "biorep01"
OUT = ROOT / "web" / "public" / "sample" / "shapes.json"

# Cells carried in the browser sample. Keep in sync with web/public/sample/cells.csv.
SAMPLE_CELLS = ("411", "328", "530")

# Canonical nucleus before fitting: semi-axes (a, a, a * FLATTEN) with a taken from
# the measured cell area. Adherent nuclei are squashed against the coverslip.
FLATTEN = 0.5

# Canonical nucleolus radius in microns, near the middle of the published range.
NUCLEOLUS_PRIOR_RADIUS = 1.2

# The two anchors pull on different things and deserve different strengths. The
# segmented centroid is a genuine independent measurement of where the cell sits, so
# it holds firm. The cell area only suggests a size, so its pull is light and the
# distances are free to squash and resize the canonical nucleus to fit them.
CENTROID_ANCHOR_WEIGHT = 2.0
SCALE_ANCHOR_WEIGHT = 0.15

# The nucleolar radius is deliberately held near its prior. chr19 approaches a
# nucleolus from one side only, which leaves radius and position degenerate: a larger
# sphere placed further away fits the same distances almost as well. Left free the
# radius simply grows until it hits the nuclear wall, making it an artifact of the
# envelope fit. Pinning it to the published size lets the distances determine the one
# thing they do constrain, which is where the nucleolus sits.
NUCLEOLUS_PRIOR_WEIGHT = 25.0

# Physical limits on a semi-axis, in microns. Where chr19 hugs a nearly flat patch of
# wall the best-fit curvature goes to zero and the axis would otherwise run to
# infinity, so the bound is stated as a nucleus size rather than a tuning multiple.
# A mammalian nucleus is a few microns across; anything past this is not a nucleus.
SEMI_AXIS_BOUNDS_UM = (1.5, 9.0)

# chr19 leaves some directions barely constrained, and without a cap the optimiser
# will crawl along those flat valleys for minutes to gain nothing.
MAX_FIT_EVALUATIONS = 600

# A nucleolus is only drawn when chr19 constrains it well enough to be worth trusting.
MAX_TRUSTED_RMS = 0.6
TRUSTED_RADIUS_RANGE = (0.5, 2.5)


def load_sample():
    """Returns per-cell spot arrays (X, Y, Z, d_lamina, d_nucleolus) and cell metadata.

    Three tables have to be joined: the core table holds coordinates, the bio table
    holds the two distances keyed by (Spot_ID, Trace_ID), and the cell table holds the
    segmented centroid and area that anchor the envelope fit.
    """
    meta = {}
    with (RAW / "4DNFITOO96GG.csv").open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split(",")
            if cols[0] in SAMPLE_CELLS:
                # Columns are Cell_ID, Cell_Size (2D um^2), FOV_ID, x_Centroid, y_Centroid.
                meta[cols[0]] = {
                    "area": float(cols[1]),
                    "centroid": (float(cols[3]), float(cols[4])),
                }

    distances = {}
    with (RAW / "4DNFI5WKRHYC.csv").open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            spot_id, trace_id, lamina, nucleolus = line.rstrip().split(",")[:4]
            distances[(spot_id, trace_id)] = (float(lamina), float(nucleolus))

    spots = defaultdict(list)
    with (RAW / "4DNFIQCHBYZ6.csv").open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split(",")
            cell_id = cols[-1]
            if cell_id not in SAMPLE_CELLS:
                continue
            lamina, nucleolus = distances[(cols[0], cols[1])]
            spots[cell_id].append(
                (float(cols[2]), float(cols[3]), float(cols[4]), lamina, nucleolus)
            )
    return {k: np.array(v) for k, v in spots.items()}, meta


def ellipsoid_depth(points, semi):
    """Signed distance from each point to an origin-centred ellipsoid, positive inside.

    The closest point on the surface satisfies x_i = q_i * s_i^2 / (s_i^2 + t) for a
    Lagrange multiplier t that solves sum((q_i * s_i / (s_i^2 + t))^2) = 1. That scalar
    equation decreases monotonically in t, so a single vectorised bisection solves it
    for every point at once instead of iterating point by point.
    """
    squared = semi ** 2
    # Below -min(s^2) the expression blows up, so that is the hard lower bracket.
    low = np.full(len(points), -squared.min() * (1 - 1e-9))
    high = np.linalg.norm(points, axis=1) * semi.max() + squared.max()
    for _ in range(60):
        mid = 0.5 * (low + high)
        residual = (((points * semi) / (squared + mid[:, None])) ** 2).sum(1) - 1.0
        shrink = residual > 0
        low = np.where(shrink, mid, low)
        high = np.where(shrink, high, mid)
    t = 0.5 * (low + high)
    foot = points * squared / (squared + t[:, None])
    unsigned = np.linalg.norm(points - foot, axis=1)
    inside = ((points / semi) ** 2).sum(1) <= 1.0
    return np.where(inside, unsigned, -unsigned)


def fit_envelope(xyz, d_lamina, centroid, area):
    """Moves, tilts, and rescales the canonical nucleus to match the lamina distances.

    Nine free parameters: a 3D centre, three semi-axis scales, and a rotation vector.
    Anchor residuals pull the in-plane centre toward the measured cell centroid and
    each scale toward one, so the ellipsoid can only drift from the canonical nucleus
    as far as the distance data insists.
    """
    equivalent_radius = np.sqrt(area / np.pi)
    base = np.array([equivalent_radius, equivalent_radius, equivalent_radius * FLATTEN])
    start = np.concatenate([
        [centroid[0], centroid[1], np.median(xyz[:, 2])], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
    ])

    def depth_at(params):
        # Rotates the spots into the ellipsoid's own frame so the axis-aligned
        # depth solver above applies to a tilted nucleus.
        rotation = Rotation.from_rotvec(params[6:9])
        return ellipsoid_depth(rotation.apply(xyz - params[:3]), base * params[3:6])

    def residuals(params):
        anchor = np.concatenate([
            CENTROID_ANCHOR_WEIGHT * (params[0:2] - centroid),
            SCALE_ANCHOR_WEIGHT * (params[3:6] - 1.0),
        ])
        return np.concatenate([depth_at(params) - d_lamina, anchor])

    # Bounds are stated in microns, so each axis converts them into its own scale.
    scale_low = SEMI_AXIS_BOUNDS_UM[0] / base
    scale_high = SEMI_AXIS_BOUNDS_UM[1] / base
    low = [-np.inf] * 3 + list(scale_low) + [-np.pi] * 3
    high = [np.inf] * 3 + list(scale_high) + [np.pi] * 3
    fit = least_squares(
        residuals, start, bounds=(low, high), max_nfev=MAX_FIT_EVALUATIONS
    )

    rotation = Rotation.from_rotvec(fit.x[6:9])
    semi = base * fit.x[3:6]
    depth = depth_at(fit.x)
    # An axis resting on its bound means the distances never determined it. The shape
    # is still worth drawing, but it is a floor or ceiling rather than a measurement.
    at_bound = bool(
        np.any(semi <= SEMI_AXIS_BOUNDS_UM[0] * 1.01)
        or np.any(semi >= SEMI_AXIS_BOUNDS_UM[1] * 0.99)
    )
    return {
        "at_bound": at_bound,
        "center": fit.x[:3],
        "semi": semi,
        # Stored local-to-world so the browser can place surface points with one multiply.
        "rotation": rotation.inv().as_matrix(),
        "rms": float(np.sqrt(np.mean((depth - d_lamina) ** 2))),
        "fraction_inside": float((depth >= 0).mean()),
        "measured_radius": float(equivalent_radius),
        "center_shift": float(np.hypot(fit.x[0] - centroid[0], fit.x[1] - centroid[1])),
    }


def fit_nucleolus(xyz, d_nucleolus, envelope):
    """Moves and resizes a canonical sphere so |p - c| - R matches the distances.

    A sphere is a fair model here because the nucleolus is a compact, roughly round
    body. One extra residual pushes it back inside the fitted envelope, since a
    nucleolus poking through the nuclear wall is not a physical answer.
    """
    world_to_local = np.linalg.inv(envelope["rotation"])
    start = np.array([*xyz.mean(0), NUCLEOLUS_PRIOR_RADIUS])

    def residuals(params):
        predicted = np.linalg.norm(xyz - params[:3], axis=1) - params[3]
        local = world_to_local @ (params[:3] - envelope["center"])
        clearance = ellipsoid_depth(local[None, :], envelope["semi"])[0] - params[3]
        return np.concatenate([
            predicted - d_nucleolus,
            [NUCLEOLUS_PRIOR_WEIGHT * (params[3] - NUCLEOLUS_PRIOR_RADIUS)],
            [5.0 * min(clearance, 0.0)],
        ])

    fit = least_squares(
        residuals,
        start,
        bounds=([-np.inf] * 3 + [0.3], [np.inf] * 3 + [3.0]),
        max_nfev=MAX_FIT_EVALUATIONS,
    )
    predicted = np.linalg.norm(xyz - fit.x[:3], axis=1) - fit.x[3]
    rms = float(np.sqrt(np.mean((predicted - d_nucleolus) ** 2)))
    radius = float(fit.x[3])
    # chr19 only pins the nucleolus down when it actually comes close to it. When the
    # nearest spot is far away the radius drifts to its bound and the fit is a guess.
    trusted = (
        rms < MAX_TRUSTED_RMS
        and TRUSTED_RADIUS_RANGE[0] <= radius <= TRUSTED_RADIUS_RANGE[1]
    )
    return {
        "center": fit.x[:3],
        "radius": radius,
        "rms": rms,
        "trusted": bool(trusted),
        "closest_spot": float(d_nucleolus.min()),
    }


def main():
    spots, meta = load_sample()
    shapes = {}
    for cell_id in SAMPLE_CELLS:
        rows = spots[cell_id]
        xyz = rows[:, 0:3]
        envelope = fit_envelope(
            xyz, rows[:, 3], meta[cell_id]["centroid"], meta[cell_id]["area"]
        )
        nucleolus = fit_nucleolus(xyz, rows[:, 4], envelope)
        shapes[cell_id] = {
            "envelope": {
                "center": envelope["center"].tolist(),
                "semi": envelope["semi"].tolist(),
                "rotation": envelope["rotation"].tolist(),
                "rms": round(envelope["rms"], 3),
                "fractionInside": round(envelope["fraction_inside"], 3),
                "measuredRadius": round(envelope["measured_radius"], 3),
                "centerShift": round(envelope["center_shift"], 3),
                "atBound": envelope["at_bound"],
            },
            "nucleolus": {
                "center": nucleolus["center"].tolist(),
                "radius": round(nucleolus["radius"], 3),
                "rms": round(nucleolus["rms"], 3),
                "trusted": nucleolus["trusted"],
                "closestSpot": round(nucleolus["closest_spot"], 3),
            },
        }
        env, nuc = shapes[cell_id]["envelope"], shapes[cell_id]["nucleolus"]
        print(
            f"cell {cell_id}: envelope semi-axes "
            f"{env['semi'][0]:.2f}/{env['semi'][1]:.2f}/{env['semi'][2]:.2f} um, "
            f"RMS {env['rms']:.3f} um, {env['fractionInside'] * 100:.0f}% of spots "
            f"inside, centre {env['centerShift']:.2f} um from the measured centroid"
            f"{' [an axis rests on its size bound]' if env['atBound'] else ''}"
        )
        print(
            f"          nucleolus r={nuc['radius']:.2f} um, RMS {nuc['rms']:.3f} um, "
            f"closest spot {nuc['closestSpot']:.2f} um, "
            f"{'trusted' if nuc['trusted'] else 'NOT trusted (will be hidden)'}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(shapes, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
