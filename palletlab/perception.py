"""From sensor data to the optimizer's state.

Two small pipelines turn raw 3-D data into the inputs the planners need:

1. **Box on the infeed conveyor → (l, w, h, yaw).** An overhead depth camera
   sees the conveyor and the top face of the box. RANSAC finds the conveyor
   plane, points well above it are the box, the height is the top face's
   distance to the plane, stray returns are dropped by a radius outlier
   filter, and PCA of the top face (refined by a minimum-area rectangle
   search) gives the yaw and the footprint. The result is snapped to the 20 mm grid and matched to the
   nearest catalogue SKU.
2. **Depth image of the pallet → heightmap.** A per-cell median over the
   depth pixels removes sensor noise before rounding to the grid.

The point clouds are synthetic (known ground truth), so the error of every
step can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import PalletSpec, Placement
from .instances import SKU_CATALOGUE


# ---------------------------------------------------------------- synthetic data


def synthetic_box_cloud(
    dims_mm: tuple[float, float, float],
    yaw_rad: float,
    center_xy_mm: tuple[float, float] = (0.0, 0.0),
    noise_mm: float = 2.0,
    points_per_m2: float = 10000.0,  # ≈ one point per cm²
    outlier_frac: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Points (N × 3, mm) an overhead camera returns: a 1.2 m × 1.2 m patch of
    conveyor at z = 0 with the box's top face at z = h, plus noise and outliers.
    """
    rng = rng or np.random.default_rng()
    l, w, h = dims_mm
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    rot = np.array([[c, -s], [s, c]])

    n_top = int(points_per_m2 * l * w / 1e6)
    local = rng.uniform([-l / 2, -w / 2], [l / 2, w / 2], size=(n_top, 2))
    top = np.column_stack([local @ rot.T + center_xy_mm, np.full(n_top, h)])

    n_floor = int(points_per_m2 * 1.44)
    floor_xy = rng.uniform(-600, 600, size=(n_floor, 2)) + center_xy_mm
    inside = np.all(np.abs((floor_xy - center_xy_mm) @ rot) <= [l / 2, w / 2], axis=1)
    floor = np.column_stack([floor_xy[~inside], np.zeros((~inside).sum())])

    cloud = np.vstack([top, floor])
    cloud += rng.normal(0.0, noise_mm, size=cloud.shape)
    n_out = int(outlier_frac * len(cloud))
    cloud[rng.choice(len(cloud), n_out, replace=False), 2] += rng.uniform(-300, 300, n_out)
    return cloud


# ---------------------------------------------------------------- estimation


def ransac_plane(
    points: np.ndarray,
    threshold_mm: float = 6.0,
    iterations: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Largest plane n·p + d = 0 (unit n, pointing up), refined by least
    squares on its inliers. Returns (n, d, inlier mask).
    """
    rng = rng or np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs((points - a) @ normal) < threshold_mm
        if inliers.sum() > best.sum():
            best = inliers
    centroid = points[best].mean(axis=0)
    # least-squares normal: eigenvector of the smallest eigenvalue of the 3 × 3
    # scatter matrix (an SVD of the N × 3 points would build an N × N matrix)
    normal = np.linalg.eigh(np.cov((points[best] - centroid).T))[1][:, 0]
    if normal[2] < 0:
        normal = -normal
    return normal, float(-normal @ centroid), best


def radius_outlier_filter(xy: np.ndarray, radius: float = 30.0, min_neighbors: int = 5) -> np.ndarray:
    """Drop points with fewer than `min_neighbors` others within `radius` mm
    (the idea of PCL's RadiusOutlierRemoval). A real surface at ~1 point/cm²
    gives each point ~28 neighbours inside 30 mm; a stray return gives ~0.
    """
    keep = np.zeros(len(xy), dtype=bool)
    for start in range(0, len(xy), 512):  # chunks keep the distance matrix small
        block = xy[start : start + 512]
        d2 = ((block[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
        keep[start : start + 512] = (d2 < radius**2).sum(axis=1) - 1 >= min_neighbors
    return xy[keep]


def refine_yaw(centered: np.ndarray, yaw0: float, span_deg: float = 12.0, step_deg: float = 0.5) -> float:
    """Minimum-area bounding rectangle near the PCA guess. PCA axes wobble
    when the face is close to square; the tightest rectangle does not.
    """
    best_yaw, best_area = yaw0, np.inf
    for delta in np.radians(np.arange(-span_deg, span_deg + 1e-9, step_deg)):
        a = yaw0 + delta
        proj = centered @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        lo, hi = np.percentile(proj, [1, 99], axis=0)
        area = float(np.prod(hi - lo))
        if area < best_area:
            best_yaw, best_area = a, area
    return best_yaw


@dataclass(frozen=True)
class BoxEstimate:
    l_mm: float
    w_mm: float
    h_mm: float
    yaw_rad: float  # of the long side, in [0, π)
    sku: str


def estimate_box(cloud: np.ndarray, unit_mm: int = 20, rng: np.random.Generator | None = None) -> BoxEstimate:
    normal, d, on_plane = ransac_plane(cloud, rng=rng)
    height = cloud @ normal + d  # signed distance to the conveyor
    above = (~on_plane) & (height > 30.0)
    # the top face is the dominant level among the points above the plane
    top_level = np.median(height[above])
    top = above & (np.abs(height - top_level) < 10.0)
    h = float(np.median(height[top]))

    xy = radius_outlier_filter(cloud[top, :2])
    centered = xy - xy.mean(axis=0)
    # Points on a rectangle are uniform, and a uniform distribution of
    # width a has variance a²/12 — so each side is √(12 · variance) along
    # its principal axis. Unlike min/max or percentiles, this barely moves
    # with noise on the edges.
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered.T))  # ascending
    major = eigvecs[:, 1]
    yaw = refine_yaw(centered, float(np.arctan2(major[1], major[0])))
    c, s_ = np.cos(yaw), np.sin(yaw)
    aligned = centered @ np.array([[c, -s_], [s_, c]])  # rotate into the box frame
    l, w = np.sqrt(12.0 * aligned.var(axis=0)).tolist()
    if w > l:
        l, w, yaw = w, l, yaw + np.pi / 2
    yaw = float(yaw % np.pi)

    snap = lambda v: unit_mm * round(v / unit_mm)
    l, w, h = snap(max(l, w)), snap(min(l, w)), snap(h)
    sku = min(
        SKU_CATALOGUE,
        key=lambda k: np.abs(np.array(sorted(SKU_CATALOGUE[k][0][:2], reverse=True) + [SKU_CATALOGUE[k][0][2]]) - [l, w, h]).sum(),
    )
    return BoxEstimate(l, w, h, yaw, sku)


# ---------------------------------------------------------------- pallet heightmap


def render_depth(spec: PalletSpec, plan: list[Placement], px_per_unit: int = 4) -> np.ndarray:
    """Ideal top-down height image (mm) of a load, `px_per_unit` pixels per cell."""
    true_hm = np.zeros((spec.length, spec.width))
    for p in plan:
        true_hm[p.x : p.x + p.dx, p.y : p.y + p.dy] = p.top
    return np.kron(true_hm * spec.unit_mm, np.ones((px_per_unit, px_per_unit)))


def heightmap_from_depth(depth_mm: np.ndarray, spec: PalletSpec, px_per_unit: int = 4) -> np.ndarray:
    """Per-cell median of the depth pixels, rounded to whole grid units."""
    cells = depth_mm.reshape(spec.length, px_per_unit, spec.width, px_per_unit)
    medians = np.median(cells, axis=(1, 3))
    return np.rint(medians / spec.unit_mm).astype(np.int32)
