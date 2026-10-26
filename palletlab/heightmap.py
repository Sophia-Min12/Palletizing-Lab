"""Heightmap state and the deepest-bottom-left (DBLF) placement rule.

A robot puts boxes down from above, so the only part of the load that matters
for the next placement is its top surface: `hm[x, y]` is the height of the
highest box covering cell (x, y). Dropping a box with footprint dx × dy at
(x, y) makes it rest at z = max(hm over the footprint), and the cells where
hm == z are exactly the ones supporting it.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .geometry import Box, PalletSpec, Placement


class HeightMap:
    def __init__(self, spec: PalletSpec, heights: np.ndarray | None = None) -> None:
        self.spec = spec
        self.heights = (
            np.zeros((spec.length, spec.width), dtype=np.int32)
            if heights is None
            else heights
        )

    def copy(self) -> HeightMap:
        return HeightMap(self.spec, self.heights.copy())

    @property
    def max_height(self) -> int:
        return int(self.heights.max())

    def rest_heights(self, dx: int, dy: int) -> np.ndarray:
        """z at which a dx × dy footprint would rest, for every corner (x, y).

        A 2-D sliding max is separable — max over rows, then over columns —
        so this costs O(L·W·(dx + dy)) instead of O(L·W·dx·dy).
        """
        rows = sliding_window_view(self.heights, dx, axis=0).max(axis=-1)
        return sliding_window_view(rows, dy, axis=1).max(axis=-1)

    def window_sums(self, a: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Sum of `a` over every dx × dy window, via a summed-area table."""
        integral = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
        integral[1:, 1:] = a.cumsum(0).cumsum(1)
        return (
            integral[dx:, dy:] - integral[:-dx, dy:]
            - integral[dx:, :-dy] + integral[:-dx, :-dy]
        )

    def positions(
        self, dx: int, dy: int, dz: int, limit: int = 1, rule: str = "dblf"
    ) -> list[tuple[tuple, int, int, int]]:
        """Up to `limit` stable, in-limit positions as (key, z, x, y), best first.

        Rules (`key` is what is minimized):
        - "dblf": deepest-bottom-left — (z, x, y).
        - "min_waste": (air trapped under the box, z, x, y). The trapped air
          is dx·dy·z − Σ hm over the footprint: space that no later box can
          ever fill, because boxes only arrive from above.
        """
        spec = self.spec
        if dx > spec.length or dy > spec.width:
            return []
        z = self.rest_heights(dx, dy)
        ok = z + dz <= spec.max_height
        if not ok.any():
            return []
        needed = math.ceil(spec.support_ratio * dx * dy - 1e-9)
        stable = ok & (z == 0)
        found: list[tuple[tuple, int, int, int]] = []
        # Walk rest heights from the lowest up. At level v the support of a
        # window is its number of cells at exactly v, which one summed-area
        # table of (hm == v) gives for all windows at once.
        for v in np.unique(z[ok]):
            if v > 0:
                at_v = ok & (z == v)
                stable |= at_v & (self.window_sums(self.heights == v, dx, dy) >= needed)
            if rule == "dblf":  # lower levels always win, so stop early
                for x, y in zip(*np.nonzero(stable & (z == v))):  # row-major = (x, y) order
                    found.append(((int(v), int(x), int(y)), int(v), int(x), int(y)))
                    if len(found) == limit:
                        return found
        if rule == "dblf":
            return found
        if rule != "min_waste":
            raise ValueError(f"unknown placement rule {rule!r}")

        xs, ys = np.nonzero(stable)
        zs = z[xs, ys]
        waste = dx * dy * zs - self.window_sums(self.heights, dx, dy)[xs, ys]
        for k in np.lexsort((ys, xs, zs, waste))[:limit]:
            key = (int(waste[k]), int(zs[k]), int(xs[k]), int(ys[k]))
            found.append((key, int(zs[k]), int(xs[k]), int(ys[k])))
        return found

    def candidates(self, box: Box, per_orientation: int = 1, rule: str = "dblf") -> list[Placement]:
        """Best stable placements of `box` over both yaw orientations, best first."""
        scored = []
        for dx, dy in box.orientations():
            for key, z, x, y in self.positions(dx, dy, box.h, per_orientation, rule):
                scored.append((key, Placement(box.id, x, y, z, dx, dy, box.h)))
        scored.sort(key=lambda kp: kp[0])
        return [p for _, p in scored]

    def best_placement(self, box: Box, rule: str = "dblf") -> Placement | None:
        found = self.candidates(box, per_orientation=1, rule=rule)
        return found[0] if found else None

    def trapped_air(self, p: Placement) -> int:
        """Empty volume left under `p` if it were placed now."""
        footprint = self.heights[p.x : p.x + p.dx, p.y : p.y + p.dy]
        return int(p.dx * p.dy * p.z - footprint.sum())

    def place(self, p: Placement) -> None:
        region = self.heights[p.x : p.x + p.dx, p.y : p.y + p.dy]
        assert region.max() == p.z, "placement must rest on the current surface"
        region[...] = p.top
