"""Problem instances: a mixed-case SKU stream and perfect-packing puzzles.

- `random_boxes`: boxes drawn from a small SKU catalogue, the way mixed
  cases arrive at a palletizing cell in a distribution centre.
- `perfect_instance`: the pallet volume is cut by random guillotine cuts, so
  the boxes are known to fill it *exactly*. The optimum (100 %) is known
  without an exact solver, which makes optimality gaps measurable on any size.
"""

from __future__ import annotations

import numpy as np

from .geometry import Box, PalletSpec

# Carton sizes in millimetres (multiples of 20 mm), with arrival frequencies.
SKU_CATALOGUE: dict[str, tuple[tuple[int, int, int], float]] = {
    "A": ((400, 300, 200), 0.20),
    "B": ((300, 200, 160), 0.18),
    "C": ((500, 400, 300), 0.12),
    "D": ((600, 400, 240), 0.10),
    "E": ((300, 300, 300), 0.10),
    "F": ((260, 200, 120), 0.12),
    "G": ((400, 400, 200), 0.10),
    "H": ((360, 240, 220), 0.08),
}


def sku_box(sku: str, box_id: int, unit_mm: int = 20) -> Box:
    (l, w, h), _ = SKU_CATALOGUE[sku]
    return Box(box_id, l // unit_mm, w // unit_mm, h // unit_mm, sku)


def sample_skus(n: int, rng: np.random.Generator) -> list[str]:
    names = list(SKU_CATALOGUE)
    probs = np.array([SKU_CATALOGUE[s][1] for s in names])
    return list(rng.choice(names, size=n, p=probs / probs.sum()))


def random_boxes(n: int, seed: int | None = None, unit_mm: int = 20) -> list[Box]:
    rng = np.random.default_rng(seed)
    return [sku_box(s, i, unit_mm) for i, s in enumerate(sample_skus(n, rng))]


def perfect_instance(
    spec: PalletSpec, n_boxes: int, min_side: int = 6, seed: int | None = None
) -> list[Box]:
    """Cut the L × W × H block into `n_boxes` boxes that tile it exactly.

    The largest piece is split each time, along a random axis that can still
    be cut, at a random position leaving both parts ≥ `min_side`. Tiling a
    full block also means every box is 100 % supported in the reference
    packing, so a stable optimum is guaranteed to exist.
    """
    rng = np.random.default_rng(seed)
    pieces = [(spec.length, spec.width, spec.max_height)]
    while len(pieces) < n_boxes:
        cuttable = [d for d in pieces if max(d) >= 2 * min_side]
        if not cuttable:
            raise ValueError("min_side too large for this many boxes")
        dims = max(cuttable, key=lambda d: d[0] * d[1] * d[2])
        pieces.remove(dims)
        axes = [a for a in range(3) if dims[a] >= 2 * min_side]
        axis = int(rng.choice(axes))
        cut = int(rng.integers(min_side, dims[axis] - min_side + 1))
        first, second = list(dims), list(dims)
        first[axis], second[axis] = cut, dims[axis] - cut
        pieces += [tuple(first), tuple(second)]

    order = rng.permutation(len(pieces))
    boxes = []
    for i, k in enumerate(order):
        l, w, h = pieces[k]
        if rng.random() < 0.5:
            l, w = w, l  # hide the reference orientation
        boxes.append(Box(i, l, w, h, "cut"))
    return boxes
