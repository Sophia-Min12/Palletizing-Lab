"""Decoder and constructive heuristics.

The decoder turns a box *order* into a plan: each box goes to its best DBLF
position over both orientations, or is left off the pallet when no stable
position fits under the height limit. Every offline method in this package
searches over orders and lets this decoder do the geometry — the classic
"sequence + decoder" split used in industrial packing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .geometry import Box, PalletSpec, Placement, load_height, utilization
from .heightmap import HeightMap


@dataclass
class PackResult:
    plan: list[Placement]
    unpacked: list[int]
    utilization: float
    height: int

    @property
    def objective(self) -> float:
        """F to minimize: unused volume, with load height as a tie-breaker."""
        return (1.0 - self.utilization) + 1e-4 * self.height


def decode(
    spec: PalletSpec,
    boxes: Sequence[Box],
    order: Sequence[int] | None = None,
    placement: str = "dblf",
) -> PackResult:
    """Place `boxes` in `order` (indices into `boxes`; default: as given),
    each at the best position under the `placement` rule.
    """
    hm = HeightMap(spec)
    plan, unpacked = [], []
    for i in order if order is not None else range(len(boxes)):
        p = hm.best_placement(boxes[i], placement)
        if p is None:
            unpacked.append(boxes[i].id)
        else:
            hm.place(p)
            plan.append(p)
    return PackResult(plan, unpacked, utilization(spec, plan), load_height(plan))


SORT_RULES: dict[str, Callable[[Box], tuple]] = {
    "volume": lambda b: (-b.volume,),
    "base_area": lambda b: (-b.l * b.w, -b.h),
    # Same-height boxes end up side by side, which builds flat layers.
    "height_area": lambda b: (-b.h, -b.l * b.w),
}


def greedy(
    spec: PalletSpec, boxes: Sequence[Box], sort_by: str = "height_area", placement: str = "dblf"
) -> PackResult:
    order = sorted(range(len(boxes)), key=lambda i: SORT_RULES[sort_by](boxes[i]))
    return decode(spec, boxes, order, placement)


def best_greedy(spec: PalletSpec, boxes: Sequence[Box], placement: str = "dblf") -> PackResult:
    """Best of all sort rules — the baseline every search method must beat."""
    return min((greedy(spec, boxes, s, placement) for s in SORT_RULES), key=lambda r: r.objective)
