"""Boxes, placements, and an independent feasibility checker.

All lengths are integers on a grid (`PalletSpec.unit_mm` millimetres per
unit), so feasibility is decided exactly — no floating-point tolerances.

A plan is an *ordered* list of placements: the order is the order in which a
robot puts the boxes down, from above. A plan is feasible when every box

1. lies inside the pallet and below the height limit,
2. overlaps no other box,
3. is placed from above: no earlier box sits over its footprint,
4. is stable when placed: it rests on the floor, or at least a fraction `α`
   of its bottom face rests on the tops of boxes placed before it.

Why α > 0.5 is enough for rigid boxes: any line through the centre of a
rectangle splits its area exactly in half, so a support region covering more
than half of the bottom face cannot lie on one side of such a line. The
centre of gravity therefore lies inside the convex hull of the support — the
static tipping condition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PalletSpec:
    length: int  # x extent, grid units
    width: int  # y extent
    max_height: int  # load height limit
    unit_mm: int = 20
    support_ratio: float = 0.75  # α

    @property
    def volume(self) -> int:
        return self.length * self.width * self.max_height


# Korean standard pallet T-11: 1100 × 1100 mm, loaded up to 1200 mm, 20 mm grid.
T11 = PalletSpec(length=55, width=55, max_height=60)


@dataclass(frozen=True)
class Box:
    id: int
    l: int
    w: int
    h: int
    sku: str = ""

    @property
    def volume(self) -> int:
        return self.l * self.w * self.h

    def orientations(self) -> list[tuple[int, int]]:
        """Footprints (dx, dy) reachable by yawing 90°; 'this side up' is kept."""
        return [(self.l, self.w)] if self.l == self.w else [(self.l, self.w), (self.w, self.l)]


@dataclass(frozen=True)
class Placement:
    box_id: int
    x: int
    y: int
    z: int
    dx: int  # extent along x after rotation
    dy: int
    dz: int

    @property
    def top(self) -> int:
        return self.z + self.dz

    @property
    def volume(self) -> int:
        return self.dx * self.dy * self.dz

    @property
    def footprint_area(self) -> int:
        return self.dx * self.dy


def footprint_overlap(a: Placement, b: Placement) -> int:
    """Area shared by the two footprints when projected onto the floor."""
    ox = min(a.x + a.dx, b.x + b.dx) - max(a.x, b.x)
    oy = min(a.y + a.dy, b.y + b.dy) - max(a.y, b.y)
    return max(ox, 0) * max(oy, 0)


def boxes_overlap(a: Placement, b: Placement) -> bool:
    """True when the open boxes intersect (touching faces is allowed)."""
    return (
        footprint_overlap(a, b) > 0
        and a.z < b.z + b.dz
        and b.z < a.z + a.dz
    )


def support_area(p: Placement, below: list[Placement]) -> int:
    """Area of p's bottom face resting on tops of `below` at exactly p.z."""
    return sum(footprint_overlap(p, q) for q in below if q.top == p.z)


def validate_plan(spec: PalletSpec, plan: list[Placement]) -> list[str]:
    """Return a list of human-readable violations; empty means feasible.

    Deliberately written from scratch with explicit geometry, independent of
    the heightmap used by the planners, so it can catch their bugs.
    """
    errors: list[str] = []
    seen: set[int] = set()
    for k, p in enumerate(plan):
        tag = f"box {p.box_id} (step {k})"
        if p.box_id in seen:
            errors.append(f"{tag}: placed twice")
        seen.add(p.box_id)
        if p.x < 0 or p.y < 0 or p.z < 0:
            errors.append(f"{tag}: negative coordinate")
        if p.x + p.dx > spec.length or p.y + p.dy > spec.width:
            errors.append(f"{tag}: outside the pallet footprint")
        if p.top > spec.max_height:
            errors.append(f"{tag}: above the height limit")

        earlier = plan[:k]
        for q in earlier:
            if boxes_overlap(p, q):
                errors.append(f"{tag}: overlaps box {q.box_id}")
            elif q.z >= p.top and footprint_overlap(p, q) > 0:
                errors.append(f"{tag}: blocked from above by box {q.box_id}")
        if p.z > 0:
            ratio = support_area(p, earlier) / p.footprint_area
            if ratio < spec.support_ratio:
                errors.append(f"{tag}: only {ratio:.0%} supported (< {spec.support_ratio:.0%})")
    return errors


def utilization(spec: PalletSpec, plan: list[Placement]) -> float:
    """Packed volume as a fraction of the usable pallet volume."""
    return sum(p.volume for p in plan) / spec.volume


def load_height(plan: list[Placement]) -> int:
    return max((p.top for p in plan), default=0)
