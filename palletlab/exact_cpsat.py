"""Exact palletizing with OR-Tools CP-SAT — the same rules, written as a model.

    maximize    Σᵢ volᵢ · bᵢ                          (packed volume)
    subject to  bᵢ ⇒ box i inside the pallet, below the height limit
                bᵢ ∧ bⱼ ⇒ i left of / right of / in front of / behind /
                          below / above j            (6-way disjunction)
                bᵢ ∧ zᵢ > 0 ⇒ Σⱼ overlapⱼᵢ · sⱼᵢ ≥ α · areaᵢ
                sⱼᵢ ⇒ bⱼ ∧ zᵢ = zⱼ + hⱼ               (j holds i up)

`overlapⱼᵢ` is the product of the x- and y-overlap lengths, each a
max(0, min(…) − max(…)), so the model needs CP-SAT's min / max /
multiplication constraints — a MILP solver would need big-M linearisations
of the same thing.

Any stable static packing can be built from above by placing boxes in order
of z: a box above i overlapping its footprint has a larger z, and boxes
supporting i end exactly at zᵢ. So the returned plan is sorted by z.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

from ortools.sat.python import cp_model

from .geometry import Box, PalletSpec, Placement, utilization


@dataclass
class ExactResult:
    plan: list[Placement]
    status: str  # OPTIMAL, FEASIBLE, or INFEASIBLE / UNKNOWN
    utilization: float
    upper_bound: float  # best proven bound on utilization
    seconds: float


def solve_exact(
    spec: PalletSpec,
    boxes: Sequence[Box],
    time_limit: float = 30.0,
    hint: Sequence[Placement] | None = None,
    workers: int = 8,
    seed: int = 0,
) -> ExactResult:
    m = cp_model.CpModel()
    L, W, H = spec.length, spec.width, spec.max_height
    n = len(boxes)
    alpha_pct = math.ceil(100 * spec.support_ratio - 1e-9)

    b, x, y, z, r, xe, ye, on_floor = [], [], [], [], [], [], [], []
    for i, bx in enumerate(boxes):
        b.append(m.new_bool_var(f"b{i}"))
        r.append(m.new_bool_var(f"r{i}") if bx.l != bx.w else m.new_constant(0))
        x.append(m.new_int_var(0, L, f"x{i}"))
        y.append(m.new_int_var(0, W, f"y{i}"))
        z.append(m.new_int_var(0, H, f"z{i}"))
        on_floor.append(m.new_bool_var(f"f{i}"))
        # extents after an optional 90° yaw
        dx = bx.l + (bx.w - bx.l) * r[i]
        dy = bx.w + (bx.l - bx.w) * r[i]
        xe.append(m.new_int_var(0, 2 * L, f"xe{i}"))
        ye.append(m.new_int_var(0, 2 * W, f"ye{i}"))
        m.add(xe[i] == x[i] + dx)
        m.add(ye[i] == y[i] + dy)
        m.add(xe[i] <= L).only_enforce_if(b[i])
        m.add(ye[i] <= W).only_enforce_if(b[i])
        m.add(z[i] + bx.h <= H).only_enforce_if(b[i])
        m.add(z[i] == 0).only_enforce_if(on_floor[i])
        # pin unused boxes so the solver does not branch on them
        for v in (x[i], y[i], z[i], r[i]):
            m.add(v == 0).only_enforce_if(~b[i])

    # no overlap: at least one of six separations for every placed pair
    for i in range(n):
        for j in range(i + 1, n):
            sep = [m.new_bool_var(f"sep{i}_{j}_{k}") for k in range(6)]
            m.add(xe[i] <= x[j]).only_enforce_if(sep[0])
            m.add(xe[j] <= x[i]).only_enforce_if(sep[1])
            m.add(ye[i] <= y[j]).only_enforce_if(sep[2])
            m.add(ye[j] <= y[i]).only_enforce_if(sep[3])
            m.add(z[i] + boxes[i].h <= z[j]).only_enforce_if(sep[4])
            m.add(z[j] + boxes[j].h <= z[i]).only_enforce_if(sep[5])
            m.add_bool_or(sep + [~b[i], ~b[j]])

    # support: α of the bottom face must rest on boxes ending exactly at zᵢ
    for i in range(n):
        area_i = boxes[i].l * boxes[i].w  # unchanged by yaw
        contributions = []
        for j in range(n):
            if j == i:
                continue
            s = m.new_bool_var(f"s{j}_{i}")
            m.add_implication(s, b[j])
            m.add(z[i] == z[j] + boxes[j].h).only_enforce_if(s)
            overlaps = []
            for lo_i, lo_j, hi_i, hi_j, size in ((x[i], x[j], xe[i], xe[j], L), (y[i], y[j], ye[i], ye[j], W)):
                hi = m.new_int_var(0, 2 * size, "")
                lo = m.new_int_var(0, size, "")
                m.add_min_equality(hi, [hi_i, hi_j])
                m.add_max_equality(lo, [lo_i, lo_j])
                diff = m.new_int_var(-size, 2 * size, "")
                m.add(diff == hi - lo)
                o = m.new_int_var(0, size, "")
                m.add_max_equality(o, [diff, 0])
                overlaps.append(o)
            area = m.new_int_var(0, L * W, f"a{j}_{i}")
            m.add_multiplication_equality(area, overlaps)
            c = m.new_int_var(0, area_i, f"c{j}_{i}")
            m.add(c <= area)
            m.add(c <= area_i * s)
            contributions.append(c)
        m.add(100 * sum(contributions) >= alpha_pct * area_i).only_enforce_if(
            [b[i], ~on_floor[i]]
        )

    m.maximize(sum(bx.volume * b[i] for i, bx in enumerate(boxes)))

    if hint:
        index = {bx.id: i for i, bx in enumerate(boxes)}
        placed = {p.box_id: p for p in hint}
        for bx in boxes:
            i = index[bx.id]
            p = placed.get(bx.id)
            m.add_hint(b[i], p is not None)
            if p is not None:
                m.add_hint(x[i], p.x)
                m.add_hint(y[i], p.y)
                m.add_hint(z[i], p.z)
                if bx.l != bx.w:
                    m.add_hint(r[i], p.dx != bx.l)
                m.add_hint(on_floor[i], p.z == 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed  # parallel search still varies run to run
    start = time.perf_counter()
    status = solver.solve(m)
    seconds = time.perf_counter() - start

    plan: list[Placement] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, bx in enumerate(boxes):
            if solver.value(b[i]):
                rotated = bool(solver.value(r[i]))
                dx, dy = (bx.w, bx.l) if rotated else (bx.l, bx.w)
                plan.append(
                    Placement(bx.id, solver.value(x[i]), solver.value(y[i]), solver.value(z[i]), dx, dy, bx.h)
                )
        plan.sort(key=lambda p: (p.z, p.x, p.y))
    return ExactResult(
        plan,
        solver.status_name(status),
        utilization(spec, plan),
        solver.best_objective_bound / spec.volume,
        seconds,
    )
