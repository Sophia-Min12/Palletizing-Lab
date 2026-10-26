"""Robot task sequencing: in which order should the robot build a given plan?

The stacking plan fixes *where* every box goes. Boxes arrive on several
infeed conveyors, so the robot's empty moves — from where it just placed a
box to the conveyor holding the next one — depend on the order. Two rules
restrict the order:

- a box must come after every box under its footprint (support, and the
  robot places from above), and
- that is all: any order respecting these precedences builds a valid plan.

Minimizing travel under precedences is the Sequential Ordering Problem (an
asymmetric TSP with precedence constraints). Solvers here: the naive
layer-by-layer order, nearest-feasible-neighbour, simulated annealing over
precedence-feasible orders, and an exact CP-SAT model with AddCircuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ortools.sat.python import cp_model

from .geometry import Box, PalletSpec, Placement, footprint_overlap

Point = tuple[float, float, float]


@dataclass(frozen=True)
class Cell:
    """Robot cell layout in millimetres; the pallet's corner is the origin."""

    home: Point = (550.0, -500.0, 1400.0)
    stations: tuple[Point, ...] = (
        (-700.0, 250.0, 750.0),  # infeed conveyor 0 (west, near)
        (-700.0, 850.0, 750.0),  # infeed conveyor 1 (west, far)
        (550.0, -700.0, 750.0),  # infeed conveyor 2 (south)
    )
    pallet_surface_mm: float = 150.0  # height of the empty pallet deck


DEFAULT_CELL = Cell()


def assign_lanes(boxes: Sequence[Box], n_lanes: int) -> dict[int, int]:
    """Which conveyor each box arrives on: one lane per SKU, round robin."""
    skus = sorted({b.sku for b in boxes})
    if len(skus) <= 1:  # unlabelled boxes: spread by id
        return {b.id: b.id % n_lanes for b in boxes}
    return {b.id: skus.index(b.sku) % n_lanes for b in boxes}


def place_point(p: Placement, spec: PalletSpec, cell: Cell) -> Point:
    u = spec.unit_mm
    return ((p.x + p.dx / 2) * u, (p.y + p.dy / 2) * u, cell.pallet_surface_mm + p.top * u)


def predecessors(plan: Sequence[Placement]) -> list[set[int]]:
    """preds[i] = indices of boxes that must be placed before plan[i]."""
    return [
        {j for j, q in enumerate(plan) if q.z < p.z and footprint_overlap(p, q) > 0}
        for p in plan
    ]


@dataclass
class SequencingProblem:
    plan: list[Placement]
    preds: list[set[int]]
    # cost[i][j]: empty move after placing i to pick j; index n is home
    cost: np.ndarray
    loaded: float  # pick → place moves, the same for every order

    @property
    def n(self) -> int:
        return len(self.plan)

    def travel(self, order: Sequence[int]) -> float:
        """Total robot travel (mm) for building the plan in `order`."""
        n = self.n
        legs = [self.cost[n, order[0]]]
        legs += [self.cost[a, b] for a, b in zip(order, order[1:])]
        legs.append(self.cost[order[-1], n])  # back home
        return float(sum(legs) + self.loaded)

    def is_feasible(self, order: Sequence[int]) -> bool:
        seen: set[int] = set()
        for i in order:
            if not self.preds[i] <= seen:
                return False
            seen.add(i)
        return len(seen) == self.n


def build_problem(
    spec: PalletSpec,
    plan: Sequence[Placement],
    lanes: dict[int, int],
    cell: Cell = DEFAULT_CELL,
) -> SequencingProblem:
    n = len(plan)
    picks = [cell.stations[lanes[p.box_id]] for p in plan]
    places = [place_point(p, spec, cell) for p in plan]
    cost = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            cost[i, j] = math.dist(places[i], picks[j])
        cost[n, i] = math.dist(cell.home, picks[i])
        cost[i, n] = math.dist(places[i], cell.home)
    loaded = sum(math.dist(a, b) for a, b in zip(picks, places))
    return SequencingProblem(list(plan), predecessors(plan), cost, loaded)


def layer_order(prob: SequencingProblem) -> list[int]:
    """Naive baseline: bottom to top, then by position."""
    return sorted(range(prob.n), key=lambda i: (prob.plan[i].z, prob.plan[i].x, prob.plan[i].y))


def nearest_feasible(prob: SequencingProblem) -> list[int]:
    """Greedy: from where the robot is, fetch the closest box it may place."""
    n, done, order, at = prob.n, set(), [], prob.n
    while len(order) < n:
        ready = [i for i in range(n) if i not in done and prob.preds[i] <= done]
        nxt = min(ready, key=lambda i: prob.cost[at, i])
        order.append(nxt)
        done.add(nxt)
        at = nxt
    return order


def anneal_order(
    prob: SequencingProblem,
    start: Sequence[int] | None = None,
    iterations: int = 20000,
    t_initial: float = 300.0,
    t_final: float = 1.0,
    seed: int | None = None,
) -> list[int]:
    """SA whose only move keeps the order feasible: lift one box out and
    reinsert it anywhere between its last predecessor and first successor.
    """
    rng = np.random.default_rng(seed)
    order = list(start) if start is not None else nearest_feasible(prob)
    succs = [{i for i in range(prob.n) if j in prob.preds[i]} for j in range(prob.n)]
    cur = prob.travel(order)
    best, best_cost = order.copy(), cur
    cooling = (t_final / t_initial) ** (1.0 / iterations)
    t = t_initial
    for _ in range(iterations):
        k = int(rng.integers(prob.n))
        item = order[k]
        rest = order[:k] + order[k + 1 :]
        pos = {v: idx for idx, v in enumerate(rest)}
        lo = max((pos[p] + 1 for p in prob.preds[item]), default=0)
        hi = min((pos[s] for s in succs[item]), default=len(rest))
        new = rest.copy()
        new.insert(int(rng.integers(lo, hi + 1)), item)
        c = prob.travel(new)
        if c <= cur or rng.random() < math.exp((cur - c) / t):
            order, cur = new, c
            if cur < best_cost:
                best, best_cost = order.copy(), cur
        t *= cooling
    return best


@dataclass
class ExactOrder:
    order: list[int]
    status: str
    travel: float
    lower_bound: float


def cpsat_order(
    prob: SequencingProblem,
    time_limit: float = 20.0,
    hint: Sequence[int] | None = None,
    seed: int = 0,
) -> ExactOrder:
    """Exact SOP: a Hamiltonian circuit through home and every box
    (AddCircuit), with ranks that must respect the precedences.
    """
    n = prob.n
    m = cp_model.CpModel()
    cost = np.rint(prob.cost).astype(int)
    # transitive closure, so arcs that would break a precedence are never built
    before = [set(p) for p in prob.preds]
    changed = True
    while changed:
        changed = False
        for i in range(n):
            closure = before[i].union(*(before[j] for j in before[i]))
            if closure != before[i]:
                before[i], changed = closure, True

    rank = [m.new_int_var(1, n, f"rank{i}") for i in range(n)]
    arcs, terms, lit = [], [], {}
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j or (i < n and j < n and j in before[i]):
                continue
            b = m.new_bool_var(f"arc{i}_{j}")
            lit[i, j] = b
            arcs.append((i, j, b))
            terms.append(int(cost[i, j]) * b)
            if i == n:
                m.add(rank[j] == 1).only_enforce_if(b)
            elif j < n:
                m.add(rank[j] == rank[i] + 1).only_enforce_if(b)
    for i in range(n):
        for j in prob.preds[i]:
            m.add(rank[j] < rank[i])
    m.add_circuit(arcs)
    m.minimize(sum(terms))

    if hint is not None:
        seq = [n, *hint, n]
        used = set(zip(seq, seq[1:]))
        for key, b in lit.items():
            m.add_hint(b, key in used)
        for r, i in enumerate(hint, start=1):
            m.add_hint(rank[i], r)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = seed
    status = solver.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactOrder([], solver.status_name(status), math.inf, solver.best_objective_bound)
    order = sorted(range(n), key=lambda i: solver.value(rank[i]))
    return ExactOrder(
        order,
        solver.status_name(status),
        prob.travel(order),
        solver.best_objective_bound + prob.loaded,
    )
