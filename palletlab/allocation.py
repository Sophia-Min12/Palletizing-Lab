"""Task allocation for a two-robot palletizing cell.

Two robots stand on opposite sides of one pallet, each fed by its own infeed
conveyor (an upstream diverter sends every box to one of them). Deciding
which robot places which box, and when, is a scheduling problem:

- each box is placed by exactly one robot that can reach its target;
- a robot does one task at a time;
- a box is placed only after every box under its footprint is in place;
- the middle strip both robots can reach is a *shared zone*: two tasks
  placing boxes there never run at the same time (collision avoidance);
- minimize the makespan — the time until the pallet is complete.

Solved by list scheduling (the usual dispatcher logic) and exactly by
CP-SAT with optional intervals, NoOverlap, and precedences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ortools.sat.python import cp_model

from .geometry import PalletSpec, Placement
from .sequencing import DEFAULT_CELL, Cell, Point, place_point, predecessors

TICKS_PER_S = 10  # schedule resolution: 0.1 s


@dataclass(frozen=True)
class Robot:
    name: str
    infeed: Point  # pick point of this robot's conveyor (mm)
    reach_x: tuple[float, float]  # x-range (mm) of targets it can reach
    speed_mm_s: float = 1500.0
    handling_s: float = 3.0  # grip + release per box


TWO_ROBOT_CELL: tuple[Robot, Robot] = (
    Robot("west", infeed=(-700.0, 550.0, 750.0), reach_x=(0.0, 750.0)),
    Robot("east", infeed=(1800.0, 550.0, 750.0), reach_x=(350.0, 1100.0)),
)


@dataclass
class AllocationProblem:
    plan: list[Placement]
    preds: list[set[int]]
    # durations[t][r] in ticks, or None when robot r cannot reach task t
    durations: list[list[int | None]]
    shared: list[bool]  # task t places a box inside the shared zone
    robots: Sequence[Robot]

    @property
    def n(self) -> int:
        return len(self.plan)


def build_allocation(
    spec: PalletSpec,
    plan: Sequence[Placement],
    robots: Sequence[Robot] = TWO_ROBOT_CELL,
    cell: Cell = DEFAULT_CELL,
) -> AllocationProblem:
    """A task's duration on robot r: infeed → target → infeed, plus handling."""
    durations, shared = [], []
    lo_shared = max(r.reach_x[0] for r in robots)
    hi_shared = min(r.reach_x[1] for r in robots)
    for p in plan:
        target = place_point(p, spec, cell)
        row = []
        for r in robots:
            if r.reach_x[0] <= target[0] <= r.reach_x[1]:
                seconds = 2 * math.dist(r.infeed, target) / r.speed_mm_s + r.handling_s
                row.append(math.ceil(seconds * TICKS_PER_S))
            else:
                row.append(None)
        if all(d is None for d in row):
            raise ValueError(f"no robot reaches box {p.box_id}")
        durations.append(row)
        shared.append(lo_shared <= target[0] <= hi_shared)
    return AllocationProblem(list(plan), predecessors(plan), durations, shared, robots)


@dataclass
class Schedule:
    start: list[int]
    robot: list[int]
    makespan: int
    status: str = "HEURISTIC"
    lower_bound: int = 0

    def end(self, prob: AllocationProblem, t: int) -> int:
        return self.start[t] + prob.durations[t][self.robot[t]]


def check_schedule(prob: AllocationProblem, s: Schedule) -> list[str]:
    """Independent feasibility check; empty list means valid."""
    errors = []
    for t in range(prob.n):
        if prob.durations[t][s.robot[t]] is None:
            errors.append(f"task {t}: robot {s.robot[t]} cannot reach it")
            continue
        for j in prob.preds[t]:
            if s.start[t] < s.end(prob, j):
                errors.append(f"task {t} starts before its support {j} is placed")

    def clash(a: int, b: int) -> bool:
        return s.start[a] < s.end(prob, b) and s.start[b] < s.end(prob, a)

    for a in range(prob.n):
        for b in range(a + 1, prob.n):
            if s.robot[a] == s.robot[b] and clash(a, b):
                errors.append(f"robot {s.robot[a]} does tasks {a} and {b} at once")
            if prob.shared[a] and prob.shared[b] and clash(a, b):
                errors.append(f"tasks {a} and {b} are in the shared zone at once")
    if s.makespan != max(s.end(prob, t) for t in range(prob.n)):
        errors.append("makespan does not match the schedule")
    return errors


def single_robot_time(prob: AllocationProblem) -> int:
    """Reference: one robot that could reach everything, using the faster
    duration of each task — the sum of work with no parallelism at all."""
    return sum(min(d for d in row if d is not None) for row in prob.durations)


def list_schedule(prob: AllocationProblem) -> Schedule:
    """Dispatch tasks bottom-up; each goes to the robot that finishes it first."""
    order = sorted(range(prob.n), key=lambda t: (prob.plan[t].z, prob.plan[t].x, prob.plan[t].y))
    free = [0] * len(prob.robots)
    shared_free = 0
    start, robot, end = [0] * prob.n, [0] * prob.n, [0] * prob.n
    for t in order:
        ready = max((end[j] for j in prob.preds[t]), default=0)
        if prob.shared[t]:
            ready = max(ready, shared_free)
        options = [
            (max(ready, free[r]) + d, r)
            for r, d in enumerate(prob.durations[t])
            if d is not None
        ]
        finish, r = min(options)
        start[t], robot[t], end[t] = finish - prob.durations[t][r], r, finish
        free[r] = finish
        if prob.shared[t]:
            shared_free = finish
    return Schedule(start, robot, max(end))


def cpsat_schedule(
    prob: AllocationProblem,
    time_limit: float = 20.0,
    hint: Schedule | None = None,
    seed: int = 0,
) -> Schedule:
    m = cp_model.CpModel()
    horizon = sum(max(d for d in row if d is not None) for row in prob.durations)
    start = [m.new_int_var(0, horizon, f"s{t}") for t in range(prob.n)]
    end = [m.new_int_var(0, horizon, f"e{t}") for t in range(prob.n)]
    use: list[dict[int, cp_model.IntVar]] = []
    per_robot: list[list] = [[] for _ in prob.robots]

    for t, row in enumerate(prob.durations):
        options = {}
        for r, d in enumerate(row):
            if d is None:
                continue
            p = m.new_bool_var(f"use{t}_{r}")
            options[r] = p
            per_robot[r].append(m.new_optional_fixed_size_interval_var(start[t], d, p, f"iv{t}_{r}"))
            m.add(end[t] == start[t] + d).only_enforce_if(p)
        m.add_exactly_one(options.values())
        use.append(options)
        for j in prob.preds[t]:
            m.add(start[t] >= end[j])

    for intervals in per_robot:
        m.add_no_overlap(intervals)
    shared_iv = []
    for t in range(prob.n):
        if prob.shared[t]:
            options = [d for d in prob.durations[t] if d is not None]
            size = m.new_int_var(min(options), max(options), f"len{t}")
            shared_iv.append(m.new_interval_var(start[t], size, end[t], f"zone{t}"))
    m.add_no_overlap(shared_iv)

    makespan = m.new_int_var(0, horizon, "makespan")
    m.add_max_equality(makespan, end)
    m.minimize(makespan)

    if hint is not None:
        for t in range(prob.n):
            m.add_hint(start[t], hint.start[t])
            for r, p in use[t].items():
                m.add_hint(p, r == hint.robot[t])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    solver.parameters.random_seed = seed
    status = solver.solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT returned {solver.status_name(status)}")
    robot = [next(r for r, p in use[t].items() if solver.value(p)) for t in range(prob.n)]
    return Schedule(
        [solver.value(s) for s in start],
        robot,
        solver.value(makespan),
        solver.status_name(status),
        int(solver.best_objective_bound),
    )
