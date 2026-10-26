"""E4 — Robot task sequencing and two-robot task allocation.

For five stacking plans (BRKGA, SKU mix), compare
- sequencing: robot travel to build the plan from three infeed conveyors;
- allocation: makespan of a two-robot cell with a shared middle zone.

Usage: python experiments/robot_tasks.py   (≈ 5 min)
"""

from __future__ import annotations

import numpy as np

from common import FIGURES, md_table, write_result
from palletlab.allocation import (
    TICKS_PER_S,
    build_allocation,
    check_schedule,
    cpsat_schedule,
    list_schedule,
    single_robot_time,
)
from palletlab.geometry import T11
from palletlab.instances import random_boxes
from palletlab.metaheuristics import brkga
from palletlab.sequencing import (
    anneal_order,
    assign_lanes,
    build_problem,
    cpsat_order,
    layer_order,
    nearest_feasible,
)
from palletlab.viz import save_gantt

SEEDS = range(5)


def main() -> None:
    seq_rows, alloc_rows = [], []
    empty = {k: [] for k in ("layer", "nearest", "SA", "CP-SAT")}
    span = {k: [] for k in ("one robot", "list", "CP-SAT")}
    optimal_seq = optimal_alloc = 0
    gantt = None
    for seed in SEEDS:
        boxes = random_boxes(60, seed=seed)
        plan = brkga(T11, boxes, max_evals=1500, seed=seed).best.plan

        prob = build_problem(T11, plan, assign_lanes(boxes, 3))
        orders = {"layer": layer_order(prob), "nearest": nearest_feasible(prob)}
        orders["SA"] = anneal_order(prob, start=orders["nearest"], seed=seed)
        exact = cpsat_order(prob, time_limit=30, hint=orders["SA"], seed=seed)
        orders["CP-SAT"] = exact.order
        optimal_seq += exact.status == "OPTIMAL"
        for k, o in orders.items():
            assert prob.is_feasible(o)
            empty[k].append((prob.travel(o) - prob.loaded) / 1000)  # metres of empty moves
        seq_rows.append([seed, len(plan)] + [f"{empty[k][-1]:.1f} m" for k in empty] + [exact.status])

        alloc = build_allocation(T11, plan)
        ls = list_schedule(alloc)
        cs = cpsat_schedule(alloc, time_limit=30, hint=ls, seed=seed)
        assert not check_schedule(alloc, ls) and not check_schedule(alloc, cs)
        optimal_alloc += cs.status == "OPTIMAL"
        span["one robot"].append(single_robot_time(alloc) / TICKS_PER_S)
        span["list"].append(ls.makespan / TICKS_PER_S)
        span["CP-SAT"].append(cs.makespan / TICKS_PER_S)
        alloc_rows.append([seed, len(plan), sum(alloc.shared)] + [f"{span[k][-1]:.1f} s" for k in span] + [cs.status])
        if gantt is None:
            gantt = (alloc, cs)
        print(seed, {k: round(v[-1], 1) for k, v in empty.items()}, {k: v[-1] for k, v in span.items()}, flush=True)

    alloc, cs = gantt
    save_gantt(
        str(FIGURES / "two_robot_schedule.png"),
        [r.name for r in alloc.robots],
        [(cs.robot[t], cs.start[t] / TICKS_PER_S, alloc.durations[t][cs.robot[t]] / TICKS_PER_S, alloc.shared[t])
         for t in range(alloc.n)],
        title=f"two-robot schedule (CP-SAT, {cs.status.lower()}): makespan {cs.makespan / TICKS_PER_S:.1f} s",
    )

    saving = 100 * (1 - np.mean(empty["CP-SAT"]) / np.mean(empty["layer"]))
    speedup = np.mean(span["one robot"]) / np.mean(span["CP-SAT"])
    list_gap = 100 * (np.mean(span["list"]) / np.mean(span["CP-SAT"]) - 1)
    body = (
        "## Sequencing — empty robot travel (placing one box → picking the next)\n\n"
        "Loaded moves (pick → place) are the same in every order and are excluded.\n\n"
        + md_table(["plan", "boxes", "layer by layer", "nearest feasible", "SA", "CP-SAT", "CP-SAT status"], seq_rows)
        + f"\n\nCP-SAT proved {optimal_seq}/{len(SEEDS)} orders optimal; empty travel is "
        f"{saving:.1f} % shorter than building layer by layer.\n\n"
        "## Allocation — two robots, shared middle zone\n\n"
        + md_table(["plan", "boxes", "shared-zone boxes", "one robot (no parallelism)", "list scheduling", "CP-SAT", "CP-SAT status"], alloc_rows)
        + f"\n\nCP-SAT proved {optimal_alloc}/{len(SEEDS)} schedules optimal. Two robots finish "
        f"{speedup:.2f}× faster than one; list scheduling is within {list_gap:.1f} % of the CP-SAT makespan.\n\n"
        "![schedule](../figures/two_robot_schedule.png)"
    )
    write_result("e4_robot_tasks", "E4 · Robot task sequencing and allocation", body)


if __name__ == "__main__":
    main()
