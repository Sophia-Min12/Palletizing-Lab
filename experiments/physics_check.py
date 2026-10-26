"""E5 — Does the geometric stability rule survive real physics (MuJoCo)?

Plans are built with different support thresholds α, then simulated: the
load settles under gravity for 1 s, then the pallet brakes at a·g for 1 s.

Usage: python experiments/physics_check.py   (≈ 3 min)
"""

from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, md_table, write_result
from palletlab.construct import best_greedy
from palletlab.geometry import T11, footprint_overlap
from palletlab.instances import random_boxes
from palletlab.sim_mujoco import simulate_plan

SEEDS = range(10)
ALPHAS = (0.3, 0.5, 0.75, 0.9)
LATERAL_G = (0.2, 0.3)


def column_profile(plan, k):
    """Follow box k down while it rests on exactly one box. Returns whether
    that chain reaches the floor (a free-standing column) and the column's
    height / narrowest base width.
    """
    cur, base = plan[k], min(plan[k].dx, plan[k].dy)
    while cur.z > 0:
        below = [q for q in plan if q.top == cur.z and footprint_overlap(cur, q) > 0]
        if len(below) != 1:
            return False, plan[k].top / base
        cur = below[0]
        base = min(base, cur.dx, cur.dy)
    return True, plan[k].top / base


def failure_analysis(g: float = 0.3) -> str:
    """Which boxes fail under braking, at the default α?"""
    column = {True: [], False: []}
    slender = {True: [], False: []}
    for seed in SEEDS:
        plan = best_greedy(T11, random_boxes(60, seed=seed)).plan
        rep = simulate_plan(T11, plan, lateral_g=g)
        failed = (rep.displacement_mm > 20) | (rep.tilt_deg > 5)
        for k in range(len(plan)):
            in_column, ratio = column_profile(plan, k)
            column[bool(failed[k])].append(in_column)
            slender[bool(failed[k])].append(ratio)
    return (
        f"**Why boxes fail at {g} g (α = {T11.support_ratio}).** "
        f"{100 * np.mean(column[True]):.0f} % of failed boxes sit in free-standing single-support "
        f"columns, against {100 * np.mean(column[False]):.0f} % of boxes that held; their columns are "
        f"more slender (height / base width {np.mean(slender[True]):.2f} vs {np.mean(slender[False]):.2f}). "
        "A rigid column tips when a/g > w/H, and w/H ≈ 1/3.3 ≈ 0.3 — the braking level where failures "
        "jump. The static support rule cannot see this: it checks each box against the one below it, "
        "not the column as a whole."
    )


def main() -> None:
    rows, settle_rate, lateral_rate = [], [], {g: [] for g in LATERAL_G}
    for alpha in ALPHAS:
        spec = dataclasses.replace(T11, support_ratio=alpha)
        boxes_total, settle, util = 0, 0, []
        lateral = {g: 0 for g in LATERAL_G}
        for seed in SEEDS:
            res = best_greedy(spec, random_boxes(60, seed=seed))
            util.append(res.utilization)
            boxes_total += len(res.plan)
            settle += simulate_plan(spec, res.plan, lateral_g=0.0).settle_failures()
            for g in LATERAL_G:
                lateral[g] += simulate_plan(spec, res.plan, lateral_g=g).failures()
        settle_rate.append(100 * settle / boxes_total)
        for g in LATERAL_G:
            lateral_rate[g].append(100 * lateral[g] / boxes_total)
        rows.append(
            [f"{alpha:.2f}", f"{100 * np.mean(util):.1f} %", boxes_total, f"{settle} ({settle_rate[-1]:.1f} %)"]
            + [f"{lateral[g]} ({lateral_rate[g][-1]:.1f} %)" for g in LATERAL_G]
        )
        print(rows[-1], flush=True)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(ALPHAS, settle_rate, "o-", label="moved > 10 mm while settling (gravity only)")
    for g in LATERAL_G:
        ax.plot(ALPHAS, lateral_rate[g], "s--", label=f"failed after {g} g braking")
    ax.set_xlabel("support threshold α used by the planner")
    ax.set_ylabel("boxes (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "physics_alpha.png", dpi=150)
    plt.close(fig)

    body = (
        f"{len(SEEDS)} SKU instances (60 boxes) per row, best greedy plans. MuJoCo 3, elliptic friction "
        "cones, μ = 0.5, 200 kg/m³ boxes. *Failed* = displaced > 20 mm or tilted > 5° at the end.\n\n"
        + md_table(["α", "utilization", "boxes", "settle failures"] + [f"failures at {g} g" for g in LATERAL_G], rows)
        + "\n\n" + failure_analysis()
        + "\n\n![physics](../figures/physics_alpha.png)"
    )
    write_result("e5_physics_check", "E5 · Physics validation in MuJoCo", body)


if __name__ == "__main__":
    main()
