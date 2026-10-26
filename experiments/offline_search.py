"""E1 — Offline stacking plans: greedy rules vs. simulated annealing vs. BRKGA.

Perfect instances have a known optimum of 100 %, so utilization *is* the
optimality gap. SKU instances hold ~1.25× the pallet volume, so the plan
also decides which boxes go on.

Usage: python experiments/offline_search.py [--figures-only]   (≈ 12 min)
"""

from __future__ import annotations

import time

import numpy as np

from common import FIGURES, md_table, write_result
from palletlab.construct import best_greedy
from palletlab.geometry import T11, validate_plan
from palletlab.instances import perfect_instance, random_boxes
from palletlab.metaheuristics import brkga, simulated_annealing
from palletlab.viz import save_convergence, save_plans, sku_colors

SEEDS = range(3)
BUDGET = 2000  # decoder evaluations per search run

INSTANCES = {
    "perfect, 20 boxes": lambda s: perfect_instance(T11, 20, min_side=8, seed=s),
    "perfect, 30 boxes": lambda s: perfect_instance(T11, 30, min_side=8, seed=s),
    "perfect, 50 boxes": lambda s: perfect_instance(T11, 50, min_side=6, seed=s),
    "SKU mix, 60 boxes": lambda s: random_boxes(60, seed=s),
}


def main() -> None:
    rows = []
    for name, make in INSTANCES.items():
        scores = {k: [] for k in ("greedy (min waste)", "greedy (DBLF)", "SA", "BRKGA")}
        seconds = {"SA": [], "BRKGA": []}
        for seed in SEEDS:
            boxes = make(seed)
            scores["greedy (min waste)"].append(best_greedy(T11, boxes, "min_waste").utilization)
            scores["greedy (DBLF)"].append(best_greedy(T11, boxes).utilization)
            for label, run in (("SA", simulated_annealing), ("BRKGA", brkga)):
                t = time.perf_counter()
                res = run(T11, boxes, max_evals=BUDGET, seed=seed)
                seconds[label].append(time.perf_counter() - t)
                assert not validate_plan(T11, res.best.plan)
                scores[label].append(res.best.utilization)
            print(name, seed, {k: round(v[-1], 3) for k, v in scores.items()}, flush=True)
        rows.append(
            [name]
            + [f"{100 * np.mean(v):.1f} %" for v in scores.values()]
            + [f"{np.mean(seconds['SA']):.0f} s / {np.mean(seconds['BRKGA']):.0f} s"]
        )

    body = (
        f"Mean utilization over {len(SEEDS)} instances per row; search methods get "
        f"{BUDGET} decoder evaluations and start from the best greedy order. Every plan "
        "passed the independent feasibility check (bounds, overlap, top-down access, "
        "≥ 75 % support).\n\n"
        + md_table(
            ["instance", "greedy (min waste)", "greedy (DBLF)", "SA", "BRKGA", "time SA / BRKGA"], rows
        )
        + "\n\n![convergence](../figures/offline_convergence.png)\n\n![plans](../figures/offline_plans.png)"
    )
    write_result("e1_offline_search", "E1 · Offline stacking plans", body)


def figures() -> None:
    """Convergence on a perfect instance; before/after plans on a SKU mix."""
    boxes = INSTANCES["perfect, 30 boxes"](0)
    greedy = best_greedy(T11, boxes)
    sa = simulated_annealing(T11, boxes, max_evals=BUDGET, seed=0)
    ga = brkga(T11, boxes, max_evals=BUDGET, seed=0)
    save_convergence(
        str(FIGURES / "offline_convergence.png"),
        {"SA": sa.utilization_history, "BRKGA": ga.utilization_history},
        greedy.utilization,
    )

    boxes = INSTANCES["SKU mix, 60 boxes"](0)
    sku_of = {b.id: b.sku for b in boxes}
    greedy = best_greedy(T11, boxes)
    ga = brkga(T11, boxes, max_evals=BUDGET, seed=0)
    save_plans(
        str(FIGURES / "offline_plans.png"),
        T11,
        [
            (f"best greedy — {100 * greedy.utilization:.1f} %, {len(greedy.plan)} boxes",
             greedy.plan, sku_colors(greedy.plan, sku_of)),
            (f"BRKGA — {100 * ga.best.utilization:.1f} %, {len(ga.best.plan)} boxes",
             ga.best.plan, sku_colors(ga.best.plan, sku_of)),
        ],
    )


if __name__ == "__main__":
    import sys

    if "--figures-only" not in sys.argv:
        main()
    figures()
