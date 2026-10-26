"""E2 — Where does the exact CP-SAT model stop scaling?

Perfect instances of growing size (optimum = 100 % known), plus SKU
instances whose optimum is unknown. CP-SAT is warm-started from the BRKGA
plan — the usual "MIP start" practice — so it can only match or improve it,
and its proven bound says how far from optimal the result can be.

Usage: python experiments/exact_vs_heuristic.py   (≈ 15 min)
"""

from __future__ import annotations

import numpy as np

from common import md_table, write_result
from palletlab.construct import best_greedy
from palletlab.exact_cpsat import solve_exact
from palletlab.geometry import T11, validate_plan
from palletlab.instances import perfect_instance, random_boxes
from palletlab.metaheuristics import brkga

SEEDS = range(3)
TIME_LIMIT = 60.0


def main() -> None:
    cases = [(f"perfect, {n} boxes", lambda s, n=n: perfect_instance(T11, n, min_side=8, seed=s)) for n in (8, 12, 16, 20, 25)]
    cases.append(("SKU mix, 60 boxes", lambda s: random_boxes(60, seed=s)))
    rows = []
    for name, make in cases:
        g, ga, ex, solved, secs, gaps = [], [], [], 0, [], []
        for seed in SEEDS:
            boxes = make(seed)
            g.append(best_greedy(T11, boxes).utilization)
            heur = brkga(T11, boxes, max_evals=2000, seed=seed).best
            ga.append(heur.utilization)
            res = solve_exact(T11, boxes, time_limit=TIME_LIMIT, hint=heur.plan, seed=seed)
            assert not validate_plan(T11, res.plan), validate_plan(T11, res.plan)[:3]
            ex.append(res.utilization)
            solved += res.status == "OPTIMAL"
            secs.append(res.seconds)
            gaps.append(res.upper_bound - res.utilization)
            print(name, seed, round(g[-1], 3), round(ga[-1], 3), res.status, round(res.utilization, 3),
                  f"bound {res.upper_bound:.3f}", f"{res.seconds:.1f}s", flush=True)
        rows.append([
            name,
            f"{100 * np.mean(g):.1f} %",
            f"{100 * np.mean(ga):.1f} %",
            f"{100 * np.mean(ex):.1f} %",
            f"{solved}/{len(SEEDS)}",
            f"{100 * np.mean(gaps):.1f} pts",
            f"{np.mean(secs):.1f} s",
        ])

    body = (
        f"{len(SEEDS)} instances per row. CP-SAT: 8 workers, {TIME_LIMIT:.0f} s limit, warm-started "
        "from BRKGA. *Proven gap* = CP-SAT's upper bound minus its solution (0 = optimality proven). "
        "Parallel CP-SAT is not bit-for-bit reproducible; reruns can differ near the time limit.\n\n"
        + md_table(["instance", "best greedy", "BRKGA (2000 evals)", "CP-SAT", "proven optimal", "proven gap", "CP-SAT time"], rows)
    )
    write_result("e2_exact_vs_heuristic", "E2 · Exact model vs. heuristics", body)


if __name__ == "__main__":
    main()
