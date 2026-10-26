"""E3 — Random (online) palletizing: what is a lookahead buffer worth, and MCTS?

Every policy sees the same 10 random SKU streams. An episode ends when no box
in the buffer can be placed stably.

Usage: python experiments/online_random.py [--figures-only]   (≈ 6 min)
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, md_table, write_result
from palletlab.geometry import T11, validate_plan
from palletlab.instances import random_boxes
from palletlab.online import BufferGreedyPolicy, GreedyPolicy, MCTSPolicy, run_episode
from palletlab.viz import save_plans, sku_colors

SEEDS = range(10)

POLICIES = [
    ("greedy, k = 1 (pure online)", lambda: GreedyPolicy(), 1),
    ("buffer greedy, k = 2", lambda: BufferGreedyPolicy(), 2),
    ("buffer greedy, k = 3", lambda: BufferGreedyPolicy(), 3),
    ("buffer greedy, k = 5", lambda: BufferGreedyPolicy(), 5),
    ("buffer greedy (min waste), k = 3", lambda: BufferGreedyPolicy("min_waste"), 3),
    ("MCTS, k = 3", lambda: MCTSPolicy(), 3),
]


def main() -> None:
    utils = {name: [] for name, _, _ in POLICIES}
    secs = {name: [] for name, _, _ in POLICIES}
    for seed in SEEDS:
        stream = random_boxes(150, seed=100 + seed)
        for name, make, k in POLICIES:
            t = time.perf_counter()
            res = run_episode(T11, stream, make(), buffer_size=k, seed=seed)
            secs[name].append((time.perf_counter() - t) / max(1, len(res.plan)))  # per box
            assert not validate_plan(T11, res.plan)
            utils[name].append(res.utilization)
        print(seed, {n: round(v[-1], 3) for n, v in utils.items()}, flush=True)

    base = np.array(utils["buffer greedy, k = 3"])
    mcts = np.array(utils["MCTS, k = 3"])
    rows = [
        [name, f"{100 * np.mean(v):.1f} %", f"{100 * np.std(v):.1f}", f"{100 * np.min(v):.1f} %",
         f"{1000 * np.mean(secs[name]):.1f} ms"]
        for name, v in utils.items()
    ]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    names = list(utils)
    ax.barh(range(len(names)), [100 * np.mean(utils[n]) for n in names],
            xerr=[100 * np.std(utils[n]) for n in names], color="#8fa6c4", edgecolor="#333333")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("pallet utilization (%) — mean ± std over 10 streams")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "online_policies.png", dpi=150)
    plt.close(fig)

    plan_figure()

    body = (
        "10 random SKU streams; k = boxes visible on the conveyor. MCTS: 3 sampled futures × "
        "16 UCT iterations per decision, greedy rollouts to the end of the episode. Every plan "
        "passed the independent feasibility check.\n\n"
        + md_table(["policy", "mean", "std (pts)", "worst", "time per box"], rows)
        + f"\n\nMCTS vs. buffer greedy at k = 3: better on {(mcts > base).sum()} of {len(base)} streams, "
        f"mean difference {100 * (mcts - base).mean():+.1f} pts.\n\n"
        "![policies](../figures/online_policies.png)\n\n![plans](../figures/online_plans.png)"
    )
    write_result("e3_online_random", "E3 · Random (online) palletizing", body)


def plan_figure() -> None:
    """Three policies on the same stream (seed 0)."""
    stream = random_boxes(150, seed=100)
    sku_of = {b.id: b.sku for b in stream}
    panels = []
    for name, make, k in POLICIES:
        if name in ("greedy, k = 1 (pure online)", "buffer greedy, k = 3", "MCTS, k = 3"):
            res = run_episode(T11, stream, make(), buffer_size=k, seed=0)
            title = f"{name}\n{100 * res.utilization:.1f} %"
            panels.append((title, res.plan, sku_colors(res.plan, sku_of)))
    save_plans(str(FIGURES / "online_plans.png"), T11, panels)


if __name__ == "__main__":
    import sys

    if "--figures-only" in sys.argv:
        plan_figure()
    else:
        main()
