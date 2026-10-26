"""Simulated annealing and a biased random-key GA over box orders.

Both search the space of orders and leave geometry to the decoder, and both
are warm-started from the best greedy rule, so they can only match or improve
on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .construct import SORT_RULES, PackResult, decode
from .geometry import Box, PalletSpec


@dataclass
class SearchResult:
    best: PackResult
    best_order: list[int]
    history: list[float] = field(default_factory=list)  # best F after each eval
    evaluations: int = 0
    utilization_history: list[float] = field(default_factory=list)  # of that best plan


def greedy_orders(boxes: Sequence[Box]) -> list[list[int]]:
    return [
        sorted(range(len(boxes)), key=lambda i: rule(boxes[i])) for rule in SORT_RULES.values()
    ]


def _neighbor(order: list[int], rng: np.random.Generator) -> list[int]:
    """Swap two boxes, move one box, or reverse a segment — equally likely."""
    n = len(order)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    new = order.copy()
    move = rng.integers(3)
    if move == 0:
        new[i], new[j] = new[j], new[i]
    elif move == 1:
        new.insert(j, new.pop(i))
    else:
        new[i : j + 1] = new[i : j + 1][::-1]
    return new


def simulated_annealing(
    spec: PalletSpec,
    boxes: Sequence[Box],
    max_evals: int = 2000,
    t_initial: float = 0.02,
    t_final: float = 1e-4,
    placement: str = "dblf",
    seed: int | None = None,
) -> SearchResult:
    """Metropolis acceptance on F = unused volume; temperature decays
    geometrically from `t_initial` to `t_final` over the evaluation budget.
    (A 0.02 change in F is 2 % of the pallet volume.)
    """
    rng = np.random.default_rng(seed)
    starts = [(decode(spec, boxes, o, placement), o) for o in greedy_orders(boxes)]
    current, order = min(starts, key=lambda s: s[0].objective)
    best, best_order = current, order
    evals = len(starts)
    history = [best.objective] * evals
    util_history = [best.utilization] * evals
    cooling = (t_final / t_initial) ** (1.0 / max(max_evals - evals, 1))
    temperature = t_initial

    while evals < max_evals:
        cand_order = _neighbor(order, rng)
        cand = decode(spec, boxes, cand_order, placement)
        evals += 1
        delta = cand.objective - current.objective
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, order = cand, cand_order
            if current.objective < best.objective:
                best, best_order = current, order
        history.append(best.objective)
        util_history.append(best.utilization)
        temperature *= cooling
    return SearchResult(best, best_order, history, evals, util_history)


def brkga(
    spec: PalletSpec,
    boxes: Sequence[Box],
    pop_size: int = 40,
    max_evals: int = 2000,
    elite_frac: float = 0.2,
    mutant_frac: float = 0.15,
    elite_bias: float = 0.7,
    placement: str = "dblf",
    seed: int | None = None,
) -> SearchResult:
    """Biased random-key GA (Gonçalves & Resende, 2013).

    A chromosome is a vector of keys in [0, 1); sorting the keys gives the
    box order. Each generation keeps the elite, injects random mutants, and
    breeds the rest by uniform crossover that takes each key from the elite
    parent with probability `elite_bias`. Every child decodes to a valid
    order, so no repair step is needed.
    """
    rng = np.random.default_rng(seed)
    n = len(boxes)
    n_elite = max(1, int(elite_frac * pop_size))
    n_mutant = max(1, int(mutant_frac * pop_size))

    keys = rng.random((pop_size, n))
    for k, order in enumerate(greedy_orders(boxes)):  # warm start
        keys[k, order] = np.arange(n) / n

    def evaluate(population: np.ndarray) -> list[PackResult]:
        return [decode(spec, boxes, np.argsort(c).tolist(), placement) for c in population]

    results = evaluate(keys)
    evals = pop_size
    history: list[float] = []
    util_history: list[float] = []
    best_so_far: PackResult | None = None

    def record(r: PackResult) -> None:
        nonlocal best_so_far
        if best_so_far is None or r.objective < best_so_far.objective:
            best_so_far = r
        history.append(best_so_far.objective)
        util_history.append(best_so_far.utilization)

    for r in results:
        record(r)

    while evals + pop_size - n_elite <= max_evals:
        rank = np.argsort([r.objective for r in results])
        elite, rest = keys[rank[:n_elite]], keys[rank[n_elite:]]
        n_children = pop_size - n_elite - n_mutant
        parents_e = elite[rng.integers(0, n_elite, n_children)]
        parents_o = rest[rng.integers(0, len(rest), n_children)]
        children = np.where(rng.random((n_children, n)) < elite_bias, parents_e, parents_o)
        newcomers = np.vstack([rng.random((n_mutant, n)), children])

        new_results = evaluate(newcomers)
        evals += len(newcomers)
        for r in new_results:
            record(r)
        keys = np.vstack([elite, newcomers])
        results = [results[i] for i in rank[:n_elite]] + new_results

    best_idx = int(np.argmin([r.objective for r in results]))
    return SearchResult(results[best_idx], np.argsort(keys[best_idx]).tolist(), history, evals, util_history)
