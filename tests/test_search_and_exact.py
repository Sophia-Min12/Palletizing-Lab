"""Metaheuristics improve on greedy; the CP-SAT model is exact and stability-aware."""

from palletlab.construct import best_greedy
from palletlab.exact_cpsat import solve_exact
from palletlab.geometry import Box, PalletSpec, T11, validate_plan
from palletlab.instances import perfect_instance
from palletlab.metaheuristics import brkga, simulated_annealing


def test_search_never_loses_to_greedy_and_stays_feasible():
    boxes = perfect_instance(T11, 20, min_side=8, seed=4)
    greedy = best_greedy(T11, boxes)
    for run in (simulated_annealing, brkga):
        res = run(T11, boxes, max_evals=300, seed=0)
        assert res.best.objective <= greedy.objective
        assert validate_plan(T11, res.best.plan) == []
        assert all(b <= a for a, b in zip(res.history, res.history[1:]))  # best-so-far curve


def test_cpsat_closes_the_gap_on_small_perfect_instances():
    boxes = perfect_instance(T11, 10, min_side=8, seed=10)
    res = solve_exact(T11, boxes, time_limit=30, hint=best_greedy(T11, boxes).plan)
    assert res.status == "OPTIMAL"
    assert res.utilization == 1.0
    assert validate_plan(T11, res.plan) == []


def test_cpsat_respects_the_support_rule():
    """Two boxes fit only stacked. A 10 × 5 box on the floor would hold the
    10 × 10 box up by just 50 %, so the only legal answer is big box below.
    """
    spec = PalletSpec(length=10, width=10, max_height=10)
    big, half = Box(0, 10, 10, 5), Box(1, 10, 5, 5)
    res = solve_exact(spec, [half, big], time_limit=10)
    assert res.status == "OPTIMAL" and res.utilization == 0.75
    z = {p.box_id: p.z for p in res.plan}
    assert z == {0: 0, 1: 5}
    assert validate_plan(spec, res.plan) == []
