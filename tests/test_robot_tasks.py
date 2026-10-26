"""Online policies, sequencing, and allocation all produce feasible answers."""

import math

from palletlab.allocation import build_allocation, check_schedule, cpsat_schedule, list_schedule
from palletlab.construct import best_greedy
from palletlab.geometry import Placement, T11, validate_plan
from palletlab.instances import random_boxes
from palletlab.online import BufferGreedyPolicy, GreedyPolicy, MCTSPolicy, run_episode
from palletlab.sequencing import (
    DEFAULT_CELL,
    anneal_order,
    assign_lanes,
    build_problem,
    cpsat_order,
    layer_order,
    nearest_feasible,
    place_point,
    predecessors,
)


def test_online_policies_build_feasible_pallets():
    stream = random_boxes(80, seed=7)
    for policy, k in ((GreedyPolicy(), 1), (BufferGreedyPolicy(), 3), (MCTSPolicy(samples=1, iterations=4), 3)):
        res = run_episode(T11, stream, policy, buffer_size=k, seed=0)
        assert res.plan and validate_plan(T11, res.plan) == []


def test_predecessors_are_the_boxes_underneath():
    plan = [Placement(0, 0, 0, 0, 10, 10, 5), Placement(1, 20, 0, 0, 10, 10, 5), Placement(2, 0, 0, 5, 10, 10, 5)]
    assert predecessors(plan) == [set(), set(), {0}]


def _small_problem():
    boxes = random_boxes(30, seed=1)
    plan = best_greedy(T11, boxes).plan
    return build_problem(T11, plan, assign_lanes(boxes, 3))


def test_travel_of_a_single_box_is_three_legs():
    p = Placement(0, 0, 0, 0, 10, 10, 5)
    prob = build_problem(T11, [p], {0: 0})
    pick, place = DEFAULT_CELL.stations[0], place_point(p, T11, DEFAULT_CELL)
    expected = math.dist(DEFAULT_CELL.home, pick) + math.dist(pick, place) + math.dist(place, DEFAULT_CELL.home)
    assert math.isclose(prob.travel([0]), expected)


def test_sequencing_orders_are_feasible_and_cpsat_is_best():
    prob = _small_problem()
    sa = anneal_order(prob, iterations=3000, seed=0)
    for order in (layer_order(prob), nearest_feasible(prob), sa):
        assert prob.is_feasible(order)
    exact = cpsat_order(prob, time_limit=20, hint=sa)
    assert prob.is_feasible(exact.order)
    assert exact.travel <= prob.travel(sa) + prob.n  # costs are rounded to whole mm in the model


def test_two_robot_schedules_are_valid_and_cpsat_is_no_worse():
    plan = best_greedy(T11, random_boxes(30, seed=2)).plan
    prob = build_allocation(T11, plan)
    ls = list_schedule(prob)
    cs = cpsat_schedule(prob, time_limit=20, hint=ls)
    assert check_schedule(prob, ls) == [] and check_schedule(prob, cs) == []
    assert cs.makespan <= ls.makespan


def test_schedule_checker_catches_a_robot_doing_two_things_at_once():
    plan = best_greedy(T11, random_boxes(30, seed=2)).plan
    prob = build_allocation(T11, plan)
    s = list_schedule(prob)
    a, b = [t for t in range(prob.n) if s.robot[t] == s.robot[0]][:2]
    s.start[b] = s.start[a]
    assert check_schedule(prob, s)
