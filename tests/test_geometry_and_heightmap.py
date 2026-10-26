"""The feasibility checker, and agreement between it and the heightmap planner."""

import dataclasses

import numpy as np
import pytest

from palletlab.construct import decode
from palletlab.geometry import PalletSpec, Placement, T11, validate_plan
from palletlab.heightmap import HeightMap
from palletlab.instances import perfect_instance, random_boxes

SMALL = PalletSpec(length=10, width=10, max_height=10)


def test_valid_hand_made_stack_passes():
    plan = [
        Placement(0, 0, 0, 0, 10, 10, 5),
        Placement(1, 0, 0, 5, 10, 8, 5),  # 80 % supported
    ]
    assert validate_plan(SMALL, plan) == []


@pytest.mark.parametrize(
    "plan, message",
    [
        ([Placement(0, 0, 0, 0, 11, 5, 5)], "outside the pallet"),
        ([Placement(0, 0, 0, 0, 5, 5, 11)], "height limit"),
        ([Placement(0, 0, 0, 0, 5, 5, 5), Placement(1, 4, 4, 0, 5, 5, 5)], "overlaps"),
        ([Placement(0, 0, 0, 0, 10, 10, 5), Placement(1, 0, 0, 5, 10, 10, 5)][:1] * 2, "placed twice"),
        # 50 % of the bottom face rests on box 0: below α = 75 %
        ([Placement(0, 0, 0, 0, 5, 10, 5), Placement(1, 0, 0, 5, 10, 10, 5)], "supported"),
        # box 1 would have to slide under box 0, which is already in place above it
        ([Placement(0, 0, 0, 5, 4, 4, 5), Placement(1, 0, 0, 0, 4, 4, 5)], "blocked from above"),
    ],
)
def test_checker_catches_each_violation(plan, message):
    errors = validate_plan(SMALL, plan)
    assert any(message in e for e in errors), errors


def test_rest_heights_match_brute_force():
    rng = np.random.default_rng(0)
    hm = HeightMap(T11, rng.integers(0, 30, size=(T11.length, T11.width)).astype(np.int32))
    fast = hm.rest_heights(7, 4)
    brute = np.array([[hm.heights[x : x + 7, y : y + 4].max() for y in range(T11.width - 3)] for x in range(T11.length - 6)])
    assert np.array_equal(fast, brute)


def test_first_box_goes_to_the_corner():
    p = HeightMap(T11).best_placement(random_boxes(1, seed=0)[0])
    assert (p.x, p.y, p.z) == (0, 0, 0)


@pytest.mark.parametrize("placement", ["dblf", "min_waste"])
@pytest.mark.parametrize("seed", range(4))
def test_heightmap_planner_agrees_with_independent_checker(placement, seed):
    """Plans the heightmap calls stable must pass the explicit-geometry checker."""
    for boxes in (random_boxes(70, seed=seed), perfect_instance(T11, 40, min_side=6, seed=seed)):
        plan = decode(T11, boxes, placement=placement).plan
        assert plan and validate_plan(T11, plan) == []


def test_support_threshold_is_respected_exactly():
    """A box straddling a step is placed only when α allows the support it gets."""
    heights = np.zeros((10, 10), dtype=np.int32)
    heights[:7, :] = 5  # 70 % of a 10 × 10 footprint sits at z = 5
    strict = HeightMap(SMALL, heights.copy())
    loose = HeightMap(dataclasses.replace(SMALL, support_ratio=0.7), heights.copy())
    assert strict.positions(10, 10, 5) == []
    assert loose.positions(10, 10, 5)[0][1:] == (5, 0, 0)


def test_perfect_instance_tiles_the_pallet():
    boxes = perfect_instance(T11, 30, min_side=8, seed=3)
    assert len(boxes) == 30
    assert sum(b.volume for b in boxes) == T11.volume
    assert min(min(b.l, b.w, b.h) for b in boxes) >= 8
