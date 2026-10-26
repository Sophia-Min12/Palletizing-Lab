"""Physics agrees with the stability rule; perception recovers known ground truth."""

import numpy as np

from palletlab.construct import best_greedy
from palletlab.geometry import Placement, T11
from palletlab.instances import SKU_CATALOGUE, random_boxes
from palletlab.perception import (
    estimate_box,
    heightmap_from_depth,
    ransac_plane,
    render_depth,
    synthetic_box_cloud,
)
from palletlab.sim_mujoco import simulate_plan


def test_supported_stack_stays_put():
    plan = [Placement(0, 0, 0, 0, 20, 15, 10), Placement(1, 0, 0, 10, 20, 15, 10)]
    report = simulate_plan(T11, plan, lateral_g=0.0)
    assert report.settle_failures() == 0
    assert report.settled_displacement_mm.max() < 3.0


def test_overhanging_box_falls():
    """30 % support — a placement the planner would reject — tips over in physics."""
    plan = [Placement(0, 0, 0, 0, 20, 15, 10), Placement(1, 14, 0, 10, 20, 15, 10)]
    report = simulate_plan(T11, plan, lateral_g=0.0)
    assert report.settled_displacement_mm[1] > 10.0


def test_planned_load_settles_without_failures():
    plan = best_greedy(T11, random_boxes(40, seed=0)).plan
    assert simulate_plan(T11, plan, lateral_g=0.0).settle_failures() == 0


def test_ransac_recovers_a_tilted_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-500, 500, size=(3000, 2))
    z = 0.1 * xy[:, 0] + 50 + rng.normal(0, 1.0, 3000)
    pts = np.column_stack([xy, z])
    pts[:300, 2] += rng.uniform(100, 400, 300)  # 10 % outliers
    normal, _, _ = ransac_plane(pts, rng=rng)
    expected = np.array([-0.1, 0.0, 1.0]) / np.linalg.norm([-0.1, 0.0, 1.0])
    assert np.allclose(normal, expected, atol=2e-3)


def test_every_sku_is_recognised_from_its_point_cloud():
    rng = np.random.default_rng(3)
    for sku, ((l, w, h), _) in SKU_CATALOGUE.items():
        cloud = synthetic_box_cloud((l, w, h), rng.uniform(0, np.pi), noise_mm=2.0, rng=rng)
        est = estimate_box(cloud, rng=rng)
        assert est.sku == sku
        assert abs(est.h_mm - h) == 0
        assert abs(est.l_mm - max(l, w)) <= 20 and abs(est.w_mm - min(l, w)) <= 20


def test_heightmap_from_noisy_depth_is_exact():
    plan = best_greedy(T11, random_boxes(60, seed=1)).plan
    rng = np.random.default_rng(0)
    depth = render_depth(T11, plan) + rng.normal(0.0, 4.0, size=(T11.length * 4, T11.width * 4))
    truth = np.rint(render_depth(T11, plan, px_per_unit=1) / T11.unit_mm).astype(int)
    assert np.array_equal(heightmap_from_depth(depth, T11), truth)
