"""E6 — Perception accuracy: box dimensions from point clouds, heightmaps
from depth images.

Usage: python experiments/perception_eval.py   (≈ 1 min)
"""

from __future__ import annotations

import time

import numpy as np

from common import md_table, write_result
from palletlab.construct import best_greedy
from palletlab.geometry import T11
from palletlab.instances import SKU_CATALOGUE, random_boxes
from palletlab.perception import estimate_box, heightmap_from_depth, render_depth, synthetic_box_cloud

BOXES_PER_LEVEL = 80


def main() -> None:
    box_rows = []
    for noise in (1.0, 2.0, 3.0, 5.0, 8.0):
        rng = np.random.default_rng(0)
        exact, sku_ok, worst, yaw_err, secs = 0, 0, 0, [], []
        for k in range(BOXES_PER_LEVEL):
            sku = list(SKU_CATALOGUE)[k % len(SKU_CATALOGUE)]
            (l, w, h), _ = SKU_CATALOGUE[sku]
            yaw = rng.uniform(0, np.pi)
            cloud = synthetic_box_cloud((l, w, h), yaw, tuple(rng.uniform(-100, 100, 2)), noise_mm=noise, rng=rng)
            t = time.perf_counter()
            est = estimate_box(cloud, rng=rng)
            secs.append(time.perf_counter() - t)
            err = max(abs(est.l_mm - max(l, w)), abs(est.w_mm - min(l, w)), abs(est.h_mm - h))
            exact += err == 0
            worst = max(worst, err)
            sku_ok += est.sku == sku
            if l != w:
                true_yaw = yaw if l > w else yaw + np.pi / 2
                yaw_err.append(np.degrees(abs((est.yaw_rad - true_yaw + np.pi / 2) % np.pi - np.pi / 2)))
        box_rows.append([
            f"{noise:.0f} mm", f"{100 * exact / BOXES_PER_LEVEL:.0f} %", f"{worst:.0f} mm",
            f"{np.mean(yaw_err):.1f}° / {np.max(yaw_err):.1f}°", f"{100 * sku_ok / BOXES_PER_LEVEL:.0f} %",
            f"{1000 * np.mean(secs):.0f} ms",
        ])
        print(box_rows[-1], flush=True)

    hm_rows = []
    for noise in (2.0, 4.0, 8.0, 12.0):
        rng = np.random.default_rng(1)
        correct = []
        for seed in range(5):
            plan = best_greedy(T11, random_boxes(60, seed=seed)).plan
            depth = render_depth(T11, plan)
            truth = np.rint(render_depth(T11, plan, px_per_unit=1) / T11.unit_mm).astype(int)
            noisy = depth + rng.normal(0.0, noise, size=depth.shape)
            correct.append((heightmap_from_depth(noisy, T11) == truth).mean())
        hm_rows.append([f"{noise:.0f} mm", f"{100 * np.mean(correct):.2f} %"])

    body = (
        "## Box on the conveyor → dimensions, yaw, SKU\n\n"
        f"{BOXES_PER_LEVEL} synthetic top-down clouds per noise level (≈ 1 point/cm², 1 % outliers), "
        "all 8 SKUs at random yaw. *Exact* = all three dimensions on the right 20 mm grid value.\n\n"
        + md_table(["depth noise σ", "dims exact", "worst dim error", "yaw error mean / max", "SKU recognised", "time"], box_rows)
        + "\n\n## Depth image → pallet heightmap\n\n"
        "4 × 4 depth pixels per 20 mm cell, 5 loads per row.\n\n"
        + md_table(["depth noise σ", "cells exactly right"], hm_rows)
    )
    write_result("e6_perception", "E6 · Perception accuracy", body)


if __name__ == "__main__":
    main()
