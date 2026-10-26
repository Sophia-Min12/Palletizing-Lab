# E5 · Physics validation in MuJoCo

10 SKU instances (60 boxes) per row, best greedy plans. MuJoCo 3, elliptic friction cones, μ = 0.5, 200 kg/m³ boxes. *Failed* = displaced > 20 mm or tilted > 5° at the end.

| α | utilization | boxes | settle failures | failures at 0.2 g | failures at 0.3 g |
|---|---|---|---|---|---|
| 0.30 | 86.6 % | 375 | 16 (4.3 %) | 41 (10.9 %) | 41 (10.9 %) |
| 0.50 | 83.9 % | 388 | 10 (2.6 %) | 25 (6.4 %) | 45 (11.6 %) |
| 0.75 | 86.1 % | 411 | 0 (0.0 %) | 30 (7.3 %) | 86 (20.9 %) |
| 0.90 | 81.2 % | 391 | 0 (0.0 %) | 42 (10.7 %) | 56 (14.3 %) |

**Why boxes fail at 0.3 g (α = 0.75).** 95 % of failed boxes sit in free-standing single-support columns, against 72 % of boxes that held; their columns are more slender (height / base width 3.27 vs 2.73). A rigid column tips when a/g > w/H, and w/H ≈ 1/3.3 ≈ 0.3 — the braking level where failures jump. The static support rule cannot see this: it checks each box against the one below it, not the column as a whole.

![physics](../figures/physics_alpha.png)
