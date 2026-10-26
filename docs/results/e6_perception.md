# E6 · Perception accuracy

## Box on the conveyor → dimensions, yaw, SKU

80 synthetic top-down clouds per noise level (≈ 1 point/cm², 1 % outliers), all 8 SKUs at random yaw. *Exact* = all three dimensions on the right 20 mm grid value.

| depth noise σ | dims exact | worst dim error | yaw error mean / max | SKU recognised | time |
|---|---|---|---|---|---|
| 1 mm | 90 % | 20 mm | 0.9° / 2.9° | 100 % | 80 ms |
| 2 mm | 90 % | 20 mm | 0.9° / 2.8° | 100 % | 80 ms |
| 3 mm | 91 % | 20 mm | 0.8° / 2.8° | 100 % | 82 ms |
| 5 mm | 90 % | 20 mm | 0.8° / 3.1° | 100 % | 76 ms |
| 8 mm | 79 % | 20 mm | 1.2° / 5.3° | 100 % | 62 ms |

## Depth image → pallet heightmap

4 × 4 depth pixels per 20 mm cell, 5 loads per row.

| depth noise σ | cells exactly right |
|---|---|
| 2 mm | 100.00 % |
| 4 mm | 100.00 % |
| 8 mm | 100.00 % |
| 12 mm | 99.40 % |
