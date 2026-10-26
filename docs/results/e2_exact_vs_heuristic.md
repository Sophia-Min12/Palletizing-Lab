# E2 · Exact model vs. heuristics

3 instances per row. CP-SAT: 8 workers, 60 s limit, warm-started from BRKGA. *Proven gap* = CP-SAT's upper bound minus its solution (0 = optimality proven). Parallel CP-SAT is not bit-for-bit reproducible; reruns can differ near the time limit.

| instance | best greedy | BRKGA (2000 evals) | CP-SAT | proven optimal | proven gap | CP-SAT time |
|---|---|---|---|---|---|---|
| perfect, 8 boxes | 89.4 % | 100.0 % | 100.0 % | 3/3 | 0.0 pts | 0.1 s |
| perfect, 12 boxes | 70.0 % | 96.8 % | 100.0 % | 3/3 | 0.0 pts | 0.3 s |
| perfect, 16 boxes | 67.3 % | 88.0 % | 100.0 % | 3/3 | 0.0 pts | 3.0 s |
| perfect, 20 boxes | 67.5 % | 88.3 % | 100.0 % | 3/3 | 0.0 pts | 13.9 s |
| perfect, 25 boxes | 67.2 % | 80.0 % | 87.0 % | 0/3 | 13.0 pts | 95.3 s |
| SKU mix, 60 boxes | 88.0 % | 93.3 % | 93.4 % | 0/3 | 27.3 pts | 60.5 s |
