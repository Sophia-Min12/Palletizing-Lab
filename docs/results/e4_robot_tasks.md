# E4 · Robot task sequencing and allocation

## Sequencing — empty robot travel (placing one box → picking the next)

Loaded moves (pick → place) are the same in every order and are excluded.

| plan | boxes | layer by layer | nearest feasible | SA | CP-SAT | CP-SAT status |
|---|---|---|---|---|---|---|
| 0 | 46 | 64.8 m | 63.5 m | 60.1 m | 59.9 m | OPTIMAL |
| 1 | 40 | 58.0 m | 52.4 m | 50.5 m | 50.5 m | OPTIMAL |
| 2 | 42 | 59.7 m | 53.9 m | 51.2 m | 51.2 m | OPTIMAL |
| 3 | 44 | 60.2 m | 57.0 m | 55.0 m | 54.9 m | OPTIMAL |
| 4 | 44 | 62.3 m | 59.1 m | 56.2 m | 56.1 m | OPTIMAL |

CP-SAT proved 5/5 orders optimal; empty travel is 10.6 % shorter than building layer by layer.

## Allocation — two robots, shared middle zone

| plan | boxes | shared-zone boxes | one robot (no parallelism) | list scheduling | CP-SAT | CP-SAT status |
|---|---|---|---|---|---|---|
| 0 | 46 | 14 | 207.0 s | 107.9 s | 105.4 s | OPTIMAL |
| 1 | 40 | 10 | 181.2 s | 95.2 s | 91.1 s | OPTIMAL |
| 2 | 42 | 13 | 190.9 s | 106.5 s | 95.8 s | OPTIMAL |
| 3 | 44 | 15 | 199.2 s | 117.8 s | 100.1 s | OPTIMAL |
| 4 | 44 | 13 | 198.4 s | 104.3 s | 100.8 s | OPTIMAL |

CP-SAT proved 5/5 schedules optimal. Two robots finish 1.98× faster than one; list scheduling is within 7.8 % of the CP-SAT makespan.

![schedule](../figures/two_robot_schedule.png)
