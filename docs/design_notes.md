# Design notes

Why each problem is modelled the way it is, and what each choice costs. The
numbers quoted here come from `docs/results/`.

---

## 1. The state: a heightmap on a 20 mm grid

**Choice.** The pallet (Korean standard T-11, 1100 × 1100 mm, 1200 mm load
height) is a 55 × 55 grid; the load is represented by its top surface
`hm[x, y]`.

**Why.** A robot places boxes *from above*, so the top surface is all that
constrains the next placement: a box with footprint `dx × dy` dropped at
`(x, y)` rests at `z = max(hm over the footprint)`, and the cells with
`hm == z` are exactly those that hold it up. The same representation is what
a top-down depth camera returns (`perception.py`), so the optimizer's state
and the sensor's output are the same object.

**Cost.** A heightmap cannot express placing a box *under* an overhang. That
is not a loss for a top-down robot — such a placement is unreachable anyway —
and the independent checker (`validate_plan`) enforces it explicitly
("blocked from above").

**Speed.** Rest heights for every position come from a separable sliding
max, O(L·W·(dx+dy)). Support counts for every position at one height level
come from one summed-area table, O(L·W). Walking levels from the lowest up
made the decoder ~6× faster than testing candidates one by one (7–13 ms per
full plan), which is what makes thousands of metaheuristic evaluations
affordable.

## 2. Stability: ≥ α of the bottom face supported, α = 0.75

**Rule.** A box rests on the floor, or at least α of its bottom face rests on
the tops of boxes already placed at exactly its height.

**Why α > 0.5 is the meaningful threshold.** Any line through the centre of a
rectangle splits its area exactly in half. A support region covering more
than half of the bottom face therefore cannot lie entirely on one side of
such a line, so the centre of gravity lies inside the convex hull of the
support — the static tipping condition for a rigid box. α = 0.75 adds
margin.

**Checked in physics.** E5 builds plans with α ∈ {0.3, 0.5, 0.75, 0.9} and
settles them in MuJoCo. The static rule holds: no box moves while settling
at α ≥ 0.75, while 2.6–4.3 % topple at α ≤ 0.5. *Transport* loads are a
different matter: after 0.3 g braking 11–21 % of boxes fail at every α, and
95 % of the failures sit in free-standing single-support columns (vs. 72 %
of the boxes that hold). The rigid-body tipping condition `a/g > w/H` is
about column slenderness, not support area — and the stricter α = 0.75
fails *most* (20.9 %), because it favours stacking same-size boxes straight
up. Static support is necessary, not sufficient; interlocking patterns and
stretch wrap exist for that reason.

**Simulation detail that mattered.** With MuJoCo's default pyramidal friction
cone, stacked boxes crept a few mm per second under sustained sideways
force even below the friction limit. Elliptic cones with `impratio = 10`
(MuJoCo's recommended setting against slip) reduced it below 1 mm.
Without that, every plan looked unstable.

## 3. Offline planning: sequence + decoder

**Encoding.** All offline methods search over *box orders*; the decoder
places each box at its best DBLF (deepest-bottom-left) position over both
yaw orientations, or leaves it off. Every order decodes to a feasible plan,
so the search never needs repair or penalties.

- **Greedy:** three sort rules (volume, base area, height-then-area).
- **Simulated annealing:** swap / move / reverse-segment neighbours,
  Metropolis acceptance, geometric cooling over the evaluation budget.
- **BRKGA** (Gonçalves & Resende 2013): chromosomes are random keys, sorted to
  get an order; elite + mutants + biased uniform crossover. Popular in
  industrial packing because every chromosome is feasible by construction.

Both search methods are warm-started from the best greedy order, so they
can only match or improve it.

**Perfect instances.** The pallet volume is cut by random guillotine cuts
into n boxes, so the optimum (100 %) is known without an exact solver — the
utilization *is* the optimality gap, at any size.

**A rule that depends on the instance.** A "minimize trapped air" placement
rule (a heightmap-minimization idea from the online-packing literature) beats
DBLF as a greedy on perfect-packing puzzles (E1: 73.6 % vs 67.5 % at 20
boxes) but loses on the realistic SKU mix (81.7 % vs 88.0 %) and in online
palletizing (E3). Puzzles reward filling every pocket; repeated carton sizes
reward the flat, layer-like surfaces DBLF builds. Both rules stay in the code
so the comparison can be rerun.

## 4. Exact model: CP-SAT

```
maximize    Σᵢ volᵢ · bᵢ
subject to  bᵢ ⇒ inside the pallet and below the height limit
            bᵢ ∧ bⱼ ⇒ one of six separations (left/right/front/behind/below/above)
            bᵢ ∧ ¬floorᵢ ⇒ Σⱼ contribⱼᵢ ≥ α · areaᵢ
            contribⱼᵢ ≤ overlap_xⱼᵢ · overlap_yⱼᵢ,   contribⱼᵢ ≤ areaᵢ · sⱼᵢ
            sⱼᵢ ⇒ bⱼ ∧ zᵢ = zⱼ + hⱼ
            overlap_x = max(0, min(xᵢ+dxᵢ, xⱼ+dxⱼ) − max(xᵢ, xⱼ))     (same for y)
```

- Yaw is a boolean per box; the footprint area is yaw-invariant, so the
  support right-hand side stays linear.
- Overlaps use CP-SAT's native min/max/multiplication constraints. A MILP
  would need big-M linearisations of the same thing — and a weak LP
  relaxation from those big-Ms is the usual reason MILP struggles on packing.
- **Any stable static packing is buildable from above** by placing boxes in
  order of z: anything above box i that overlaps its footprint has a larger
  z, and boxes supporting i end exactly at zᵢ. So the model needs no
  sequencing variables, and the returned plan is sorted by z.
- **Warm start:** the BRKGA plan is passed as a hint (the "MIP start"
  practice), so CP-SAT starts from a good incumbent and spends its time
  improving or proving.

**Where it stops scaling** (E2): perfect instances up to 20 boxes are solved
to proven optimality (100 %, 14 s on average at 20 boxes) where BRKGA stops
at 88 %; at 25 boxes no run proves optimality within 60 s (87 %); at 60
boxes CP-SAT barely moves the BRKGA warm start (93.3 % → 93.4 %) and cannot
tighten the trivial bound at all. One 25-box run overran the 60 s limit to
166 s, so the limit is not a hard wall-clock cut-off. This is the
practical division of labour: exact methods for small sub-problems (one
layer, one pallet zone, validation of heuristics), metaheuristics for full
pallets.

## 5. Random (online) palletizing

Boxes arrive one at a time; the robot sees the next k on the conveyor and
must commit each placement. The episode ends when no visible box can be
placed stably — the termination rule used in the online 3D bin-packing
literature.

- **Buffer greedy:** best DBLF placement over all k visible boxes.
- **MCTS:** the unseen stream is *sampled* from the SKU distribution. Each
  sample makes the problem deterministic (determinization); a UCT tree is
  grown per sample with rollouts that play the episode *to the end*, and
  root visit counts are summed over samples — a sample-average approximation
  of the stochastic problem.

**Lesson from the first MCTS version.** Valuing a node by the volume packed
in the next 6 steps made MCTS *worse* than greedy: what ends an episode is
the surface becoming unusable, which a short horizon cannot see. Rolling out
to the end, and valuing the final utilization — the quantity actually
scored — fixed the direction. Even so, the average gain over buffer greedy
is small (E3: +1.9 points, 5 of 10 streams); the clearer gain is the worst
case (52 % vs 42 %), at ~750× the compute. The buffer itself is the larger
lever: k = 1 → 3 → 5 gives 40.5 % → 59.8 % → 67.7 %.

## 6. Robot task sequencing (Sequential Ordering Problem)

The plan fixes where boxes go; the order is free except that a box comes
after every box under its footprint. With boxes arriving on three infeed
conveyors, the robot's *empty* moves (place → next pick) depend on the
order: an asymmetric TSP with precedences.

- Nearest feasible neighbour; SA whose only move keeps feasibility (lift
  one box, reinsert it between its last predecessor and first successor).
- **CP-SAT:** `AddCircuit` over home + boxes, rank variables tied to arcs,
  precedences as rank inequalities; arcs that would violate the transitive
  closure of the precedences are never created.

## 7. Two-robot task allocation

Two robots on opposite sides, each with its own infeed; each can reach part
of the pallet, and the middle strip is shared. Each box is assigned to one
robot that reaches it; robots do one task at a time; supports first; no
two shared-zone tasks at the same time (collision avoidance); minimize
makespan.

- **List scheduling:** bottom-up dispatch to whichever robot finishes the task
  first.
- **CP-SAT:** one optional interval per (task, eligible robot), `ExactlyOne`
  per task, `NoOverlap` per robot, `NoOverlap` over shared-zone tasks,
  precedences on task start/end.

CP-SAT proves the optimal makespan on every E4 instance in seconds, and list
scheduling lands 3–18 % above it (7.8 % on average): greedy dispatch makes
the shared zone a bottleneck that an exact schedule routes around. The
exact model is both the better scheduler and the certificate that says how
good the dispatcher is.

## 8. Perception → state

- **Box on the conveyor:** RANSAC plane (conveyor) → points above it → top
  face by height band → radius outlier removal → PCA, refined by a
  minimum-area-rectangle search over ±12° → footprint from
  `side = √(12 · variance)` (exact for uniform points on a rectangle, and far
  less sensitive to edge noise than min/max) → snap to the 20 mm grid →
  nearest SKU.
- **Pallet:** per-cell median of the depth pixels, rounded to grid units.

The accuracy limit is point density (~1 point/cm²), not sensor noise:
dimension accuracy barely changes between 1 mm and 5 mm noise (E6).

## 9. Limitations and next steps

- **Transport stability** as an objective, not just a check: penalize tall
  slender columns, prefer interlocking, and use the simulator inside the
  search (e.g. re-rank the GA's top plans by simulated displacement).
- **Weight:** heavy boxes low, load limits on boxes underneath, and the
  pallet's centre of gravity.
- **Robot reachability and collision** of the gripper with neighbouring
  boxes, from a kinematic model rather than a reach range.
- **Learning:** a learned value function (or policy) to replace MCTS
  rollouts — the slowest part of the online planner.
- **Real data:** SKU statistics and measured friction instead of a synthetic
  catalogue; `sim_mujoco.py` is where those would be calibrated.
