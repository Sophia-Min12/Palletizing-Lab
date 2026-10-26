"""palletlab — mixed-case palletizing and robot task optimization.

Modules, in pipeline order:
    geometry        boxes, placements, independent feasibility checker
    instances       SKU streams and perfect-packing puzzles
    heightmap       placement engine (DBLF / min-waste rules)
    construct       decoder + greedy sort rules
    metaheuristics  simulated annealing, BRKGA
    exact_cpsat     exact CP-SAT model
    online          random palletizing: greedy, buffer, MCTS
    sequencing      robot build order (sequential ordering problem)
    allocation      two-robot task allocation and scheduling
    sim_mujoco      physics validation in MuJoCo
    perception      point cloud / depth → planner inputs
    viz             figures
"""

__version__ = "0.1.0"
