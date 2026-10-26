"""Online ("random") palletizing: boxes arrive one at a time, unknown in advance.

The robot sees the next `buffer_size` boxes on the conveyor (a lookahead
buffer) and must commit each placement before seeing the rest of the stream.
The episode ends when no box in the buffer can be placed stably.

Policies
--------
- `GreedyPolicy`: best DBLF placement of the box at the head of the buffer
  (buffer_size = 1 is pure online).
- `BufferGreedyPolicy`: best DBLF placement over every box in the buffer.
- `MCTSPolicy`: UCT search whose rollouts finish the episode. The unseen
  part of the stream is *sampled* from the SKU distribution
  (determinization): each sample turns the problem deterministic, a UCT tree
  is grown per sample, and root visit counts are summed across samples
  before committing — a sample-average approximation of the stochastic
  problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .geometry import Box, PalletSpec, Placement, utilization
from .heightmap import HeightMap
from .instances import sample_skus, sku_box


class Policy(Protocol):
    def choose(self, hm: HeightMap, buffer: list[Box], rng: np.random.Generator) -> Placement | None: ...


@dataclass
class EpisodeResult:
    plan: list[Placement]
    utilization: float
    boxes_seen: int


def run_episode(
    spec: PalletSpec,
    stream: Sequence[Box],
    policy: Policy,
    buffer_size: int = 1,
    seed: int | None = None,
) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    hm = HeightMap(spec)
    buffer = list(stream[:buffer_size])
    nxt = buffer_size
    plan: list[Placement] = []
    while buffer:
        p = policy.choose(hm, buffer, rng)
        if p is None:
            break
        hm.place(p)
        plan.append(p)
        buffer = [b for b in buffer if b.id != p.box_id]
        if nxt < len(stream):
            buffer.append(stream[nxt])
            nxt += 1
    return EpisodeResult(plan, utilization(spec, plan), nxt)


class GreedyPolicy:
    """Place the box at the head of the buffer at its best position."""

    def __init__(self, placement: str = "dblf") -> None:
        self.placement = placement

    def choose(self, hm: HeightMap, buffer: list[Box], rng: np.random.Generator | None) -> Placement | None:
        return hm.best_placement(buffer[0], self.placement)


class BufferGreedyPolicy:
    """Pick the best (box, position) over the whole buffer.

    With DBLF, the lowest resting height wins and ties go to the bigger box.
    With min-waste, placements are compared by trapped air *per unit of
    volume placed*, so a big box that wastes a little can beat a small box
    that wastes nothing.
    """

    def __init__(self, placement: str = "dblf") -> None:
        self.placement = placement

    def choose(self, hm: HeightMap, buffer: list[Box], rng: np.random.Generator | None) -> Placement | None:
        found = [p for b in buffer if (p := hm.best_placement(b, self.placement)) is not None]
        if self.placement == "dblf":
            return min(found, key=lambda p: (p.z, -p.volume, p.x, p.y), default=None)
        return min(found, key=lambda p: (hm.trapped_air(p) / p.volume, p.z, -p.volume), default=None)


def candidate_actions(hm: HeightMap, buffer: list[Box], per_box: int, placement: str) -> list[Placement]:
    """A few good placements per buffered box — the MCTS branching set."""
    actions = []
    for b in buffer:
        actions.extend(hm.candidates(b, per_orientation=per_box, rule=placement)[:per_box])
    return actions


@dataclass
class _Node:
    hm: HeightMap
    buffer: list[Box]
    arrivals: int  # how many sampled boxes have joined the buffer so far
    packed: int  # volume placed since the root
    actions: list[Placement] | None = None
    children: dict[int, _Node] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0


class MCTSPolicy:
    """Determinized UCT whose rollouts play the episode to the end.

    Each determinization fixes one sampled future stream, which makes the
    problem deterministic; a UCT tree is grown on it, and the value of a
    node is the *final* utilization reached by greedy rollouts from there —
    the quantity the episode is actually scored on (counted from the root:
    what is already on the pallet is the same for every action). Root visit
    counts are summed over determinizations before committing.

    Args:
        samples: number of sampled futures (determinizations).
        iterations: UCT iterations per sample.
        per_box: candidate positions per buffered box (branching control).
        c: UCB1 exploration constant; values are utilizations in [0, 1],
           and good and bad moves typically differ by a few percent.
    """

    def __init__(
        self,
        samples: int = 3,
        iterations: int = 16,
        per_box: int = 2,
        c: float = 0.05,
        placement: str = "dblf",
        unit_mm: int = 20,
    ) -> None:
        self.samples, self.iterations = samples, iterations
        self.per_box, self.c, self.unit_mm = per_box, c, unit_mm
        self.placement = placement

    def _child(self, node: _Node, p: Placement, future: list[Box]) -> _Node:
        hm = node.hm.copy()
        hm.place(p)
        buffer = [b for b in node.buffer if b.id != p.box_id]
        arrivals = node.arrivals
        if arrivals < len(future):
            buffer.append(future[arrivals])
            arrivals += 1
        return _Node(hm, buffer, arrivals, node.packed + p.volume)

    def _rollout(self, node: _Node, future: list[Box]) -> float:
        greedy = BufferGreedyPolicy(self.placement)
        hm, buffer, arrivals, packed = node.hm.copy(), list(node.buffer), node.arrivals, node.packed
        while buffer:
            p = greedy.choose(hm, buffer, None)
            if p is None:
                break
            hm.place(p)
            packed += p.volume
            buffer = [b for b in buffer if b.id != p.box_id]
            if arrivals < len(future):
                buffer.append(future[arrivals])
                arrivals += 1
        return packed / hm.spec.volume

    def _simulate(self, node: _Node, future: list[Box]) -> float:
        if node.actions is None:
            node.actions = candidate_actions(node.hm, node.buffer, self.per_box, self.placement)
        if not node.actions:  # episode over in this branch
            value = node.packed / node.hm.spec.volume
        elif len(node.children) < len(node.actions):  # expand one new child
            k = len(node.children)
            child = self._child(node, node.actions[k], future)
            node.children[k] = child
            value = self._rollout(child, future)
            child.visits += 1
            child.value += value
        else:  # descend by UCB1
            log_n = math.log(node.visits)
            k = max(
                node.children,
                key=lambda a: node.children[a].value / node.children[a].visits
                + self.c * math.sqrt(log_n / node.children[a].visits),
            )
            value = self._simulate(node.children[k], future)
        node.visits += 1
        node.value += value
        return value

    def choose(self, hm: HeightMap, buffer: list[Box], rng: np.random.Generator) -> Placement | None:
        actions = candidate_actions(hm, buffer, self.per_box, self.placement)
        if len(actions) <= 1:
            return actions[0] if actions else None
        visits = np.zeros(len(actions))
        values = np.zeros(len(actions))
        # a pallet holds well under 100 boxes, so 100 sampled arrivals suffice
        horizon = 100
        for _ in range(self.samples):
            future = [
                sku_box(sku, 10**6 + k, self.unit_mm)  # ids never clash with real boxes
                for k, sku in enumerate(sample_skus(horizon, rng))
            ]
            root = _Node(hm, list(buffer), 0, 0, actions=actions)
            for _ in range(self.iterations):
                self._simulate(root, future)
            for k, child in root.children.items():
                visits[k] += child.visits
                values[k] += child.value
        # most visited wins; mean value breaks ties
        best = max(range(len(actions)), key=lambda k: (visits[k], values[k] / max(visits[k], 1)))
        return actions[best]
