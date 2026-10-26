"""Figures: 3-D pallet loads, search convergence, and robot schedules."""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .geometry import PalletSpec, Placement  # noqa: E402

PALETTE = ["#c8a27a", "#9fb7a0", "#8fa6c4", "#d6b25e", "#c48f8f", "#a99bc4", "#7fb3b0", "#c7a0b8"]


def _cuboid_faces(x, y, z, dx, dy, dz):
    p = [
        (x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z),
        (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz),
    ]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[p[i] for i in f] for f in idx]


def draw_plan(ax, spec: PalletSpec, plan: Sequence[Placement], colors: Sequence[str] | None = None, title: str = ""):
    u = spec.unit_mm
    # One collection for every face, so matplotlib depth-sorts faces across
    # boxes (separate collections are sorted only as whole boxes and overlap
    # in the wrong order).
    faces = _cuboid_faces(0, 0, -150, spec.length * u, spec.width * u, 150)
    face_colors = ["#8a6d4b"] * 6
    for k, p in enumerate(plan):
        faces += _cuboid_faces(p.x * u, p.y * u, p.z * u, p.dx * u, p.dy * u, p.dz * u)
        face_colors += [colors[k] if colors else PALETTE[k % len(PALETTE)]] * 6
    ax.add_collection3d(
        Poly3DCollection(faces, facecolors=face_colors, edgecolor="#333333", linewidth=0.35, alpha=1.0)
    )
    ax.set_xlim(0, spec.length * u)
    ax.set_ylim(0, spec.width * u)
    ax.set_zlim(-150, spec.max_height * u)
    ax.set_box_aspect((spec.length, spec.width, spec.max_height + 150 / u))
    ax.set_xlabel("x (mm)", fontsize=7)
    ax.set_ylabel("y (mm)", fontsize=7)
    ax.set_zlabel("z (mm)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=24, azim=-58)
    if title:
        ax.set_title(title, fontsize=9)


def sku_colors(plan: Sequence[Placement], sku_of: dict[int, str]) -> list[str]:
    skus = sorted(set(sku_of.values()))
    return [PALETTE[skus.index(sku_of[p.box_id]) % len(PALETTE)] for p in plan]


def save_plans(path: str, spec: PalletSpec, panels: Sequence[tuple[str, Sequence[Placement], Sequence[str] | None]]) -> None:
    fig = plt.figure(figsize=(4.2 * len(panels), 4.8))
    for k, (title, plan, colors) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d")
        draw_plan(ax, spec, plan, colors, title)
    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.04, top=0.88, wspace=0.05)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_convergence(path: str, curves: dict[str, Sequence[float]], baseline: float | None = None) -> None:
    """curves: best utilization so far, per decoder evaluation."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, values in curves.items():
        ax.plot(range(1, len(values) + 1), [100 * v for v in values], label=name, linewidth=1.6)
    if baseline is not None:
        ax.axhline(100 * baseline, color="gray", linestyle="--", linewidth=1, label="best greedy")
    ax.set_xlabel("decoder evaluations")
    ax.set_ylabel("best utilization so far (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gantt(path: str, rows: Sequence[str], tasks: Sequence[tuple[int, float, float, bool]], title: str = "") -> None:
    """tasks: (row index, start s, duration s, in shared zone)."""
    fig, ax = plt.subplots(figsize=(8, 2.2))
    for row, start, dur, shared in tasks:
        ax.barh(row, dur, left=start, height=0.6,
                color="#c48f8f" if shared else "#8fa6c4", edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("time (s)  —  red: task inside the shared zone")
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
