"""Physics check of a stacking plan in MuJoCo.

The planners decide stability with a geometric rule (≥ α of each bottom face
supported). This module asks the physics engine instead: every box becomes a
free rigid body at its planned pose, the load settles under gravity, and the
pallet is then pushed sideways — a forklift braking or a truck cornering —
by tilting gravity in the pallet's frame. Boxes that end up far from their
planned pose are counted as failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

from .geometry import PalletSpec, Placement


@dataclass
class SimReport:
    displacement_mm: np.ndarray  # per box, after the whole test
    tilt_deg: np.ndarray
    settled_displacement_mm: np.ndarray  # per box, after settling only

    def failures(self, tol_mm: float = 20.0, tol_deg: float = 5.0) -> int:
        return int(np.sum((self.displacement_mm > tol_mm) | (self.tilt_deg > tol_deg)))

    def settle_failures(self, tol_mm: float = 10.0) -> int:
        return int(np.sum(self.settled_displacement_mm > tol_mm))


def plan_to_mjcf(
    spec: PalletSpec,
    plan: Sequence[Placement],
    friction: float = 0.5,
    density: float = 200.0,
    gap_mm: float = 0.5,
) -> str:
    """MJCF scene: a static pallet deck (top face at z = 0) and one free box per
    placement. Boxes start `gap_mm` above their planned height so they never
    begin interpenetrating.
    """
    u = spec.unit_mm / 1000.0  # grid unit in metres
    gap = gap_mm / 1000.0
    L, W = spec.length * u, spec.width * u
    bodies = []
    for k, p in enumerate(plan):
        hx, hy, hz = p.dx * u / 2, p.dy * u / 2, p.dz * u / 2
        cx, cy = p.x * u + hx, p.y * u + hy
        cz = p.z * u + hz + gap
        bodies.append(
            f'<body name="box{k}" pos="{cx:.5f} {cy:.5f} {cz:.5f}">'
            f'<freejoint/><geom type="box" size="{hx:.5f} {hy:.5f} {hz:.5f}" '
            f'density="{density}" friction="{friction} 0.005 0.0001" rgba="0.76 0.6 0.42 1"/></body>'
        )
    return f"""
<mujoco>
  <!-- elliptic friction cones + impratio: MuJoCo's recommended setting against
       contact creep; the default pyramid cone lets stacked boxes drift mm/s -->
  <option timestep="0.002" iterations="50" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="deck" type="box" pos="{L / 2:.4f} {W / 2:.4f} -0.075" size="{L / 2:.4f} {W / 2:.4f} 0.075"
          friction="{friction} 0.005 0.0001" rgba="0.55 0.42 0.25 1"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>
"""


def _tilt_deg(quat: np.ndarray) -> np.ndarray:
    """Angle between each box's z axis and the vertical."""
    w, x, y, z = quat.T
    cos_tilt = 1 - 2 * (x**2 + y**2)  # (R · e_z) · e_z for unit quaternions
    return np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))


def simulate_plan(
    spec: PalletSpec,
    plan: Sequence[Placement],
    settle_s: float = 1.0,
    lateral_g: float = 0.3,
    lateral_s: float = 1.0,
    friction: float = 0.5,
) -> SimReport:
    if not plan:
        empty = np.zeros(0)
        return SimReport(empty, empty, empty)
    model = mujoco.MjModel.from_xml_string(plan_to_mjcf(spec, plan, friction))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = data.xpos[1:].copy()  # body 0 is the world

    for _ in range(int(settle_s / model.opt.timestep)):
        mujoco.mj_step(model, data)
    settled = data.xpos[1:].copy()

    # in the pallet's frame, braking at a·g looks like gravity tilted by a
    model.opt.gravity[:] = (lateral_g * 9.81, 0.0, -9.81)
    for _ in range(int(lateral_s / model.opt.timestep)):
        mujoco.mj_step(model, data)

    return SimReport(
        displacement_mm=np.linalg.norm(data.xpos[1:] - start, axis=1) * 1000,
        tilt_deg=_tilt_deg(data.xquat[1:]),
        settled_displacement_mm=np.linalg.norm(settled - start, axis=1) * 1000,
    )
