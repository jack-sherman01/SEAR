"""Carry-and-place transport controller.

Stateless (phase inferred from observation):

1. transit  -- move at carry_height with carry_speed toward target_x
2. descend  -- approach surface at approach_speed once laterally aligned
3. place    -- maintain contact at target; task completes when tool_slip < 1.0

Safety tension: higher carry_speed → higher lateral acceleration → more
inertial slip during transit (fn=0, CBF inactive → domain gap).
"""

from __future__ import annotations

import numpy as np

from sear.policies.base import Policy, register_policy
from sear.sim.env import Observation

LATERAL_TOL = 0.015  # m, switch from transit to descend


@register_policy
class CarryTransportPolicy(Policy):
    kind = "carry_transport"
    DEFAULTS = {
        "carry_speed": 0.20,      # m/s, transit speed
        "carry_height": 0.08,     # m, height during transit
        "approach_speed": 0.10,   # m/s, descent speed
        "grip_force": 25.0,       # N, downward press at place
        "contact_threshold": 2.0, # N, below this is treated as free-space
        "pos_gain": 6.0,          # 1/s, position → desired velocity
        "vel_gain": 60.0,         # N·s/m, velocity tracking
        "yield_to_human": 0.0,    # >0.5: brake when human in zone
    }
    PARAM_BOUNDS = {
        "carry_speed":       (0.01, 2.0),
        "carry_height":      (0.01, 0.3),
        "approach_speed":    (0.01, 0.5),
        "grip_force":        (0.0, 80.0),
        "contact_threshold": (0.5, 20.0),
        "pos_gain":          (0.5, 30.0),
        "vel_gain":          (5.0, 400.0),
        "yield_to_human":    (0.0, 1.0),
    }

    def required_tools(self) -> tuple[str, ...]:
        return ("gripper.hold", "gripper.transport")

    def act(self, obs: Observation) -> np.ndarray:
        k = self.params
        p, v = obs.position, obs.velocity

        if k["yield_to_human"] > 0.5 and obs.human_in_zone:
            return -k["vel_gain"] * v

        dx = obs.target_x - p[0]
        in_contact = obs.contact_force > k["contact_threshold"] if obs.sensor_ok else False

        if in_contact:
            # Phase 3: place — hold grip and correct laterally
            ux = float(np.clip(k["vel_gain"] * (k["pos_gain"] * dx - v[0]), -10.0, 10.0))
            uy = -k["grip_force"] - 0.1 * k["vel_gain"] * v[1]
            return np.array([ux, uy])

        if abs(dx) > LATERAL_TOL:
            # Phase 1: transit at carry_height — drives inertial slip
            target = np.array([obs.target_x, k["carry_height"]])
            v_des = k["pos_gain"] * (target - p)
            speed = float(np.linalg.norm(v_des))
            if speed > k["carry_speed"]:
                v_des *= k["carry_speed"] / speed
            return k["vel_gain"] * (v_des - v)

        # Phase 2: descend
        v_des = np.array([k["pos_gain"] * dx, -k["approach_speed"]])
        return k["vel_gain"] * (v_des - v)
