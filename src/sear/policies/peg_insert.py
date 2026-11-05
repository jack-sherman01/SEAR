"""Approach-align-insert controller for the peg-in-hole task.

Phases (inferred from observation, stateless):

1. transit  -- velocity-track toward a hover waypoint above the hole
2. descend  -- move down slowly once laterally aligned
3. insert   -- force feed-forward ``insert_force`` with tight lateral
               centring once contact is detected or y <= 0
"""

from __future__ import annotations

import math

import numpy as np

from sear.policies.base import Policy, register_policy
from sear.sim.env import Observation

INSERT_LATERAL_MAX = 10.0  # N, lateral centring correction while inserting


@register_policy
class PegInsertPolicy(Policy):
    kind = "peg_insert"
    DEFAULTS = {
        "approach_speed": 0.10,    # m/s, transit and lateral alignment
        "insert_speed": 0.05,      # m/s, slow descent to hole entrance
        "hover_height": 0.04,      # m above surface for alignment
        "insert_force": 15.0,      # N, downward force during insertion
        "pos_gain": 6.0,           # 1/s, position → desired velocity
        "vel_gain": 60.0,          # N s/m, velocity tracking
        "contact_threshold": 1.5,  # N, switches to insertion mode
        "yield_to_human": 0.0,     # >0.5: brake when human in zone
    }
    PARAM_BOUNDS = {
        "approach_speed": (0.01, 0.5),
        "insert_speed": (0.01, 0.3),
        "hover_height": (0.0, 0.2),
        "insert_force": (0.0, 80.0),
        "pos_gain": (0.5, 30.0),
        "vel_gain": (5.0, 400.0),
        "contact_threshold": (0.1, 10.0),
        "yield_to_human": (0.0, 1.0),
    }

    def required_tools(self) -> tuple[str, ...]:
        return ("gripper.hold", "probe.insert")

    def act(self, obs: Observation) -> np.ndarray:
        k = self.params
        p, v = obs.position, obs.velocity

        if k["yield_to_human"] > 0.5 and obs.human_in_zone:
            return -k["vel_gain"] * v

        force = obs.contact_force if math.isfinite(obs.contact_force) else 0.0
        dx = obs.target_x - p[0]

        # Phase 3: in contact or already below surface → insertion
        if force > k["contact_threshold"] or p[1] <= 0.002:
            ux = float(np.clip(k["vel_gain"] * (k["pos_gain"] * dx - v[0]),
                               -INSERT_LATERAL_MAX, INSERT_LATERAL_MAX))
            uy = -k["insert_force"] - 0.1 * k["vel_gain"] * v[1]
            return np.array([ux, uy])

        # Phase 1: transit to hover waypoint if not yet aligned / at height
        if abs(dx) > 0.005 or p[1] > k["hover_height"] + 0.01:
            v_des = k["pos_gain"] * (np.array([obs.target_x, k["hover_height"]]) - p)
            speed = float(np.linalg.norm(v_des))
            if speed > k["approach_speed"]:
                v_des *= k["approach_speed"] / speed
            return k["vel_gain"] * (v_des - v)

        # Phase 2: aligned at hover height → descend slowly to hole entrance
        v_des = np.array([k["pos_gain"] * dx, -k["insert_speed"]])
        speed = float(np.linalg.norm(v_des))
        if speed > k["insert_speed"] * 1.5:
            v_des *= k["insert_speed"] * 1.5 / speed
        return k["vel_gain"] * (v_des - v)
