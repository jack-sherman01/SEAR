"""Approach-and-press fastening controller.

Stateless (phase is inferred from the observation), which also makes it
amenable to grid-based certificate checks:

1. transit  -- velocity-tracking toward a hover waypoint above the target
2. descend  -- move down at ``approach_speed`` until contact is sensed
3. press    -- force feed-forward ``press_force`` with lateral alignment
"""

from __future__ import annotations

import math

import numpy as np

from sear.policies.base import Policy, register_policy
from sear.sim.env import Observation


@register_policy
class ApproachPressPolicy(Policy):
    kind = "approach_press"
    DEFAULTS = {
        "approach_speed": 0.15,  # m/s
        "hover_height": 0.04,  # m
        "press_force": 20.0,  # N
        "pos_gain": 6.0,  # 1/s, position error -> desired velocity
        "vel_gain": 60.0,  # N s/m, velocity tracking
        "contact_threshold": 2.0,  # N
        "yield_to_human": 0.0,  # >0.5: brake when a human is in the zone
        "high_power": 0.0,  # >0.5: request the high-power fastening mode
    }
    PARAM_BOUNDS = {
        "approach_speed": (0.01, 1.5),
        "hover_height": (0.0, 0.2),
        "press_force": (0.0, 120.0),
        "pos_gain": (0.5, 30.0),
        "vel_gain": (5.0, 400.0),
        "contact_threshold": (0.1, 20.0),
        "yield_to_human": (0.0, 1.0),
        "high_power": (0.0, 1.0),
    }

    def required_tools(self) -> tuple[str, ...]:
        tools = ("gripper.hold", "fastener.drive")
        return tools + (("fastener.high_power",) if self.params["high_power"] > 0.5 else ())

    def act(self, obs: Observation) -> np.ndarray:
        k = self.params
        p, v = obs.position, obs.velocity
        if k["yield_to_human"] > 0.5 and obs.human_in_zone:
            return -k["vel_gain"] * v
        force = obs.contact_force if math.isfinite(obs.contact_force) else 0.0
        dx = obs.target_x - p[0]
        if force > k["contact_threshold"]:
            ux = k["vel_gain"] * (k["pos_gain"] * dx - v[0])
            uy = -k["press_force"] - 0.1 * k["vel_gain"] * v[1]
            return np.array([ux, uy])
        if abs(dx) > 0.005:
            v_des = k["pos_gain"] * (np.array([obs.target_x, k["hover_height"]]) - p)
        else:
            v_des = np.array([k["pos_gain"] * dx, -k["approach_speed"]])
        speed = float(np.linalg.norm(v_des))
        if speed > k["approach_speed"]:
            v_des *= k["approach_speed"] / speed
        return k["vel_gain"] * (v_des - v)
