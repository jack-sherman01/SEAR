"""Verified fallback controllers."""

from __future__ import annotations

import math

import numpy as np

from sear.sim.env import Observation


class SafeStopController:
    """Brake to rest; if in contact, back off the surface first.

    Safe by construction for the nominal model: the commanded force opposes
    velocity (energy decreasing) and the only non-dissipative term pushes
    *away* from the work surface.
    """

    def __init__(self, brake_gain: float = 60.0, retract_force: float = 6.0, contact_threshold: float = 1.0) -> None:
        self.brake_gain = brake_gain
        self.retract_force = retract_force
        self.contact_threshold = contact_threshold

    def reset(self) -> None:
        pass

    def required_tools(self) -> tuple[str, ...]:
        return ()

    def act(self, obs: Observation) -> np.ndarray:
        u = -self.brake_gain * obs.velocity
        force = obs.contact_force
        if not math.isfinite(force) or force > self.contact_threshold or obs.position[1] < 0.0:
            u = u + np.array([0.0, self.retract_force])
        return u
