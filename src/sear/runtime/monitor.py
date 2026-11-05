"""Runtime monitor: decides when the candidate may not keep control.

Triggers (proposal WP3): unreliable sensing, a constraint approaching its
boundary, exit from the certified operating domain, infeasible safety filter,
and human intrusion into the protected zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sear.runtime.qp import QPResult
from sear.sim.env import Observation

# Reasons that release once cleared; all others latch for the rest of the episode.
NON_LATCHING = frozenset({"human_in_zone"})


@dataclass(frozen=True)
class CertifiedDomain:
    """Operating envelope over which the deployed certificate is valid."""

    position_low: tuple[float, float] = (-0.15, -0.02)
    position_high: tuple[float, float] = (0.65, 0.40)
    max_speed: float = 0.6
    max_measured_force: float = 40.0

    def violations(self, obs: Observation) -> list[str]:
        reasons = []
        p = obs.position
        if np.any(p < np.asarray(self.position_low)) or np.any(p > np.asarray(self.position_high)):
            reasons.append("domain_exit:position")
        if float(np.linalg.norm(obs.velocity)) > self.max_speed:
            reasons.append("domain_exit:speed")
        if math.isfinite(obs.contact_force) and obs.contact_force > self.max_measured_force:
            reasons.append("domain_exit:force")
        return reasons


@dataclass(frozen=True)
class MonitorStatus:
    ok: bool
    reasons: tuple[str, ...] = ()

    @property
    def latching(self) -> bool:
        return any(r not in NON_LATCHING for r in self.reasons)


@dataclass
class RuntimeMonitor:
    domain: CertifiedDomain = CertifiedDomain()
    force_guard: float = 34.0  # N: approach-to-boundary threshold for contact force
    slip_guard: float = 0.3  # normalized slip threshold (contract limit 0.5)

    def check(self, obs: Observation, qp: QPResult | None) -> MonitorStatus:
        reasons: list[str] = []
        if obs.human_in_zone:
            reasons.append("human_in_zone")
        if not obs.sensor_ok or not math.isfinite(obs.contact_force):
            reasons.append("sensor_fault")
        if qp is not None and not qp.feasible:
            reasons.append("filter_infeasible")
        if math.isfinite(obs.contact_force) and obs.contact_force > self.force_guard:
            reasons.append("force_near_limit")
        if obs.tool_slip > self.slip_guard:
            reasons.append("tool_slip_near_limit")
        reasons.extend(self.domain.violations(obs))
        return MonitorStatus(not reasons, tuple(reasons))
