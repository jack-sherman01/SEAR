"""Simplex-style runtime assurance (proposal WP3).

Per control step:

1. the candidate policy proposes ``u_agent``;
2. the CBF-QP filter returns the closest admissible ``u*``
       min ||u - u_agent||_H^2  s.t.  L_f h + L_g h u + alpha(h) >= 0,  |u| <= u_max;
3. the runtime monitor may hand control to a verified fallback (safe stop or
   the last verified baseline policy).

Modes are recorded per step: 0 = nominal, 1 = filtered, 2 = fallback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from sear.contracts.contract import Contract
from sear.runtime.cbf import Barrier, NominalModel, barriers_from_contract
from sear.runtime.fallback import SafeStopController
from sear.runtime.monitor import RuntimeMonitor
from sear.runtime.qp import QPResult, solve_min_norm_qp
from sear.sim.env import Observation


def domain_from_contract(contract: Contract) -> "CertifiedDomain":
    """Derive a CertifiedDomain from the contract's workspace and speed limits."""
    from sear.contracts.constraints import SignalBound, WorkspaceBox
    from sear.runtime.monitor import CertifiedDomain
    pos_low: tuple[float, ...] = (-0.15, -0.02)
    pos_high: tuple[float, ...] = (0.65, 0.40)
    max_speed = 0.6
    max_force = 40.0
    for c in contract.of_type(WorkspaceBox):
        if c.signal == "ee_pos":
            pos_low = tuple(float(v) for v in c.low)
            pos_high = tuple(float(v) for v in c.high)
    for c in contract.of_type(SignalBound):
        if type(c) is SignalBound and c.signal == "ee_speed" and c.upper is not None:
            max_speed = float(c.upper)
        if type(c) is SignalBound and c.signal == "contact_force" and c.upper is not None:
            max_force = float(c.upper)
    return CertifiedDomain(position_low=pos_low, position_high=pos_high,
                           max_speed=max_speed, max_measured_force=max_force)


class RTAMode(IntEnum):
    NOMINAL = 0
    FILTERED = 1
    FALLBACK = 2


@dataclass
class RuntimeAssuranceConfig:
    use_filter: bool = True
    use_monitor: bool = True
    u_max: float = 60.0  # actuator saturation (norm); the QP uses the inscribed box
    position_margin: float = 0.0005
    force_fraction: float = 0.85
    speed_fraction: float = 0.9
    latency_steps: int = 2  # actuation delay the filter compensates for (certified bound)
    dt: float = 0.01
    model: NominalModel = field(default_factory=NominalModel)
    monitor: RuntimeMonitor = field(default_factory=RuntimeMonitor)


class RuntimeAssurance:
    def __init__(self, contract: Contract, config: RuntimeAssuranceConfig | None = None, fallback=None) -> None:
        self.contract = contract
        self.config = config or RuntimeAssuranceConfig()
        self.fallback = fallback or SafeStopController()
        self.reset()

    # ----------------------------------------------------------------- API
    def reset(self) -> None:
        self._barriers: list[Barrier] | None = None
        self._latched = False
        self._modes: list[int] = []
        self._intervention: list[float] = []
        self._reasons: dict[str, int] = {}
        self._first_fallback_t: float | None = None
        self._pending: deque[np.ndarray] = deque(
            (np.zeros(2) for _ in range(self.config.latency_steps)), maxlen=max(1, self.config.latency_steps))
        self.fallback.reset()

    def barriers(self, obs: Observation) -> list[Barrier]:
        if self._barriers is None:
            c = self.config
            self._barriers = barriers_from_contract(
                self.contract, c.model, obs.obstacle_center, obs.obstacle_radius,
                position_margin=c.position_margin, force_fraction=c.force_fraction,
                speed_fraction=c.speed_fraction)
        return self._barriers

    def predict(self, obs: Observation, f_ext: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """State at which the next command takes effect (latency compensation).

        Commands already sent but not yet applied are rolled through the
        nominal model; external force is held at its measured value.
        """
        p, v = obs.position.copy(), obs.velocity.copy()
        if self.config.latency_steps <= 0:
            return p, v
        m = self.config.model
        for u in self._pending:
            v = v + m.drift(v, f_ext) * self.config.dt + u / m.mass * self.config.dt
            p = p + v * self.config.dt
        return p, v

    def solve(self, obs: Observation, u_agent: np.ndarray) -> QPResult:
        f_meas = obs.contact_force if np.isfinite(obs.contact_force) else 0.0
        f_ext = np.array([0.0, f_meas])
        p, v = self.predict(obs, f_ext)
        rows, rhs = [], []
        for bar in self.barriers(obs):
            a, b = bar.constraint(p, v, f_ext, self.config.model)
            rows.append(a)
            rhs.append(b)
        box = self.config.u_max / np.sqrt(2.0)
        for i in range(2):
            e = np.eye(2)[i]
            rows += [e, -e]
            rhs += [-box, -box]
        return solve_min_norm_qp(u_agent, np.eye(2), np.array(rows), np.array(rhs))

    def filter(self, obs: Observation, u_agent: np.ndarray) -> np.ndarray:
        cfg = self.config
        u_agent = np.asarray(u_agent, float)
        qp = self.solve(obs, u_agent) if cfg.use_filter else None
        if cfg.use_monitor:
            status = cfg.monitor.check(obs, qp)
            if status.latching:
                self._latched = True
            if self._latched or not status.ok:
                for r in status.reasons:
                    self._reasons[r] = self._reasons.get(r, 0) + 1
                return self._record(obs, RTAMode.FALLBACK, u_agent, self.fallback.act(obs))
        if qp is None:
            return self._record(obs, RTAMode.NOMINAL, u_agent, u_agent)
        changed = float(np.linalg.norm(qp.u - u_agent)) > 1e-6
        return self._record(obs, RTAMode.FILTERED if changed else RTAMode.NOMINAL, u_agent, qp.u)

    def episode_signals(self) -> dict[str, list[float]]:
        return {"rta_mode": list(map(float, self._modes)), "rta_intervention": list(self._intervention)}

    def summary(self) -> dict:
        modes = np.asarray(self._modes)
        n = max(1, len(modes))
        return {
            "steps": int(len(modes)),
            "filtered_fraction": float(np.sum(modes == RTAMode.FILTERED) / n),
            "fallback_fraction": float(np.sum(modes == RTAMode.FALLBACK) / n),
            "fallback_used": bool(np.any(modes == RTAMode.FALLBACK)),
            "first_fallback_t": self._first_fallback_t,
            "fallback_reasons": dict(self._reasons),
        }

    # ------------------------------------------------------------- helpers
    def _record(self, obs: Observation, mode: RTAMode, u_agent: np.ndarray, u: np.ndarray) -> np.ndarray:
        if mode is RTAMode.FALLBACK and self._first_fallback_t is None:
            self._first_fallback_t = obs.t
        self._modes.append(int(mode))
        self._intervention.append(float(np.linalg.norm(u - u_agent)))
        if self.config.latency_steps > 0:
            self._pending.append(np.asarray(u, float))
        return u
