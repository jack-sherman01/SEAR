"""Adaptive latency-compensation for hardware CBF-QP deployment.

The simulation RuntimeAssurance uses a fixed ``latency_steps`` parameter.
On real hardware, loop latency varies with control load, OS scheduling, and
network round-trips.  This module provides:

  * :class:`LatencyMonitor`     — rolling-window latency tracker
  * :class:`AdaptiveLatencyConfig` — policy for translating measured latency
    to barrier margin tightening and latency_steps adjustment
  * :func:`tighten_margins`     — compute per-barrier margin increment given
    measured latency and barrier decay rate α

The barrier margin tightening formula (from the paper, §9):

    Δh_margin = α · δt_latency

where α is the barrier decay rate (``RuntimeAssuranceConfig.model``'s
characteristic decay, typically ``damping / mass``) and ``δt_latency`` is the
measured end-to-end latency (sense → compute → actuate).  This guarantees
that any state prediction error introduced by latency is absorbed before the
QP constraint is applied.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


@dataclass
class LatencyMeasurement:
    t_sense: float    # wall time when observation was read
    t_command: float  # wall time when command was sent
    loop_dt: float    # nominal control period


@dataclass
class AdaptiveLatencyConfig:
    """Policy for how measured latency maps to safety-margin tightening.

    Args:
        window: number of recent steps used for rolling average
        alpha: barrier decay rate used for margin computation (rad/s or 1/s);
               default matches damping/mass for the nominal nailing model (1.6)
        margin_min: minimum additional margin added even at zero latency (m)
        margin_max: cap on additional margin to avoid over-conservative QP (m)
        fallback_latency_s: if rolling average latency exceeds this threshold,
            trigger RuntimeAssurance fallback
        adapt_latency_steps: if True, update ``RuntimeAssuranceConfig.latency_steps``
            based on measured loop latency / nominal dt
    """

    window: int = 50
    alpha: float = 1.6          # damping/mass for default NominalModel(4.0/2.5)
    margin_min: float = 0.0005  # 0.5 mm floor (matches RuntimeAssuranceConfig.position_margin)
    margin_max: float = 0.010   # 10 mm cap
    fallback_latency_s: float = 0.020   # 20 ms → trigger safe stop
    adapt_latency_steps: bool = True


class LatencyMonitor:
    """Tracks control-loop latency and derives adaptive CBF margin increments.

    Usage::

        monitor = LatencyMonitor(config)

        # inside the control loop:
        t_sense = time.perf_counter()
        obs = robot.observe()
        u = rta.filter(obs, policy(obs))
        t_cmd = time.perf_counter()
        robot.command(u)
        meas = monitor.record(t_sense, t_cmd, dt=0.002)

        margin_extra = monitor.margin_increment()   # add to position_margin
        if monitor.should_fallback():
            robot.safe_stop()
    """

    def __init__(self, config: AdaptiveLatencyConfig | None = None) -> None:
        self.config = config or AdaptiveLatencyConfig()
        self._window: Deque[float] = deque(maxlen=self.config.window)
        self._latency_steps_estimate: int = 1
        self._fallback: bool = False

    def record(self, t_sense: float, t_command: float, dt: float) -> LatencyMeasurement:
        latency = t_command - t_sense
        self._window.append(latency)
        cfg = self.config
        if cfg.adapt_latency_steps and dt > 0:
            steps = max(1, round(self.mean_latency / dt))
            self._latency_steps_estimate = steps
        if self.mean_latency > cfg.fallback_latency_s:
            self._fallback = True
        return LatencyMeasurement(t_sense=t_sense, t_command=t_command, loop_dt=dt)

    @property
    def mean_latency(self) -> float:
        return float(np.mean(self._window)) if self._window else 0.0

    @property
    def max_latency(self) -> float:
        return float(max(self._window)) if self._window else 0.0

    @property
    def latency_steps(self) -> int:
        return self._latency_steps_estimate

    def margin_increment(self) -> float:
        """Additional position margin to add to CBF constraints due to latency.

        Δh_margin = clip(α · mean_latency, margin_min, margin_max)
        """
        cfg = self.config
        delta = cfg.alpha * self.mean_latency
        return float(np.clip(delta, cfg.margin_min, cfg.margin_max))

    def should_fallback(self) -> bool:
        return self._fallback

    def reset_fallback(self) -> None:
        self._fallback = False

    def summary(self) -> dict:
        return {
            "mean_latency_ms": self.mean_latency * 1e3,
            "max_latency_ms": self.max_latency * 1e3,
            "latency_steps": self.latency_steps,
            "margin_increment_m": self.margin_increment(),
            "fallback_triggered": self._fallback,
            "n_samples": len(self._window),
        }


def tighten_margins(
    base_margin: float,
    latency_s: float,
    alpha: float,
    margin_min: float = 0.0005,
    margin_max: float = 0.010,
) -> float:
    """Compute tightened position margin given measured latency.

    Args:
        base_margin: nominal position margin from RuntimeAssuranceConfig
        latency_s: measured end-to-end control-loop latency (seconds)
        alpha: barrier decay rate (1/s), typically damping/mass
        margin_min: minimum total margin
        margin_max: maximum total margin

    Returns:
        Total position margin = base_margin + clip(alpha * latency_s, 0, margin_max)
    """
    extra = float(np.clip(alpha * latency_s, 0.0, margin_max))
    return max(base_margin + extra, margin_min)


def apply_latency_compensation(
    rta_config: "RuntimeAssuranceConfig",  # type: ignore[name-defined]
    monitor: LatencyMonitor,
) -> "RuntimeAssuranceConfig":  # type: ignore[name-defined]
    """Return a copy of RuntimeAssuranceConfig with latency-adjusted parameters.

    Call this once per control step (or whenever latency statistics update)
    to keep the CBF margin current.  The original config is not mutated.
    """
    from dataclasses import replace
    cfg = rta_config
    new_margin = tighten_margins(
        base_margin=cfg.position_margin,
        latency_s=monitor.mean_latency,
        alpha=monitor.config.alpha,
        margin_min=monitor.config.margin_min,
        margin_max=monitor.config.margin_max,
    )
    return replace(cfg,
                   position_margin=new_margin,
                   latency_steps=monitor.latency_steps)
