"""Execution traces evaluated by safety contracts and metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Trajectory:
    """Time series of named signals plus discrete events (e.g. tool calls).

    ``signals["t"]`` holds sample times; every other signal has the same
    leading dimension.
    """

    dt: float
    signals: dict[str, np.ndarray]
    events: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "simulation"
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def t(self) -> np.ndarray:
        return self.signals["t"]

    def __len__(self) -> int:
        return int(len(self.signals.get("t", ())))

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self) else 0.0

    def has(self, name: str) -> bool:
        return name in self.signals

    def get(self, name: str) -> np.ndarray:
        return self.signals[name]


class TrajectoryRecorder:
    def __init__(self, dt: float, mode: str = "simulation") -> None:
        self.dt = dt
        self.mode = mode
        self._rows: dict[str, list[Any]] = defaultdict(list)
        self._events: list[dict[str, Any]] = []

    def record(self, **signals: Any) -> None:
        for name, value in signals.items():
            self._rows[name].append(value)

    def event(self, **event: Any) -> None:
        self._events.append(dict(event))

    def finalize(self, **info: Any) -> Trajectory:
        signals = {name: np.asarray(values, dtype=float) for name, values in self._rows.items()}
        lengths = {len(v) for v in signals.values()}
        if len(lengths) > 1:
            raise ValueError(f"signals have inconsistent lengths: {lengths}")
        return Trajectory(dt=self.dt, signals=signals, events=list(self._events), mode=self.mode, info=dict(info))
