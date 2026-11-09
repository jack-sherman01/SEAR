"""Mock RobotInterface for unit testing and simulation-in-the-loop runs.

Wraps a :class:`sear.sim.env.ContactToolEnv` so that the hardware control
loop in ``sear.robots`` can be exercised without physical hardware.  The mock
also exposes a :class:`MockSafetyInterlock` whose ``engaged`` state can be set
programmatically to test e-stop / guarded-mode paths.
"""

from __future__ import annotations

import numpy as np

from sear.robots.base import RobotInterface, SafetyInterlock
from sear.sim.env import ContactToolEnv, Observation
from sear.sim.scenario import Scenario


class MockSafetyInterlock(SafetyInterlock):
    """Software-controlled interlock for testing."""

    def __init__(self) -> None:
        self._engaged = False

    def engage(self) -> None:
        self._engaged = True

    def release(self) -> None:
        self._engaged = False

    def engaged(self) -> bool:
        return self._engaged


class MockRobot(RobotInterface):
    """Thin wrapper around :class:`ContactToolEnv` implementing :class:`RobotInterface`.

    Args:
        scenario: the scenario to simulate
        allowed_tools: tool permissions passed to the environment
        dt: control period (seconds)
    """

    def __init__(
        self,
        scenario: Scenario | None = None,
        dt: float = 0.002,
    ) -> None:
        sc = scenario or Scenario("mock", seed=0)
        self._env = ContactToolEnv(sc)
        self._env.reset()
        self._dt = dt
        self._obs: Observation = self._env._observe()
        self._stopped = False
        self._mode = "normal"
        self.interlock = MockSafetyInterlock()
        self.commanded: list[np.ndarray] = []  # history of commanded actions

    def observe(self) -> Observation:
        return self._obs

    def command(self, u: np.ndarray) -> None:
        if self._stopped or self.interlock.engaged():
            return
        self.commanded.append(np.asarray(u, dtype=float))
        self._obs = self._env.step(u)

    def safe_stop(self) -> None:
        self._stopped = True

    def set_mode(self, mode: str) -> None:
        allowed = {"normal", "low_energy", "guarded"}
        if mode not in allowed:
            raise ValueError(f"unknown mode {mode!r}; choose from {allowed}")
        self._mode = mode

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def mode(self) -> str:
        return self._mode

    def reset(self) -> None:
        self._env.reset()
        self._obs = self._env._observe()
        self._stopped = False
        self._mode = "normal"
        self.commanded.clear()
