"""Interface a real manipulator driver must implement.

The simulator (:class:`sear.sim.env.ContactToolEnv`) mirrors this interface so
that policies, runtime assurance and the evolution loop are unchanged when
moving to hardware.  Real-robot drivers must additionally honour hardware
interlocks (e-stop, guarded fixtures, reduced-energy modes) that sit *below*
this software stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sear.sim.env import Observation


class SafetyInterlock(ABC):
    @abstractmethod
    def engaged(self) -> bool:
        """True if a hardware e-stop / interlock is active."""


class RobotInterface(ABC):
    @abstractmethod
    def observe(self) -> Observation: ...

    @abstractmethod
    def command(self, u: np.ndarray) -> None:
        """Send a (filtered) end-effector force/velocity command."""

    @abstractmethod
    def safe_stop(self) -> None:
        """Bring the robot to a controlled stop (category 1/2 stop)."""

    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """Switch operating mode, e.g. ``low_energy``; must be operator-approved."""
