"""Franka Panda hardware driver stub.

Implements :class:`~sear.robots.base.RobotInterface` for the Franka Panda
manipulator used in the nailing experiments (§9 of the paper).

**This is a stub.**  It documents the expected driver interface and raises
``NotImplementedError`` on all methods until the real libfranka / franka_ros
integration is wired in.  The stub is importable without any Franka SDK so
that the rest of the codebase can be tested without hardware.

Franka-specific notes (for the eventual implementer):
- Use ``franka_ros`` or ``frankx`` for Cartesian impedance control.
- The control loop runs at 1 kHz internally; the SEAR loop steps at 500 Hz
  (dt=0.002 s) by calling ``robot.command()`` every other Franka tick.
- F/T is read from the internal joint-torque model; external ATI sensor values
  should be fused via ``observe()`` if available.
- The hardware e-stop is wired through the :class:`FrankaSafetyInterlock`;
  ``safe_stop()`` must call the libfranka stop motion before returning.
"""

from __future__ import annotations

import numpy as np

from sear.robots.base import RobotInterface, SafetyInterlock
from sear.sim.env import Observation


class FrankaSafetyInterlock(SafetyInterlock):
    """Reflects the Franka's internal error / e-stop state.

    Call :meth:`sync` to update from the robot status.
    """

    def __init__(self) -> None:
        self._engaged = False

    def sync(self, robot_state: object) -> None:
        """Update interlock state from a libfranka ``RobotState`` object."""
        raise NotImplementedError("wire to libfranka RobotState.robot_mode")

    def engaged(self) -> bool:
        return self._engaged


class FrankaRobot(RobotInterface):
    """Franka Panda hardware interface (stub).

    Args:
        robot_ip: IP address of the Franka Control Interface
        dt: nominal control period (default 0.002 s = 500 Hz)
    """

    def __init__(self, robot_ip: str, dt: float = 0.002) -> None:
        self._ip = robot_ip
        self._dt = dt
        self.interlock = FrankaSafetyInterlock()

    def connect(self) -> None:
        """Open a connection to the Franka Control Interface."""
        raise NotImplementedError(
            "import franka_robot (libfranka Python bindings) and call Robot(self._ip)"
        )

    def observe(self) -> Observation:
        """Read current robot state and return an Observation.

        Maps:
          - Cartesian TCP position → position (2D projection for planar task)
          - Cartesian TCP velocity → velocity
          - External torque estimate (or ATI F/T) → contact_force
          - Gripper slip sensor → tool_slip
        """
        raise NotImplementedError("read from libfranka RobotState")

    def command(self, u: np.ndarray) -> None:
        """Send a 2-D Cartesian force/velocity command.

        ``u[0]`` = lateral (x) force/velocity, ``u[1]`` = vertical (z) force/velocity.
        The Franka impedance controller tracks these as a Cartesian wrench target.
        """
        raise NotImplementedError("send via libfranka CartesianImpedanceController")

    def safe_stop(self) -> None:
        """Execute a category-1 controlled stop (decelerate to zero, then power hold)."""
        raise NotImplementedError("call libfranka robot.stop()")

    def set_mode(self, mode: str) -> None:
        """Switch operating mode.

        ``low_energy``  — reduce Cartesian velocity limit to 0.1 m/s
        ``guarded``     — require operator confirmation before each trial
        ``normal``      — full contract-specified limits
        """
        raise NotImplementedError("set impedance gains and velocity limits via libfranka")
