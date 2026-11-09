"""UR5e hardware driver stub.

Implements :class:`~sear.robots.base.RobotInterface` for the Universal Robots
UR5e used in the peg-insertion experiments (§9 of the paper).

**This is a stub.**  All methods raise ``NotImplementedError`` until the real
``ur_rtde`` integration is wired in.  The stub is importable without any UR SDK.

UR5e-specific notes (for the eventual implementer):
- Use ``ur_rtde`` (Python bindings to RTDE protocol) for 500 Hz control.
- The peg-insertion task uses TCP force control; map Fz → vertical command,
  Fx/Fy → lateral compliance.
- F/T is read from the UR internal wrench estimate (or external ATI if mounted).
- ``safe_stop()`` must call ``rtde_c.servoStop()`` before returning.
"""

from __future__ import annotations

import numpy as np

from sear.robots.base import RobotInterface, SafetyInterlock
from sear.sim.env import Observation


class UR5eSafetyInterlock(SafetyInterlock):
    """Reflects the UR5e's safety state (normal / reduced / stopped / fault).

    Call :meth:`sync` to update from a ``ur_rtde`` dashboard client.
    """

    def __init__(self) -> None:
        self._engaged = False

    def sync(self, safety_status: str) -> None:
        """Update from UR5e safety status string (e.g. from dashboardclient)."""
        self._engaged = safety_status not in {"Normal", "Reduced"}

    def engaged(self) -> bool:
        return self._engaged


class UR5eRobot(RobotInterface):
    """UR5e hardware interface (stub).

    Args:
        robot_ip: IP address of the UR5e controller
        dt: nominal control period (default 0.002 s = 500 Hz)
    """

    def __init__(self, robot_ip: str, dt: float = 0.002) -> None:
        self._ip = robot_ip
        self._dt = dt
        self.interlock = UR5eSafetyInterlock()

    def connect(self) -> None:
        """Open RTDE receive and control connections."""
        raise NotImplementedError(
            "import rtde_receive, rtde_control from ur_rtde and open connections"
        )

    def observe(self) -> Observation:
        """Read current robot state and return an Observation.

        Maps:
          - TCP position (z-axis = insertion depth) → position
          - TCP velocity → velocity
          - Fz from RTDE wrench → contact_force
          - Peg-slip encoder (if present) → tool_slip; else 0.0
        """
        raise NotImplementedError("read from ur_rtde RTDEReceiveInterface")

    def command(self, u: np.ndarray) -> None:
        """Send a 2-D force/velocity command.

        ``u[0]`` = lateral (x) force/velocity, ``u[1]`` = insertion (z) force/velocity.
        Uses ``servoL`` or force-mode depending on the current operating mode.
        """
        raise NotImplementedError("send via ur_rtde RTDEControlInterface.servoL / forceMode")

    def safe_stop(self) -> None:
        """Call servoStop and then halt the RTDE control connection."""
        raise NotImplementedError("call rtde_c.servoStop(); rtde_c.stopScript()")

    def set_mode(self, mode: str) -> None:
        """Switch operating mode.

        ``low_energy``  — reduce TCP speed limit and insertion force cap
        ``guarded``     — pause before each trial, await operator confirmation
        ``normal``      — full contract-specified limits
        """
        raise NotImplementedError("update RTDE force-mode parameters and speed limits")
