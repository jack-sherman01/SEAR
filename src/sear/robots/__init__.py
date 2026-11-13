"""Hardware abstraction for real-robot experiments (M9--M10)."""

from sear.robots.base import RobotInterface, SafetyInterlock
from sear.robots.franka import FrankaRobot, FrankaSafetyInterlock
from sear.robots.mock import MockRobot, MockSafetyInterlock
from sear.robots.rollout import run_hardware_episode
from sear.robots.ur5e import UR5eRobot, UR5eSafetyInterlock

__all__ = [
    "FrankaRobot",
    "FrankaSafetyInterlock",
    "MockRobot",
    "MockSafetyInterlock",
    "RobotInterface",
    "SafetyInterlock",
    "UR5eRobot",
    "UR5eSafetyInterlock",
    "run_hardware_episode",
]
