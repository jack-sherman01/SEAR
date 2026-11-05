"""Scenario and dynamics-parameter definitions for the contact/tool simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from sear.core.hashing import content_hash


@dataclass(frozen=True)
class DynamicsParams:
    """Uncertain dynamics parameters (the proposal's vartheta) and sensing quality."""

    mass: float = 2.0  # effective end-effector mass [kg]
    payload: float = 0.5  # tool mass [kg]
    damping: float = 4.0  # joint/air viscous damping [N s/m]
    surface_stiffness: float = 5000.0  # [N/m]
    surface_damping: float = 40.0  # [N s/m]
    surface_friction: float = 0.4
    grip_force: float = 35.0  # normal force the grasp tolerates before the tool slips [N]
    slip_gain: float = 0.08  # slip accumulated per N of excess force per second
    drive_threshold: float = 10.0  # force needed to advance the fastener [N]
    drive_rate: float = 0.06  # fastener progress per N of excess force per second
    u_max: float = 60.0  # actuator saturation [N]
    latency_steps: int = 1
    disturbance_std: float = 0.5  # process force noise [N]
    force_noise_std: float = 0.3  # F/T sensor noise [N]
    pos_noise_std: float = 0.0005  # state-estimation noise [m]
    vel_noise_std: float = 0.002  # [m/s]


@dataclass(frozen=True)
class Scenario:
    """One reproducible episode configuration."""

    scenario_id: str
    seed: int
    params: DynamicsParams = field(default_factory=DynamicsParams)
    start: tuple[float, float] = (0.0, 0.2)
    target_x: float = 0.3
    obstacle_center: tuple[float, float] = (0.15, 0.03)
    obstacle_radius: float = 0.025
    human_enter_t: float | None = None
    human_exit_t: float | None = None
    sensor_dropout_t: float | None = None
    horizon: float = 5.0
    tags: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Scenario":
        d = dict(d)
        d["params"] = DynamicsParams(**d.get("params", {}))
        for key in ("start", "obstacle_center", "tags"):
            if key in d:
                d[key] = tuple(d[key])
        return cls(**d)
