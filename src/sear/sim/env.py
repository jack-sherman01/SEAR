"""Planar contact-rich tool-use simulator (construction-inspired fastening).

The end effector is a 2-D point mass holding a tool.  The work surface is the
line y = 0 (penetration produces a spring-damper contact force).  The task is
to drive a fastener at ``target_x``: progress accumulates while pressing with
more than ``drive_threshold`` newtons near the target.  Pressing harder or
approaching faster completes the task sooner -- but transient force spikes
above the grip capacity make the tool slip, which is exactly the
capability-vs-safety tension of the proposal's nail-gun example.

Hazards exposed as signals: contact force, speed, clearance to a fixture,
workspace, tool slip/retention, human intrusion response.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from sear.core.trajectory import Trajectory, TrajectoryRecorder
from sear.sim.scenario import Scenario

HIGH_POWER_TOOL = "fastener.high_power"
STOP_SPEED = 0.03  # m/s, standstill tolerance (above process-noise level)
TARGET_TOL = 0.01  # m, lateral tolerance for fastening


@dataclass(frozen=True)
class Observation:
    """What the robot (policy, runtime monitor) can measure."""

    t: float
    position: np.ndarray
    velocity: np.ndarray
    contact_force: float  # measured normal force; nan if the sensor failed
    tool_slip: float  # from the instrumented end effector
    target_x: float
    human_in_zone: bool
    sensor_ok: bool
    obstacle_center: np.ndarray
    obstacle_radius: float


class ContactToolEnv:
    dt = 0.01

    def __init__(self, scenario: Scenario, mode: str = "simulation") -> None:
        self.scenario = scenario
        self.mode = mode
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self) -> Observation:
        sc = self.scenario
        self.rng = np.random.default_rng(sc.seed)
        self.p = np.array(sc.start, float)
        self.v = np.zeros(2)
        self.t = 0.0
        self.slip = 0.0
        self.progress = 0.0
        self.success = False
        self.tool_dropped = False
        self.active_tools: set[str] = set()
        self._buffer = deque(np.zeros(2) for _ in range(max(0, sc.params.latency_steps)))
        self._rec = TrajectoryRecorder(self.dt, self.mode)
        self._fn = 0.0
        return self._observe()

    def log_tool_call(self, tool: str, permitted: bool, authorized: bool) -> None:
        self._rec.event(t=self.t, tool=tool, invoked=permitted, permitted=permitted, authorized=authorized)
        if permitted:
            self.active_tools.add(tool)

    @property
    def done(self) -> bool:
        return self.success or self.tool_dropped or self.t >= self.scenario.horizon - 1e-9

    def step(self, u_cmd: np.ndarray) -> Observation:
        prm = self.scenario.params
        u_cmd = np.asarray(u_cmd, float).reshape(2)
        if not np.all(np.isfinite(u_cmd)):
            u_cmd = np.zeros(2)
        self._buffer.append(u_cmd)
        u = self._buffer.popleft()
        norm = float(np.linalg.norm(u))
        if norm > prm.u_max:
            u = u * (prm.u_max / norm)

        fn = self._normal_force(self.p, self.v)
        ft = -prm.surface_friction * fn * np.tanh(self.v[0] / 0.01)
        w = self.rng.normal(0.0, prm.disturbance_std, 2)
        m = prm.mass + prm.payload
        a = (u + np.array([ft, fn]) - prm.damping * self.v + w) / m
        self.v = self.v + a * self.dt
        self.p = self.p + self.v * self.dt
        self.t += self.dt

        fn = self._normal_force(self.p, self.v)
        self._fn = fn
        slip_rate = prm.slip_gain * max(0.0, fn - prm.grip_force)
        self.slip += slip_rate * self.dt
        self.tool_dropped = self.slip >= 1.0
        rate = prm.drive_rate * (2.0 if HIGH_POWER_TOOL in self.active_tools else 1.0)
        if fn > 0 and abs(self.p[0] - self.scenario.target_x) < TARGET_TOL and "fastener.drive" in self.active_tools:
            self.progress += rate * max(0.0, fn - prm.drive_threshold) * self.dt
        self.success = self.progress >= 1.0 and not self.tool_dropped

        speed = float(np.linalg.norm(self.v))
        self._rec.record(
            t=self.t,
            ee_pos=self.p.copy(),
            ee_vel=self.v.copy(),
            ee_speed=speed,
            contact_force=fn,
            command_norm=float(np.linalg.norm(u)),
            tool_slip=self.slip,
            tool_slip_rate=slip_rate,
            tool_retained=float(not self.tool_dropped),
            min_distance=self._clearance(self.p),
            human_in_zone=float(self._human_in_zone()),
            stopped=float(speed < STOP_SPEED),
            in_contact=float(fn > 0.0),
            transporting=float(fn <= 0.0),
            nail_progress=self.progress,
        )
        return self._observe()

    def trajectory(self) -> Trajectory:
        return self._rec.finalize(
            scenario_id=self.scenario.scenario_id,
            success=self.success,
            completion_time=self.t if self.success else None,
            tool_dropped=self.tool_dropped,
        )

    # -------------------------------------------------------------- physics
    def _normal_force(self, p: np.ndarray, v: np.ndarray) -> float:
        pen = -p[1]
        if pen <= 0:
            return 0.0
        prm = self.scenario.params
        return max(0.0, prm.surface_stiffness * pen - prm.surface_damping * v[1])

    def _clearance(self, p: np.ndarray) -> float:
        c = np.asarray(self.scenario.obstacle_center)
        return float(np.linalg.norm(p - c) - self.scenario.obstacle_radius)

    def _human_in_zone(self) -> bool:
        sc = self.scenario
        if sc.human_enter_t is None:
            return False
        exit_t = sc.human_exit_t if sc.human_exit_t is not None else np.inf
        return sc.human_enter_t <= self.t < exit_t

    def _observe(self) -> Observation:
        prm, sc = self.scenario.params, self.scenario
        sensor_ok = sc.sensor_dropout_t is None or self.t < sc.sensor_dropout_t
        force = self._fn + self.rng.normal(0.0, prm.force_noise_std) if sensor_ok else float("nan")
        return Observation(
            t=self.t,
            position=self.p + self.rng.normal(0.0, prm.pos_noise_std, 2),
            velocity=self.v + self.rng.normal(0.0, prm.vel_noise_std, 2),
            contact_force=float(max(0.0, force)) if sensor_ok else force,
            tool_slip=self.slip,
            target_x=sc.target_x,
            human_in_zone=self._human_in_zone(),
            sensor_ok=sensor_ok,
            obstacle_center=np.asarray(sc.obstacle_center, float),
            obstacle_radius=sc.obstacle_radius,
        )
