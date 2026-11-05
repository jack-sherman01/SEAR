"""Planar peg-in-hole simulator (proposal §6.1 task 1: constrained assembly).

The end effector carries a cylindrical peg (radius ``peg_radius``) that must
be inserted into a hole (radius ``hole_radius``) in a rigid surface at y = 0,
centered at ``target_x``.

Physics zones
-------------
* y > 0 (above surface): free space — standard point-mass dynamics.
* y ≤ 0, |x − target_x| > hole_radius (rim): spring-damper surface contact,
  the peg cannot enter.
* y ≤ 0, |x − target_x| ≤ hole_radius (in hole): lateral wall-contact force
  when |x − target_x| > hole_radius − peg_radius; floor stop at
  y = −target_depth.

Task
----
Progress = insertion depth (−y when centered in hole).
Success when depth ≥ target_depth − 0.003 m.

Contract signals
----------------
``contact_force``  total contact magnitude (rim/floor/wall)
``lateral_force``  |fn_x| wall contact component (jamming risk)
``ee_speed``       speed magnitude
``ee_pos``         position [x, y]
``human_in_zone``  binary trigger
``stopped``        speed < STOP_SPEED
``transporting``   not in contact (above surface, moving to hole)
``nail_progress``  depth / target_depth  (0 → 1, for TaskEvaluator compatibility)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from sear.core.trajectory import Trajectory, TrajectoryRecorder
from sear.sim.env import Observation
from sear.sim.scenario import Scenario

STOP_SPEED = 0.03  # m/s standstill tolerance


class PegEnv:
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
        self.depth = 0.0
        self.success = False
        self.active_tools: set[str] = set()
        self._buffer = deque(np.zeros(2) for _ in range(max(0, sc.params.latency_steps)))
        self._rec = TrajectoryRecorder(self.dt, self.mode)
        self._fn_y = 0.0
        self._fn_x = 0.0
        return self._observe()

    def log_tool_call(self, tool: str, permitted: bool, authorized: bool) -> None:
        self._rec.event(t=self.t, tool=tool, invoked=permitted, permitted=permitted, authorized=authorized)
        if permitted:
            self.active_tools.add(tool)

    @property
    def done(self) -> bool:
        return self.success or self.t >= self.scenario.horizon - 1e-9

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

        fn_y, fn_x = self._contact_forces(self.p, self.v)
        # Tangential friction on the floor/rim (opposes lateral slip)
        ft = -prm.surface_friction * fn_y * np.tanh(self.v[0] / 0.01)
        w = self.rng.normal(0.0, prm.disturbance_std, 2)
        m = prm.mass + prm.payload
        a = (u + np.array([fn_x + ft, fn_y]) - prm.damping * self.v + w) / m
        self.v = self.v + a * self.dt
        self.p = self.p + self.v * self.dt
        self.t += self.dt

        fn_y, fn_x = self._contact_forces(self.p, self.v)
        self._fn_y, self._fn_x = fn_y, fn_x
        contact_force = fn_y + abs(fn_x)
        lateral_force = abs(fn_x)

        clearance = prm.hole_radius - prm.peg_radius
        in_hole = abs(self.p[0] - self.scenario.target_x) <= clearance
        self.depth = max(0.0, -self.p[1]) if in_hole else 0.0
        self.success = self.depth >= prm.target_depth - 0.003

        speed = float(np.linalg.norm(self.v))
        in_contact = contact_force > 0.1
        self._rec.record(
            t=self.t,
            ee_pos=self.p.copy(),
            ee_vel=self.v.copy(),
            ee_speed=speed,
            contact_force=contact_force,
            lateral_force=lateral_force,
            command_norm=float(np.linalg.norm(u)),
            tool_slip=0.0,
            tool_slip_rate=0.0,
            tool_retained=1.0,
            min_distance=self._clearance(self.p),
            human_in_zone=float(self._human_in_zone()),
            stopped=float(speed < STOP_SPEED),
            in_contact=float(in_contact),
            transporting=float(not in_contact),
            nail_progress=self.depth / max(prm.target_depth, 1e-9),
        )
        return self._observe()

    def trajectory(self) -> Trajectory:
        return self._rec.finalize(
            scenario_id=self.scenario.scenario_id,
            success=self.success,
            completion_time=self.t if self.success else None,
            tool_dropped=False,
        )

    # -------------------------------------------------------------- physics
    def _contact_forces(self, p: np.ndarray, v: np.ndarray) -> tuple[float, float]:
        """(fn_y [+y = up], fn_x [pushes peg toward hole centre])."""
        prm = self.scenario.params
        x, y = float(p[0]), float(p[1])

        if y > 0:
            return 0.0, 0.0

        clearance = prm.hole_radius - prm.peg_radius
        dx = x - self.scenario.target_x
        in_hole = abs(dx) <= prm.hole_radius

        if not in_hole:
            # Rim contact: peg sits on rim surface
            fn_y = max(0.0, prm.surface_stiffness * (-y) - prm.surface_damping * v[1])
            return fn_y, 0.0

        # Inside hole opening
        # Lateral wall contact when off-centre
        offset = abs(dx) - clearance
        fn_x = 0.0
        if offset > 0:
            fn_x = -prm.wall_stiffness * offset * float(np.sign(dx))  # pushes toward centre

        # Floor stop at y = −target_depth
        overshoot = (-y) - prm.target_depth
        fn_y = 0.0
        if overshoot > 0:
            fn_y = max(0.0, prm.surface_stiffness * overshoot - prm.surface_damping * v[1])

        # Wall friction resists downward motion (Coulomb approximation)
        if abs(fn_x) > 0 and v[1] < 0:
            fn_y += prm.surface_friction * abs(fn_x)

        return fn_y, fn_x

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
        force_meas = (self._fn_y + abs(self._fn_x) + self.rng.normal(0.0, prm.force_noise_std)
                      if sensor_ok else float("nan"))
        return Observation(
            t=self.t,
            position=self.p + self.rng.normal(0.0, prm.pos_noise_std, 2),
            velocity=self.v + self.rng.normal(0.0, prm.vel_noise_std, 2),
            contact_force=float(max(0.0, force_meas)) if sensor_ok else force_meas,
            tool_slip=0.0,
            target_x=sc.target_x,
            human_in_zone=self._human_in_zone(),
            sensor_ok=sensor_ok,
            obstacle_center=np.asarray(sc.obstacle_center, float),
            obstacle_radius=sc.obstacle_radius,
        )
