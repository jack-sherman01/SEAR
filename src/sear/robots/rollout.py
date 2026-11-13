"""Closed-loop hardware episode execution.

Hardware analog of :func:`sear.sim.rollout.run_episode`.  Uses a
:class:`~sear.robots.base.RobotInterface` instead of a simulator, records
all :class:`~sear.sim.env.Observation` fields into a
:class:`~sear.core.trajectory.Trajectory`, and integrates
:class:`~sear.runtime.latency.LatencyMonitor` for adaptive CBF-margin
tightening.

Returned :class:`~sear.core.trajectory.Trajectory` is signal-compatible with
the simulator output, so the same contract evaluator, metrics, and
visualisation code work unchanged.

Typical usage::

    from sear.robots import MockRobot
    from sear.runtime import RuntimeAssurance, LatencyMonitor
    from sear.contracts import Contract

    robot = MockRobot(scenario)
    contract = Contract.from_yaml(CONTRACT_YAML)
    rta = RuntimeAssurance(contract)
    monitor = LatencyMonitor()

    # optional 20-step warm-up to seed the latency estimate
    _warmup(robot, monitor, dt=0.002)

    traj = run_hardware_episode(robot, policy, runtime=rta, monitor=monitor, dt=0.002)
    result = contract.evaluate(traj)
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from sear.core.trajectory import Trajectory, TrajectoryRecorder
from sear.robots.base import RobotInterface, SafetyInterlock
from sear.runtime.assurance import RuntimeAssuranceConfig
from sear.runtime.latency import LatencyMonitor, apply_latency_compensation
from sear.sim.env import Observation
from sear.sim.rollout import ActionFilter, Controller


def run_hardware_episode(
    robot: RobotInterface,
    policy: Controller,
    *,
    runtime: ActionFilter | None = None,
    monitor: LatencyMonitor | None = None,
    max_steps: int = 1000,
    dt: float = 0.002,
    done_fn: Callable[[Observation, int], bool] | None = None,
    extra_signals: Callable[[Observation, int], dict[str, float]] | None = None,
    allowed_tools: Iterable[str] | None = None,
) -> Trajectory:
    """Run one closed-loop episode on a physical (or mock) robot.

    Args:
        robot: hardware driver implementing :class:`RobotInterface`
        policy: controller implementing ``.act(obs)`` and ``.reset()``
        runtime: optional CBF-QP safety filter (same as simulation)
        monitor: optional latency tracker; updates ``runtime.config`` margin
            each step if both are provided
        max_steps: hard step limit (episode terminates after this many steps
            regardless of ``done_fn``)
        dt: nominal control period (seconds); used for time-stamping and
            latency-step computation
        done_fn: ``(obs, step) -> bool``; return True to end the episode
            early.  If None, only ``max_steps`` or a safe-stop terminates.
        extra_signals: ``(obs, step) -> dict[str, float]``; inject
            hardware-specific signals (e.g. lateral F/T, insertion depth)
            that are not derivable from :class:`Observation` alone.
            These are recorded verbatim into the trajectory.
        allowed_tools: tool permissions; logged on the first step

    Returns:
        :class:`~sear.core.trajectory.Trajectory` with the same signal names
        as the simulator, plus ``"latency_ms"`` and ``"rta_mode"`` if
        ``monitor`` / ``runtime`` are provided.

    The returned trajectory's ``info["hardware"]`` dict contains:
        - ``n_steps``: actual steps executed
        - ``safe_stopped``: True if a safe-stop terminated the episode
        - ``latency``: :meth:`LatencyMonitor.summary` dict (if monitor given)
        - ``runtime``: :meth:`ActionFilter.summary` dict (if runtime given)
    """
    policy.reset()
    if runtime is not None:
        runtime.reset()

    # Snapshot base RTA config before the episode so latency compensation
    # always adjusts from the original margin, never compounds.
    base_rta_cfg: RuntimeAssuranceConfig | None = None
    if monitor is not None and runtime is not None and hasattr(runtime, "config"):
        from dataclasses import replace as _replace
        base_rta_cfg = _replace(runtime.config)

    rec = TrajectoryRecorder(dt=dt, mode="hardware")
    allowed = None if allowed_tools is None else set(allowed_tools)

    # Log tool calls (first step only, mirrors sim behaviour)
    if allowed is not None and hasattr(robot, "_env"):
        for tool in policy.required_tools():
            robot._env.log_tool_call(tool, permitted=tool in allowed, authorized=False)

    safe_stopped = False
    obs = robot.observe()

    for step in range(max_steps):
        t_sense = time.perf_counter()
        obs = robot.observe()

        u = np.asarray(policy.act(obs), float)

        if runtime is not None:
            u = runtime.filter(obs, u)

        t_cmd = time.perf_counter()

        if not safe_stopped:
            robot.command(u)

        # Latency tracking and adaptive CBF-margin update
        if monitor is not None:
            monitor.record(t_sense, t_cmd, dt=dt)
            if base_rta_cfg is not None and runtime is not None:
                new_cfg = apply_latency_compensation(base_rta_cfg, monitor)
                runtime.config.position_margin = new_cfg.position_margin
                # Clear cached barriers so next filter() call recomputes with new margin
                if hasattr(runtime, "_barriers"):
                    runtime._barriers = None

            if monitor.should_fallback():
                robot.safe_stop()
                safe_stopped = True

        # Check hardware interlock
        interlock: SafetyInterlock | None = getattr(robot, "interlock", None)
        if interlock is not None and interlock.engaged():
            robot.safe_stop()
            safe_stopped = True

        # Record signals derivable from observation
        speed = float(np.linalg.norm(obs.velocity))
        prev_slip = float(rec._rows["tool_slip"][-1]) if rec._rows.get("tool_slip") else 0.0
        slip_rate = (obs.tool_slip - prev_slip) / dt
        min_dist = float(np.linalg.norm(obs.position - obs.obstacle_center) - obs.obstacle_radius)

        rec.record(
            t=obs.t if obs.t > 0.0 else step * dt,
            ee_pos=np.asarray(obs.position, float),
            ee_vel=np.asarray(obs.velocity, float),
            ee_speed=speed,
            contact_force=obs.contact_force if np.isfinite(obs.contact_force) else 0.0,
            tool_slip=obs.tool_slip,
            tool_slip_rate=slip_rate,
            tool_retained=float(obs.tool_slip < 1.0),
            min_distance=min_dist,
            human_in_zone=float(obs.human_in_zone),
            stopped=float(speed < 0.03),
            in_contact=float(np.isfinite(obs.contact_force) and obs.contact_force > 0.0),
            sensor_ok=float(obs.sensor_ok),
            command_norm=float(np.linalg.norm(u)),
        )

        if monitor is not None:
            rec.record(latency_ms=monitor.mean_latency * 1e3)

        if extra_signals is not None:
            rec.record(**extra_signals(obs, step))

        if runtime is not None and hasattr(runtime, "_modes") and runtime._modes:
            rec.record(rta_mode=float(runtime._modes[-1]))

        if safe_stopped:
            break

        if done_fn is not None and done_fn(obs, step):
            break

    # Flush remaining runtime signals
    if runtime is not None:
        for name, values in runtime.episode_signals().items():
            if name not in rec._rows:
                rec._rows[name] = list(np.asarray(values, float))

    info: dict = {
        "n_steps": step + 1,
        "safe_stopped": safe_stopped,
    }
    if monitor is not None:
        info["latency"] = monitor.summary()
    if runtime is not None:
        info["runtime"] = runtime.summary()

    return rec.finalize(**info)
