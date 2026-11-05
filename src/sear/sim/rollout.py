"""Closed-loop episode execution."""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

from sear.core.trajectory import Trajectory
from sear.sim.env import ContactToolEnv, Observation
from sear.sim.peg_env import PegEnv
from sear.sim.scenario import Scenario


class ActionFilter(Protocol):
    """Runtime-assurance hook: may modify or replace the policy's action."""

    def reset(self) -> None: ...

    def filter(self, obs: Observation, u_agent: np.ndarray) -> np.ndarray: ...

    def episode_signals(self) -> dict[str, list[float]]: ...

    def summary(self) -> dict: ...


class Controller(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: Observation) -> np.ndarray: ...

    def required_tools(self) -> tuple[str, ...]: ...


def run_episode(
    scenario: Scenario,
    policy: Controller,
    *,
    runtime: ActionFilter | None = None,
    allowed_tools: Iterable[str] | None = None,
    operator_approvals: Iterable[str] = (),
    mode: str = "simulation",
) -> Trajectory:
    """Run one episode.

    Tool calls are gated by the state's tool permissions (``allowed_tools``);
    whether each call carried an operator authorization is logged for the
    contract's operational constraints.
    """
    env = (PegEnv(scenario, mode=mode) if getattr(scenario, "task_type", "nailing") == "peg_insertion"
           else ContactToolEnv(scenario, mode=mode))
    obs = env.reset()
    policy.reset()
    if runtime is not None:
        runtime.reset()
    allowed = None if allowed_tools is None else set(allowed_tools)
    approvals = set(operator_approvals)
    for tool in policy.required_tools():
        env.log_tool_call(tool, permitted=allowed is None or tool in allowed, authorized=tool in approvals)

    while not env.done:
        u = np.asarray(policy.act(obs), float)
        if runtime is not None:
            u = runtime.filter(obs, u)
        obs = env.step(u)

    traj = env.trajectory()
    if runtime is not None:
        for name, values in runtime.episode_signals().items():
            traj.signals[name] = np.asarray(values, float)
        traj.info["runtime"] = runtime.summary()
    return traj
