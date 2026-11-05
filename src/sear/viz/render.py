"""Render trajectory animations from final deployed states.

For each method in a run, loads the final deployed SystemState, re-runs
``n_episodes`` benchmark scenarios, and saves one animation per episode to
``<run_dir>/<method>/videos/``.

Video format: MP4 (requires ffmpeg) with GIF fallback.

Frame layout:
  Left panel   — 2D workspace: EE path (colour-coded by RTA mode), obstacle,
                 target, hole/nail marker
  Right panels — force and speed time series with constraint limits
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np

from sear.benchmark.suites import Benchmark
from sear.contracts.oracle import SafetyOracle
from sear.contracts.signing import DEV_KEY, load_key, read_signed
from sear.core.state import SystemState
from sear.core.versioning import VersionStore
from sear.evolution.evaluator import DEFAULT_EVALUATOR
from sear.policies.base import build_policy
from sear.sim.rollout import run_episode
from sear.sim.scenario import Scenario

# RTA mode colours (matches RTAMode enum: 0=nominal,1=filtered,2=fallback)
MODE_COLORS = ["#2196F3", "#FF9800", "#F44336"]
MODE_LABELS = ["nominal", "filtered", "fallback"]


def _load_config(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text())


def _load_final_state(method_dir: Path) -> SystemState | None:
    result_file = method_dir / "result.json"
    if not result_file.exists():
        return None
    result = json.loads(result_file.read_text())
    records = result.get("records", [])
    if not records:
        return None
    final_digest = records[-1]["deployed_state"]
    store_dir = method_dir / "states"
    if not store_dir.exists():
        return None
    store = VersionStore(store_dir)
    if final_digest not in store:
        return None
    return store.get(final_digest)


def _select_scenarios(cfg: dict, n: int) -> list[Scenario]:
    task = cfg.get("task", "nailing")
    bench = Benchmark.build(n_per_suite=max(n, 4), seed=cfg.get("seed", 0), task=task)
    nominal = list(bench["in_distribution"])[:max(1, n // 2)]
    adv = list(bench["adversarial"])[:max(1, n - len(nominal))]
    return nominal + adv


def _render_frame(ax_ws, ax_force, ax_speed, ee_pos, rta_modes, contact_force,
                  ee_speed, t_arr, frame_idx, scenario: Scenario, force_limit: float,
                  speed_limit: float, task: str) -> None:
    ax_ws.cla()
    ax_force.cla()
    ax_speed.cla()

    # -- workspace --
    pos = np.asarray(ee_pos[:frame_idx + 1])
    modes = rta_modes[:frame_idx + 1] if rta_modes else [0] * (frame_idx + 1)

    for i in range(len(pos) - 1):
        c = MODE_COLORS[int(min(modes[i], 2))]
        ax_ws.plot(pos[i:i + 2, 0], pos[i:i + 2, 1], color=c, linewidth=1.2, alpha=0.7)

    # current EE
    ax_ws.scatter([pos[-1, 0]], [pos[-1, 1]], color="k", s=25, zorder=5)

    # obstacle
    obs = mpatches.Circle(scenario.obstacle_center, scenario.obstacle_radius,
                           color="#FF5722", alpha=0.25, linewidth=1)
    ax_ws.add_patch(obs)

    if task == "peg_insertion":
        # draw hole
        hole_r = scenario.params.hole_radius
        ax_ws.axhline(0, color="#555", linewidth=1.5)
        hole = mpatches.FancyArrow(scenario.target_x - hole_r, 0, 2 * hole_r, 0,
                                    color="white", width=0.003, head_width=0)
        ax_ws.add_patch(hole)
        # draw floor of hole
        ax_ws.plot([scenario.target_x - hole_r, scenario.target_x + hole_r],
                   [-scenario.params.target_depth, -scenario.params.target_depth],
                   color="#555", linewidth=1)
    else:
        # nailing: draw surface line and target
        ax_ws.axhline(0, color="#555", linewidth=1.5)
        ax_ws.axvline(scenario.target_x, color="#aaa", linewidth=1, linestyle="--")

    ax_ws.set_xlim(-0.2, 0.75)
    if task == "peg_insertion":
        ax_ws.set_ylim(-0.08, 0.35)
    else:
        ax_ws.set_ylim(-0.03, 0.35)
    ax_ws.set_aspect("equal")
    ax_ws.set_title(f"t={t_arr[frame_idx]:.2f}s  {scenario.scenario_id}", fontsize=8)
    ax_ws.set_xlabel("x (m)", fontsize=7)
    ax_ws.set_ylabel("y (m)", fontsize=7)
    ax_ws.tick_params(labelsize=6)

    # legend patches
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(MODE_COLORS, MODE_LABELS)]
    ax_ws.legend(handles=patches, fontsize=6, loc="upper right")

    # -- force time series --
    t_now = t_arr[:frame_idx + 1]
    f_now = contact_force[:frame_idx + 1]
    ax_force.plot(t_now, f_now, color="#E91E63", linewidth=1.2)
    ax_force.axhline(force_limit, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_force.set_xlim(0, t_arr[-1] + 0.1)
    ax_force.set_ylim(0, max(float(np.max(contact_force)) * 1.1, force_limit * 1.2))
    ax_force.set_ylabel("Contact force (N)", fontsize=7)
    ax_force.set_xlabel("t (s)", fontsize=7)
    ax_force.tick_params(labelsize=6)
    ax_force.grid(True, alpha=0.25)

    # -- speed time series --
    s_now = ee_speed[:frame_idx + 1]
    ax_speed.plot(t_now, s_now, color="#009688", linewidth=1.2)
    ax_speed.axhline(speed_limit, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_speed.set_xlim(0, t_arr[-1] + 0.1)
    ax_speed.set_ylim(0, max(float(np.max(ee_speed)) * 1.1, speed_limit * 1.2))
    ax_speed.set_ylabel("Speed (m/s)", fontsize=7)
    ax_speed.set_xlabel("t (s)", fontsize=7)
    ax_speed.tick_params(labelsize=6)
    ax_speed.grid(True, alpha=0.25)


def _animate_trajectory(traj, scenario: Scenario, out_path: Path, task: str,
                         force_limit: float = 40.0, speed_limit: float = 0.6,
                         fps: int = 20, stride: int = 5) -> Path:
    t_arr = traj.get("t")
    ee_pos = traj.get("ee_pos") if traj.has("ee_pos") else np.zeros((len(t_arr), 2))
    contact_force = traj.get("contact_force") if traj.has("contact_force") else np.zeros(len(t_arr))
    ee_speed = traj.get("ee_speed") if traj.has("ee_speed") else np.zeros(len(t_arr))
    rta_mode = traj.get("rta_mode").astype(int).tolist() if traj.has("rta_mode") else []

    # Replace NaN contact_force with 0
    contact_force = np.where(np.isfinite(contact_force), contact_force, 0.0)

    frames = list(range(0, len(t_arr), stride))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    ax_ws, ax_force, ax_speed = axes
    fig.patch.set_facecolor("#fafafa")

    def update(fi):
        idx = frames[fi]
        _render_frame(ax_ws, ax_force, ax_speed, ee_pos, rta_mode, contact_force,
                      ee_speed, t_arr, idx, scenario, force_limit, speed_limit, task)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=800)
        mp4_path = out_path.with_suffix(".mp4")
        ani.save(str(mp4_path), writer=writer, dpi=100)
        plt.close(fig)
        return mp4_path
    except Exception:
        pass  # fall back to GIF
    try:
        gif_path = out_path.with_suffix(".gif")
        ani.save(str(gif_path), writer="pillow", fps=fps // 2, dpi=80)
        plt.close(fig)
        return gif_path
    finally:
        plt.close(fig)


def render_episodes(run_dir: str | Path, n_episodes: int = 3,
                    methods: list[str] | None = None, fps: int = 20) -> list[Path]:
    """Re-run N episodes with each method's final state and save animations.

    Returns list of saved video paths.
    """
    run_dir = Path(run_dir)
    cfg = _load_config(run_dir)
    task = cfg.get("task", "nailing")
    contract_path = cfg.get("contract")
    key = load_key(key_file=cfg.get("contract_key_file"))
    signed = read_signed(contract_path, key)
    oracle = SafetyOracle(signed, key)
    contract = signed.contract

    # Derive force and speed limits from contract for rendering
    from sear.contracts.constraints import SignalBound
    force_limit, speed_limit = 40.0, 0.6
    for c in contract.constraints:
        if isinstance(c, SignalBound) and c.signal == "contact_force" and c.upper:
            force_limit = float(c.upper)
        if isinstance(c, SignalBound) and c.signal == "lateral_force" and c.upper:
            force_limit = float(c.upper)
        if isinstance(c, SignalBound) and c.signal == "ee_speed" and c.upper:
            speed_limit = float(c.upper)

    scenarios = _select_scenarios(cfg, n_episodes)
    method_dirs = [d for d in sorted(run_dir.iterdir()) if d.is_dir() and d.name != "figures"]
    if methods:
        method_dirs = [d for d in method_dirs if d.name in methods]

    outputs: list[Path] = []
    for method_dir in method_dirs:
        state = _load_final_state(method_dir)
        if state is None:
            continue
        policy = build_policy(state.policy)
        vid_dir = method_dir / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        for i, sc in enumerate(scenarios):
            traj = run_episode(sc, policy, allowed_tools=frozenset(state.tools.allowed),
                               operator_approvals=tuple(cfg.get("operator_approvals", ())))
            out = vid_dir / f"ep{i:02d}_{sc.scenario_id}"
            try:
                saved = _animate_trajectory(traj, sc, out, task, force_limit, speed_limit, fps)
                outputs.append(saved)
            except Exception as exc:
                print(f"  [render] {method_dir.name}/ep{i:02d}: {exc}")
    return outputs
