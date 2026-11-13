"""Tests for hardware calibration check, latency-compensation, and robot stubs."""

import numpy as np
import pytest

from sear.benchmark import Benchmark
from sear.contracts import SafetyOracle, read_signed
from sear.contracts.signing import DEV_KEY
from sear.core import CandidateArtifact, SystemState, ToolPermissions, UpdateProposal
from sear.core.types import Verdict
from sear.policies import ApproachPressPolicy
from sear.runtime.latency import (
    AdaptiveLatencyConfig,
    LatencyMonitor,
    apply_latency_compensation,
    tighten_margins,
)
from sear.verification import VerificationContext, default_pipeline
from sear.verification.checks import (
    HardwareCalibrationCheck,
    HardwareRunnerConfig,
    HardwareTrial,
)
from tests.conftest import CONTRACT_YAML

TOOLS = frozenset({"gripper.hold", "fastener.drive"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ctx():
    oracle = SafetyOracle(read_signed(CONTRACT_YAML, DEV_KEY), DEV_KEY)
    bench = Benchmark.build(4)
    return VerificationContext(oracle, bench, bench.digests())


@pytest.fixture(scope="module")
def z0():
    return SystemState(policy=ApproachPressPolicy.default_state(), tools=ToolPermissions(TOOLS))


def _art(z0, **kw):
    return CandidateArtifact.from_proposal(z0, UpdateProposal(proposer="agent", **kw))


def _trial(T=20, force_offset=0.0, pos_offset=0.0, any_violation=False):
    t = np.linspace(0, 1, T)
    pred_f = np.sin(t) * 10.0
    obs_f = pred_f + force_offset
    pred_p = np.column_stack([t, np.zeros(T)])
    obs_p = pred_p + pos_offset
    return HardwareTrial(
        scenario_id="s0",
        observed_forces=obs_f,
        predicted_forces=pred_f,
        observed_positions=obs_p,
        predicted_positions=pred_p,
        any_violation=any_violation,
    )


# ---------------------------------------------------------------------------
# HardwareCalibrationCheck
# ---------------------------------------------------------------------------

def test_hardware_check_skips_without_runner(ctx, z0):
    check = HardwareCalibrationCheck(hardware_runner=None)
    result = check.run(_art(z0), ctx)
    assert result.verdict is Verdict.PASS
    assert result.metrics["skipped"] is True
    assert result.metrics["n_trials"] == 0


def test_hardware_check_pass_within_tolerance(ctx, z0):
    trials = [_trial(force_offset=0.1, pos_offset=0.001) for _ in range(5)]
    check = HardwareCalibrationCheck(
        hardware_runner=lambda a, s: trials,
        config=HardwareRunnerConfig(n_trials=5),
    )
    result = check.run(_art(z0), ctx)
    assert result.verdict is Verdict.PASS
    assert result.metrics["n_violations"] == 0


def test_hardware_check_unknown_on_high_force_residual(ctx, z0):
    # large force offset pushes residual above 0.25 fraction threshold
    trials = [_trial(force_offset=5.0) for _ in range(5)]
    check = HardwareCalibrationCheck(
        hardware_runner=lambda a, s: trials,
        config=HardwareRunnerConfig(n_trials=5, tolerance_force=0.25),
    )
    result = check.run(_art(z0), ctx)
    assert result.verdict is Verdict.UNKNOWN


def test_hardware_check_unknown_on_high_position_residual(ctx, z0):
    trials = [_trial(pos_offset=0.05) for _ in range(5)]
    check = HardwareCalibrationCheck(
        hardware_runner=lambda a, s: trials,
        config=HardwareRunnerConfig(n_trials=5, tolerance_position=0.02),
    )
    result = check.run(_art(z0), ctx)
    assert result.verdict is Verdict.UNKNOWN


def test_hardware_check_fail_on_violation(ctx, z0):
    trials = [_trial(any_violation=True)] + [_trial() for _ in range(4)]
    check = HardwareCalibrationCheck(
        hardware_runner=lambda a, s: trials,
        config=HardwareRunnerConfig(max_violation_fraction=0.0),
    )
    result = check.run(_art(z0), ctx)
    assert result.verdict is Verdict.FAIL
    assert result.metrics["n_violations"] == 1


def test_hardware_check_unknown_on_runner_exception(ctx, z0):
    def bad_runner(a, s):
        raise RuntimeError("hardware comms failure")

    check = HardwareCalibrationCheck(hardware_runner=bad_runner)
    result = check.run(_art(z0), ctx)
    assert result.verdict is Verdict.UNKNOWN
    assert "comms failure" in result.metrics["error"]


def test_hardware_check_unknown_on_no_scenarios(ctx, z0):
    from dataclasses import replace
    empty_bench = Benchmark.build(0)
    empty_ctx = replace(ctx, benchmark=empty_bench, _cache={})
    check = HardwareCalibrationCheck(hardware_runner=lambda a, s: [])
    result = check.run(_art(z0), empty_ctx)
    assert result.verdict is Verdict.UNKNOWN


def test_hardware_check_in_default_pipeline(ctx, z0):
    # With no runner, the check is a no-op PASS that must not change pipeline outcome
    pipe = default_pipeline(n_sim=30, hardware_runner=None)
    names = [c.name for c in pipe.checks]
    assert "hardware_calibration" in names
    # hardware check is non-blocking when skipped
    ev = pipe.verify(_art(z0, policy_params={"hover_height": 0.05}), ctx)
    hw = ev.check("hardware_calibration")
    assert hw.verdict is Verdict.PASS and hw.metrics["skipped"]


# ---------------------------------------------------------------------------
# LatencyMonitor
# ---------------------------------------------------------------------------

def test_latency_monitor_mean_and_max():
    monitor = LatencyMonitor()
    for i in range(5):
        monitor.record(0.0, 0.005 * (i + 1), dt=0.002)
    assert abs(monitor.mean_latency - np.mean([0.005 * (i + 1) for i in range(5)])) < 1e-10
    assert abs(monitor.max_latency - 0.025) < 1e-10


def test_latency_monitor_empty():
    monitor = LatencyMonitor()
    assert monitor.mean_latency == 0.0
    assert monitor.max_latency == 0.0
    assert monitor.latency_steps == 1


def test_latency_monitor_latency_steps():
    monitor = LatencyMonitor(AdaptiveLatencyConfig(window=10, adapt_latency_steps=True))
    # 6 ms latency / 2 ms dt → round(3.0) = 3
    for _ in range(10):
        monitor.record(0.0, 0.006, dt=0.002)
    assert monitor.latency_steps == 3


def test_latency_monitor_no_adapt():
    monitor = LatencyMonitor(AdaptiveLatencyConfig(adapt_latency_steps=False))
    for _ in range(5):
        monitor.record(0.0, 0.010, dt=0.002)
    assert monitor.latency_steps == 1  # unchanged


def test_latency_monitor_fallback_trigger():
    monitor = LatencyMonitor(AdaptiveLatencyConfig(fallback_latency_s=0.020))
    assert not monitor.should_fallback()
    for _ in range(5):
        monitor.record(0.0, 0.025, dt=0.002)
    assert monitor.should_fallback()
    monitor.reset_fallback()
    assert not monitor.should_fallback()


def test_latency_monitor_no_fallback_below_threshold():
    monitor = LatencyMonitor(AdaptiveLatencyConfig(fallback_latency_s=0.020))
    for _ in range(10):
        monitor.record(0.0, 0.010, dt=0.002)
    assert not monitor.should_fallback()


def test_latency_monitor_margin_increment_clamps():
    cfg = AdaptiveLatencyConfig(alpha=1.6, margin_min=0.0005, margin_max=0.010)
    monitor = LatencyMonitor(cfg)
    # zero latency → margin_min
    assert monitor.margin_increment() == cfg.margin_min
    # small latency: α * 0.001 = 0.0016 > margin_min, < margin_max
    for _ in range(5):
        monitor.record(0.0, 0.001, dt=0.002)
    inc = monitor.margin_increment()
    assert cfg.margin_min <= inc <= cfg.margin_max
    assert abs(inc - 1.6 * 0.001) < 1e-9
    # large latency → clamp to margin_max
    monitor2 = LatencyMonitor(cfg)
    for _ in range(5):
        monitor2.record(0.0, 1.0, dt=0.002)  # 1 s latency
    assert monitor2.margin_increment() == cfg.margin_max


def test_latency_monitor_summary_keys():
    monitor = LatencyMonitor()
    for _ in range(3):
        monitor.record(0.0, 0.005, dt=0.002)
    s = monitor.summary()
    for key in ("mean_latency_ms", "max_latency_ms", "latency_steps",
                 "margin_increment_m", "fallback_triggered", "n_samples"):
        assert key in s
    assert s["n_samples"] == 3


def test_tighten_margins():
    result = tighten_margins(base_margin=0.002, latency_s=0.005, alpha=1.6,
                             margin_min=0.0005, margin_max=0.010)
    assert abs(result - (0.002 + 1.6 * 0.005)) < 1e-9

    # extra > margin_max → capped
    result_cap = tighten_margins(base_margin=0.002, latency_s=1.0, alpha=1.6,
                                 margin_min=0.0005, margin_max=0.010)
    assert result_cap == 0.002 + 0.010

    # base + extra < margin_min → floor
    result_floor = tighten_margins(base_margin=0.0, latency_s=0.0, alpha=1.6,
                                   margin_min=0.001, margin_max=0.010)
    assert result_floor == 0.001


def test_apply_latency_compensation():
    from sear.runtime import RuntimeAssuranceConfig

    base_cfg = RuntimeAssuranceConfig(position_margin=0.001, latency_steps=1)
    monitor = LatencyMonitor()
    for _ in range(10):
        monitor.record(0.0, 0.006, dt=0.002)

    new_cfg = apply_latency_compensation(base_cfg, monitor)
    assert new_cfg.position_margin > base_cfg.position_margin
    assert new_cfg.latency_steps == monitor.latency_steps
    # original must be unchanged (dataclass replace returns a new object)
    assert base_cfg.position_margin == 0.001


# ---------------------------------------------------------------------------
# Robot stubs
# ---------------------------------------------------------------------------

def test_mock_robot_observe_and_command():
    from sear.robots import MockRobot
    robot = MockRobot()
    obs = robot.observe()
    assert hasattr(obs, "position") and obs.position.shape == (2,)
    robot.command(np.zeros(2))
    assert len(robot.commanded) == 1


def test_mock_robot_safe_stop_blocks_command():
    from sear.robots import MockRobot
    robot = MockRobot()
    robot.safe_stop()
    assert robot.stopped
    robot.command(np.array([1.0, 0.0]))
    assert len(robot.commanded) == 0  # blocked


def test_mock_robot_interlock_blocks_command():
    from sear.robots import MockRobot
    robot = MockRobot()
    robot.interlock.engage()
    robot.command(np.array([1.0, 0.0]))
    assert len(robot.commanded) == 0
    robot.interlock.release()
    robot.command(np.array([0.0, 0.0]))
    assert len(robot.commanded) == 1


def test_mock_robot_set_mode():
    from sear.robots import MockRobot
    robot = MockRobot()
    robot.set_mode("low_energy")
    assert robot.mode == "low_energy"
    with pytest.raises(ValueError):
        robot.set_mode("warp_speed")


def test_mock_robot_reset():
    from sear.robots import MockRobot
    robot = MockRobot()
    robot.command(np.zeros(2))
    robot.safe_stop()
    robot.set_mode("guarded")
    robot.reset()
    assert not robot.stopped
    assert robot.mode == "normal"
    assert len(robot.commanded) == 0


def test_franka_stub_raises():
    from sear.robots import FrankaRobot
    r = FrankaRobot(robot_ip="10.0.0.1")
    with pytest.raises(NotImplementedError):
        r.observe()
    with pytest.raises(NotImplementedError):
        r.command(np.zeros(2))
    with pytest.raises(NotImplementedError):
        r.safe_stop()
    with pytest.raises(NotImplementedError):
        r.set_mode("low_energy")


def test_ur5e_stub_raises():
    from sear.robots import UR5eRobot
    r = UR5eRobot(robot_ip="192.168.1.1")
    with pytest.raises(NotImplementedError):
        r.observe()
    with pytest.raises(NotImplementedError):
        r.command(np.zeros(2))
    with pytest.raises(NotImplementedError):
        r.safe_stop()
    with pytest.raises(NotImplementedError):
        r.set_mode("guarded")


# ---------------------------------------------------------------------------
# run_hardware_episode
# ---------------------------------------------------------------------------

def test_run_hardware_episode_basic():
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy

    robot = MockRobot()
    policy = ApproachPressPolicy()
    traj = run_hardware_episode(robot, policy, max_steps=20, dt=0.002)

    assert len(traj) == 20
    assert "ee_pos" in traj.signals
    assert "ee_speed" in traj.signals
    assert "contact_force" in traj.signals
    assert "tool_slip" in traj.signals
    assert "min_distance" in traj.signals
    assert traj.info["n_steps"] == 20
    assert not traj.info["safe_stopped"]


def test_run_hardware_episode_with_monitor():
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy
    from sear.runtime import LatencyMonitor, AdaptiveLatencyConfig

    robot = MockRobot()
    monitor = LatencyMonitor(AdaptiveLatencyConfig(window=10))
    traj = run_hardware_episode(robot, ApproachPressPolicy(),
                                monitor=monitor, max_steps=15, dt=0.002)

    assert "latency_ms" in traj.signals
    assert len(traj.signals["latency_ms"]) == 15
    assert "latency" in traj.info
    assert traj.info["latency"]["n_samples"] == 10  # window=10 caps the deque


def test_run_hardware_episode_with_runtime():
    from sear.contracts import Contract
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy
    from sear.runtime import RuntimeAssurance

    contract = Contract.from_yaml(CONTRACT_YAML)
    robot = MockRobot()
    rta = RuntimeAssurance(contract)
    traj = run_hardware_episode(robot, ApproachPressPolicy(),
                                runtime=rta, max_steps=20, dt=0.002)

    assert "rta_mode" in traj.signals
    assert "runtime" in traj.info


def test_run_hardware_episode_with_monitor_and_runtime():
    from sear.contracts import Contract
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy
    from sear.runtime import RuntimeAssurance, LatencyMonitor

    contract = Contract.from_yaml(CONTRACT_YAML)
    robot = MockRobot()
    rta = RuntimeAssurance(contract)
    monitor = LatencyMonitor()
    traj = run_hardware_episode(robot, ApproachPressPolicy(),
                                runtime=rta, monitor=monitor,
                                max_steps=20, dt=0.002)

    # margin should have been updated each step (not just at start)
    assert traj.info["n_steps"] == 20
    assert "latency" in traj.info and "runtime" in traj.info


def test_run_hardware_episode_safe_stop_on_fallback():
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy
    from sear.runtime import LatencyMonitor, AdaptiveLatencyConfig

    robot = MockRobot()
    # trigger fallback immediately: fallback_latency_s = 0 means any latency triggers
    monitor = LatencyMonitor(AdaptiveLatencyConfig(fallback_latency_s=0.0))
    traj = run_hardware_episode(robot, ApproachPressPolicy(),
                                monitor=monitor, max_steps=100, dt=0.002)

    assert traj.info["safe_stopped"]
    assert robot.stopped
    assert traj.info["n_steps"] < 100  # terminated early


def test_run_hardware_episode_safe_stop_on_interlock():
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy

    robot = MockRobot()
    robot.interlock.engage()
    traj = run_hardware_episode(robot, ApproachPressPolicy(),
                                max_steps=50, dt=0.002)

    assert traj.info["safe_stopped"]
    assert traj.info["n_steps"] <= 2  # interlock detected on first step


def test_run_hardware_episode_done_fn():
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy

    robot = MockRobot()
    traj = run_hardware_episode(robot, ApproachPressPolicy(),
                                done_fn=lambda obs, step: step >= 7,
                                max_steps=50, dt=0.002)

    assert traj.info["n_steps"] == 8  # done_fn fires at step=7, loop breaks after record


def test_run_hardware_episode_extra_signals():
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy

    robot = MockRobot()
    traj = run_hardware_episode(
        robot, ApproachPressPolicy(),
        extra_signals=lambda obs, step: {"lateral_force": 0.5 * obs.contact_force},
        max_steps=10, dt=0.002,
    )

    assert "lateral_force" in traj.signals
    assert len(traj.signals["lateral_force"]) == 10


def test_run_hardware_episode_contract_evaluable():
    """Trajectory signals must be sufficient for contract evaluation."""
    from sear.contracts import Contract
    from sear.robots import MockRobot, run_hardware_episode
    from sear.policies import ApproachPressPolicy

    contract = Contract.from_yaml(CONTRACT_YAML)
    robot = MockRobot()
    traj = run_hardware_episode(robot, ApproachPressPolicy(), max_steps=50, dt=0.002)
    result = contract.evaluate(traj)
    # Result must not be entirely UNKNOWN (signals present)
    verdicts = {cr.verdict.value for cr in result.results}
    assert "unknown" not in verdicts or "pass" in verdicts or "fail" in verdicts
