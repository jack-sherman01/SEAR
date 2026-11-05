"""Tests for the peg-in-hole task: physics, policy, contract, suites, evolution."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sear.benchmark.suites import (
    _PEG_ADVERSARIAL_KINDS,
    Benchmark,
    peg_adversarial,
    peg_in_distribution,
    peg_training_scenarios,
)
from sear.contracts import Contract, SafetyOracle, read_signed
from sear.contracts.signing import DEV_KEY
from sear.evolution import build_loop, make_genesis
from sear.policies import PegInsertPolicy
from sear.sim import DynamicsParams, PegEnv, Scenario, run_episode
from sear.sim.scenario import Scenario

PEG_CONTRACT_YAML = Path(__file__).parents[1] / "configs" / "contracts" / "peg_insertion.yaml"
PEG_TOOLS = frozenset({"gripper.hold", "probe.insert"})

SMOKE_CFG = {
    "task": "peg_insertion",
    "contract": str(PEG_CONTRACT_YAML),
    "seed": 0,
    "generations": 2,
    "trials_per_generation": 3,
    "benchmark_size": 6,
    "proposer": {"n_candidates": 2, "trials": 2},
    "adversary": {"schedule": {1: "evaluator_tamper"}},
    "verification": {"epsilon": 0.15, "n_sim": 20},
}


@pytest.fixture(scope="module")
def peg_oracle():
    return SafetyOracle(read_signed(PEG_CONTRACT_YAML, DEV_KEY), DEV_KEY)


# ---------------------------------------------------------------------- physics

def _make_scenario(**overrides) -> Scenario:
    return Scenario("peg-t", 0, task_type="peg_insertion", **overrides)


def test_no_contact_above_surface():
    sc = _make_scenario()
    env = PegEnv(sc)
    env.reset()
    env.p = np.array([sc.target_x, 0.05])
    fn_y, fn_x = env._contact_forces(env.p, np.zeros(2))
    assert fn_y == 0.0 and fn_x == 0.0


def test_rim_contact_pushes_up():
    sc = _make_scenario()
    env = PegEnv(sc)
    env.reset()
    # Place peg outside hole, below surface
    env.p = np.array([sc.target_x + 0.10, -0.005])
    fn_y, fn_x = env._contact_forces(env.p, np.array([0.0, -0.1]))
    assert fn_y > 0.0  # surface pushes up
    assert fn_x == 0.0  # no lateral force outside hole


def test_wall_contact_in_hole():
    sc = _make_scenario()
    env = PegEnv(sc)
    env.reset()
    prm = sc.params
    # Inside hole opening but off-centre beyond clearance
    clearance = prm.hole_radius - prm.peg_radius
    env.p = np.array([sc.target_x + clearance + 0.001, -0.005])
    fn_y, fn_x = env._contact_forces(env.p, np.zeros(2))
    assert abs(fn_x) > 0.0  # wall pushes toward centre
    assert fn_x < 0.0        # peg is to the right of centre, force is leftward


def test_floor_stop_at_target_depth():
    sc = _make_scenario()
    env = PegEnv(sc)
    env.reset()
    prm = sc.params
    overshoot = 0.002
    env.p = np.array([sc.target_x, -(prm.target_depth + overshoot)])
    fn_y, fn_x = env._contact_forces(env.p, np.array([0.0, -0.1]))
    assert fn_y > 0.0  # floor pushes up


def test_centered_insertion_no_lateral_force():
    sc = _make_scenario()
    env = PegEnv(sc)
    env.reset()
    env.p = np.array([sc.target_x, -0.010])  # centred, mid-insertion
    fn_y, fn_x = env._contact_forces(env.p, np.zeros(2))
    assert fn_x == 0.0
    assert fn_y == 0.0  # above floor, no wall


# ---------------------------------------------------------------------- policy

def test_peg_policy_succeeds_nominal():
    sc = peg_in_distribution(1, seed=0)[0]
    traj = run_episode(sc, PegInsertPolicy(), allowed_tools=PEG_TOOLS)
    assert traj.info["success"], "peg insertion should succeed on a nominal scenario"


def test_peg_episodes_are_reproducible():
    sc = peg_in_distribution(1, seed=1)[0]
    a = run_episode(sc, PegInsertPolicy(), allowed_tools=PEG_TOOLS)
    b = run_episode(sc, PegInsertPolicy(), allowed_tools=PEG_TOOLS)
    np.testing.assert_allclose(a.get("ee_pos"), b.get("ee_pos"))


def test_peg_aggressive_violates_contract():
    contract = Contract.from_yaml(PEG_CONTRACT_YAML)
    # Very high force and speed → violates at least one physical constraint
    fast = PegInsertPolicy({"insert_force": 60.0, "insert_speed": 0.3, "approach_speed": 0.5})
    traj = run_episode(peg_in_distribution(1, seed=0)[0], fast, allowed_tools=PEG_TOOLS)
    assert contract.evaluate(traj).any_violation


# -------------------------------------------------------------------- suites

def test_peg_suites_are_deterministic():
    a = Benchmark.build(6, seed=0, task="peg_insertion")
    b = Benchmark.build(6, seed=0, task="peg_insertion")
    assert a.digests() == b.digests()
    assert not Benchmark.build(6, seed=1, task="peg_insertion").verify(a.digests())


def test_peg_adversarial_covers_all_kinds():
    bench = Benchmark.build(len(_PEG_ADVERSARIAL_KINDS), task="peg_insertion")
    kinds = {s.tags[1] for s in bench["adversarial"]}
    assert kinds == set(_PEG_ADVERSARIAL_KINDS)


def test_peg_training_seeds_disjoint_from_test():
    bench = Benchmark.build(12, task="peg_insertion")
    test_seeds = {s.seed for suite in bench.suites.values() for s in suite}
    train_seeds = {s.seed for g in range(20) for s in peg_training_scenarios(8, g)}
    assert not test_seeds & train_seeds


def test_peg_in_distribution_has_correct_task_type():
    scenarios = peg_in_distribution(4)
    for sc in scenarios:
        assert sc.task_type == "peg_insertion"
        assert sc.params.target_depth == pytest.approx(0.040)
        assert sc.params.peg_radius == pytest.approx(0.010)


# ---------------------------------------------------------------- evolution

@pytest.mark.parametrize("method", ["B0", "SPE"])
def test_peg_loop_runs_and_records(method, peg_oracle):
    res = build_loop(method, SMOKE_CFG, peg_oracle, DEV_KEY).run()
    assert len(res.records) == SMOKE_CFG["generations"] + 1
    assert res.ledger_valid
    s = res.summary()
    if method == "SPE":
        assert s["unsafe_admitted"] == 0


def test_peg_make_genesis_uses_correct_defaults():
    z = make_genesis({"task": "peg_insertion"})
    assert z.policy.kind == "peg_insert"
    assert "gripper.hold" in z.tools.allowed
    assert "probe.insert" in z.tools.allowed
    assert "fastener.drive" not in z.tools.allowed
