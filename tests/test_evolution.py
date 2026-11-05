import numpy as np
import pytest

from sear.contracts import SafetyOracle, read_signed
from sear.contracts.signing import DEV_KEY
from sear.evolution import (
    METHODS,
    AdversarialProposer,
    ExternalAgentProposer,
    Feedback,
    GreedyTuningProposer,
    TaskEvaluator,
    build_loop,
    make_genesis,
)
from sear.policies import ApproachPressPolicy
from sear.sim import Scenario, run_episode
from tests.conftest import CONTRACT_YAML

SMOKE = {
    "contract": str(CONTRACT_YAML),
    "seed": 0,
    "generations": 2,
    "trials_per_generation": 3,
    "benchmark_size": 6,
    "proposer": {"n_candidates": 2, "trials": 2},
    "adversary": {"schedule": {1: "tool_escalation"}},
    "verification": {"epsilon": 0.15, "n_sim": 20},
}


@pytest.fixture(scope="module")
def oracle():
    return SafetyOracle(read_signed(CONTRACT_YAML, DEV_KEY), DEV_KEY)


def test_task_evaluator_prefers_aggressive_policy():
    """The motivating example: the task evaluator rewards the unsafe update."""
    ev = TaskEvaluator()
    tools = {"gripper.hold", "fastener.drive"}
    safe = run_episode(Scenario("s", 0), ApproachPressPolicy(), allowed_tools=tools)
    fast = run_episode(Scenario("s", 0), ApproachPressPolicy({"press_force": 60, "approach_speed": 0.8}),
                       allowed_tools=tools)
    assert ev.episode_score(fast) > ev.episode_score(safe)


def test_adversarial_proposer_schedule():
    z0 = make_genesis({})
    adv = AdversarialProposer(GreedyTuningProposer(n_candidates=1, trials=1), schedule={0: "protected_write"})
    fb = Feedback(generation=0, state=z0, task_score=0.0, success_rate=1.0)
    prop = adv.propose(fb, np.random.default_rng(0))
    assert prop.touched_paths and adv.injected == {0: "protected_write"}
    with pytest.raises(ValueError):
        AdversarialProposer(GreedyTuningProposer(), schedule={0: "nonsense"})


def test_trust_region_ignores_injected_attacks():
    z0 = make_genesis({})
    agent = GreedyTuningProposer(n_candidates=1, trials=1, step=0.3)
    adv = AdversarialProposer(agent, schedule={1: "aggressive"})
    adv.propose(Feedback(1, z0, 0.0, 1.0), np.random.default_rng(0))
    adv.observe(Feedback(2, z0, 0.0, 1.0, last_decision="reject"))  # attack was rejected
    assert agent.step == pytest.approx(0.3)
    adv.observe(Feedback(3, z0, 0.0, 1.0, last_decision="reject"))  # agent's own proposal rejected
    assert agent.step == pytest.approx(0.15)
    adv.observe(Feedback(4, z0, 0.0, 1.0, last_decision="admit"))
    assert agent.step == pytest.approx(0.225)


def test_external_agent_adapter():
    def agent(request):
        assert "param_bounds" in request
        return {"policy_params": {"press_force": request["policy"]["params"]["press_force"] + 1},
                "description": "llm suggestion", "memory_append": [{"note": "tried +1N"}]}
    z0 = make_genesis({})
    prop = ExternalAgentProposer(agent).propose(Feedback(0, z0, 0.0, 1.0), np.random.default_rng(0))
    assert prop.policy_params["press_force"] == z0.policy.params["press_force"] + 1
    assert prop.memory_append == ({"note": "tried +1N"},)


@pytest.mark.parametrize("method", ["B0", "SPE"])
def test_loop_runs_and_records(method, oracle):
    res = build_loop(method, SMOKE, oracle, DEV_KEY).run()
    assert len(res.records) == SMOKE["generations"] + 1
    assert res.ledger_valid
    s = res.summary()
    assert s["unsafe_proposals"] >= 1
    if method == "SPE":
        assert s["unsafe_admitted"] == 0
        assert res.records[1].decision == "reject"
    else:
        assert res.records[1].decision == "admit_unverified"


def test_run_persists_ledger_and_states(oracle, tmp_path):
    from sear.lineage import EvidenceLedger
    build_loop("SPE", {**SMOKE, "generations": 1}, oracle, DEV_KEY, tmp_path).run()
    ledger = EvidenceLedger(tmp_path / "SPE" / "ledger.jsonl")  # reload from disk
    assert len(ledger) >= 2 and ledger.verify_chain()
    assert list((tmp_path / "SPE" / "states").glob("*.json"))


def test_cli_verify_ledger(tmp_path, capsys):
    from sear.cli import main
    assert main(["verify-ledger", str(tmp_path / "missing.jsonl")]) == 1


def test_all_methods_defined():
    assert set(METHODS) == {"B0", "B1", "B2", "B3", "SPE"}
