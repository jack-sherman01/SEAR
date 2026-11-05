import numpy as np
import pytest

from sear.contracts import Contract
from sear.policies import ApproachPressPolicy, build_policy
from sear.sim import Scenario, run_episode
from tests.conftest import CONTRACT_YAML

TOOLS = {"gripper.hold", "fastener.drive"}


def test_default_policy_is_safe_and_succeeds():
    contract = Contract.from_yaml(CONTRACT_YAML)
    traj = run_episode(Scenario("s", 0), ApproachPressPolicy(), allowed_tools=TOOLS)
    assert traj.info["success"]
    assert not contract.evaluate(traj).any_violation


def test_aggressive_policy_is_faster_but_unsafe():
    contract = Contract.from_yaml(CONTRACT_YAML)
    base = run_episode(Scenario("s", 0), ApproachPressPolicy(), allowed_tools=TOOLS)
    fast = run_episode(Scenario("s", 0), ApproachPressPolicy({"press_force": 60, "approach_speed": 0.8}),
                       allowed_tools=TOOLS)
    assert fast.info["success"] and fast.duration < base.duration
    assert "contact_force_limit" in contract.evaluate(fast).violated


def test_episodes_are_reproducible():
    a = run_episode(Scenario("s", 3), ApproachPressPolicy(), allowed_tools=TOOLS)
    b = run_episode(Scenario("s", 3), ApproachPressPolicy(), allowed_tools=TOOLS)
    np.testing.assert_allclose(a.get("ee_pos"), b.get("ee_pos"))


def test_tool_gating_and_authorization_events():
    policy = ApproachPressPolicy({"high_power": 1.0})
    denied = run_episode(Scenario("s", 0), policy, allowed_tools=TOOLS)
    hp = [e for e in denied.events if e["tool"] == "fastener.high_power"]
    assert hp and not hp[0]["invoked"]
    granted = run_episode(Scenario("s", 0), policy, allowed_tools=TOOLS | {"fastener.high_power"})
    contract = Contract.from_yaml(CONTRACT_YAML)
    assert "high_risk_tool_authorization" in contract.evaluate(granted).violated
    approved = run_episode(Scenario("s", 0), policy, allowed_tools=TOOLS | {"fastener.high_power"},
                           operator_approvals={"fastener.high_power"})
    assert "high_risk_tool_authorization" not in contract.evaluate(approved).violated


def test_build_policy_rejects_unknown_params():
    with pytest.raises(ValueError):
        build_policy(ApproachPressPolicy.default_state().__class__(kind="approach_press", params={"bogus": 1}))
