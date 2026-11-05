import numpy as np
from scipy.optimize import minimize

from sear.contracts import Contract
from sear.policies import ApproachPressPolicy
from sear.runtime import RuntimeAssurance, solve_min_norm_qp
from sear.sim import Scenario, run_episode
from tests.conftest import CONTRACT_YAML

TOOLS = {"gripper.hold", "fastener.drive"}


def test_qp_matches_reference_solver():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.normal(size=(4, 2))
        b = rng.normal(size=4) - 2.0  # feasible around the origin
        u_ref = rng.normal(size=2) * 3
        res = solve_min_norm_qp(u_ref, np.eye(2), A, b)
        ref = minimize(lambda u: np.sum((u - u_ref) ** 2), np.zeros(2), method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda u: A @ u - b}])
        assert res.feasible
        np.testing.assert_allclose(res.u, ref.x, atol=1e-3)


def test_qp_reports_infeasibility():
    A = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.array([1.0, 1.0])  # u >= 1 and u <= -1
    assert not solve_min_norm_qp(np.zeros(2), np.eye(2), A, b).feasible


def test_rta_makes_aggressive_policy_safe():
    contract = Contract.from_yaml(CONTRACT_YAML)
    policy = ApproachPressPolicy({"press_force": 60.0, "approach_speed": 0.8})
    raw = run_episode(Scenario("s", 0), policy, allowed_tools=TOOLS)
    safe = run_episode(Scenario("s", 0), policy, runtime=RuntimeAssurance(contract), allowed_tools=TOOLS)
    assert contract.evaluate(raw).any_violation
    assert not contract.evaluate(safe).any_violation
    assert safe.info["runtime"]["filtered_fraction"] > 0
    assert "rta_mode" in safe.signals


def test_rta_safe_stop_on_human_intrusion():
    contract = Contract.from_yaml(CONTRACT_YAML)
    sc = Scenario("h", 1, human_enter_t=1.0, human_exit_t=2.0)
    raw = run_episode(sc, ApproachPressPolicy(), allowed_tools=TOOLS)
    safe = run_episode(sc, ApproachPressPolicy(), runtime=RuntimeAssurance(contract), allowed_tools=TOOLS)
    assert "human_safe_stop" in contract.evaluate(raw).violated
    assert "human_safe_stop" not in contract.evaluate(safe).violated
    assert safe.info["runtime"]["fallback_reasons"].get("human_in_zone", 0) > 0
    after_exit = safe.get("rta_mode")[safe.t > 2.2]
    assert np.any(after_exit < 2)  # human intrusion does not latch; control resumes


def test_rta_latches_on_sensor_fault():
    contract = Contract.from_yaml(CONTRACT_YAML)
    sc = Scenario("f", 2, sensor_dropout_t=1.0)
    traj = run_episode(sc, ApproachPressPolicy(), runtime=RuntimeAssurance(contract), allowed_tools=TOOLS)
    modes = traj.get("rta_mode")
    first = int(np.argmax(modes == 2))
    assert np.all(modes[first:] == 2)
