import pytest

from sear.contracts import Contract
from sear.metrics import (
    aggregate,
    clopper_pearson,
    clopper_pearson_upper,
    poisson_rate_upper,
    summarize_episode,
    wilson_interval,
)
from sear.policies import ApproachPressPolicy
from sear.sim import Scenario, run_episode
from tests.conftest import CONTRACT_YAML


def test_zero_failures_is_not_zero_risk():
    ub = clopper_pearson_upper(0, 100, 0.95)
    assert ub == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)  # ~ 3/n rule
    assert 0.029 < ub < 0.031
    assert clopper_pearson_upper(5, 5) == 1.0


def test_intervals_contain_point_estimate():
    lo, hi = clopper_pearson(3, 20)
    assert lo < 0.15 < hi
    lo, hi = wilson_interval(3, 20)
    assert lo < 0.15 < hi
    assert poisson_rate_upper(0, 1.0) == pytest.approx(2.9957, rel=1e-3)


def test_episode_summary_and_aggregate():
    contract = Contract.from_yaml(CONTRACT_YAML)
    outcomes = []
    for params in ({}, {"press_force": 60.0, "approach_speed": 0.8}):
        traj = run_episode(Scenario("s", 0), ApproachPressPolicy(params),
                           allowed_tools={"gripper.hold", "fastener.drive"})
        outcomes.append(summarize_episode(traj, contract.evaluate(traj), contract))
    safe, unsafe = outcomes
    assert not safe.any_violation and safe.safety_margin > 0
    assert unsafe.force_violation and unsafe.force_excess > 0 and unsafe.safety_margin < 0
    m = aggregate(outcomes, "x")
    assert m.n == 2 and m.violation_rate == 0.5 and m.success_rate == 1.0
    assert m.violation_upper > 0.5
