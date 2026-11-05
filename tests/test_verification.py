import pytest

from sear.benchmark import Benchmark
from sear.contracts import SafetyOracle, read_signed
from sear.contracts.signing import DEV_KEY
from sear.core import CandidateArtifact, EvaluatorSpec, SystemState, ToolPermissions, UpdateProposal, Verdict
from sear.policies import ApproachPressPolicy
from sear.verification import AdmissionDecision, VerificationContext, VerificationPipeline, default_pipeline
from sear.verification.checks import (
    BarrierCertificateCheck,
    EvaluatorChangeCheck,
    IntegrityCheck,
    InterfaceCheck,
    PermissionCheck,
    ProtectedPathCheck,
    ReplayRegressionCheck,
)
from sear.verification.evidence import CheckResult
from sear.core.types import GuaranteeLevel
from tests.conftest import CONTRACT_YAML

TOOLS = frozenset({"gripper.hold", "fastener.drive"})


@pytest.fixture(scope="module")
def ctx():
    oracle = SafetyOracle(read_signed(CONTRACT_YAML, DEV_KEY), DEV_KEY)
    bench = Benchmark.build(6)
    return VerificationContext(oracle, bench, bench.digests())


@pytest.fixture(scope="module")
def z0():
    return SystemState(policy=ApproachPressPolicy.default_state(), tools=ToolPermissions(TOOLS))


def _art(z0, **kw):
    return CandidateArtifact.from_proposal(z0, UpdateProposal(proposer="agent", **kw))


def test_decision_rule():
    def r(v, mandatory=True):
        return CheckResult("c", v, GuaranteeLevel.EMPIRICAL, mandatory, "")
    decide = VerificationPipeline.decide
    assert decide([r(Verdict.PASS), r(Verdict.UNKNOWN, mandatory=False)]) is AdmissionDecision.ADMIT
    assert decide([r(Verdict.PASS), r(Verdict.UNKNOWN)]) is AdmissionDecision.RESTRICT
    assert decide([r(Verdict.UNKNOWN), r(Verdict.FAIL)]) is AdmissionDecision.REJECT
    assert decide([]) is AdmissionDecision.RESTRICT


def test_static_checks(ctx, z0):
    assert ProtectedPathCheck().run(_art(z0, touched_paths=("configs/contracts/x.yaml",)), ctx).verdict is Verdict.FAIL
    assert ProtectedPathCheck().run(_art(z0, touched_paths=("notes/agent.md",)), ctx).verdict is Verdict.PASS
    esc = _art(z0, tools=TOOLS | {"fastener.high_power"})
    assert PermissionCheck().run(esc, ctx).verdict is Verdict.FAIL
    assert PermissionCheck().run(_art(z0, tools=TOOLS | {"robot.shell"}), ctx).verdict is Verdict.FAIL
    ev = _art(z0, evaluator=EvaluatorSpec(version="2"))
    assert EvaluatorChangeCheck().run(ev, ctx).verdict is Verdict.UNKNOWN
    assert InterfaceCheck().run(_art(z0, policy_params={"press_force": 500.0}), ctx).verdict is Verdict.FAIL
    assert IntegrityCheck().run(_art(z0), ctx).verdict is Verdict.PASS


def test_operator_approval_allows_high_risk_tool(ctx, z0):
    from dataclasses import replace
    approved = replace(ctx, operator_approvals=frozenset({"fastener.high_power"}))
    esc = _art(z0, tools=TOOLS | {"fastener.high_power"})
    assert PermissionCheck().run(esc, approved).verdict is Verdict.PASS


def test_integrity_detects_benchmark_change(ctx, z0):
    from dataclasses import replace
    tampered = replace(ctx, benchmark=Benchmark.build(6, seed=5))
    assert IntegrityCheck().run(_art(z0), tampered).verdict is Verdict.FAIL


def test_regression_rejects_aggressive_update(ctx, z0):
    res = ReplayRegressionCheck().run(_art(z0, policy_params={"press_force": 60.0, "approach_speed": 0.8}), ctx)
    assert res.verdict is Verdict.FAIL and res.failing_scenarios
    assert ReplayRegressionCheck().run(_art(z0, policy_params={"hover_height": 0.05}), ctx).verdict is Verdict.PASS


def test_certificate_never_fails(ctx, z0):
    res = BarrierCertificateCheck(mode="raw", grid=4).run(_art(z0, policy_params={"press_force": 60.0}), ctx)
    assert res.verdict in (Verdict.PASS, Verdict.UNKNOWN)
    assert not res.mandatory


def test_pipeline_end_to_end(ctx, z0):
    small = default_pipeline(n_sim=20).verify(_art(z0, policy_params={"hover_height": 0.05}), ctx)
    assert small.decision is AdmissionDecision.RESTRICT  # 0/20 -> bound 0.109 > eps: not enough evidence
    pipe = default_pipeline(n_sim=30)
    good = pipe.verify(_art(z0, policy_params={"hover_height": 0.05}), ctx)
    assert good.decision is AdmissionDecision.ADMIT
    assert good.check("statistical_safety").level is GuaranteeLevel.PROBABILISTIC
    bad = pipe.verify(_art(z0, touched_paths=("src/sear/contracts/oracle.py",)), ctx)
    assert bad.decision is AdmissionDecision.REJECT
    assert len(bad.checks) < len(good.checks)  # fail-fast skipped expensive tiers
    assert good.verify(good.sign(b"k"), b"k") and not good.verify(good.sign(b"k"), b"other")
