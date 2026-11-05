import json

import pytest

from sear.benchmark import Benchmark
from sear.core import CandidateArtifact, EvaluatorSpec, SystemState, ToolPermissions, UpdateProposal
from sear.lineage import (
    DeploymentManager,
    EvidenceLedger,
    LedgerIntegrityError,
    RecordKind,
    SafetyRegressionRetriever,
)
from sear.policies import ApproachPressPolicy


def _z0():
    return SystemState(policy=ApproachPressPolicy.default_state(), tools=ToolPermissions({"gripper.hold"}))


def test_ledger_chain_and_tamper_detection(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = EvidenceLedger(path)
    led.append(RecordKind.GENESIS, {"a": 1})
    led.append(RecordKind.REJECTED, {"b": 2})
    assert led.verify_chain()
    assert len(EvidenceLedger(path)) == 2

    lines = path.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["b"] = 3
    path.write_text(lines[0] + "\n" + json.dumps(rec) + "\n")
    with pytest.raises(LedgerIntegrityError):
        EvidenceLedger(path)

    path.write_text(lines[1] + "\n")  # deleting history breaks the chain too
    with pytest.raises(LedgerIntegrityError):
        EvidenceLedger(path)


def test_view_is_query_only():
    led = EvidenceLedger()
    led.append(RecordKind.REJECTED, {"x": 1})
    view = led.view()
    assert not hasattr(view, "append")
    assert len(view.query([RecordKind.REJECTED])) == 1
    assert view.latest(RecordKind.ADMITTED) is None


def test_retrieval_by_similar_param_delta():
    led = EvidenceLedger()
    failing = [s.to_dict() for s in Benchmark.build(2)["adversarial"]]
    led.append(RecordKind.REJECTED, {"param_delta": {"press_force": 20.0}, "changed_components": ["policy"],
                                     "failing_scenarios": failing})
    r = SafetyRegressionRetriever(led.view())
    z0 = _z0()
    same_dir = CandidateArtifact.from_proposal(z0, UpdateProposal("a", policy_params={"press_force": 25.0}))
    opposite = CandidateArtifact.from_proposal(z0, UpdateProposal("a", policy_params={"press_force": 15.0}))
    assert len(r.retrieve(same_dir)) == 2
    assert r.retrieve(opposite) == []
    led.append(RecordKind.ROLLBACK, {"changed_components": ["evaluator"], "failing_scenarios": failing[:1]})
    ev = CandidateArtifact.from_proposal(z0, UpdateProposal("a", evaluator=EvaluatorSpec(version="9")))
    assert len(r.retrieve(ev)) == 1


def test_deployment_and_rollback():
    led = EvidenceLedger()
    z0 = _z0()
    mgr = DeploymentManager(z0, led)
    z1 = CandidateArtifact.from_proposal(z0, UpdateProposal("a", policy_params={"press_force": 25.0})).candidate
    mgr.deploy(z1, "evidence-digest")
    assert mgr.current is z1
    target, latency = mgr.rollback("violations in deployment")
    assert target is z0 and mgr.current is z0 and latency >= 0
    assert [r.kind for r in led.records()] == [RecordKind.GENESIS, RecordKind.DEPLOYED, RecordKind.ROLLBACK]
    with pytest.raises(RuntimeError):
        mgr.rollback("again")
