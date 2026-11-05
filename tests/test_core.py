import pytest

from sear.core import (
    CandidateArtifact,
    EvaluatorSpec,
    PolicyState,
    SystemState,
    ToolPermissions,
    UpdateProposal,
    Verdict,
    VersionStore,
    apply_update,
)


def _genesis() -> SystemState:
    return SystemState(
        policy=PolicyState(kind="approach_press", params={"press_force": 20.0, "approach_speed": 0.15}),
        tools=ToolPermissions(allowed={"gripper.hold"}),
    )


def test_verdict_combine_is_conservative():
    assert Verdict.combine([]) is Verdict.UNKNOWN
    assert Verdict.combine([Verdict.PASS, Verdict.UNKNOWN]) is Verdict.UNKNOWN
    assert Verdict.combine([Verdict.PASS, Verdict.FAIL, Verdict.UNKNOWN]) is Verdict.FAIL
    assert Verdict.combine([Verdict.PASS, Verdict.PASS]) is Verdict.PASS


def test_state_is_immutable_and_hash_stable():
    z = _genesis()
    with pytest.raises(TypeError):
        z.policy.params["press_force"] = 99.0  # type: ignore[index]
    assert z.digest == _genesis().digest
    assert SystemState.from_dict(z.to_dict()).digest == z.digest


def test_apply_update_tracks_provenance_and_changes():
    z0 = _genesis()
    delta = UpdateProposal(proposer="agent", policy_params={"press_force": 30.0},
                           evaluator=EvaluatorSpec(version="2"))
    art = CandidateArtifact.from_proposal(z0, delta)
    z1 = art.candidate
    assert z1.generation == 1 and z1.provenance.parent == z0.digest
    assert z1.policy.params["press_force"] == 30.0
    assert z1.policy.params["approach_speed"] == 0.15
    assert set(art.changed_components()) == {"policy", "evaluator"}
    assert art.param_delta() == {"press_force": 10.0}
    assert apply_update(z0, delta).digest == z1.digest


def test_version_store_ancestry(tmp_path):
    store = VersionStore(tmp_path)
    z0 = _genesis()
    z1 = apply_update(z0, UpdateProposal(proposer="a", policy_params={"press_force": 25.0}))
    z2 = apply_update(z1, UpdateProposal(proposer="a", policy_params={"press_force": 26.0}))
    for z in (z0, z1, z2):
        store.put(z)
    assert [s.generation for s in store.ancestry(z2.digest)] == [2, 1, 0]
    reloaded = VersionStore(tmp_path)
    assert len(reloaded) == 3 and z2.digest in reloaded
