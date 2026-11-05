"""Update proposals Delta_k, the update operator U, and auditable candidate artifacts.

Proposal, Sec. 3:  z_{k+1} = U(z_k, Delta_k)
Proposal, WP2:     each update is packaged as an auditable artifact (diff,
                   checkpoint, data lineage, evaluator version, tool
                   permissions, declared change-impact summary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from sear import __version__
from sear.core.hashing import content_hash, to_jsonable
from sear.core.state import (
    MUTABLE_COMPONENTS,
    DataState,
    EvaluatorSpec,
    MemoryState,
    PolicyState,
    Provenance,
    RewardSpec,
    SystemState,
    ToolPermissions,
)


@dataclass(frozen=True)
class UpdateProposal:
    """Delta_k -- a change proposed by the (untrusted) improvement agent.

    ``None`` fields mean "unchanged".  ``touched_paths`` lists repository files
    the update would modify; it is checked against protected paths.
    """

    proposer: str
    description: str = ""
    policy_params: Mapping[str, float] | None = None
    memory_append: tuple[Mapping[str, Any], ...] = ()
    new_dataset_ids: tuple[str, ...] = ()
    new_episodes: int = 0
    tools: frozenset[str] | None = None
    reward_weights: Mapping[str, float] | None = None
    evaluator: EvaluatorSpec | None = None
    touched_paths: tuple[str, ...] = ()
    declared_impact: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def apply_update(parent: SystemState, delta: UpdateProposal) -> SystemState:
    """The update operator U(z_k, Delta_k)."""
    params = dict(parent.policy.params)
    if delta.policy_params:
        params.update(delta.policy_params)
    reward_weights = dict(parent.reward.weights)
    if delta.reward_weights:
        reward_weights.update(delta.reward_weights)
    return SystemState(
        policy=PolicyState(kind=parent.policy.kind, params=params),
        memory=MemoryState(entries=parent.memory.entries + tuple(delta.memory_append)),
        data=DataState(
            dataset_ids=parent.data.dataset_ids + tuple(delta.new_dataset_ids),
            num_episodes=parent.data.num_episodes + int(delta.new_episodes),
        ),
        tools=ToolPermissions(allowed=delta.tools if delta.tools is not None else parent.tools.allowed),
        reward=RewardSpec(name=parent.reward.name, weights=reward_weights),
        evaluator=delta.evaluator if delta.evaluator is not None else parent.evaluator,
        provenance=Provenance(
            generation=parent.provenance.generation + 1,
            parent=parent.digest,
            proposer=delta.proposer,
            description=delta.description,
            software_version=__version__,
        ),
    )


@dataclass(frozen=True)
class CandidateArtifact:
    """An auditable package: parent, candidate, and the proposal that links them."""

    parent: SystemState
    candidate: SystemState
    proposal: UpdateProposal

    @classmethod
    def from_proposal(cls, parent: SystemState, proposal: UpdateProposal) -> "CandidateArtifact":
        return cls(parent=parent, candidate=apply_update(parent, proposal), proposal=proposal)

    @property
    def digest(self) -> str:
        return content_hash(
            {"parent": self.parent.digest, "candidate": self.candidate.digest, "proposal": to_jsonable(self.proposal)}
        )

    def changed_components(self) -> tuple[str, ...]:
        ph, ch = self.parent.component_hashes(), self.candidate.component_hashes()
        return tuple(c for c in MUTABLE_COMPONENTS if ph[c] != ch[c])

    def param_delta(self) -> dict[str, float]:
        old, new = self.parent.policy.params, self.candidate.policy.params
        keys = set(old) | set(new)
        return {k: new.get(k, 0.0) - old.get(k, 0.0) for k in sorted(keys) if new.get(k) != old.get(k)}

    def summary(self) -> dict[str, Any]:
        return {
            "artifact": self.digest,
            "parent": self.parent.digest,
            "candidate": self.candidate.digest,
            "generation": self.candidate.generation,
            "proposer": self.proposal.proposer,
            "description": self.proposal.description,
            "declared_impact": self.proposal.declared_impact,
            "changed_components": list(self.changed_components()),
            "param_delta": self.param_delta(),
            "touched_paths": list(self.proposal.touched_paths),
            "tools": sorted(self.candidate.tools.allowed),
            "evaluator": to_jsonable(self.candidate.evaluator),
        }
