"""Core abstractions: the versioned agentic system state and its updates."""

from sear.core.hashing import canonical_json, content_hash, to_jsonable
from sear.core.state import (
    DataState,
    EvaluatorSpec,
    MemoryState,
    PolicyState,
    Provenance,
    RewardSpec,
    SystemState,
    ToolPermissions,
)
from sear.core.trajectory import Trajectory, TrajectoryRecorder
from sear.core.types import GuaranteeLevel, OperatingMode, Verdict
from sear.core.update import CandidateArtifact, UpdateProposal, apply_update
from sear.core.versioning import VersionStore

__all__ = [
    "CandidateArtifact",
    "DataState",
    "EvaluatorSpec",
    "GuaranteeLevel",
    "MemoryState",
    "OperatingMode",
    "PolicyState",
    "Provenance",
    "RewardSpec",
    "SystemState",
    "ToolPermissions",
    "Trajectory",
    "TrajectoryRecorder",
    "UpdateProposal",
    "Verdict",
    "VersionStore",
    "apply_update",
    "canonical_json",
    "content_hash",
    "to_jsonable",
]
