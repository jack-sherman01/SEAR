"""The complete agentic system state.

Proposal, Sec. 3:  z_k = (theta_k, m_k, d_k, tau_k, r_k, e_k, P_k)

=================  ==========================  =========================
symbol             component                   class
=================  ==========================  =========================
theta_k            policy / controller         :class:`PolicyState`
m_k                persistent agent memory     :class:`MemoryState`
d_k                training / rollout data     :class:`DataState`
tau_k              tool / API permissions      :class:`ToolPermissions`
r_k                reward / task spec          :class:`RewardSpec`
e_k                (task) evaluator            :class:`EvaluatorSpec`
P_k                software & model provenance :class:`Provenance`
=================  ==========================  =========================

All components are frozen; an update produces a *new* state (see
:func:`sear.core.update.apply_update`).  Note that the *safety* evaluator is
deliberately not part of z_k: it lives in :mod:`sear.contracts` outside the
agent's writable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

from sear import __version__
from sear.core.hashing import content_hash, to_jsonable

#: Components that an update may change (provenance always changes).
MUTABLE_COMPONENTS = ("policy", "memory", "data", "tools", "reward", "evaluator")


def _frozen(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(mapping or {}))


@dataclass(frozen=True)
class PolicyState:
    """theta_k -- policy kind plus its (flat, numeric) parameters."""

    kind: str
    params: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _frozen({k: float(v) for k, v in dict(self.params).items()}))


@dataclass(frozen=True)
class MemoryState:
    """m_k -- append-only notes / lessons the agent keeps across generations."""

    entries: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(_frozen(e) for e in self.entries))


@dataclass(frozen=True)
class DataState:
    """d_k -- lineage of the data used to produce theta_k."""

    dataset_ids: tuple[str, ...] = ()
    num_episodes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_ids", tuple(self.dataset_ids))


@dataclass(frozen=True)
class ToolPermissions:
    """tau_k -- tools / APIs the policy may invoke."""

    allowed: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed", frozenset(self.allowed))


@dataclass(frozen=True)
class RewardSpec:
    """r_k -- task objective (never contains safety constraints)."""

    name: str = "task"
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _frozen({k: float(v) for k, v in dict(self.weights).items()}))


@dataclass(frozen=True)
class EvaluatorSpec:
    """e_k -- the agent-side *task* evaluator (it labels task success)."""

    name: str = "task_evaluator"
    version: str = "1"
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _frozen(self.config))


@dataclass(frozen=True)
class Provenance:
    """P_k -- where this state came from."""

    generation: int = 0
    parent: str | None = None
    proposer: str = "human"
    description: str = "genesis"
    software_version: str = __version__


@dataclass(frozen=True)
class SystemState:
    """z_k -- the complete, versioned state of an evolving robot system."""

    policy: PolicyState
    memory: MemoryState = MemoryState()
    data: DataState = DataState()
    tools: ToolPermissions = ToolPermissions()
    reward: RewardSpec = RewardSpec()
    evaluator: EvaluatorSpec = EvaluatorSpec()
    provenance: Provenance = Provenance()

    # ------------------------------------------------------------------ hashing
    def component_hashes(self) -> dict[str, str]:
        names = MUTABLE_COMPONENTS + ("provenance",)
        return {name: content_hash(getattr(self, name)) for name in names}

    @property
    def digest(self) -> str:
        return content_hash(self.component_hashes())

    @property
    def generation(self) -> int:
        return self.provenance.generation

    def evolve(self, **changes: Any) -> "SystemState":
        return replace(self, **changes)

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SystemState":
        return cls(
            policy=PolicyState(**d["policy"]),
            memory=MemoryState(entries=tuple(d["memory"]["entries"])),
            data=DataState(dataset_ids=tuple(d["data"]["dataset_ids"]), num_episodes=d["data"]["num_episodes"]),
            tools=ToolPermissions(allowed=frozenset(d["tools"]["allowed"])),
            reward=RewardSpec(**d["reward"]),
            evaluator=EvaluatorSpec(**d["evaluator"]),
            provenance=Provenance(**d["provenance"]),
        )
