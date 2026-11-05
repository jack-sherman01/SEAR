"""Policy interface and registry.

A policy is built from a :class:`~sear.core.state.PolicyState` (theta_k).  Each
policy class declares parameter defaults and admissible bounds; the bounds are
part of the interface invariants checked by the verifier.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Mapping

import numpy as np

from sear.core.state import PolicyState
from sear.sim.env import Observation

_REGISTRY: dict[str, type["Policy"]] = {}


class Policy(ABC):
    kind: ClassVar[str]
    DEFAULTS: ClassVar[dict[str, float]]
    PARAM_BOUNDS: ClassVar[dict[str, tuple[float, float]]]

    def __init__(self, params: Mapping[str, float] | None = None) -> None:
        merged = dict(self.DEFAULTS)
        merged.update(params or {})
        unknown = set(merged) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown parameters for {self.kind}: {sorted(unknown)}")
        self.params = merged

    @abstractmethod
    def act(self, obs: Observation) -> np.ndarray: ...

    def reset(self) -> None:
        """Clear per-episode internal state (if any)."""

    def required_tools(self) -> tuple[str, ...]:
        return ()

    @classmethod
    def default_state(cls) -> PolicyState:
        return PolicyState(kind=cls.kind, params=dict(cls.DEFAULTS))


def register_policy(cls: type[Policy]) -> type[Policy]:
    _REGISTRY[cls.kind] = cls
    return cls


def policy_class(kind: str) -> type[Policy]:
    try:
        return _REGISTRY[kind]
    except KeyError as exc:
        raise ValueError(f"unknown policy kind {kind!r}") from exc


def build_policy(state: PolicyState) -> Policy:
    return policy_class(state.kind)(state.params)
