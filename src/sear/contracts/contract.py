"""Safety contracts C: a named, versioned set of constraints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from sear.contracts.constraints import (
    Constraint,
    ConstraintCategory,
    ConstraintResult,
    constraint_from_dict,
)
from sear.core.hashing import content_hash
from sear.core.trajectory import Trajectory
from sear.core.types import Verdict


@dataclass(frozen=True)
class ContractReport:
    contract: str
    contract_digest: str
    results: tuple[ConstraintResult, ...]

    @property
    def verdict(self) -> Verdict:
        return Verdict.combine(r.verdict for r in self.results)

    @property
    def violated(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.results if r.verdict is Verdict.FAIL)

    @property
    def unknown(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.results if r.verdict is Verdict.UNKNOWN)

    @property
    def any_violation(self) -> bool:
        return bool(self.violated)

    def result(self, name: str) -> ConstraintResult:
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(name)

    def min_normalized_margin(self, categories: Iterable[ConstraintCategory] | None = None) -> float:
        cats = set(categories) if categories is not None else None
        vals = [r.normalized_margin for r in self.results
                if (cats is None or r.category in cats) and math.isfinite(r.normalized_margin)]
        return min(vals) if vals else math.nan


@dataclass(frozen=True)
class Contract:
    name: str
    version: str
    constraints: tuple[Constraint, ...]
    description: str = ""

    def __post_init__(self) -> None:
        names = [c.name for c in self.constraints]
        if len(names) != len(set(names)):
            raise ValueError("constraint names must be unique")

    @property
    def digest(self) -> str:
        return content_hash(self.to_dict())

    def get(self, name: str) -> Constraint:
        for c in self.constraints:
            if c.name == name:
                return c
        raise KeyError(name)

    def of_type(self, cls: type) -> list[Any]:
        return [c for c in self.constraints if isinstance(c, cls)]

    def evaluate(self, traj: Trajectory) -> ContractReport:
        return ContractReport(self.name, self.digest, tuple(c.evaluate(traj) for c in self.constraints))

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "constraints": [c.to_dict() for c in self.constraints],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Contract":
        return cls(
            name=d["name"],
            version=str(d["version"]),
            description=d.get("description", ""),
            constraints=tuple(constraint_from_dict(c) for c in d["constraints"]),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Contract":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))
