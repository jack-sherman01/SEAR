"""Base class for verification checks."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, ClassVar, Iterable, Mapping

from sear.core.types import GuaranteeLevel, Verdict
from sear.core.update import CandidateArtifact
from sear.sim.scenario import Scenario
from sear.verification.context import VerificationContext
from sear.verification.evidence import CheckResult


class Check(ABC):
    name: ClassVar[str]
    level: ClassVar[GuaranteeLevel]
    tier: ClassVar[int]  # cheap checks first: 0 static, 1 replay, 2 simulation, 3 certificate, 4 statistics

    def __init__(self, mandatory: bool = True) -> None:
        self.mandatory = mandatory

    def run(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        start = time.perf_counter()
        try:
            result = self.evaluate(artifact, ctx)
        except Exception as exc:  # an erroring check is *unknown*, never pass
            result = self.result(Verdict.UNKNOWN, f"check raised {type(exc).__name__}: {exc}")
        return replace(result, elapsed_s=time.perf_counter() - start)

    @abstractmethod
    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult: ...

    def result(self, verdict: Verdict, summary: str, *, metrics: Mapping[str, Any] | None = None,
               assumptions: Iterable[str] = (), failing: Iterable[Scenario] = (),
               level: GuaranteeLevel | None = None) -> CheckResult:
        return CheckResult(
            name=self.name,
            verdict=verdict,
            level=level or self.level,
            mandatory=self.mandatory,
            summary=summary,
            metrics=dict(metrics or {}),
            assumptions=tuple(assumptions),
            failing_scenarios=tuple(s.to_dict() for s in failing),
        )
