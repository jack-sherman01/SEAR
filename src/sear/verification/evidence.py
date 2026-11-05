"""Evidence bundles produced by the verifier.

Evidence(z_{k+1}) |= C  is represented by an :class:`Evidence` object whose
decision is ADMIT.  Evidence is content-hashed and HMAC-signed by the verifier
so that it can be independently re-checked later (WP4 lineage).
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from sear import __version__
from sear.core.hashing import content_hash, to_jsonable
from sear.core.types import GuaranteeLevel, Verdict


class AdmissionDecision(str, Enum):
    ADMIT = "admit"  # all mandatory checks PASS -> may be deployed
    RESTRICT = "restrict"  # no FAIL, but some mandatory UNKNOWN -> simulation / low-energy only
    REJECT = "reject"  # some mandatory check FAILed


@dataclass(frozen=True)
class CheckResult:
    name: str
    verdict: Verdict
    level: GuaranteeLevel
    mandatory: bool
    summary: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    failing_scenarios: tuple[Mapping[str, Any], ...] = ()
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class Evidence:
    artifact_digest: str
    parent_digest: str
    candidate_digest: str
    contract_digest: str
    benchmark_digests: Mapping[str, str]
    checks: tuple[CheckResult, ...]
    decision: AdmissionDecision
    created_at: float = field(default_factory=time.time)
    verifier_version: str = __version__

    @property
    def admitted(self) -> bool:
        return self.decision is AdmissionDecision.ADMIT

    @property
    def digest(self) -> str:
        return content_hash(self.to_dict())

    @property
    def elapsed_s(self) -> float:
        return sum(c.elapsed_s for c in self.checks)

    def check(self, name: str) -> CheckResult:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def failing_scenarios(self) -> list[Mapping[str, Any]]:
        seen, out = set(), []
        for c in self.checks:
            for s in c.failing_scenarios:
                if s["scenario_id"] not in seen:
                    seen.add(s["scenario_id"])
                    out.append(s)
        return out

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def sign(self, key: bytes) -> str:
        return hmac.new(key, self.digest.encode(), hashlib.sha256).hexdigest()

    def verify(self, signature: str, key: bytes) -> bool:
        return hmac.compare_digest(signature, self.sign(key))
