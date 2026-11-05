"""Safety-regression retrieval (proposal WP4).

For each new candidate, select historical failures that are *relevant* to the
change it makes and add their scenarios to the regression suite.  Relevance:

* cosine similarity between the candidate's policy-parameter delta and the
  delta of a past rejected / rolled-back update (deltas are normalized by the
  policy's declared parameter ranges), or
* overlap in the non-policy components changed (evaluator, tools, ...).

The hypothesis to test is that this reduces repeated failure and unsafe
rediscovery across generations.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from sear.core.update import CandidateArtifact
from sear.lineage.ledger import LedgerView, RecordKind
from sear.policies.base import policy_class
from sear.sim.scenario import Scenario

FAILURE_KINDS = (RecordKind.REJECTED, RecordKind.ROLLBACK)


def _normalized(delta: Mapping[str, float], kind: str, keys: list[str]) -> np.ndarray:
    bounds = policy_class(kind).PARAM_BOUNDS
    return np.array([delta.get(k, 0.0) / max(1e-9, bounds[k][1] - bounds[k][0]) if k in bounds else 0.0
                     for k in keys])


class SafetyRegressionRetriever:
    def __init__(self, ledger: LedgerView, similarity_threshold: float = 0.5, max_scenarios: int = 12) -> None:
        self.ledger = ledger
        self.similarity_threshold = similarity_threshold
        self.max_scenarios = max_scenarios

    def similarity(self, artifact: CandidateArtifact, payload: Mapping) -> float:
        kind = artifact.candidate.policy.kind
        delta, past = artifact.param_delta(), payload.get("param_delta", {}) or {}
        keys = sorted(set(delta) | set(past))
        score = 0.0
        if keys:
            a, b = _normalized(delta, kind, keys), _normalized(past, kind, keys)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                score = float(a @ b / (na * nb))
        other = set(artifact.changed_components()) - {"policy", "memory", "data"}
        if other & (set(payload.get("changed_components", ())) - {"policy", "memory", "data"}):
            score = max(score, 1.0)
        return score

    def retrieve(self, artifact: CandidateArtifact) -> list[Scenario]:
        scored = []
        for rec in self.ledger.query(FAILURE_KINDS):
            sim = self.similarity(artifact, rec.payload)
            if sim >= self.similarity_threshold:
                scored.append((sim, rec.index, rec.payload.get("failing_scenarios", [])))
        scored.sort(key=lambda x: (-x[0], -x[1]))  # most similar, then most recent
        out: dict[str, Scenario] = {}
        for _, _, scenarios in scored:
            for s in scenarios:
                sc = Scenario.from_dict(s)
                out.setdefault(sc.scenario_id, sc)
                if len(out) >= self.max_scenarios:
                    return list(out.values())
        return list(out.values())
