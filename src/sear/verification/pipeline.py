"""Tiered candidate-update verification pipeline (proposal WP2).

Admit(z_k, z_{k+1}) = 1  only if every mandatory check PASSes.  Any mandatory
UNKNOWN restricts the candidate to simulation / low-energy testing; any
mandatory FAIL rejects it.  Cheap tiers run first and, with ``fail_fast``,
a mandatory FAIL skips the expensive tiers.
"""

from __future__ import annotations

from typing import Sequence

from sear.core.types import Verdict
from sear.core.update import CandidateArtifact
from sear.verification.checks import (
    BarrierCertificateCheck,
    Check,
    EvaluatorChangeCheck,
    IntegrityCheck,
    InterfaceCheck,
    PermissionCheck,
    ProtectedPathCheck,
    RandomizedSimulationCheck,
    ReplayRegressionCheck,
    StatisticalSafetyCheck,
)
from sear.verification.context import VerificationContext
from sear.verification.evidence import AdmissionDecision, CheckResult, Evidence


class VerificationPipeline:
    def __init__(self, checks: Sequence[Check], fail_fast: bool = True) -> None:
        self.checks = sorted(checks, key=lambda c: c.tier)
        self.fail_fast = fail_fast

    @staticmethod
    def decide(results: Sequence[CheckResult]) -> AdmissionDecision:
        mandatory = [r.verdict for r in results if r.mandatory]
        verdict = Verdict.combine(mandatory)
        if verdict is Verdict.FAIL:
            return AdmissionDecision.REJECT
        if verdict is Verdict.UNKNOWN:
            return AdmissionDecision.RESTRICT
        return AdmissionDecision.ADMIT

    def verify(self, artifact: CandidateArtifact, ctx: VerificationContext) -> Evidence:
        results: list[CheckResult] = []
        for check in self.checks:
            result = check.run(artifact, ctx)
            results.append(result)
            if self.fail_fast and result.mandatory and result.verdict is Verdict.FAIL:
                break
        return Evidence(
            artifact_digest=artifact.digest,
            parent_digest=artifact.parent.digest,
            candidate_digest=artifact.candidate.digest,
            contract_digest=ctx.oracle.digest,
            benchmark_digests=dict(ctx.pinned_benchmark),
            checks=tuple(results),
            decision=self.decide(results),
        )


def default_pipeline(*, statistical_with_runtime: bool = False, epsilon: float = 0.1, confidence: float = 0.9,
                     n_sim: int = 60, certificate_mode: str = "filtered", fail_fast: bool = True) -> VerificationPipeline:
    """The five-step verifier of the proposal.

    Regression and severe-event simulation always test the *raw* candidate
    (it must not rely on the filter for nominal safety); the statistical
    bound is computed for the configuration that will actually be deployed.
    """
    return VerificationPipeline(
        [
            IntegrityCheck(),
            ProtectedPathCheck(),
            PermissionCheck(),
            EvaluatorChangeCheck(),
            InterfaceCheck(),
            ReplayRegressionCheck(),
            RandomizedSimulationCheck(n=n_sim),
            BarrierCertificateCheck(mode=certificate_mode, mandatory=False),
            StatisticalSafetyCheck(epsilon=epsilon, confidence=confidence, n=n_sim,
                                   with_runtime=statistical_with_runtime),
        ],
        fail_fast=fail_fast,
    )
