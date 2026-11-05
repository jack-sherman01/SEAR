"""Tier 2: closed-loop simulation with parameter randomization and disturbances."""

from __future__ import annotations

import math

from sear.benchmark.suites import randomized_scenarios
from sear.core.types import GuaranteeLevel, Verdict
from sear.core.update import CandidateArtifact
from sear.metrics.episode import EpisodeOutcome
from sear.verification.checks.base import Check
from sear.verification.context import VerificationContext
from sear.verification.evidence import CheckResult


def sim_key(digest: str, n: int, with_runtime: bool) -> str:
    return f"sim:{digest}:{n}:{int(with_runtime)}"


def randomized_outcomes(artifact: CandidateArtifact, ctx: VerificationContext, n: int,
                        with_runtime: bool) -> tuple[list, list[EpisodeOutcome]]:
    """Randomized rollouts of the candidate, shared between checks via ``ctx.shared``."""
    key = sim_key(artifact.candidate.digest, n, with_runtime)
    if key not in ctx.shared:
        seed = int(artifact.digest[:8], 16) % 1_000_000  # fresh, but reproducible per artifact
        scenarios = randomized_scenarios(n, seed)
        ctx.shared[key] = (scenarios, ctx.run(artifact.candidate, scenarios, with_runtime))
    return ctx.shared[key]


class RandomizedSimulationCheck(Check):
    """FAIL on any *severe* event: tool loss, collision, or force beyond
    ``severe_force_factor`` x the contract limit."""

    name = "randomized_simulation"
    level = GuaranteeLevel.EMPIRICAL
    tier = 2

    def __init__(self, n: int = 60, severe_force_factor: float = 1.5, with_runtime: bool = False,
                 mandatory: bool = True) -> None:
        super().__init__(mandatory)
        self.n = n
        self.severe_force_factor = severe_force_factor
        self.with_runtime = with_runtime

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        scenarios, outcomes = randomized_outcomes(artifact, ctx, self.n, self.with_runtime)
        limit = next((c.upper for c in ctx.oracle.contract.constraints
                      if getattr(c, "signal", None) == "contact_force" and getattr(c, "upper", None)), math.inf)
        severe = [s for s, o in zip(scenarios, outcomes)
                  if o.tool_dropped or (o.min_distance < 0) or o.peak_force > self.severe_force_factor * limit]
        metrics = {
            "n": len(outcomes),
            "severe": len(severe),
            "violations": sum(o.any_violation for o in outcomes),
            "success_rate": sum(o.success for o in outcomes) / max(1, len(outcomes)),
        }
        if severe:
            return self.result(Verdict.FAIL, f"{len(severe)} severe events in {len(outcomes)} randomized rollouts",
                               metrics=metrics, failing=severe)
        return self.result(Verdict.PASS, f"no severe events in {len(outcomes)} randomized rollouts",
                           metrics=metrics, assumptions=("simulator fidelity; sim-to-real gap not covered",))
