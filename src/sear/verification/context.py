"""Everything the verifier needs, held *outside* the agent's writable state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from sear.benchmark.suites import Benchmark
from sear.contracts.oracle import SafetyOracle
from sear.core.state import SystemState
from sear.metrics.episode import EpisodeOutcome, summarize_episode
from sear.policies.base import build_policy
from sear.runtime.monitor import CertifiedDomain
from sear.sim.rollout import ActionFilter, run_episode
from sear.sim.scenario import Scenario

#: Files no candidate update may touch (contracts, frozen tests, oracle, verifier, ledger).
DEFAULT_PROTECTED_PATHS = (
    "configs/contracts/*",
    "src/sear/contracts/*",
    "src/sear/verification/*",
    "src/sear/verification/checks/*",
    "src/sear/benchmark/*",
    "src/sear/lineage/*",
    "runs/*/ledger.jsonl",
)


@dataclass
class VerificationContext:
    oracle: SafetyOracle
    benchmark: Benchmark
    pinned_benchmark: Mapping[str, str]
    operator_approvals: frozenset[str] = frozenset()
    approved_evaluators: frozenset[str] = frozenset()  # "name:version" reviewed by a human
    protected_paths: tuple[str, ...] = DEFAULT_PROTECTED_PATHS
    runtime_factory: Callable[[], ActionFilter] | None = None
    certified_domain: CertifiedDomain = field(default_factory=CertifiedDomain)
    extra_regression: tuple[Scenario, ...] = ()  # filled by lineage retrieval
    mode: str = "simulation"
    shared: dict[str, Any] = field(default_factory=dict)
    _cache: dict[tuple[str, str, bool], EpisodeOutcome] = field(default_factory=dict, repr=False)

    def run(self, state: SystemState, scenarios: Iterable[Scenario], with_runtime: bool = False) -> list[EpisodeOutcome]:
        """Roll out ``state`` on ``scenarios`` and score with the independent oracle (cached)."""
        if with_runtime and self.runtime_factory is None:
            raise ValueError("verification with runtime assurance requested but no runtime_factory given")
        outcomes = []
        policy = None
        for sc in scenarios:
            key = (state.digest, sc.digest, with_runtime)
            if key not in self._cache:
                policy = policy or build_policy(state.policy)
                traj = run_episode(
                    sc, policy,
                    runtime=self.runtime_factory() if with_runtime else None,
                    allowed_tools=state.tools.allowed,
                    operator_approvals=self.operator_approvals,
                    mode=self.mode,
                )
                self._cache[key] = summarize_episode(traj, self.oracle.evaluate(traj), self.oracle.contract)
            outcomes.append(self._cache[key])
        return outcomes

    def regression_scenarios(self, suites: Sequence[str]) -> list[Scenario]:
        scenarios = [s for name in suites for s in self.benchmark[name]]
        known = {s.scenario_id for s in scenarios}
        return scenarios + [s for s in self.extra_regression if s.scenario_id not in known]
