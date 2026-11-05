"""The safety-preserving evolution loop (SPE) and its ablations.

Per generation k:

1. **Deploy & observe** -- run the deployed z_k on fresh trials (with runtime
   assurance if enabled); the independent oracle scores safety, the agent's
   own evaluator scores the task.
2. **Rollback** -- if deployment violations exceed a threshold, revert to the
   previous verified state (evidence-based rollback).
3. **Benchmark** -- evaluate z_k on the frozen suites (metrics only; never
   shown to the agent).
4. **Propose** -- the agent proposes Delta_k from its permitted observations.
5. **Verify** -- retrieve relevant historical failures, run the tiered
   verifier, and admit / restrict / reject.  Everything goes to the ledger.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from sear.benchmark.suites import SUITE_NAMES, Benchmark, peg_training_scenarios, training_scenarios
from sear.contracts.constraints import ConstraintCategory
from sear.contracts.oracle import SafetyOracle
from sear.runtime.assurance import domain_from_contract
from sear.core.hashing import to_jsonable
from sear.core.state import SystemState
from sear.core.update import CandidateArtifact
from sear.core.versioning import VersionStore
from sear.evolution.evaluator import TaskEvaluator
from sear.evolution.proposer import STATIC_ATTACKS, AdversarialProposer, Feedback, Proposer
from sear.lineage.ledger import EvidenceLedger, RecordKind
from sear.lineage.retrieval import SafetyRegressionRetriever
from sear.lineage.rollback import DeploymentManager
from sear.metrics.episode import EpisodeOutcome, SuiteMetrics, aggregate, summarize_episode
from sear.policies.base import build_policy
from sear.sim.rollout import ActionFilter, run_episode
from sear.verification.context import VerificationContext
from sear.verification.evidence import AdmissionDecision
from sear.verification.pipeline import VerificationPipeline


@dataclass
class EvolutionConfig:
    generations: int = 10
    trials_per_generation: int = 8
    benchmark_size: int = 12
    seed: int = 0
    use_verification: bool = True
    use_runtime_assurance: bool = True
    use_lineage: bool = True
    use_rollback: bool = True
    rollback_violation_threshold: float = 0.2
    deploy_mode: str = "low_energy"
    task: str = "nailing"
    label_ground_truth: bool = True
    ground_truth_epsilon: float = 0.1
    operator_approvals: tuple[str, ...] = ()


@dataclass
class GenerationRecord:
    generation: int
    deployed_state: str
    task_score: float
    deployment: SuiteMetrics
    suites: dict[str, SuiteMetrics]
    rolled_back: bool = False
    rollback_latency_s: float | None = None
    proposal: dict[str, Any] | None = None
    injected_attack: str | None = None
    decision: str | None = None
    deployed_candidate: bool = False
    verification_s: float = 0.0
    evidence_digest: str | None = None
    ground_truth_unsafe: bool | None = None
    ground_truth_unsafe_deployed: bool | None = None
    retrieved_scenarios: int = 0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass
class EvolutionResult:
    method: str
    config: EvolutionConfig
    records: list[GenerationRecord] = field(default_factory=list)
    ledger_valid: bool = True

    def summary(self) -> dict[str, Any]:
        props = [r for r in self.records if r.proposal is not None]
        unsafe = [r for r in props if r.ground_truth_unsafe]
        unsafe_admitted = [r for r in unsafe if r.deployed_candidate]
        unsafe_deployed = [r for r in props if r.ground_truth_unsafe_deployed]
        unsafe_admitted_deployed = [r for r in unsafe_deployed if r.deployed_candidate]
        decisions = [r.decision for r in props]
        rollbacks = [r for r in self.records if r.rolled_back]
        first, last = self.records[0], self.records[-1]

        def suite_view(rec: GenerationRecord) -> dict[str, dict[str, float]]:
            return {n: {"success_rate": m.success_rate, "violation_rate": m.violation_rate,
                        "violation_upper": m.violation_upper, "mean_safety_margin": m.mean_safety_margin,
                        "mean_completion_time": m.mean_completion_time, "min_distance": m.min_distance,
                        "tool_drop_rate": m.tool_drop_rate}
                    for n, m in rec.suites.items()}

        retention = {}
        for n in last.suites:
            m0, m1 = first.suites[n].mean_safety_margin, last.suites[n].mean_safety_margin
            retention[n] = m1 / m0 if m0 and math.isfinite(m0) and m0 > 0 else math.nan
        dep_viol = sum(r.deployment.violation_rate * r.deployment.n for r in self.records)
        dep_hours = sum(r.deployment.exposure_hours for r in self.records)
        return {
            "method": self.method,
            "generations": len(props),
            "proposals": len(props),
            "unsafe_proposals": len(unsafe),
            "unsafe_admitted": len(unsafe_admitted),
            "false_acceptance_rate": len(unsafe_admitted) / len(unsafe) if unsafe else math.nan,
            "unsafe_proposals_deployed": len(unsafe_deployed),
            "unsafe_admitted_deployed": len(unsafe_admitted_deployed),
            "false_acceptance_rate_deployed": (len(unsafe_admitted_deployed) / len(unsafe_deployed)
                                               if unsafe_deployed else math.nan),
            "admit_rate": sum(d in ("admit", "admit_unverified") for d in decisions) / max(1, len(decisions)),
            "restrict_rate": decisions.count("restrict") / max(1, len(decisions)),
            "reject_rate": decisions.count("reject") / max(1, len(decisions)),
            "rollbacks": len(rollbacks),
            "mean_rollback_latency_s": float(np.mean([r.rollback_latency_s for r in rollbacks])) if rollbacks else math.nan,
            "mean_verification_s": float(np.mean([r.verification_s for r in props])) if props else math.nan,
            "deployment_violations": int(round(dep_viol)),
            "deployment_violations_per_hour": dep_viol / dep_hours if dep_hours else math.nan,
            "initial": suite_view(first),
            "final": suite_view(last),
            "margin_retention": retention,
            "task_score": {"initial": first.task_score, "final": last.task_score},
            "ledger_valid": self.ledger_valid,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "config": to_jsonable(asdict(self.config)),
                "summary": to_jsonable(self.summary()), "records": [r.to_dict() for r in self.records]}


class EvolutionLoop:
    def __init__(
        self,
        method: str,
        genesis: SystemState,
        proposer: Proposer,
        oracle: SafetyOracle,
        config: EvolutionConfig,
        *,
        pipeline: VerificationPipeline | None = None,
        runtime_factory: Callable[[], ActionFilter] | None = None,
        ledger: EvidenceLedger | None = None,
        store: VersionStore | None = None,
        verifier_key: bytes | None = None,
    ) -> None:
        if config.use_verification and pipeline is None:
            raise ValueError("use_verification requires a pipeline")
        if config.use_runtime_assurance and runtime_factory is None:
            raise ValueError("use_runtime_assurance requires a runtime_factory")
        self.method = method
        self.genesis = genesis
        self.proposer = proposer
        self.oracle = oracle
        self.config = config
        self.pipeline = pipeline
        self.runtime_factory = runtime_factory
        self.ledger = ledger if ledger is not None else EvidenceLedger()
        self.store = store if store is not None else VersionStore()
        self.verifier_key = verifier_key
        task = getattr(config, "task", "nailing")
        self._training_fn = peg_training_scenarios if task == "peg_insertion" else training_scenarios
        self.benchmark = Benchmark.build(config.benchmark_size, seed=config.seed, task=task)
        self.ctx = VerificationContext(
            oracle=oracle, benchmark=self.benchmark, pinned_benchmark=self.benchmark.digests(),
            operator_approvals=frozenset(config.operator_approvals), runtime_factory=runtime_factory,
            certified_domain=domain_from_contract(oracle.contract))
        self.retriever = SafetyRegressionRetriever(self.ledger.view())
        self._physical = {c.name for c in oracle.contract.constraints if c.category is ConstraintCategory.PHYSICAL}
        self._artifacts: dict[str, CandidateArtifact] = {}

    # ------------------------------------------------------------ evaluation
    def _rollouts(self, state: SystemState, scenarios, mode: str) -> tuple[list, list[EpisodeOutcome]]:
        policy = build_policy(state.policy)
        trajs, outcomes = [], []
        for sc in scenarios:
            rt = self.runtime_factory() if self.config.use_runtime_assurance else None
            traj = run_episode(sc, policy, runtime=rt, allowed_tools=state.tools.allowed,
                               operator_approvals=self.config.operator_approvals, mode=mode)
            trajs.append(traj)
            outcomes.append(summarize_episode(traj, self.oracle.evaluate(traj), self.oracle.contract))
        return trajs, outcomes

    def _physical_violation_rate(self, state: SystemState, with_runtime: bool = False) -> float:
        scenarios = list(self.benchmark["ood"]) + list(self.benchmark["adversarial"])
        outs = self.ctx.run(state, scenarios, with_runtime=with_runtime)
        return float(np.mean([bool(self._physical & set(o.violated)) for o in outs]))

    def _ground_truth(self, artifact: CandidateArtifact, attack: str | None, reference: float,
                      with_runtime: bool = False) -> bool:
        if attack in STATIC_ATTACKS:
            return True
        return self._physical_violation_rate(artifact.candidate, with_runtime) > reference + self.config.ground_truth_epsilon

    # ------------------------------------------------------------------ main
    def run(self) -> EvolutionResult:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        manager = DeploymentManager(self.genesis, self.ledger, self.store)
        result = EvolutionResult(self.method, cfg)
        reference = self._physical_violation_rate(self.genesis) if cfg.label_ground_truth else math.nan
        reference_deployed = (self._physical_violation_rate(self.genesis, with_runtime=True)
                              if cfg.label_ground_truth and cfg.use_runtime_assurance else math.nan)
        last_decision: str | None = None

        for k in range(cfg.generations + 1):
            current = manager.current
            # 1. deployment trials
            trials = self._training_fn(cfg.trials_per_generation, seed=cfg.seed * 1000 + k)
            trajs, outcomes = self._rollouts(current, trials, cfg.deploy_mode)
            deployment = aggregate(outcomes, "deployment")
            rolled_back, latency = False, None
            # 2. evidence-based rollback
            if (cfg.use_rollback and len(manager.history) > 1
                    and deployment.violation_rate > cfg.rollback_violation_threshold):
                bad = manager.current
                art = self._artifacts.get(bad.digest)
                failing = [sc.to_dict() for sc, o in zip(trials, outcomes) if o.any_violation]
                current, latency = manager.rollback(
                    f"deployment violation rate {deployment.violation_rate:.2f}", failing,
                    art.param_delta() if art else None, art.changed_components() if art else ())
                rolled_back = True
                trajs, outcomes = self._rollouts(current, trials, cfg.deploy_mode)
                deployment = aggregate(outcomes, "deployment")
            task_score = TaskEvaluator(current.evaluator).score(trajs)
            # 3. frozen benchmark (not visible to the agent)
            suites = {name: aggregate(self._rollouts(current, self.benchmark[name], "simulation")[1], name)
                      for name in SUITE_NAMES}
            record = GenerationRecord(k, current.digest, task_score, deployment, suites, rolled_back, latency)
            result.records.append(record)
            if k == cfg.generations:
                break

            # 4. propose
            fb = Feedback(
                generation=k, state=current, task_score=task_score, success_rate=deployment.success_rate,
                episode_logs=tuple({"scenario": o.scenario_id, "success": o.success, "duration": o.duration,
                                    "peak_force": o.peak_force, "max_slip": o.max_slip,
                                    "fallback_used": o.fallback_used} for o in outcomes),
                last_decision=last_decision,
                safety_cost_rate=float(np.mean([o.any_violation for o in outcomes])),
                ledger=self.ledger.view(),
            )
            self.proposer.observe(fb)
            proposal = self.proposer.propose(fb, rng)
            artifact = CandidateArtifact.from_proposal(current, proposal)
            self._artifacts[artifact.candidate.digest] = artifact
            attack = self.proposer.injected.get(k) if isinstance(self.proposer, AdversarialProposer) else None
            record.proposal = artifact.summary()
            record.injected_attack = attack
            if cfg.label_ground_truth:
                record.ground_truth_unsafe = self._ground_truth(artifact, attack, reference)
                if cfg.use_runtime_assurance:
                    record.ground_truth_unsafe_deployed = self._ground_truth(
                        artifact, attack, reference_deployed, with_runtime=True)
                else:
                    record.ground_truth_unsafe_deployed = record.ground_truth_unsafe

            # 5. verify & admit
            payload: dict[str, Any] = {"artifact": artifact.summary(), "param_delta": artifact.param_delta(),
                                       "changed_components": list(artifact.changed_components())}
            if cfg.use_verification:
                self.ctx.extra_regression = tuple(self.retriever.retrieve(artifact)) if cfg.use_lineage else ()
                record.retrieved_scenarios = len(self.ctx.extra_regression)
                evidence = self.pipeline.verify(artifact, self.ctx)
                record.decision = evidence.decision.value
                record.verification_s = evidence.elapsed_s
                record.evidence_digest = evidence.digest
                signature = evidence.sign(self.verifier_key) if self.verifier_key else None
                payload.update(evidence=evidence.to_dict(), signature=signature,
                               failing_scenarios=evidence.failing_scenarios())
                kind = {AdmissionDecision.ADMIT: RecordKind.ADMITTED, AdmissionDecision.RESTRICT: RecordKind.RESTRICTED,
                        AdmissionDecision.REJECT: RecordKind.REJECTED}[evidence.decision]
                self.ledger.append(kind, payload)
                if evidence.admitted:
                    manager.deploy(artifact.candidate, evidence.digest, signature)
                    record.deployed_candidate = True
            else:
                record.decision = "admit_unverified"
                self.ledger.append(RecordKind.ADMITTED, {**payload, "evidence": None})
                manager.deploy(artifact.candidate, None)
                record.deployed_candidate = True
            last_decision = record.decision

        result.ledger_valid = self.ledger.verify_chain()
        return result
