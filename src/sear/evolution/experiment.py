"""Experiment assembly: baselines B0-B3 and full SPE from a config dict / YAML.

==========  ======================  ============  =========  =====================
method      update verification    runtime RTA   lineage    notes
==========  ======================  ============  =========  =====================
B0          no                      no            no         raw self-improvement
B1          no                      yes           no         fixed runtime filter
B2          yes (raw statistics)    no            no
B3          no                      yes           no         Lagrangian Safe-RL agent
SPE         yes (deployed stats)    yes           yes        + rollback
==========  ======================  ============  =========  =====================
"""

from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from sear.contracts.constraints import ConstraintCategory
from sear.contracts.oracle import SafetyOracle
from sear.contracts.signing import load_key, read_signed
from sear.core.state import DataState, PolicyState, SystemState, ToolPermissions
from sear.core.versioning import VersionStore
from sear.evolution.evaluator import DEFAULT_EVALUATOR
from sear.evolution.loop import EvolutionConfig, EvolutionLoop, EvolutionResult
from sear.evolution.proposer import AdversarialProposer, GreedyTuningProposer
from sear.lineage.ledger import EvidenceLedger
from sear.policies.base import policy_class
from sear.runtime.assurance import RuntimeAssurance
from sear.verification.pipeline import default_pipeline

METHODS: dict[str, dict[str, Any]] = {
    "B0": dict(use_verification=False, use_runtime_assurance=False, use_lineage=False, use_rollback=False),
    "B1": dict(use_verification=False, use_runtime_assurance=True, use_lineage=False, use_rollback=False),
    "B2": dict(use_verification=True, use_runtime_assurance=False, use_lineage=False, use_rollback=False),
    "B3": dict(use_verification=False, use_runtime_assurance=True, use_lineage=False, use_rollback=False,
               safe_rl=True),
    "SPE": dict(use_verification=True, use_runtime_assurance=True, use_lineage=True, use_rollback=True),
}


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def make_genesis(cfg: Mapping[str, Any]) -> SystemState:
    g = cfg.get("genesis", {})
    cls = policy_class(g.get("policy", "approach_press"))
    params = {**cls.DEFAULTS, **g.get("params", {})}
    return SystemState(
        policy=PolicyState(cls.kind, params),
        data=DataState(dataset_ids=("human-demo-v0",)),
        tools=ToolPermissions(frozenset(g.get("tools", ["gripper.hold", "fastener.drive"]))),
        evaluator=DEFAULT_EVALUATOR,
    )


def build_loop(method: str, cfg: Mapping[str, Any], oracle: SafetyOracle, key: bytes,
               out_dir: Path | None = None) -> EvolutionLoop:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    flags = dict(METHODS[method])
    safe_rl = flags.pop("safe_rl", False)
    names = {f.name for f in fields(EvolutionConfig)}
    evo = EvolutionConfig(**{**{k: v for k, v in cfg.items() if k in names}, **flags})

    contract = oracle.contract
    physical = {c.name for c in contract.constraints if c.category is ConstraintCategory.PHYSICAL}

    def cost_fn(traj) -> float:
        return float(bool(physical & set(contract.evaluate(traj).violated)))

    p = cfg.get("proposer", {})
    base = GreedyTuningProposer(n_candidates=p.get("n_candidates", 6), trials=p.get("trials", 4),
                                step=p.get("step", 0.3), cost_fn=cost_fn if safe_rl else None, lagrangian=safe_rl)
    adv = cfg.get("adversary", {})
    proposer = AdversarialProposer(base, schedule=adv.get("schedule"), rate=adv.get("rate", 0.0))

    v = cfg.get("verification", {})
    pipeline = default_pipeline(statistical_with_runtime=evo.use_runtime_assurance, epsilon=v.get("epsilon", 0.1),
                                confidence=v.get("confidence", 0.9), n_sim=v.get("n_sim", 60),
                                certificate_mode=v.get("certificate_mode", "filtered"))
    run_dir = out_dir / method if out_dir else None
    return EvolutionLoop(
        method, make_genesis(cfg), proposer, oracle, evo,
        pipeline=pipeline if evo.use_verification else None,
        runtime_factory=(lambda: RuntimeAssurance(contract)) if evo.use_runtime_assurance else None,
        ledger=EvidenceLedger(run_dir / "ledger.jsonl" if run_dir else None),
        store=VersionStore(run_dir / "states" if run_dir else None),
        verifier_key=key,
    )


def run_experiment(cfg: Mapping[str, Any], methods: Sequence[str] | None = None,
                   out_dir: str | Path | None = None, log=print) -> dict[str, EvolutionResult]:
    key = load_key(key_file=cfg.get("contract_key_file"))
    oracle = SafetyOracle(read_signed(cfg["contract"], key), key)
    methods = list(methods or cfg.get("methods", list(METHODS)))
    out = Path(out_dir) if out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    results = {}
    for m in methods:
        start = time.perf_counter()
        loop = build_loop(m, cfg, oracle, key, out)
        res = loop.run()
        results[m] = res
        s = res.summary()
        log(f"[{m}] {time.perf_counter() - start:.1f}s  unsafe admitted {s['unsafe_admitted']}/{s['unsafe_proposals']}"
            f"  deployment violations {s['deployment_violations']}  rollbacks {s['rollbacks']}"
            f"  final in-dist success {s['final']['in_distribution']['success_rate']:.2f}")
        if out is not None:
            (out / m).mkdir(parents=True, exist_ok=True)
            (out / m / "result.json").write_text(json.dumps(res.to_dict(), indent=2))
    return results
