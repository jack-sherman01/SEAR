# SEAR

**Safe Evolution of Agentic Robots: Verifiable Safety Invariants for Self-Improving Physical Agents**

A robot may be safe in generation *k* and unsafe after an apparently beneficial
update to its policy, memory, evaluator or tools. SEAR is a framework for
admitting autonomous updates only with independently checkable evidence, so
that task capability can improve across generations while physical safety
violations and safety-margin degradation stay bounded.

## Concepts

| Proposal | Code |
|---|---|
| system state z_k = (θ, m, d, τ, r, e, P) | `sear.core.SystemState` |
| update operator z_{k+1} = U(z_k, Δ_k) | `sear.core.apply_update`, `CandidateArtifact` |
| immutable safety contract 𝔠 (WP1) | `sear.contracts` — YAML + HMAC signature, `SafetyOracle` |
| Admit(z_k, z_{k+1}) ⇒ Evidence ⊨ 𝔠 (WP2) | `sear.verification` — tiered checks → `Evidence` |
| runtime assurance, CBF-QP, fallback (WP3) | `sear.runtime` — `RuntimeAssurance` |
| evidence lineage, retrieval, rollback (WP4) | `sear.lineage` — hash-chained ledger |
| multi-generation protocol and baselines | `sear.evolution` — `EvolutionLoop`, B0–B3, SPE |

Every verdict is `PASS`, `FAIL` or `UNKNOWN`, and `UNKNOWN` is never treated as
safe. All evidence is tagged as **deterministic** (under stated model
assumptions), **probabilistic** (exact Clopper–Pearson bounds) or **empirical**.

## Layout

```
src/sear/
  core/          versioned system state, update operator, content hashing, trajectories
  contracts/     constraint language (physical / temporal / operational), signing, oracle
  sim/           planar contact-rich tool-use simulator (surrogate fastening task)
  policies/      parametric controllers (θ_k); bounds are interface invariants
  benchmark/     frozen, content-pinned suites: in-dist / held-out / OOD / adversarial
  runtime/       latency-aware HOCBF-QP safety filter, runtime monitor, safe stop
  verification/  checks: integrity, protected paths, permissions, evaluator change,
                 interface, replay regression, randomized sim, barrier certificate,
                 statistical bound → admit / restrict / reject
  lineage/       append-only hash-chained ledger, safety-regression retrieval, rollback
  metrics/       episode & suite metrics, Clopper–Pearson / Wilson / Poisson bounds
  evolution/     task evaluator, agents (greedy, Safe-RL, adversarial, external/LLM), loop
  robots/        hardware interface for real-manipulator experiments
  cli.py         `sear` command
configs/
  contracts/     signed safety contracts
  experiments/   study configurations
tests/
```

## Quick start

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate

pytest                                                    # test suite
sear run configs/experiments/smoke.yaml                   # ~20 s sanity run (B0 vs SPE)
sear run configs/experiments/nailing_default.yaml         # full study, all methods
sear report runs/nailing_default                          # comparison table
sear verify-ledger runs/nailing_default/SPE/ledger.jsonl  # audit the evidence chain
```

## Methods compared

| | update verification | runtime assurance | lineage + rollback | agent |
|---|---|---|---|---|
| **B0** | – | – | – | task-greedy |
| **B1** | – | ✓ | – | task-greedy |
| **B2** | ✓ | – | – | task-greedy |
| **B3** | – | ✓ | – | Lagrangian Safe-RL |
| **SPE** | ✓ | ✓ | ✓ | task-greedy |

The adversary injects scheduled unsafe updates: evaluator tampering, tool
escalation, protected-file writes, aggressive parameters, memory poisoning.
Reported metrics are those of proposal Sec. 6.4: false-acceptance rate of
unsafe updates, reject and rollback rates, deployment violations per hour,
per-suite success and violation rates with upper confidence bounds,
safety-margin retention, and verification cost.

## Safety contracts

Contracts live in `configs/contracts/` and are owned by the human safety
authority. After editing one, re-sign it:

```bash
export SEAR_CONTRACT_KEY=...        # never available to the improvement agent
sear sign-contract configs/contracts/nailing_surrogate.yaml
```

The committed signature uses the built-in **development key**, which is for
simulation only.

## Plugging in a real agent

`sear.evolution.ExternalAgentProposer` wraps any callable (for example an LLM
coding agent). The callable receives the permitted observations as a dict and
returns `UpdateProposal` fields. Everything it proposes goes through the same
verifier, runtime assurance and ledger.

## Status

This is a simulation-first implementation (proposal milestones M1–M8). The
real-robot work (M9–M10) will implement `sear.robots.RobotInterface` on top
of hardware interlocks.
