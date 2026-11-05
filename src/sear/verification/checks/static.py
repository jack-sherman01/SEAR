"""Tier 0: static checks on permissions, protected files and interface invariants."""

from __future__ import annotations

import math
from fnmatch import fnmatch

from sear.contracts.constraints import AllowedTools, OperatorAuthorization
from sear.contracts.signing import ContractIntegrityError
from sear.core.types import GuaranteeLevel, Verdict
from sear.core.update import CandidateArtifact
from sear.policies.base import policy_class
from sear.verification.checks.base import Check
from sear.verification.context import VerificationContext
from sear.verification.evidence import CheckResult


class IntegrityCheck(Check):
    """The safety oracle and the frozen benchmark must be exactly as pinned."""

    name = "integrity"
    level = GuaranteeLevel.DETERMINISTIC
    tier = 0

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        try:
            ctx.oracle.check_integrity()
        except ContractIntegrityError as exc:
            return self.result(Verdict.FAIL, f"safety contract integrity: {exc}")
        if not ctx.benchmark.verify(ctx.pinned_benchmark):
            return self.result(Verdict.FAIL, "frozen benchmark suites differ from pinned digests")
        return self.result(Verdict.PASS, "contract signature and benchmark digests verified")


class ProtectedPathCheck(Check):
    name = "protected_paths"
    level = GuaranteeLevel.DETERMINISTIC
    tier = 0

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        hits = sorted({p for p in artifact.proposal.touched_paths
                       for pattern in ctx.protected_paths if fnmatch(p, pattern)})
        if hits:
            return self.result(Verdict.FAIL, f"update touches protected paths: {hits}", metrics={"paths": hits})
        return self.result(Verdict.PASS, "no protected path touched")


class PermissionCheck(Check):
    """No tool escalation beyond the contract; high-risk tools need operator approval."""

    name = "tool_permissions"
    level = GuaranteeLevel.DETERMINISTIC
    tier = 0

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        contract = ctx.oracle.contract
        added = sorted(artifact.candidate.tools.allowed - artifact.parent.tools.allowed)
        whitelist = {t for c in contract.of_type(AllowedTools) for t in c.allowed}
        needs_auth = {t for c in contract.of_type(OperatorAuthorization) for t in c.tools}
        problems = []
        for tool in added:
            if whitelist and tool not in whitelist:
                problems.append(f"{tool}: not in contract whitelist")
            elif tool in needs_auth and tool not in ctx.operator_approvals:
                problems.append(f"{tool}: requires operator authorization")
        if problems:
            return self.result(Verdict.FAIL, "; ".join(problems), metrics={"added": added})
        return self.result(Verdict.PASS, f"tools added: {added or 'none'}", metrics={"added": added})


class EvaluatorChangeCheck(Check):
    """Changes to the task evaluator or reward need independent human review.

    The safety oracle is separate and cannot be changed by the agent, but a
    modified task evaluator can still mislabel outcomes and steer future
    evolution, so it is never silently accepted: the verdict is UNKNOWN
    until the new version is in ``ctx.approved_evaluators``.
    """

    name = "evaluator_change"
    level = GuaranteeLevel.DETERMINISTIC
    tier = 0

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        changed = [c for c in artifact.changed_components() if c in ("evaluator", "reward")]
        if not changed:
            return self.result(Verdict.PASS, "evaluator and reward unchanged")
        ev = artifact.candidate.evaluator
        if "reward" not in changed and f"{ev.name}:{ev.version}" in ctx.approved_evaluators:
            return self.result(Verdict.PASS, f"evaluator {ev.name}:{ev.version} approved by reviewer")
        return self.result(Verdict.UNKNOWN, f"unreviewed change to {changed}; requires human review",
                           metrics={"changed": changed})


class InterfaceCheck(Check):
    """Policy kind unchanged, parameters known, finite and within declared bounds."""

    name = "interface_invariants"
    level = GuaranteeLevel.DETERMINISTIC
    tier = 0

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        old, new = artifact.parent.policy, artifact.candidate.policy
        if old.kind != new.kind:
            return self.result(Verdict.FAIL, f"policy kind changed {old.kind} -> {new.kind}")
        cls = policy_class(new.kind)
        problems = []
        for k, v in new.params.items():
            if k not in cls.PARAM_BOUNDS:
                problems.append(f"unknown parameter {k}")
            elif not math.isfinite(v):
                problems.append(f"{k} not finite")
            else:
                lo, hi = cls.PARAM_BOUNDS[k]
                if not lo <= v <= hi:
                    problems.append(f"{k}={v:g} outside [{lo:g}, {hi:g}]")
        if problems:
            return self.result(Verdict.FAIL, "; ".join(problems))
        return self.result(Verdict.PASS, "interface invariants hold")
