"""Deployment history and evidence-based rollback."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from sear.core.state import SystemState
from sear.core.versioning import VersionStore
from sear.lineage.ledger import EvidenceLedger, RecordKind


class DeploymentManager:
    """Tracks which verified state is deployed and restores earlier ones.

    The stack only ever contains states that were deployed with evidence (or
    the human-approved genesis), so a rollback target is always a
    previously-verified system.
    """

    def __init__(self, genesis: SystemState, ledger: EvidenceLedger, store: VersionStore | None = None) -> None:
        self.ledger = ledger
        self.store = store or VersionStore()
        self.store.put(genesis)
        self._stack: list[SystemState] = [genesis]
        ledger.append(RecordKind.GENESIS, {"state": genesis.digest, "generation": genesis.generation,
                                           "policy": genesis.policy})

    @property
    def current(self) -> SystemState:
        return self._stack[-1]

    @property
    def history(self) -> tuple[SystemState, ...]:
        return tuple(self._stack)

    def deploy(self, state: SystemState, evidence_digest: str | None, signature: str | None = None) -> None:
        self.store.put(state)
        self._stack.append(state)
        self.ledger.append(RecordKind.DEPLOYED, {"state": state.digest, "generation": state.generation,
                                                 "evidence": evidence_digest, "signature": signature})

    def rollback(self, reason: str, failing_scenarios: Sequence[Mapping[str, Any]] = (),
                 param_delta: Mapping[str, float] | None = None,
                 changed_components: Sequence[str] = ()) -> tuple[SystemState, float]:
        """Revert to the previous verified state.  Returns (state, latency_s)."""
        start = time.perf_counter()
        if len(self._stack) == 1:
            raise RuntimeError("cannot roll back past genesis")
        bad = self._stack.pop()
        target = self._stack[-1]
        latency = time.perf_counter() - start
        self.ledger.append(RecordKind.ROLLBACK, {
            "from": bad.digest, "to": target.digest, "reason": reason, "latency_s": latency,
            "failing_scenarios": list(failing_scenarios), "param_delta": dict(param_delta or {}),
            "changed_components": list(changed_components),
        })
        return target, latency
