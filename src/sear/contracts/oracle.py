"""The independent safety oracle.

Task reward/evaluation (agent-modifiable, part of z_k) and safety evaluation
(this class, outside z_k) are strictly separated so that optimizing task
success cannot rewrite the safety specification.
"""

from __future__ import annotations

from sear.contracts.contract import Contract, ContractReport
from sear.contracts.signing import ContractIntegrityError, SignedContract, verify_signature
from sear.core.trajectory import Trajectory


class SafetyOracle:
    def __init__(self, signed: SignedContract, key: bytes) -> None:
        if not verify_signature(signed, key):
            raise ContractIntegrityError("refusing to build oracle from an unverified contract")
        self._signed = signed
        self._key = key
        self._pinned_digest = signed.digest

    @property
    def contract(self) -> Contract:
        return self._signed.contract

    @property
    def digest(self) -> str:
        return self._pinned_digest

    def check_integrity(self) -> None:
        if self._signed.contract.digest != self._pinned_digest or not verify_signature(self._signed, self._key):
            raise ContractIntegrityError("safety contract changed since the oracle was created")

    def evaluate(self, traj: Trajectory) -> ContractReport:
        self.check_integrity()
        return self._signed.contract.evaluate(traj)
