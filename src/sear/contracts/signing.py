"""Signing and integrity verification for contracts.

A contract is stored as ``<name>.yaml`` next to ``<name>.yaml.sig`` (JSON with
digest + HMAC-SHA256).  The signing key is held by the human safety authority,
never by the improvement agent.  In a real deployment the contract directory is
mounted read-only outside the agent's workspace; here we additionally chmod the
files read-only and re-verify on every load.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from sear.contracts.contract import Contract

KEY_ENV = "SEAR_CONTRACT_KEY"
DEV_KEY = b"sear-dev-key-not-for-hardware"


class ContractIntegrityError(RuntimeError):
    """Raised when a contract's signature or digest does not verify."""


def load_key(env: str = KEY_ENV, key_file: str | Path | None = None) -> bytes:
    if key_file is not None:
        return Path(key_file).read_bytes().strip()
    value = os.environ.get(env)
    if value:
        return value.encode()
    warnings.warn(f"{env} not set; using the development key (simulation only)", stacklevel=2)
    return DEV_KEY


@dataclass(frozen=True)
class SignedContract:
    contract: Contract
    signature: str
    key_id: str = "default"

    @property
    def digest(self) -> str:
        return self.contract.digest


def _mac(digest: str, key: bytes) -> str:
    return hmac.new(key, digest.encode(), hashlib.sha256).hexdigest()


def sign_contract(contract: Contract, key: bytes, key_id: str = "default") -> SignedContract:
    return SignedContract(contract, _mac(contract.digest, key), key_id)


def verify_signature(signed: SignedContract, key: bytes) -> bool:
    return hmac.compare_digest(signed.signature, _mac(signed.contract.digest, key))


def write_signed(contract_yaml: str | Path, key: bytes, key_id: str = "default") -> Path:
    """Sign a YAML contract in place, writing ``<file>.sig`` and making both read-only."""
    path = Path(contract_yaml)
    signed = sign_contract(Contract.from_yaml(path), key, key_id)
    sig = path.with_name(path.name + ".sig")
    if sig.exists():
        sig.chmod(0o644)
    sig.write_text(json.dumps({"digest": signed.digest, "signature": signed.signature, "key_id": key_id}, indent=2))
    for p in (path, sig):
        p.chmod(0o444)
    return sig


def read_signed(contract_yaml: str | Path, key: bytes) -> SignedContract:
    path = Path(contract_yaml)
    sig_path = path.with_name(path.name + ".sig")
    if not sig_path.exists():
        raise ContractIntegrityError(f"missing signature file {sig_path}; run `sear sign-contract {path}`")
    meta = json.loads(sig_path.read_text())
    signed = SignedContract(Contract.from_yaml(path), meta["signature"], meta.get("key_id", "default"))
    if signed.digest != meta["digest"]:
        raise ContractIntegrityError(f"contract {path} was modified after signing")
    if not verify_signature(signed, key):
        raise ContractIntegrityError(f"invalid signature for contract {path}")
    return signed
