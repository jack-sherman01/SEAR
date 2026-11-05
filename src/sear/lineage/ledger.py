"""Append-only, hash-chained evidence ledger (proposal WP4).

Every accepted, restricted and rejected update, every deployment and every
rollback is recorded with its test conditions, failure modes, certificate
scope and rollback point.  Each record commits to its predecessor's hash, so
deleting or rewriting history is detectable by :meth:`EvidenceLedger.verify_chain`.

The agent only ever receives a :class:`LedgerView` (query-only).  In a real
deployment the ledger file lives outside the agent's sandbox; the in-process
separation here documents the interface, not a security boundary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from sear.core.hashing import content_hash, to_jsonable

GENESIS_HASH = "0" * 64


class RecordKind(str, Enum):
    GENESIS = "genesis"
    ADMITTED = "admitted"
    RESTRICTED = "restricted"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class LedgerRecord:
    index: int
    kind: RecordKind
    timestamp: float
    payload: Mapping[str, Any]
    prev_hash: str
    record_hash: str

    @staticmethod
    def compute_hash(index: int, kind: RecordKind, timestamp: float, payload: Mapping[str, Any], prev: str) -> str:
        return content_hash({"index": index, "kind": kind.value, "timestamp": timestamp,
                             "payload": payload, "prev_hash": prev})

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LedgerRecord":
        return cls(d["index"], RecordKind(d["kind"]), d["timestamp"], d["payload"], d["prev_hash"], d["record_hash"])


class LedgerIntegrityError(RuntimeError):
    pass


class EvidenceLedger:
    def __init__(self, path: str | Path | None = None) -> None:
        self._records: list[LedgerRecord] = []
        self._path = Path(path) if path is not None else None
        if self._path is not None and self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    self._records.append(LedgerRecord.from_dict(json.loads(line)))
            if not self.verify_chain():
                raise LedgerIntegrityError(f"ledger {self._path} failed hash-chain verification")

    def append(self, kind: RecordKind, payload: Mapping[str, Any]) -> LedgerRecord:
        payload = to_jsonable(payload)
        index = len(self._records)
        prev = self._records[-1].record_hash if self._records else GENESIS_HASH
        ts = time.time()
        rec = LedgerRecord(index, kind, ts, payload, prev, LedgerRecord.compute_hash(index, kind, ts, payload, prev))
        self._records.append(rec)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as fh:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
        return rec

    def records(self) -> tuple[LedgerRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def verify_chain(self) -> bool:
        prev = GENESIS_HASH
        for i, r in enumerate(self._records):
            if r.index != i or r.prev_hash != prev:
                return False
            if r.record_hash != LedgerRecord.compute_hash(r.index, r.kind, r.timestamp, r.payload, r.prev_hash):
                return False
            prev = r.record_hash
        return True

    def view(self) -> "LedgerView":
        return LedgerView(self)


class LedgerView:
    """Read-only query interface handed to the improvement agent."""

    __slots__ = ("_records_fn",)

    def __init__(self, ledger: EvidenceLedger) -> None:
        self._records_fn: Callable[[], tuple[LedgerRecord, ...]] = ledger.records

    def __len__(self) -> int:
        return len(self._records_fn())

    def query(self, kinds: Iterable[RecordKind] | None = None,
              where: Callable[[LedgerRecord], bool] | None = None) -> list[LedgerRecord]:
        ks = set(kinds) if kinds is not None else None
        return [r for r in self._records_fn() if (ks is None or r.kind in ks) and (where is None or where(r))]

    def latest(self, kind: RecordKind | None = None) -> LedgerRecord | None:
        found = self.query([kind] if kind else None)
        return found[-1] if found else None
