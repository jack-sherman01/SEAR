"""Content-addressed store of system states."""

from __future__ import annotations

import json
from pathlib import Path

from sear.core.state import SystemState


class VersionStore:
    """Stores every system state ever produced, keyed by digest.

    When ``root`` is given, states are also written as ``<root>/<digest>.json``
    and never overwritten.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._states: dict[str, SystemState] = {}
        self._root = Path(root) if root is not None else None
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            for path in sorted(self._root.glob("*.json")):
                state = SystemState.from_dict(json.loads(path.read_text()))
                if state.digest != path.stem:
                    raise ValueError(f"state file {path} does not match its digest")
                self._states[state.digest] = state

    def put(self, state: SystemState) -> str:
        digest = state.digest
        if digest not in self._states:
            self._states[digest] = state
            if self._root is not None:
                path = self._root / f"{digest}.json"
                if not path.exists():
                    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True))
        return digest

    def get(self, digest: str) -> SystemState:
        return self._states[digest]

    def __contains__(self, digest: object) -> bool:
        return digest in self._states

    def __len__(self) -> int:
        return len(self._states)

    def ancestry(self, digest: str) -> list[SystemState]:
        """States from ``digest`` back to the oldest stored ancestor."""
        chain: list[SystemState] = []
        current: str | None = digest
        while current is not None and current in self._states:
            state = self._states[current]
            chain.append(state)
            current = state.provenance.parent
        return chain
