"""Shared enumerations.

The proposal insists that deterministic guarantees, probabilistic guarantees and
empirical evidence "must never be conflated"; :class:`GuaranteeLevel` tags every
piece of evidence accordingly.  :class:`Verdict` has an explicit ``UNKNOWN`` so
that an unverifiable property is never silently treated as safe.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"

    @classmethod
    def combine(cls, verdicts: Iterable["Verdict"]) -> "Verdict":
        """Conservative conjunction: FAIL dominates UNKNOWN dominates PASS.

        An empty collection is UNKNOWN (no evidence is not evidence of safety).
        """
        vs = list(verdicts)
        if not vs:
            return cls.UNKNOWN
        if any(v is cls.FAIL for v in vs):
            return cls.FAIL
        if any(v is cls.UNKNOWN for v in vs):
            return cls.UNKNOWN
        return cls.PASS


class GuaranteeLevel(str, Enum):
    DETERMINISTIC = "deterministic"  # holds under stated model/disturbance assumptions
    PROBABILISTIC = "probabilistic"  # holds w.p. >= 1-delta under statistical assumptions
    EMPIRICAL = "empirical"  # observed on finite tests only


class OperatingMode(str, Enum):
    SIMULATION = "simulation"
    LOW_ENERGY = "low_energy"
    FULL = "full"
