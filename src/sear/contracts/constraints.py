"""Constraint language for safety contracts (WP1).

Three layers, as in the proposal:

* physical    -- force/torque bounds, velocity limits, collision distance,
                 workspace boundaries, tool retention
* temporal    -- e.g. "a human entering a protected zone must trigger a safe
                 stop within 0.3 s"; "tool must stay secured during transport"
* operational -- allowed tool calls, maximum duration, approved operating
                 modes, mandatory operator authorization

Every constraint maps a :class:`~sear.core.trajectory.Trajectory` to a
:class:`ConstraintResult` with a quantitative *robustness margin* (>= 0 means
satisfied).  A missing or invalid signal yields ``Verdict.UNKNOWN`` -- never PASS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Mapping

import numpy as np

from sear.core.hashing import to_jsonable
from sear.core.trajectory import Trajectory
from sear.core.types import Verdict


class ConstraintCategory(str, Enum):
    PHYSICAL = "physical"
    TEMPORAL = "temporal"
    OPERATIONAL = "operational"


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    category: ConstraintCategory
    verdict: Verdict
    margin: float  # robustness in constraint units; nan if unknown, inf if vacuous
    normalized_margin: float  # margin / scale, comparable across constraints
    violations: int
    severity: float  # largest violation magnitude (>= 0)
    first_violation_t: float | None = None
    detail: str = ""

    @property
    def violated(self) -> bool:
        return self.verdict is Verdict.FAIL


_REGISTRY: dict[str, type["Constraint"]] = {}


def register(cls: type["Constraint"]) -> type["Constraint"]:
    _REGISTRY[cls.TYPE] = cls
    return cls


def constraint_from_dict(d: Mapping[str, Any]) -> "Constraint":
    try:
        cls = _REGISTRY[d["type"]]
    except KeyError as exc:
        raise ValueError(f"unknown constraint type {d.get('type')!r}") from exc
    return cls.from_dict(d)


@dataclass(frozen=True, kw_only=True)
class Constraint:
    TYPE: ClassVar[str] = ""
    DEFAULT_CATEGORY: ClassVar[ConstraintCategory] = ConstraintCategory.PHYSICAL

    name: str
    category: ConstraintCategory | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.category is None:
            object.__setattr__(self, "category", self.DEFAULT_CATEGORY)
        else:
            object.__setattr__(self, "category", ConstraintCategory(self.category))

    def evaluate(self, traj: Trajectory) -> ConstraintResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        d = to_jsonable(self)
        d["type"] = self.TYPE
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Constraint":
        names = {f.name for f in fields(cls)}
        kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**kwargs)

    # ---------------------------------------------------------------- helpers
    def _result(self, verdict: Verdict, margin: float, scale: float, violations: int = 0,
                severity: float = 0.0, first_t: float | None = None, detail: str = "") -> ConstraintResult:
        scale = abs(scale) if scale else 1.0
        norm = margin / scale if math.isfinite(margin) else margin
        return ConstraintResult(self.name, self.category, verdict, float(margin), float(norm),
                                int(violations), float(severity), first_t, detail)

    def _unknown(self, detail: str) -> ConstraintResult:
        return self._result(Verdict.UNKNOWN, math.nan, 1.0, detail=detail)

    def _from_margins(self, margins: np.ndarray, t: np.ndarray, scale: float,
                      active: np.ndarray | None = None) -> ConstraintResult:
        if active is not None:
            margins, t = margins[active], t[active]
            if margins.size == 0:
                return self._result(Verdict.PASS, math.inf, scale, detail="condition never active")
        if margins.size == 0:
            return self._unknown("empty trajectory")
        bad = margins < 0
        if bad.any():
            idx = int(np.argmax(bad))
            sev = float(-np.nanmin(margins))
            return self._result(Verdict.FAIL, float(np.nanmin(margins)), scale, int(bad.sum()), sev, float(t[idx]))
        if np.isnan(margins).any():
            return self._unknown("signal contains invalid samples")
        return self._result(Verdict.PASS, float(margins.min()), scale)


# ============================================================== physical
@register
@dataclass(frozen=True, kw_only=True)
class SignalBound(Constraint):
    """lower <= signal(t) <= upper for all t (vector signals use their norm)."""

    TYPE: ClassVar[str] = "signal_bound"
    signal: str
    upper: float | None = None
    lower: float | None = None

    def _scale(self) -> float:
        bounds = [abs(b) for b in (self.upper, self.lower) if b is not None and b != 0]
        return min(bounds) if bounds else 1.0

    def margins(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if x.ndim > 1:
            x = np.linalg.norm(x, axis=-1)
        m = np.full(x.shape, np.inf)
        if self.upper is not None:
            m = np.minimum(m, self.upper - x)
        if self.lower is not None:
            m = np.minimum(m, x - self.lower)
        return m

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        if not traj.has(self.signal):
            return self._unknown(f"signal {self.signal!r} missing")
        return self._from_margins(self.margins(traj.get(self.signal)), traj.t, self._scale())


@register
@dataclass(frozen=True, kw_only=True)
class WorkspaceBox(Constraint):
    """low <= signal(t) <= high componentwise (e.g. end-effector position)."""

    TYPE: ClassVar[str] = "workspace_box"
    signal: str = "ee_pos"
    low: tuple[float, ...]
    high: tuple[float, ...]

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        if not traj.has(self.signal):
            return self._unknown(f"signal {self.signal!r} missing")
        x = np.atleast_2d(traj.get(self.signal))
        lo, hi = np.asarray(self.low, float), np.asarray(self.high, float)
        if x.shape[-1] != lo.size:
            return self._unknown("dimension mismatch")
        m = np.minimum(x - lo, hi - x).min(axis=-1)
        return self._from_margins(m, traj.t, float(np.min(hi - lo) / 2))


# ============================================================== temporal
@register
@dataclass(frozen=True, kw_only=True)
class ConditionalBound(SignalBound):
    """Bound that must hold whenever ``condition`` signal is active (> 0.5).

    Example: tool slip rate must be ~0 while *transporting* the tool.
    """

    TYPE: ClassVar[str] = "conditional_bound"
    DEFAULT_CATEGORY: ClassVar[ConstraintCategory] = ConstraintCategory.TEMPORAL
    condition: str

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        for s in (self.signal, self.condition):
            if not traj.has(s):
                return self._unknown(f"signal {s!r} missing")
        active = np.asarray(traj.get(self.condition)) > 0.5
        return self._from_margins(self.margins(traj.get(self.signal)), traj.t, self._scale(), active)


@register
@dataclass(frozen=True, kw_only=True)
class ResponseWithin(Constraint):
    """Whenever ``trigger`` becomes active, ``response`` must hold from at most
    ``max_delay`` seconds later until the trigger ends."""

    TYPE: ClassVar[str] = "response_within"
    DEFAULT_CATEGORY: ClassVar[ConstraintCategory] = ConstraintCategory.TEMPORAL
    trigger: str
    response: str
    max_delay: float

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        for s in (self.trigger, self.response):
            if not traj.has(s):
                return self._unknown(f"signal {s!r} missing")
        trig = np.asarray(traj.get(self.trigger)) > 0.5
        resp = np.asarray(traj.get(self.response)) > 0.5
        t = traj.t
        if not trig.any():
            return self._result(Verdict.PASS, math.inf, self.max_delay, detail="never triggered")
        violations, worst, first_t = 0, math.inf, None
        edges = np.flatnonzero(np.diff(np.concatenate(([0], trig.astype(int), [0]))))
        for start, end in zip(edges[::2], edges[1::2]):
            deadline = start
            while deadline < end and t[deadline] - t[start] < self.max_delay - 1e-9:
                deadline += 1
            late = ~resp[deadline:end]
            if late.any():
                violations += int(late.sum())
                worst = min(worst, -float(late.sum()) * traj.dt)
                if first_t is None:
                    first_t = float(t[deadline + int(np.argmax(late))])
            else:
                hits = np.flatnonzero(resp[start:end])
                reaction = float(t[start + hits[0]] - t[start]) if hits.size else 0.0
                worst = min(worst, self.max_delay - reaction)
        verdict = Verdict.FAIL if violations else Verdict.PASS
        return self._result(verdict, worst, self.max_delay, violations, max(0.0, -worst), first_t)


# =========================================================== operational
def _event_result(c: Constraint, bad: list[dict[str, Any]], detail: str) -> ConstraintResult:
    if bad:
        return c._result(Verdict.FAIL, -float(len(bad)), 1.0, len(bad), float(len(bad)),
                         float(bad[0].get("t", 0.0)), detail)
    return c._result(Verdict.PASS, 1.0, 1.0)


@register
@dataclass(frozen=True, kw_only=True)
class AllowedTools(Constraint):
    TYPE: ClassVar[str] = "allowed_tools"
    DEFAULT_CATEGORY: ClassVar[ConstraintCategory] = ConstraintCategory.OPERATIONAL
    allowed: tuple[str, ...]

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        bad = [e for e in traj.events if "tool" in e and e.get("invoked", True) and e["tool"] not in self.allowed]
        return _event_result(self, bad, f"disallowed tools: {sorted({e['tool'] for e in bad})}")


@register
@dataclass(frozen=True, kw_only=True)
class OperatorAuthorization(Constraint):
    """Listed tools may only be invoked with an operator authorization."""

    TYPE: ClassVar[str] = "operator_authorization"
    DEFAULT_CATEGORY: ClassVar[ConstraintCategory] = ConstraintCategory.OPERATIONAL
    tools: tuple[str, ...]

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        bad = [e for e in traj.events
               if e.get("tool") in self.tools and e.get("invoked", True) and not e.get("authorized", False)]
        return _event_result(self, bad, "unauthorized high-risk tool use")


@register
@dataclass(frozen=True, kw_only=True)
class MaxDuration(Constraint):
    TYPE: ClassVar[str] = "max_duration"
    DEFAULT_CATEGORY: ClassVar[ConstraintCategory] = ConstraintCategory.OPERATIONAL
    seconds: float

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        if not len(traj):
            return self._unknown("empty trajectory")
        m = self.seconds - traj.duration
        return self._result(Verdict.PASS if m >= 0 else Verdict.FAIL, m, self.seconds,
                            int(m < 0), max(0.0, -m), self.seconds if m < 0 else None)


@register
@dataclass(frozen=True, kw_only=True)
class ApprovedModes(Constraint):
    TYPE: ClassVar[str] = "approved_modes"
    DEFAULT_CATEGORY: ClassVar[ConstraintCategory] = ConstraintCategory.OPERATIONAL
    modes: tuple[str, ...]

    def evaluate(self, traj: Trajectory) -> ConstraintResult:
        ok = traj.mode in self.modes
        return self._result(Verdict.PASS if ok else Verdict.FAIL, 1.0 if ok else -1.0, 1.0,
                            int(not ok), float(not ok), 0.0 if not ok else None,
                            "" if ok else f"mode {traj.mode!r} not approved")
