"""Per-episode and per-suite safety/task metrics (proposal Sec. 6.4)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from sear.contracts.constraints import ConstraintCategory, SignalBound
from sear.contracts.contract import Contract, ContractReport
from sear.core.trajectory import Trajectory
from sear.metrics.stats import clopper_pearson_upper, mean_ci, poisson_rate_upper, wilson_interval

SAFETY_CATEGORIES = (ConstraintCategory.PHYSICAL, ConstraintCategory.TEMPORAL)


@dataclass(frozen=True)
class EpisodeOutcome:
    scenario_id: str
    success: bool
    completion_time: float | None
    duration: float
    any_violation: bool
    violated: tuple[str, ...]
    unknown: tuple[str, ...]
    peak_force: float
    force_violation: bool
    force_excess: float
    min_distance: float
    tool_dropped: bool
    max_slip: float
    safety_margin: float  # min normalized robustness over physical+temporal constraints
    fallback_used: bool
    filtered_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _force_limit(contract: Contract) -> tuple[str | None, float]:
    for c in contract.of_type(SignalBound):
        if type(c) is SignalBound and c.signal == "contact_force" and c.upper is not None:
            return c.name, c.upper
    return None, math.inf


def summarize_episode(traj: Trajectory, report: ContractReport, contract: Contract) -> EpisodeOutcome:
    force_name, force_limit = _force_limit(contract)
    peak = float(np.max(traj.get("contact_force"))) if traj.has("contact_force") and len(traj) else math.nan
    rt = traj.info.get("runtime", {})
    return EpisodeOutcome(
        scenario_id=str(traj.info.get("scenario_id", "")),
        success=bool(traj.info.get("success", False)),
        completion_time=traj.info.get("completion_time"),
        duration=traj.duration,
        any_violation=report.any_violation,
        violated=report.violated,
        unknown=report.unknown,
        peak_force=peak,
        force_violation=force_name in report.violated if force_name else False,
        force_excess=max(0.0, peak - force_limit) if math.isfinite(peak) else math.nan,
        min_distance=float(np.min(traj.get("min_distance"))) if traj.has("min_distance") and len(traj) else math.nan,
        tool_dropped=bool(traj.info.get("tool_dropped", False)),
        max_slip=float(np.max(traj.get("tool_slip"))) if traj.has("tool_slip") and len(traj) else math.nan,
        safety_margin=report.min_normalized_margin(SAFETY_CATEGORIES),
        fallback_used=bool(rt.get("fallback_used", False)),
        filtered_fraction=float(rt.get("filtered_fraction", 0.0)),
    )


@dataclass(frozen=True)
class SuiteMetrics:
    suite: str
    n: int
    success_rate: float
    success_ci: tuple[float, float]
    violation_rate: float
    violation_upper: float  # one-sided Clopper-Pearson bound
    force_violation_rate: float
    mean_force_excess: float
    tool_drop_rate: float
    min_distance: float
    mean_safety_margin: float
    exposure_hours: float
    violation_events_per_hour: float
    violation_events_per_hour_upper: float
    mean_completion_time: float
    fallback_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate(outcomes: Sequence[EpisodeOutcome], suite: str = "", confidence: float = 0.95) -> SuiteMetrics:
    n = len(outcomes)
    succ = sum(o.success for o in outcomes)
    viol = sum(o.any_violation for o in outcomes)
    exposure_h = sum(o.duration for o in outcomes) / 3600.0
    times = [o.completion_time for o in outcomes if o.completion_time is not None]
    excess = [o.force_excess for o in outcomes if o.force_violation]
    dists = [o.min_distance for o in outcomes if math.isfinite(o.min_distance)]
    return SuiteMetrics(
        suite=suite,
        n=n,
        success_rate=succ / n if n else math.nan,
        success_ci=wilson_interval(succ, n, confidence),
        violation_rate=viol / n if n else math.nan,
        violation_upper=clopper_pearson_upper(viol, n, confidence),
        force_violation_rate=sum(o.force_violation for o in outcomes) / n if n else math.nan,
        mean_force_excess=float(np.mean(excess)) if excess else 0.0,
        tool_drop_rate=sum(o.tool_dropped for o in outcomes) / n if n else math.nan,
        min_distance=min(dists) if dists else math.nan,
        mean_safety_margin=mean_ci([o.safety_margin for o in outcomes], confidence)[0],
        exposure_hours=exposure_h,
        violation_events_per_hour=viol / exposure_h if exposure_h > 0 else math.nan,
        violation_events_per_hour_upper=poisson_rate_upper(viol, exposure_h, confidence),
        mean_completion_time=float(np.mean(times)) if times else math.nan,
        fallback_rate=sum(o.fallback_used for o in outcomes) / n if n else math.nan,
    )
