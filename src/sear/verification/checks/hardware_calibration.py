"""Tier 3.5: sim-to-real calibration check.

Before admitting a candidate update to the real robot, run a small number of
hardware trials on a guarded testbed fixture and verify that the observed
force/position profiles are within the tolerance band predicted by the
simulator.  If the residual exceeds the threshold, return UNKNOWN (which
causes the pipeline to RESTRICT rather than ADMIT, keeping the update in
low-energy / guarded mode until residual is bounded).

When no hardware runner is configured (simulation-only mode), the check
returns PASS with a note that hardware validation was skipped.  This lets
the same pipeline object run in both simulation and hardware contexts without
modification.

Tier numbering: 3.5 is encoded as integer 3 so it sorts between the existing
Tier-3 (BarrierCertificateCheck) and Tier-4 (StatisticalSafetyCheck).  The
subclass overrides this with the class attribute ``tier = 3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from sear.core.types import GuaranteeLevel, Verdict
from sear.core.update import CandidateArtifact
from sear.sim.scenario import Scenario
from sear.verification.checks.base import Check
from sear.verification.context import VerificationContext
from sear.verification.evidence import CheckResult


@dataclass
class HardwareTrial:
    """Outcome of one hardware trial: observed and predicted trajectories."""
    scenario_id: str
    observed_forces: np.ndarray   # shape (T,) — wrist F/T magnitude
    predicted_forces: np.ndarray  # shape (T,) — simulator prediction
    observed_positions: np.ndarray   # shape (T, D)
    predicted_positions: np.ndarray  # shape (T, D)
    any_violation: bool = False
    notes: str = ""


@dataclass
class HardwareRunnerConfig:
    n_trials: int = 5
    tolerance_force: float = 0.25   # max allowed RMS residual as fraction of max predicted force
    tolerance_position: float = 0.02  # metres RMS
    max_violation_fraction: float = 0.0  # any hardware violation → FAIL immediately


HardwareRunner = Callable[[CandidateArtifact, Sequence[Scenario]], list[HardwareTrial]]


class HardwareCalibrationCheck(Check):
    """Sim-to-real calibration check (Tier 3.5).

    Runs the candidate policy on the physical robot for ``config.n_trials``
    episodes and compares observed trajectories to simulator predictions.

    Verdict:
        PASS    — all trials within tolerance and no violations
        FAIL    — any trial produced a hardware safety violation
        UNKNOWN — residual exceeds tolerance or hardware runner raised an error
                  (causes RESTRICT, not deployment-block)

    When ``hardware_runner`` is None the check is a no-op (returns PASS with
    ``GuaranteeLevel.EMPIRICAL`` and skipped=True in metrics).
    """

    name = "hardware_calibration"
    level = GuaranteeLevel.EMPIRICAL
    tier = 3   # sorts between certificate (3) and statistical (4); tie-broken by insertion order

    def __init__(
        self,
        hardware_runner: HardwareRunner | None = None,
        config: HardwareRunnerConfig | None = None,
        mandatory: bool = True,
    ) -> None:
        super().__init__(mandatory)
        self.hardware_runner = hardware_runner
        self.config = config or HardwareRunnerConfig()

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        if self.hardware_runner is None:
            return self.result(
                Verdict.PASS,
                "hardware runner not configured — hardware validation skipped (simulation-only mode)",
                metrics={"skipped": True, "n_trials": 0},
                assumptions=("hardware validation skipped; sim-to-real gap unquantified",),
            )

        # Run hardware trials on the in-distribution benchmark subset
        try:
            scenarios = list(ctx.benchmark["in_distribution"])[: self.config.n_trials]
        except (KeyError, TypeError):
            scenarios = []
        if not scenarios:
            return self.result(Verdict.UNKNOWN, "no in-distribution scenarios available for hardware trials",
                               metrics={"skipped": True})

        try:
            trials = self.hardware_runner(artifact, scenarios)
        except Exception as exc:
            return self.result(Verdict.UNKNOWN, f"hardware runner raised {type(exc).__name__}: {exc}",
                               metrics={"error": str(exc)})

        return self._score(trials)

    def _score(self, trials: list[HardwareTrial]) -> CheckResult:
        cfg = self.config
        if not trials:
            return self.result(Verdict.UNKNOWN, "hardware runner returned no trials")

        violation_trials = [t for t in trials if t.any_violation]
        if len(violation_trials) / len(trials) > cfg.max_violation_fraction:
            return self.result(
                Verdict.FAIL,
                f"{len(violation_trials)}/{len(trials)} hardware trials produced safety violations",
                metrics={"n_trials": len(trials), "n_violations": len(violation_trials)},
            )

        force_residuals, pos_residuals = [], []
        for t in trials:
            if len(t.observed_forces) and len(t.predicted_forces):
                n = min(len(t.observed_forces), len(t.predicted_forces))
                pred_max = float(np.max(np.abs(t.predicted_forces[:n]))) or 1.0
                rms = float(np.sqrt(np.mean((t.observed_forces[:n] - t.predicted_forces[:n]) ** 2)))
                force_residuals.append(rms / pred_max)
            if len(t.observed_positions) and len(t.predicted_positions):
                n = min(len(t.observed_positions), len(t.predicted_positions))
                rms = float(np.sqrt(np.mean(
                    np.sum((t.observed_positions[:n] - t.predicted_positions[:n]) ** 2, axis=-1)
                )))
                pos_residuals.append(rms)

        mean_force_res = float(np.mean(force_residuals)) if force_residuals else float("nan")
        mean_pos_res = float(np.mean(pos_residuals)) if pos_residuals else float("nan")
        metrics = {
            "n_trials": len(trials),
            "n_violations": len(violation_trials),
            "mean_force_residual_fraction": mean_force_res,
            "mean_position_residual_m": mean_pos_res,
            "force_tolerance": cfg.tolerance_force,
            "position_tolerance_m": cfg.tolerance_position,
        }

        problems = []
        if np.isfinite(mean_force_res) and mean_force_res > cfg.tolerance_force:
            problems.append(
                f"force residual {mean_force_res:.3f} > tolerance {cfg.tolerance_force}"
            )
        if np.isfinite(mean_pos_res) and mean_pos_res > cfg.tolerance_position:
            problems.append(
                f"position residual {mean_pos_res:.4f} m > tolerance {cfg.tolerance_position} m"
            )

        if problems:
            return self.result(
                Verdict.UNKNOWN,
                "sim-to-real residual exceeds tolerance: " + "; ".join(problems),
                metrics=metrics,
                assumptions=("hardware trials on guarded testbed fixture",),
            )

        return self.result(
            Verdict.PASS,
            f"{len(trials)} hardware trials within tolerance "
            f"(force {mean_force_res:.3f}, pos {mean_pos_res:.4f} m)",
            metrics=metrics,
            assumptions=(
                "hardware trials on guarded testbed fixture",
                "force/position profiles representative of deployment environment",
            ),
        )
