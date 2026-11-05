"""Tier 3: grid-based barrier-certificate check (where model assumptions permit).

Over a grid covering the certified operating domain and restricted to the safe
set {h_i >= 0}, evaluate the CBF condition residual

    r(x) = min_i  (a_i(x) . u - b_i(x)) / ||a_i(x)||

for either the *raw* candidate action (``mode="raw"``) or the best admissible
action of the runtime filter (``mode="filtered"``, i.e. QP feasibility).  If
min r exceeds a Lipschitz bound times the grid half-spacing, the condition
holds on the whole domain, which certifies forward invariance of the safe set
*under the stated assumptions*.  Otherwise the result is UNKNOWN -- the CBF
condition is only sufficient, so failing it is not evidence of unsafety.
"""

from __future__ import annotations

import itertools

import numpy as np

from sear.core.types import GuaranteeLevel, Verdict
from sear.core.update import CandidateArtifact
from sear.policies.base import build_policy
from sear.runtime.cbf import NominalModel, barriers_from_contract
from sear.sim.env import Observation
from sear.sim.scenario import DynamicsParams, Scenario
from sear.verification.checks.base import Check
from sear.verification.context import VerificationContext
from sear.verification.evidence import CheckResult


class BarrierCertificateCheck(Check):
    name = "barrier_certificate"
    level = GuaranteeLevel.DETERMINISTIC
    tier = 3

    def __init__(self, mode: str = "filtered", grid: int = 7, u_samples: int = 15, u_max: float = 60.0,
                 model: NominalModel | None = None, mandatory: bool = False) -> None:
        super().__init__(mandatory)
        if mode not in ("raw", "filtered"):
            raise ValueError("mode must be 'raw' or 'filtered'")
        self.mode = mode
        self.grid = grid
        self.u_max = u_max
        self.model = model or NominalModel()
        box = u_max / np.sqrt(2.0)
        g = np.linspace(-box, box, u_samples)
        self._u_grid = np.array(list(itertools.product(g, g)))

    def evaluate(self, artifact: CandidateArtifact, ctx: VerificationContext) -> CheckResult:
        dom, model = ctx.certified_domain, self.model
        ref = Scenario("certificate", 0)
        barriers = barriers_from_contract(ctx.oracle.contract, model, np.asarray(ref.obstacle_center),
                                          ref.obstacle_radius)
        if not barriers:
            return self.result(Verdict.UNKNOWN, "contract yields no barrier functions")
        surface_damping = DynamicsParams().surface_damping
        policy = build_policy(artifact.candidate.policy) if self.mode == "raw" else None

        axes = [np.linspace(dom.position_low[0], dom.position_high[0], self.grid),
                np.linspace(dom.position_low[1], dom.position_high[1], self.grid),
                np.linspace(-dom.max_speed, dom.max_speed, self.grid),
                np.linspace(-dom.max_speed, dom.max_speed, self.grid)]
        values = np.full((self.grid,) * 4, np.nan)
        for idx in np.ndindex(values.shape):
            p = np.array([axes[0][idx[0]], axes[1][idx[1]]])
            v = np.array([axes[2][idx[2]], axes[3][idx[3]]])
            if any(b.h(p, v) < 0 for b in barriers):
                continue
            pen = -p[1]
            fn = max(0.0, model.certified_stiffness * pen - surface_damping * v[1]) if pen > 0 else 0.0
            f_ext = np.array([0.0, fn])
            rows = [b.constraint(p, v, f_ext, model) for b in barriers]
            A = np.array([a for a, _ in rows])
            rhs = np.array([b for _, b in rows])
            norms = np.maximum(np.linalg.norm(A, axis=1), 1e-12)
            A, rhs = A / norms[:, None], rhs / norms
            if policy is not None:
                obs = Observation(0.0, p, v, fn, 0.0, ref.target_x, False, True,
                                  np.asarray(ref.obstacle_center), ref.obstacle_radius)
                u = np.asarray(policy.act(obs), float)
                n = np.linalg.norm(u)
                u = u * (self.u_max / n) if n > self.u_max else u
                values[idx] = float(np.min(A @ u - rhs))
            else:
                values[idx] = float(np.max(np.min(A @ self._u_grid.T - rhs[:, None], axis=0)))

        valid = np.isfinite(values)
        if not valid.any():
            return self.result(Verdict.UNKNOWN, "no grid point inside the safe set")
        min_val = float(np.nanmin(values))
        bound = 0.0
        for axis, ax in enumerate(axes):
            spacing = float(ax[1] - ax[0])
            diffs = np.abs(np.diff(values, axis=axis))
            lip = float(np.nanmax(diffs)) / spacing if np.isfinite(diffs).any() else 0.0
            bound += lip * spacing / 2.0
        metrics = {"mode": self.mode, "points": int(valid.sum()), "min_residual": min_val,
                   "lipschitz_bound": bound, "fraction_violating": float(np.mean(values[valid] < 0))}
        assumptions = (
            f"nominal model m={model.mass}, c={model.damping}; surface stiffness <= {model.certified_stiffness}",
            "Lipschitz constant estimated from grid differences",
            "certified domain: " + repr(dom),
            "actuator saturation |u_i| <= u_max/sqrt(2); latency and estimation error covered by margins only",
        )
        if min_val > 0 and min_val >= bound:
            return self.result(Verdict.PASS, f"CBF condition holds on domain ({self.mode})", metrics=metrics,
                               assumptions=assumptions)
        return self.result(Verdict.UNKNOWN, f"certificate inconclusive ({self.mode}): min residual {min_val:.3g}, "
                           f"Lipschitz margin {bound:.3g}", metrics=metrics, assumptions=assumptions)
