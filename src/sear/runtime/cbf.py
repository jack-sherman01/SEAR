"""Control barrier functions for the end-effector model.

Nominal control-affine model (double integrator with external force):

    m a = u + f_ext - c v,      x = (p, v)

Position constraints have relative degree 2 and use exponential high-order
CBFs (HOCBF):  h_ddot + (a1 + a2) h_dot + a1 a2 h >= 0.  Speed limits have
relative degree 1:  h_dot + a h >= 0.  Every barrier yields one linear
constraint ``a . u >= b`` for the QP filter (proposal WP3).

``margin`` shrinks each safe set to absorb state-estimation error, latency and
model mismatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from sear.contracts.constraints import SignalBound, WorkspaceBox
from sear.contracts.contract import Contract


@dataclass(frozen=True)
class NominalModel:
    mass: float = 2.5
    damping: float = 4.0
    certified_stiffness: float = 6000.0  # stiffest surface the force barrier is certified for [N/m]

    def drift(self, v: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        """Acceleration with zero input."""
        return (f_ext - self.damping * v) / self.mass


class Barrier(ABC):
    name: str

    @abstractmethod
    def h(self, p: np.ndarray, v: np.ndarray) -> float:
        """Barrier value of the *outer* safe set (>= 0 safe)."""

    @abstractmethod
    def constraint(self, p: np.ndarray, v: np.ndarray, f_ext: np.ndarray,
                   model: NominalModel) -> tuple[np.ndarray, float]:
        """Return (a, b) such that the CBF condition is ``a . u >= b``."""


@dataclass
class HalfPlaneBarrier(Barrier):
    """n . p >= offset (HOCBF)."""

    name: str
    normal: np.ndarray
    offset: float
    margin: float = 0.0
    alpha1: float = 15.0
    alpha2: float = 15.0

    def h(self, p, v):
        return float(self.normal @ p - self.offset - self.margin)

    def constraint(self, p, v, f_ext, model):
        n = self.normal
        h, hd = self.h(p, v), float(n @ v)
        rhs = -float(n @ model.drift(v, f_ext)) - (self.alpha1 + self.alpha2) * hd - self.alpha1 * self.alpha2 * h
        return n / model.mass, rhs


@dataclass
class SphereAvoidBarrier(Barrier):
    """||p - c|| >= radius (HOCBF); the centre may be updated from perception."""

    name: str
    center: np.ndarray
    radius: float
    margin: float = 0.0
    alpha1: float = 10.0
    alpha2: float = 10.0

    def h(self, p, v):
        return float(np.linalg.norm(p - self.center) - self.radius - self.margin)

    def constraint(self, p, v, f_ext, model):
        r = p - self.center
        dist = max(float(np.linalg.norm(r)), 1e-9)
        n = r / dist
        hd = float(n @ v)
        curvature = (float(v @ v) - hd * hd) / dist  # d/dt of n contributes (|v|^2 - (n.v)^2)/|r|
        h = self.h(p, v)
        rhs = (-float(n @ model.drift(v, f_ext)) - curvature
               - (self.alpha1 + self.alpha2) * hd - self.alpha1 * self.alpha2 * h)
        return n / model.mass, rhs


@dataclass
class SpeedBarrier(Barrier):
    """|v| <= v_max (relative degree 1)."""

    name: str
    v_max: float
    alpha: float = 20.0

    def h(self, p, v):
        return float(self.v_max**2 - v @ v)

    def constraint(self, p, v, f_ext, model):
        # h_dot = -2 v . a = -2 v . (u + f_ext - c v)/m  >= -alpha h
        a = -2.0 * v / model.mass
        rhs = -self.alpha * self.h(p, v) + 2.0 * float(v @ model.drift(v, f_ext))
        return a, rhs


def barriers_from_contract(contract: Contract, model: NominalModel, obstacle_center: np.ndarray,
                           obstacle_radius: float, *, position_margin: float = 0.001,
                           force_fraction: float = 0.85, speed_fraction: float = 0.9) -> list[Barrier]:
    """Translate the contract's physical constraints into barriers.

    * ``contact_force`` upper bound F  ->  penetration bound p_y >= -F'/k_cert
    * ``ee_speed`` upper bound         ->  speed barrier
    * ``min_distance`` lower bound d   ->  sphere avoidance with radius r + d
    * workspace box                    ->  four half-planes
    """
    out: list[Barrier] = []
    for c in contract.of_type(SignalBound):
        if type(c) is not SignalBound:
            continue
        if c.signal == "contact_force" and c.upper is not None:
            pen_max = force_fraction * c.upper / model.certified_stiffness
            out.append(HalfPlaneBarrier(c.name, np.array([0.0, 1.0]), -pen_max, position_margin))
        elif c.signal == "ee_speed" and c.upper is not None:
            out.append(SpeedBarrier(c.name, speed_fraction * c.upper))
        elif c.signal == "min_distance" and c.lower is not None:
            out.append(SphereAvoidBarrier(c.name, np.asarray(obstacle_center, float),
                                          obstacle_radius + c.lower, position_margin))
    for box in contract.of_type(WorkspaceBox):
        if box.signal != "ee_pos":
            continue
        lo, hi = np.asarray(box.low, float), np.asarray(box.high, float)
        for i in range(len(lo)):
            e = np.eye(len(lo))[i]
            out.append(HalfPlaneBarrier(f"{box.name}[{i}]_low", e, lo[i], position_margin))
            out.append(HalfPlaneBarrier(f"{box.name}[{i}]_high", -e, -hi[i], position_margin))
    return out
