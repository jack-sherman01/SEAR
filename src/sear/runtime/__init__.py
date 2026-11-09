"""WP3: runtime assurance and safe fallback."""

from sear.runtime.assurance import RTAMode, RuntimeAssurance, RuntimeAssuranceConfig, domain_from_contract
from sear.runtime.cbf import (
    Barrier,
    HalfPlaneBarrier,
    NominalModel,
    SpeedBarrier,
    SphereAvoidBarrier,
    barriers_from_contract,
)
from sear.runtime.fallback import SafeStopController
from sear.runtime.monitor import CertifiedDomain, MonitorStatus, RuntimeMonitor
from sear.runtime.qp import QPResult, solve_min_norm_qp

__all__ = [
    "Barrier",
    "CertifiedDomain",
    "HalfPlaneBarrier",
    "MonitorStatus",
    "NominalModel",
    "QPResult",
    "RTAMode",
    "RuntimeAssurance",
    "RuntimeAssuranceConfig",
    "RuntimeMonitor",
    "SafeStopController",
    "SpeedBarrier",
    "SphereAvoidBarrier",
    "barriers_from_contract",
    "domain_from_contract",
    "solve_min_norm_qp",
]
