"""WP2: evolution-aware candidate verification."""

from sear.verification.context import DEFAULT_PROTECTED_PATHS, VerificationContext
from sear.verification.evidence import AdmissionDecision, CheckResult, Evidence
from sear.verification.pipeline import VerificationPipeline, default_pipeline

__all__ = [
    "DEFAULT_PROTECTED_PATHS",
    "AdmissionDecision",
    "CheckResult",
    "Evidence",
    "VerificationContext",
    "VerificationPipeline",
    "default_pipeline",
]
