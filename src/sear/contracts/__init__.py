"""WP1: immutable, machine-checkable safety contracts."""

from sear.contracts.constraints import (
    AllowedTools,
    ApprovedModes,
    ConditionalBound,
    Constraint,
    ConstraintCategory,
    ConstraintResult,
    MaxDuration,
    OperatorAuthorization,
    ResponseWithin,
    SignalBound,
    WorkspaceBox,
    constraint_from_dict,
)
from sear.contracts.contract import Contract, ContractReport
from sear.contracts.oracle import SafetyOracle
from sear.contracts.signing import (
    ContractIntegrityError,
    SignedContract,
    load_key,
    read_signed,
    sign_contract,
    verify_signature,
    write_signed,
)

__all__ = [
    "AllowedTools",
    "ApprovedModes",
    "ConditionalBound",
    "Constraint",
    "ConstraintCategory",
    "ConstraintResult",
    "Contract",
    "ContractIntegrityError",
    "ContractReport",
    "MaxDuration",
    "OperatorAuthorization",
    "ResponseWithin",
    "SafetyOracle",
    "SignalBound",
    "SignedContract",
    "WorkspaceBox",
    "constraint_from_dict",
    "load_key",
    "read_signed",
    "sign_contract",
    "verify_signature",
    "write_signed",
]
