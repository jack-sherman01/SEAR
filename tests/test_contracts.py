import math
from pathlib import Path

import numpy as np
import pytest

from sear.contracts import (
    Contract,
    ContractIntegrityError,
    ResponseWithin,
    SafetyOracle,
    SignalBound,
    read_signed,
    sign_contract,
    write_signed,
)
from sear.core import TrajectoryRecorder, Verdict

from tests.conftest import CONTRACT_YAML
KEY = b"test-key"


def _traj(force, human=None, stopped=None, dt=0.1):
    rec = TrajectoryRecorder(dt)
    for i, f in enumerate(force):
        row = {"t": i * dt, "contact_force": f}
        if human is not None:
            row.update(human_in_zone=human[i], stopped=stopped[i])
        rec.record(**row)
    return rec.finalize()


def test_signal_bound_margins_and_unknown():
    c = SignalBound(name="f", signal="contact_force", upper=40.0)
    ok = c.evaluate(_traj([10, 20, 30]))
    assert ok.verdict is Verdict.PASS and ok.margin == pytest.approx(10.0)
    bad = c.evaluate(_traj([10, 50, 45]))
    assert bad.verdict is Verdict.FAIL and bad.violations == 2 and bad.severity == pytest.approx(10.0)
    assert bad.first_violation_t == pytest.approx(0.1)
    missing = SignalBound(name="v", signal="ee_speed", upper=1.0).evaluate(_traj([1.0]))
    assert missing.verdict is Verdict.UNKNOWN and math.isnan(missing.margin)
    nan = c.evaluate(_traj([10, float("nan")]))
    assert nan.verdict is Verdict.UNKNOWN


def test_response_within():
    c = ResponseWithin(name="stop", trigger="human_in_zone", response="stopped", max_delay=0.2)
    human = [0, 1, 1, 1, 1, 0]
    good = c.evaluate(_traj([0] * 6, human, [0, 0, 1, 1, 1, 0]))
    assert good.verdict is Verdict.PASS
    late = c.evaluate(_traj([0] * 6, human, [0, 0, 0, 0, 1, 0]))
    assert late.verdict is Verdict.FAIL and late.violations == 1


def test_contract_roundtrip_and_digest():
    contract = Contract.from_yaml(CONTRACT_YAML)
    again = Contract.from_dict(contract.to_dict())
    assert again.digest == contract.digest
    assert len(contract.constraints) == 11


def test_signing_detects_tampering(tmp_path):
    src = tmp_path / "c.yaml"
    src.write_text(CONTRACT_YAML.read_text())
    write_signed(src, KEY)
    signed = read_signed(src, KEY)
    with pytest.raises(ContractIntegrityError):
        read_signed(src, b"wrong-key")
    src.chmod(0o644)
    src.write_text(src.read_text().replace("upper: 40.0", "upper: 400.0"))
    with pytest.raises(ContractIntegrityError):
        read_signed(src, KEY)
    oracle = SafetyOracle(signed, KEY)
    report = oracle.evaluate(_traj([10, 50]))
    assert "contact_force_limit" in report.violated


def test_oracle_rejects_unsigned():
    contract = Contract.from_yaml(CONTRACT_YAML)
    signed = sign_contract(contract, b"other")
    with pytest.raises(ContractIntegrityError):
        SafetyOracle(signed, KEY)


def test_report_min_margin():
    contract = Contract(name="c", version="1", constraints=(
        SignalBound(name="f", signal="contact_force", upper=40.0),))
    report = contract.evaluate(_traj(np.array([0.0, 20.0])))
    assert report.min_normalized_margin() == pytest.approx(0.5)
