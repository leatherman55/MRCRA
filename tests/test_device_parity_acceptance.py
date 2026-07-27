from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import mrrn.device_parity_acceptance as parity


def test_unavailable_device_is_explicitly_untested_not_passed(monkeypatch):
    monkeypatch.setattr(parity, "_available", lambda _: False)
    result = parity._run_device("cuda")
    assert result.status == "untested_unavailable"
    assert result.passed is None
    assert result.criteria == ()


def test_report_fails_if_any_available_backend_fails(monkeypatch):
    passed = parity.DeviceParityResult(
        "cpu", "tested", "cpu", (), {}, True
    )
    failed = parity.DeviceParityResult(
        "mps", "failed", "mps", (), {}, False, "synthetic"
    )
    unavailable = parity.DeviceParityResult(
        "cuda", "untested_unavailable", "unavailable", (), {}, None
    )
    results = iter((passed, failed, unavailable))
    monkeypatch.setattr(parity, "_run_device", lambda _: next(results))
    report = parity.run_device_parity_acceptance()
    assert not report.passed
    assert report.results[2].passed is None


def test_cpu_integrated_device_parity_acceptance_passes():
    result = parity._run_device("cpu")
    assert result.status == "tested", result.error
    assert result.passed, result
    assert len(result.criteria) >= 12
    assert all(item.passed for item in result.criteria)
    assert result.telemetry["cstm_substrate_vjps"] == 1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not locally available",
)
def test_mps_integrated_device_parity_acceptance_passes():
    result = parity._run_device("mps")
    assert result.status == "tested", result.error
    assert result.passed, result
