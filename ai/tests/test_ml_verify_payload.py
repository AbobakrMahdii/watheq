"""Unit tests for ml_verify payload mapping."""

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from api.services.verification_steps_service import ml_verify


@pytest.mark.unit
def test_ml_verify_uses_weighted_overall_confidence(monkeypatch):
    fake_stdout = json.dumps(
        {
            "decision": "FAILED",
            "overall_confidence": 0.6999,
            "pass_threshold": 0.70,
            "failed_elements": ["stamp"],
            "elements": {"stamp": {"status": "FAILED", "score": 0.4}},
            "element_results": {"stamp": {"status": "FAILED", "score": 0.4}},
        }
    )

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = ml_verify(Path("dummy.jpg"), "identity")

    assert result["final_decision"] == "FAILED"
    assert result["authenticity_percent"] == pytest.approx(69.99, abs=1e-6)
    assert result["pass_threshold_percent"] == pytest.approx(70.0, abs=1e-6)
    assert result["failed_elements"] == ["stamp"]
    assert "stamp" in result["element_results"]


@pytest.mark.unit
def test_ml_verify_falls_back_to_legacy_average_when_overall_missing(monkeypatch):
    fake_stdout = json.dumps(
        {
            "decision": "PASSED",
            "pass_threshold": 0.70,
            "failed_elements": [],
            "element_results": {
                "logo": {"status": "PASSED", "score": 0.6},
                "stamp": {"status": "PASSED", "score": 0.8},
            },
        }
    )

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=fake_stdout, stderr="")

    monkeypatch.setattr("subprocess.run", _fake_run)

    result = ml_verify(Path("dummy.jpg"), "identity")

    assert result["final_decision"] == "PASSED"
    assert result["authenticity_percent"] == pytest.approx(70.0, abs=1e-6)
    assert result["pass_threshold_percent"] == pytest.approx(70.0, abs=1e-6)
