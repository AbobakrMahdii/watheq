"""Decision tests for ai.verify_document binary 70% gate."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ai import verify_document as vd


class _StaticClassifier:
    def __init__(self, score: float) -> None:
        self._score = score

    def predict(self, _img: np.ndarray) -> float:
        return self._score


def _write_dummy_image(path: Path) -> None:
    img = np.full((200, 320, 3), 255, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, "Failed to write dummy image"


@pytest.mark.unit
def test_binary_decision_passes_at_70_percent(monkeypatch, tmp_path):
    img_path = tmp_path / "doc.jpg"
    _write_dummy_image(img_path)

    layout = {
        "elements": {
            "seal": {
                "class_name": "stamp",
                "roi": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
                "tolerance": 0.1,
                "weight": 1.0,
                "critical": False,
                "type": "visual",
            }
        },
        "text_regions": {},
        "thresholds": {"pass_score": 0.70, "suspicious_score": 0.70},
    }
    monkeypatch.setattr(vd, "_load_layout_config", lambda _doc_type: layout)
    monkeypatch.setattr(
        vd,
        "_load_classifier",
        lambda _doc_type, _class_name: _StaticClassifier(0.5714285714),
    )

    result = vd.verify(str(img_path), "identity")

    assert result["decision"] == "PASSED"
    assert result["overall_confidence"] == pytest.approx(0.70, abs=1e-4)
    assert result["decision"] in {"PASSED", "FAILED"}


@pytest.mark.unit
def test_binary_decision_fails_below_70_percent(monkeypatch, tmp_path):
    img_path = tmp_path / "doc.jpg"
    _write_dummy_image(img_path)

    layout = {
        "elements": {
            "seal": {
                "class_name": "stamp",
                "roi": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
                "tolerance": 0.1,
                "weight": 1.0,
                "critical": False,
                "type": "visual",
            }
        },
        "text_regions": {},
        "thresholds": {"pass_score": 0.70, "suspicious_score": 0.70},
    }
    monkeypatch.setattr(vd, "_load_layout_config", lambda _doc_type: layout)
    monkeypatch.setattr(
        vd,
        "_load_classifier",
        lambda _doc_type, _class_name: _StaticClassifier(0.5712857143),
    )

    result = vd.verify(str(img_path), "identity")

    assert result["decision"] == "FAILED"
    assert result["overall_confidence"] == pytest.approx(0.6999, abs=1e-4)
    assert result["decision"] in {"PASSED", "FAILED"}


@pytest.mark.unit
def test_critical_element_failure_overrides_high_overall(monkeypatch, tmp_path):
    img_path = tmp_path / "doc.jpg"
    _write_dummy_image(img_path)

    layout = {
        "elements": {
            "critical_stamp": {
                "class_name": "stamp",
                "roi": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
                "tolerance": 0.1,
                "weight": 1.0,
                "critical": True,
                "type": "visual",
            },
            "support_logo": {
                "class_name": "logo",
                "roi": {"x": 0.5, "y": 0.1, "w": 0.2, "h": 0.2},
                "tolerance": 0.1,
                "weight": 5.0,
                "critical": False,
                "type": "visual",
            },
        },
        "text_regions": {},
        "thresholds": {"pass_score": 0.70, "suspicious_score": 0.70},
    }
    score_map = {"stamp": 0.0, "logo": 1.0}

    def _load_classifier(_doc_type: str, class_name: str):
        return _StaticClassifier(score_map[class_name])

    monkeypatch.setattr(vd, "_load_layout_config", lambda _doc_type: layout)
    monkeypatch.setattr(vd, "_load_classifier", _load_classifier)

    result = vd.verify(str(img_path), "identity")

    assert result["overall_confidence"] >= 0.70
    assert result["decision"] == "FAILED"
    assert any("Critical elements failed" in msg for msg in result["anomalies"])
