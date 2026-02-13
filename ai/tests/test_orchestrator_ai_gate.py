"""Unit test for orchestrator AI stage hard gate."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from api.models import VerificationStage, VerificationStatus
from api.services import verification_orchestrator as orch


class _FakeVerifications:
    def __init__(self) -> None:
        self.updates = []

    async def update_one(self, verification_id: int, payload: dict):
        self.updates.append((verification_id, payload))
        return 1


class _FakeSteps:
    def __init__(self) -> None:
        self.inserts = []
        self.updates = []

    async def insert_one(self, payload: dict):
        self.inserts.append(payload)
        return len(self.inserts)

    async def update_one(self, step_id: int, payload: dict):
        self.updates.append((step_id, payload))
        return 1


class _FakeDB:
    async def fetch_one(self, query: str, values=None):
        if "SELECT folder_name FROM document_types" in query:
            return {"folder_name": "identity"}
        if "SELECT user_id FROM verifications" in query:
            return {"user_id": 10}
        if "SELECT name FROM users" in query:
            return {"name": "Test User"}
        if "SELECT name FROM document_types" in query:
            return {"name": "National ID"}
        return None


class _FakeBus:
    def __init__(self) -> None:
        self.events = []

    async def broadcast(self, payload: dict):
        self.events.append(payload)


def _write_dummy_image(path: Path) -> None:
    img = np.full((120, 200, 3), 255, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, "Failed to write dummy image"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_orchestrator_stops_pipeline_when_ai_fails(monkeypatch, tmp_path):
    fake_verifications = _FakeVerifications()
    fake_steps = _FakeSteps()
    fake_bus = _FakeBus()
    fake_db = _FakeDB()
    calls = {"data_verification": 0, "blockchain": 0}

    async def _fake_persist_notification(**_kwargs):
        return 1

    def _document_image_quality(_front):
        return {"brightness_ok": True, "blur_ok": True}

    def _document_crop(_front, rectified_path):
        _write_dummy_image(Path(rectified_path))
        return {"ok": True}

    def _layout_gating_verify(_rectified):
        return {"layout_status": "PASS", "reason": "", "artifacts": {}}

    def _document_face_extraction(_rectified, out_path):
        _write_dummy_image(Path(out_path))
        return {"ok": True}

    def _face_matching(_face_src, _person, _rectified):
        return {
            "accepted": True,
            "similarity_percent": 88.0,
            "accept_threshold_percent": 80.0,
        }

    def _ocr_verify(_src):
        return {"text": "01310001042"}

    def _ml_verify(_src, _doc_folder):
        return {
            "final_decision": "FAILED",
            "authenticity_percent": 69.0,
            "pass_threshold_percent": 70.0,
            "failed_elements": ["stamp"],
            "element_results": {},
        }

    async def _data_verification(**_kwargs):
        calls["data_verification"] += 1
        return {"data_match": True}

    def _blockchain_verify(*_args, **_kwargs):
        calls["blockchain"] += 1
        return {"cid": "cid123"}

    monkeypatch.setattr(orch, "_db", fake_db)
    monkeypatch.setattr(orch, "get_verifications_collection", lambda: fake_verifications)
    monkeypatch.setattr(orch, "get_verification_steps_collection", lambda: fake_steps)
    monkeypatch.setattr(orch, "notification_bus", fake_bus)
    monkeypatch.setattr(orch, "persist_notification", _fake_persist_notification)

    monkeypatch.setattr(orch, "document_image_quality", _document_image_quality)
    monkeypatch.setattr(orch, "document_crop", _document_crop)
    monkeypatch.setattr(orch, "layout_gating_verify", _layout_gating_verify)
    monkeypatch.setattr(orch, "document_face_extraction", _document_face_extraction)
    monkeypatch.setattr(orch, "face_matching", _face_matching)
    monkeypatch.setattr(orch, "ocr_verify", _ocr_verify)
    monkeypatch.setattr(orch, "ml_verify", _ml_verify)
    monkeypatch.setattr(orch, "data_verification", _data_verification)
    monkeypatch.setattr(orch, "blockchain_verify", _blockchain_verify)

    front = tmp_path / "front.jpg"
    person = tmp_path / "person.jpg"
    _write_dummy_image(front)
    _write_dummy_image(person)

    orchestrator = orch.VerificationOrchestrator()
    payload = orch.VerificationInput(
        verification_id=123,
        document_front_path=front,
        document_back_path=None,
        person_image_path=person,
        document_type_id=1,
        owner_email="user@example.com",
    )

    await orchestrator.run(payload)

    assert calls["data_verification"] == 0
    assert calls["blockchain"] == 0

    executed_stages = [item["stage"] for item in fake_steps.inserts]
    assert VerificationStage.DATA_VERIFICATION.value not in executed_stages
    assert VerificationStage.BLOCKCHAIN.value not in executed_stages

    failed_updates = [
        update
        for _, update in fake_verifications.updates
        if update.get("status") == VerificationStatus.FAILED.value
    ]
    assert failed_updates, "Expected verification to fail at AI gate"
    last_failed = failed_updates[-1]
    assert last_failed.get("current_stage") == VerificationStage.AI_VERIFICATION.value
    assert "AI_VALIDATION_FAILED" in (last_failed.get("error_message") or "")

    raw_result_data = last_failed.get("result_data")
    assert isinstance(raw_result_data, str)
    parsed = json.loads(raw_result_data)
    assert parsed.get("failure_reason_code") == "AI_VALIDATION_FAILED"
