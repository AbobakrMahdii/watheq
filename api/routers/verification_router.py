from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, Form
import cv2
import numpy as np

from api.database import get_verifications_collection, get_verification_steps_collection
from api.models import VerificationPublic, VerificationStepPublic, VerificationStatus
from api.security import get_current_user
from api.services.verification_orchestrator import VerificationOrchestrator, VerificationInput

router = APIRouter(prefix="/api/v1/verifications", tags=["Verifications"])


def _save_upload(upload: UploadFile, folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / filename
    dest.write_bytes(upload.file.read())
    return dest


@router.post("/start", response_model=VerificationPublic)
async def start_verification(
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    document_type_id: int = Form(...),
    document_image_front: UploadFile = File(...),
    person_image: UploadFile = File(...),
    document_image_back: Optional[UploadFile] = File(None),
    liveness_data: Optional[str] = Form(None),
):
    user_id = int(current_user.get("sub")) if str(current_user.get("sub")).isdigit() else None
    if user_id is None:
        raise HTTPException(status_code=400, detail="Invalid user id")

    verifications = get_verifications_collection()
    now = datetime.now(timezone.utc)
    verification_id = await verifications.insert_one(
        {
            "user_id": user_id,
            "document_type_id": document_type_id,
            "status": VerificationStatus.PENDING.value,
            "current_stage": None,
            "error_message": None,
            "start_time": now,
            "end_time": None,
            "result_data": None,
        }
    )

    storage_dir = Path(__file__).resolve().parents[2] / "storage" / "verifications" / str(verification_id)
    debug_dir = storage_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    front_bytes = document_image_front.file.read()
    front_path = storage_dir / "document_front"
    front_path.write_bytes(front_bytes)
    image = cv2.imdecode(np.frombuffer(front_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid document image")
    print(f"[SERVER] received shape: {image.shape[1]}x{image.shape[0]}")
    print(f"[SERVER] received bytes: {len(front_bytes)}")
    cv2.imwrite(str(debug_dir / "input.jpg"), image)
    cv2.imwrite(str(debug_dir / "server_received.jpg"), image)
    person_path = _save_upload(person_image, storage_dir, "person_image")
    back_path = None
    if document_image_back is not None:
        back_path = _save_upload(document_image_back, storage_dir, "document_back")

    parsed_liveness = None
    if liveness_data:
        try:
            import json as _json

            parsed_liveness = _json.loads(liveness_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid liveness data")

    orchestrator = VerificationOrchestrator()
    background_tasks.add_task(
        orchestrator.run,
        VerificationInput(
            verification_id=verification_id,
            document_front_path=front_path,
            document_back_path=back_path,
            person_image_path=person_path,
            document_type_id=document_type_id,
            owner_email=current_user.get("email") or "",
            liveness_data=parsed_liveness,
        ),
    )

    item = await verifications.find_one(verification_id)
    return item


@router.get("/{verification_id}", response_model=VerificationPublic)
async def get_verification(verification_id: int, current_user=Depends(get_current_user)):
    verifications = get_verifications_collection()
    item = await verifications.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")
    if item.get("user_id") != int(current_user.get("sub")):
        raise HTTPException(status_code=403, detail="Forbidden")
    return item


@router.get("/{verification_id}/steps", response_model=list[VerificationStepPublic])
async def get_verification_steps(verification_id: int, current_user=Depends(get_current_user)):
    verifications = get_verifications_collection()
    item = await verifications.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")
    if item.get("user_id") != int(current_user.get("sub")):
        raise HTTPException(status_code=403, detail="Forbidden")

    steps = get_verification_steps_collection()
    return await steps.list_by_verification(verification_id)
