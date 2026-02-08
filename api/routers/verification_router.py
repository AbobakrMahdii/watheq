from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    Form,
    Query,
)
import cv2
import io
import numpy as np
from PIL import Image, ImageOps

from api.database import get_verifications_collection, get_verification_steps_collection
from api.models import (
    VerificationListWithStatsResponse,
    VerificationPublic,
    VerificationStepPublic,
    VerificationStatus,
)
from api.security import get_current_user
from api.services.verification_orchestrator import (
    VerificationOrchestrator,
    VerificationInput,
)

router = APIRouter(prefix="/api/v1/verifications", tags=["Verifications"])


def _normalize_exif_orientation(raw_bytes: bytes) -> bytes:
    """Apply EXIF rotation to raw JPEG bytes and return re-encoded bytes.

    Phone cameras often store the image in landscape pixel orientation
    with an EXIF tag indicating the display rotation.  OpenCV's
    ``cv2.imdecode`` ignores EXIF, so the image appears rotated/flipped.
    This function bakes the rotation into the actual pixels.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return raw_bytes


def _save_upload(upload: UploadFile, folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    raw = upload.file.read()
    normalized = _normalize_exif_orientation(raw)
    dest = folder / filename
    dest.write_bytes(normalized)
    return dest


@router.post("/start", response_model=VerificationPublic)
async def start_verification(
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user),
    document_type_id: int = Form(...),
    document_image_front: UploadFile = File(...),
    person_image: UploadFile = File(...),
    document_image_back: Optional[UploadFile] = File(None),
    liveness_data: Optional[str] = Form(None),  # kept for client backward compat
):
    user_id = (
        int(current_user.get("sub")) if str(current_user.get("sub")).isdigit() else None
    )
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

    storage_dir = (
        Path(__file__).resolve().parents[2]
        / "storage"
        / "verifications"
        / str(verification_id)
    )
    debug_dir = storage_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    front_bytes_raw = document_image_front.file.read()
    front_bytes = _normalize_exif_orientation(front_bytes_raw)
    front_path = storage_dir / "document_front.jpg"
    front_path.write_bytes(front_bytes)
    image = cv2.imdecode(np.frombuffer(front_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid document image")
    print(f"[SERVER] received shape: {image.shape[1]}x{image.shape[0]}")
    print(f"[SERVER] received bytes (raw={len(front_bytes_raw)}, normalized={len(front_bytes)})")
    cv2.imwrite(str(debug_dir / "input.jpg"), image)
    cv2.imwrite(str(debug_dir / "server_received.jpg"), image)
    person_path = _save_upload(person_image, storage_dir, "person_image.jpg")
    back_path = None
    if document_image_back is not None:
        back_path = _save_upload(document_image_back, storage_dir, "document_back")

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
        ),
    )

    item = await verifications.find_one(verification_id)
    return item


@router.get("/my", response_model=VerificationListWithStatsResponse)
async def list_my_verifications(
    current_user=Depends(get_current_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    user_id = (
        int(current_user.get("sub")) if str(current_user.get("sub")).isdigit() else None
    )
    if user_id is None:
        raise HTTPException(status_code=400, detail="Invalid user id")

    verifications = get_verifications_collection()
    offset = (page - 1) * page_size
    items = await verifications.list_by_user(user_id, limit=page_size, offset=offset)
    total = await verifications.count(user_id=user_id)
    status_counts = await verifications.count_by_status(user_id)

    normalized_status_counts = {
        "SUCCESS": status_counts.get(VerificationStatus.SUCCESS.value, 0),
        "FAILED": status_counts.get(VerificationStatus.FAILED.value, 0),
        "RUNNING": status_counts.get(VerificationStatus.RUNNING.value, 0),
        "PENDING": status_counts.get(VerificationStatus.PENDING.value, 0),
    }

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items,
        "status_counts": normalized_status_counts,
    }


@router.get("/{verification_id}", response_model=VerificationPublic)
async def get_verification(
    verification_id: int, current_user=Depends(get_current_user)
):
    verifications = get_verifications_collection()
    item = await verifications.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")
    if item.get("user_id") != int(current_user.get("sub")):
        raise HTTPException(status_code=403, detail="Forbidden")
    return item


@router.get("/{verification_id}/steps", response_model=list[VerificationStepPublic])
async def get_verification_steps(
    verification_id: int, current_user=Depends(get_current_user)
):
    verifications = get_verifications_collection()
    item = await verifications.find_one(verification_id)
    if not item:
        raise HTTPException(status_code=404, detail="Verification not found")
    if item.get("user_id") != int(current_user.get("sub")):
        raise HTTPException(status_code=403, detail="Forbidden")

    steps = get_verification_steps_collection()
    return await steps.list_by_verification(verification_id)
