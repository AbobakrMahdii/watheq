"""Blockchain Router — واجهة البلوكتشين

Endpoints for document registration and verification on MultiChain + IPFS:
- POST /upload — upload, hash-check, IPFS pin, publish to chain
- GET /documents — list all chain records
- GET /documents/{id} — lookup by document ID
- GET /verify/{hash} — verify a document by SHA-256 hash
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.security import get_current_user
from ledger.ipfs_service import IPFSService
from api.services.multichain_service import (
    json_to_hex,
    list_stream_items,
    publish_to_stream,
    get_item_by_key,
)
from api.services.hash_service import sha256_bytes
from api.database import get_document_hashes_collection

router = APIRouter(
    prefix="/api/v1/blockchain",
    tags=["Blockchain"],
    dependencies=[Depends(get_current_user)],
)

# نستخدم IPFS خارج السلسلة لتخزين الملف نفسه وتخزين CID فقط على البلوكشين.
_ipfs: Optional[IPFSService] = None
_ipfs_err: Optional[str] = None
try:
    _ipfs = IPFSService()
except Exception as exc:
    _ipfs = None
    _ipfs_err = str(exc)


@router.post("/upload")
async def upload_and_publish(
    file: UploadFile = File(...), current_user=Depends(get_current_user)
):
    """
    1) يحسب بصمة SHA-256 للمحتوى (hash) لمنع التكرار.
    2) يتحقق من عدم وجود hash في قاعدة البيانات، وإلا يرجع 409.
    3) يرفع الملف إلى IPFS -> يرجع CID.
    4) يبني metadata موسعة ثم يحولها HEX.
    5) ينشر على MultiChain stream 'documents' (key=document_id).
    6) يحفظ hash + CID في قاعدة البيانات.
    """
    if _ipfs is None:
        raise HTTPException(status_code=503, detail=f"IPFS unavailable: {_ipfs_err}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="ملف فارغ")

    # 4.5.1 بصمة المحتوى
    file_hash = sha256_bytes(data)
    hashes = get_document_hashes_collection()

    # 4.5.4 منع التكرار قبل الرفع
    existing = await hashes.find_by_hash(file_hash)
    if existing:
        raise HTTPException(
            status_code=409,
            detail="هذه الوثيقة مسجلة مسبقًا (تم العثور على نفس البصمة).",
        )

    document_id = str(uuid.uuid4())

    # 4.5.5 رفع إلى IPFS بعد التأكد من عدم التكرار
    try:
        cid = _ipfs.pin_bytes(data, filename=file.filename or "file")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IPFS error: {exc}")

    metadata = {
        "document_id": document_id,
        "hash": file_hash,
        "ipfs_cid": cid,
        "filename": file.filename,
        "content_type": file.content_type,
        "user": current_user.get("email") or current_user.get("sub"),
        "timestamp": int(time.time()),
        "processing_result": None,
    }

    # 4.5.6 التسجيل في البلوكشين (data = metadata HEX, key = document_id)
    try:
        hex_payload = json_to_hex(metadata)
        publish_to_stream(document_id, hex_payload)
        # 4.5.3 تخزين البصمة في قاعدة البيانات بعد نجاح الرفع والنشر
        await hashes.insert_one(document_id, file_hash, cid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MultiChain error: {exc}")

    return {
        "document_id": document_id,
        "cid": cid,
        "hash": file_hash,
        "metadata": metadata,
    }


@router.get("/documents")
def list_documents():
    """
    يرجع كل العناصر من stream 'documents' مع فك ترميز HEX إلى JSON.
    """
    try:
        items = list_stream_items()
        return {"items": items}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MultiChain error: {exc}")


@router.get("/documents/{document_id}")
def get_document(document_id: str):
    """
    استعلام عبر document_id:
    - قراءة آخر عنصر في stream بنفس المفتاح
    - فك HEX إلى JSON وإرجاعه
    """
    try:
        item = get_item_by_key(document_id)
        if not item:
            raise HTTPException(status_code=404, detail="Document not found on chain")
        return item
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MultiChain error: {exc}")


@router.get("/verify/{file_hash}")
async def verify_by_hash(file_hash: str):
    """
    التحقق مما إذا كانت وثيقة (عبر بصمة SHA-256) مسجلة على البلوكتشين + IPFS.
    يرجع بيانات التسجيل إذا وُجدت، أو 404.
    """
    hashes = get_document_hashes_collection()
    record = await hashes.find_by_hash(file_hash)
    if not record:
        raise HTTPException(
            status_code=404,
            detail="لا يوجد سجل لهذه البصمة على البلوكتشين",
        )

    # Fetch full metadata from MultiChain if possible
    chain_data = None
    try:
        chain_data = get_item_by_key(record.get("document_id", ""))
    except Exception:
        pass

    return {
        "verified": True,
        "document_id": record.get("document_id"),
        "hash": file_hash,
        "cid": record.get("cid"),
        "chain_data": chain_data,
    }
