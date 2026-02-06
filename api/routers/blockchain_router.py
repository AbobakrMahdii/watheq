from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.security import get_current_user
from ledger.ipfs_service import IPFSService
from api.services.multichain_service import (
    json_to_hex,
    list_stream_items,
    publish_to_stream,
)

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
async def upload_and_publish(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    """
    1) يرفع الملف إلى IPFS -> يرجع CID
    2) يبني metadata (ipfs_hash، اسم الملف، النوع، المستخدم، timestamp)
    3) يحول JSON إلى HEX
    4) ينشر على MultiChain stream 'documents' باستخدام multichain-cli
    """
    if _ipfs is None:
        raise HTTPException(status_code=503, detail=f"IPFS unavailable: {_ipfs_err}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="ملف فارغ")

    try:
        cid = _ipfs.pin_bytes(data, filename=file.filename or "file")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"IPFS error: {exc}")

    metadata = {
        "ipfs_hash": cid,
        "filename": file.filename,
        "content_type": file.content_type,
        "user": current_user.get("email") or current_user.get("sub"),
        "timestamp": int(time.time()),
    }

    try:
        hex_payload = json_to_hex(metadata)
        publish_to_stream(file.filename or cid, hex_payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MultiChain error: {exc}")

    return {"cid": cid, "metadata": metadata}


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
