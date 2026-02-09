"""Blockchain Router — واجهة البلوكتشين

Full CRUD endpoints for MultiChain + IPFS:
- POST /upload — upload, hash-check, IPFS pin, publish to chain
- POST /publish — manual publish raw JSON to any stream
- POST /hash — compute SHA-256 of uploaded file
- GET /documents — list all chain records
- GET /documents/{id} — lookup by document ID
- GET /verify/{hash} — verify a document by SHA-256 hash
- GET /info — chain status (getinfo)
- GET /block-count — current block height
- GET /blocks/{height} — block details
- GET /tx/{txid} — transaction details
- GET /streams — list all streams
- GET /streams/{name}/keys — stream keys
- GET /streams/{name}/items — stream items
- GET /streams/{name}/publishers — stream publishers
- GET /peers — connected peers
- GET /permissions — address permissions
- IPFS: /ipfs/health, /ipfs/node-id, /ipfs/pins, /ipfs/repo-stat, /ipfs/pin, DELETE /ipfs/pin/{cid}
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, Body, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.security import get_current_user
from ledger.ipfs_service import IPFSService
from api.services.multichain_service import (
    json_to_hex,
    list_stream_items,
    publish_to_stream,
    get_item_by_key,
    get_chain_info,
    get_block_count,
    get_block,
    get_raw_transaction,
    list_streams,
    list_stream_keys,
    list_stream_items_by_stream,
    list_stream_publishers,
    get_peer_info,
    list_permissions,
    publish_raw_json,
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


# ═══════════════════════════════════════════════════════════════════
# MultiChain — READ endpoints
# ═══════════════════════════════════════════════════════════════════

@router.get("/info")
def chain_info():
    """معلومات سلسلة البلوكتشين (getinfo)."""
    try:
        return get_chain_info()
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/block-count")
def block_count():
    """ارتفاع البلوك الحالي."""
    try:
        height = get_block_count()
        return {"block_count": height}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/blocks/{height}")
def block_details(height: str):
    """تفاصيل بلوك حسب ارتفاعه أو hash."""
    try:
        return get_block(height)
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/tx/{txid}")
def transaction_details(txid: str):
    """تفاصيل معاملة حسب txid."""
    try:
        return get_raw_transaction(txid)
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/streams")
def list_all_streams():
    """قائمة جميع الـ streams على السلسلة."""
    try:
        return {"streams": list_streams()}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/streams/{name}/keys")
def stream_keys(name: str):
    """مفاتيح stream معين."""
    try:
        return {"keys": list_stream_keys(name)}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/streams/{name}/items")
def stream_items(name: str):
    """عناصر stream معين (مع فك ترميز البيانات)."""
    try:
        return {"items": list_stream_items_by_stream(name)}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/streams/{name}/publishers")
def stream_publishers(name: str):
    """ناشرو stream معين."""
    try:
        return {"publishers": list_stream_publishers(name)}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/peers")
def peers():
    """الأقران المتصلون بالعقدة."""
    try:
        return {"peers": get_peer_info()}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


@router.get("/permissions")
def permissions():
    """صلاحيات العناوين على السلسلة."""
    try:
        return {"permissions": list_permissions()}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


# ═══════════════════════════════════════════════════════════════════
# MultiChain — CREATE (manual publish)
# ═══════════════════════════════════════════════════════════════════

class ManualPublishBody(BaseModel):
    stream: str = "documents"
    key: str
    data: dict


@router.post("/publish")
def manual_publish(body: ManualPublishBody):
    """نشر بيانات JSON يدوياً على أي stream. يرجع txid."""
    try:
        txid = publish_raw_json(body.stream, body.key, body.data)
        return {"txid": txid, "stream": body.stream, "key": body.key}
    except Exception as exc:
        raise HTTPException(502, f"MultiChain error: {exc}")


# ═══════════════════════════════════════════════════════════════════
# Hash utility
# ═══════════════════════════════════════════════════════════════════

@router.post("/hash")
async def compute_hash(file: UploadFile = File(...)):
    """حساب بصمة SHA-256 لملف مرفوع."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    return {"hash": sha256_bytes(data), "filename": file.filename, "size": len(data)}


# ═══════════════════════════════════════════════════════════════════
# IPFS — Extended endpoints
# ═══════════════════════════════════════════════════════════════════

@router.get("/ipfs/health")
def ipfs_health():
    """فحص اتصال IPFS."""
    if _ipfs is None:
        return {"healthy": False, "error": _ipfs_err}
    return {"healthy": _ipfs.healthy()}


@router.get("/ipfs/node-id")
def ipfs_node_id():
    """هوية عقدة IPFS."""
    if _ipfs is None:
        raise HTTPException(503, f"IPFS unavailable: {_ipfs_err}")
    try:
        return _ipfs.node_id()
    except Exception as exc:
        raise HTTPException(502, f"IPFS error: {exc}")


@router.get("/ipfs/pins")
def ipfs_pins():
    """قائمة جميع الملفات المثبتة على IPFS."""
    if _ipfs is None:
        raise HTTPException(503, f"IPFS unavailable: {_ipfs_err}")
    try:
        pins = _ipfs.list_pins()
        return {"pins": pins, "count": len(pins)}
    except Exception as exc:
        raise HTTPException(502, f"IPFS error: {exc}")


@router.get("/ipfs/repo-stat")
def ipfs_repo_stat():
    """إحصائيات مخزن IPFS (المساحة المستخدمة ...)."""
    if _ipfs is None:
        raise HTTPException(503, f"IPFS unavailable: {_ipfs_err}")
    try:
        return _ipfs.repo_stat()
    except Exception as exc:
        raise HTTPException(502, f"IPFS error: {exc}")


@router.post("/ipfs/pin")
async def ipfs_pin_file(file: UploadFile = File(...)):
    """رفع وتثبيت ملف على IPFS فقط (بدون بلوكتشين). يرجع CID."""
    if _ipfs is None:
        raise HTTPException(503, f"IPFS unavailable: {_ipfs_err}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        cid = _ipfs.pin_bytes(data, filename=file.filename or "file")
        return {"cid": cid, "filename": file.filename, "size": len(data)}
    except Exception as exc:
        raise HTTPException(502, f"IPFS error: {exc}")


@router.delete("/ipfs/pin/{cid}")
def ipfs_unpin(cid: str):
    """إلغاء تثبيت CID من IPFS."""
    if _ipfs is None:
        raise HTTPException(503, f"IPFS unavailable: {_ipfs_err}")
    try:
        result = _ipfs.unpin(cid)
        return {"unpinned": True, "cid": cid, "result": result}
    except Exception as exc:
        raise HTTPException(502, f"IPFS error: {exc}")
