from fastapi import APIRouter, UploadFile, File, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ledger.ipfs_service import IPFSService
from api.services.audit_log_service import log_file_event

router = APIRouter(
    prefix="/api/v1/ipfs",
    tags=["IPFS"],
)

logger = logging.getLogger("api.ipfs")

# Try to create IPFS service at startup; if it fails keep `ipfs` as None
ipfs: Optional[IPFSService] = None
_ipfs_init_error: Optional[str] = None
try:
    ipfs = IPFSService()
except Exception as e:
    ipfs = None
    _ipfs_init_error = str(e)
    logger.warning("IPFS not available at startup: %s", _ipfs_init_error)


@router.get("/health")
def health_check():
    """Return IPFS connection status (503 if unavailable)."""
    if ipfs is None:
        return JSONResponse(status_code=503, content={"available": False, "error": _ipfs_init_error})

    try:
        # Using HTTP-based IPFSService (no ipfshttpclient).
        return {"available": True, "ok": ipfs.healthy()}
    except Exception as e:
        logger.error("IPFS health check failed: %s", e)
        return JSONResponse(status_code=503, content={"available": False, "error": str(e)})


@router.post("/pin-file")
async def pin_file(request: Request, file: UploadFile = File(...)):
    """Upload a file to IPFS and return CID. Returns 503 if IPFS not available."""
    if ipfs is None:
        raise HTTPException(status_code=503, detail="IPFS service not available")

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        cid = ipfs.pin_bytes(data, filename=file.filename or "file")

        request.state.audit_logged = True
        await log_file_event(
            request,
            status="success",
            operation_type="Upload",
            module="ipfs",
            file_name=file.filename,
            file_size=len(data),
            file_cid=cid,
            file_url=f"/api/v1/ipfs/file/{cid}",
        )
        return {
            "cid": cid,
            "filename": file.filename,
        }
    except Exception as e:
        logger.exception("Failed to pin file to IPFS")
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason=f"IPFS error: {e}",
            operation_type="Upload",
            module="ipfs",
            file_name=file.filename,
            file_size=len(data) if "data" in locals() and data is not None else None,
        )
        raise HTTPException(status_code=500, detail=f"IPFS error: {e}")


@router.get("/file/{cid}")
def get_file(cid: str):
    """Return file content from IPFS. Returns 503 if IPFS not available."""
    if ipfs is None:
        raise HTTPException(status_code=503, detail="IPFS service not available")

    try:
        content = ipfs.get_file(cid)
        return Response(content=content, media_type="application/octet-stream")
    except Exception as e:
        logger.exception("Failed to get file from IPFS: %s", cid)
        raise HTTPException(status_code=500, detail=f"IPFS error: {e}")
