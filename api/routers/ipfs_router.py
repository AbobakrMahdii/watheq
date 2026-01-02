from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from ledger.ipfs_service import IPFSService

router = APIRouter(
    prefix="/ipfs",
    tags=["ipfs"],
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
        return {"available": True, "version": ipfs.client.version()}
    except Exception as e:
        logger.error("IPFS health check failed: %s", e)
        return JSONResponse(status_code=503, content={"available": False, "error": str(e)})


@router.post("/pin-file")
async def pin_file(file: UploadFile = File(...)):
    """Upload a file to IPFS and return CID. Returns 503 if IPFS not available."""
    if ipfs is None:
        raise HTTPException(status_code=503, detail="IPFS service not available")

    try:
        data = await file.read()
        cid = ipfs.client.add_bytes(data)

        return {
            "cid": cid,
            "filename": file.filename,
        }
    except Exception as e:
        logger.exception("Failed to pin file to IPFS")
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
