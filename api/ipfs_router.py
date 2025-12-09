from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from ledger.ipfs_service import IPFSService

router = APIRouter(
    prefix="/ipfs",
    tags=["ipfs"],
)

ipfs = IPFSService()


@router.get("/health")
def health_check():
    """
    يرجع حالة اتصال IPFS
    """
    try:
        return {"healthy": ipfs.healthy()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pin-file")
async def pin_file(file: UploadFile = File(...)):
    """
    تستقبل ملف من العميل، ترفعه إلى IPFS، وترجع CID
    """
    try:
        data = await file.read()
        cid = ipfs.client.add_bytes(data)

        return {
            "cid": cid,
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPFS error: {e}")


@router.get("/file/{cid}")
def get_file(cid: str):
    """
    ترجّع محتوى الملف من IPFS
    """
    try:
        content = ipfs.get_file(cid)
        return Response(content=content, media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPFS error: {e}")
