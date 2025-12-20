from fastapi import APIRouter, UploadFile, File, HTTPException
import requests

router = APIRouter(prefix="/ipfs", tags=["ipfs"])

IPFS_API = "http://127.0.0.1:5001/api/v0"

@router.get("/health")
def health():
    r = requests.post(f"{IPFS_API}/version", timeout=5)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=r.text)
    return {"healthy": True, "ipfs": r.json()}

@router.post("/pin-file")
async def pin_file(file: UploadFile = File(...)):
    content = await file.read()
    files = {"file": (file.filename, content)}
    r = requests.post(f"{IPFS_API}/add", files=files, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=r.text)

    data = r.json()           # مثال: {"Name":"test.txt","Hash":"CID","Size":"..."}
    return {"filename": data.get("Name"), "cid": data.get("Hash")}

@router.get("/file/{cid}")
def get_file(cid: str):
    r = requests.post(f"{IPFS_API}/cat", params={"arg": cid}, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=r.text)
    return {"cid": cid, "data": r.text}
