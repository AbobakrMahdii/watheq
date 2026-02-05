from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import os

from schemas import DocumentMetadata, Owner
from utils import sha256_file, sha256_bytes, now_utc
from ipfs_service import add_file_to_ipfs, download_from_ipfs
from blockchain_service import publish_metadata, get_by_key

app = FastAPI(title="Watheq Blockchain + IPFS API")

# =======================
# Upload Document
# =======================
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_id: str = Form(...),
    owner_id: str = Form(...),
    owner_name: str = Form(...),
    document_type: str = Form(...),
    issue_date: str = Form(...),
    expiry_date: str = Form(...)
):
    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        file_hash = sha256_file(tmp_path)

        ipfs_res = add_file_to_ipfs(tmp_path)
        cid = ipfs_res["cid"]
        size = ipfs_res["size"]

        metadata = DocumentMetadata(
            document_id=document_id,
            cid=cid,
            owner=Owner(id=owner_id, name=owner_name),
            document_type=document_type,
            issue_date=issue_date,
            expiry_date=expiry_date,
            file_hash_sha256=file_hash,
            file_name=file.filename,
            file_size=size,
            created_at=now_utc()
        )

        txid = publish_metadata(document_id, metadata.model_dump())

        return {
            "status": "success",
            "txid": txid,
            "cid": cid,
            "metadata": metadata.model_dump()
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# =======================
# Get Metadata
# =======================
@app.get("/documents/{document_id}")
def get_document(document_id: str):
    items = get_by_key(document_id)
    if not items:
        return {"error": "Document not found"}

    data = items[-1]["data"]["json"]
    return {
        "document_id": document_id,
        "metadata": data,
        "ipfs_url": f"https://ipfs.io/ipfs/{data['cid']}"
    }

# =======================
# Verify Document
# =======================
@app.get("/documents/{document_id}/verify")
def verify_document(document_id: str):
    items = get_by_key(document_id)
    if not items:
        return {"error": "Document not found"}

    data = items[-1]["data"]["json"]
    cid = data["cid"]
    stored_hash = data["file_hash_sha256"]

    file_bytes = download_from_ipfs(cid)
    computed_hash = sha256_bytes(file_bytes)

    return {
        "document_id": document_id,
        "cid": cid,
        "stored_hash": stored_hash,
        "computed_hash": computed_hash,
        "valid": stored_hash == computed_hash
    }
