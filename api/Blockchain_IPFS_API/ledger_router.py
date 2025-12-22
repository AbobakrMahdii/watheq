from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.fabric_service import fabric_invoke, fabric_query

router = APIRouter(prefix="/ledger", tags=["ledger"])

CHANNEL = "mychannel"
CHAINCODE = "watheq"
ORG = 1  # نخليها Org1 كبداية

class CreateDocBody(BaseModel):
    doc_id: str
    cid: str
    filename: str
    owner: str
    sha256: str

@router.get("/health")
def health():
    return {"ok": True, "channel": CHANNEL, "chaincode": CHAINCODE, "org": ORG}

@router.post("/docs")
def create_doc(body: CreateDocBody):
    try:
        out = fabric_invoke(
            ORG, CHANNEL, CHAINCODE,
            "CreateDoc",
            [body.doc_id, body.cid, body.filename, body.owner, body.sha256],
        )
        return {"ok": True, "invoke_output": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/docs/{doc_id}")
def read_doc(doc_id: str):
    try:
        out = fabric_query(ORG, CHANNEL, CHAINCODE, "ReadDoc", [doc_id])
        # غالبًا out بيرجع JSON من التشين كود
        return {"ok": True, "result": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
