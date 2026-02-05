from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class Owner(BaseModel):
    id: str
    name: str

class DocumentMetadata(BaseModel):
    document_id: str
    cid: str
    owner: Owner
    document_type: str
    issue_date: date
    expiry_date: date
    file_hash_sha256: str
    file_name: str
    file_size: int
    created_at: datetime
    note: Optional[str] = None
