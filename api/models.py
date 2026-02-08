from typing import Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    name: str
    username: Optional[str] = None
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class UserInDB(BaseModel):
    id: Optional[str]
    name: str
    email: EmailStr
    hashed_password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class DocumentTypeBase(BaseModel):
    name: str
    folder_name: str  # Required - maps to ai/data/refrences/{folder_name}
    is_active: bool = True
    requires_back_image: bool = False


class DocumentTypeCreate(DocumentTypeBase):
    pass


class DocumentTypeUpdate(DocumentTypeBase):
    name: Optional[str] = None
    folder_name: Optional[str] = None
    is_active: Optional[bool] = None
    requires_back_image: Optional[bool] = None


class DocumentTypeInDB(DocumentTypeBase):
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DocumentTypePublic(DocumentTypeBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AuditLogPublic(BaseModel):
    id: int
    operation_id: str
    operation_type: str
    status: str
    failure_reason: Optional[str] = None
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    service: Optional[str] = None
    module: Optional[str] = None
    path: Optional[str] = None
    method: Optional[str] = None
    file_name: Optional[str] = None
    file_ext: Optional[str] = None
    file_size: Optional[int] = None
    file_cid: Optional[str] = None
    file_url: Optional[str] = None
    extra_data: Optional[dict[str, Any]] = None
    created_at: datetime


class AuditLogListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[AuditLogPublic]


class VerificationStage(str, Enum):
    DOCUMENT_IMAGE_QUALITY = "DOCUMENT_IMAGE_QUALITY"
    DOCUMENT_CROPPING = "DOCUMENT_CROPPING"
    DOCUMENT_FACE_EXTRACTION = "DOCUMENT_FACE_EXTRACTION"
    FACE_MATCHING = "FACE_MATCHING"
    OCR = "OCR"
    AI_VERIFICATION = "AI_VERIFICATION"
    DATA_VERIFICATION = "DATA_VERIFICATION"
    BLOCKCHAIN = "BLOCKCHAIN"


class VerificationStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class VerificationCreateRequest(BaseModel):
    document_type_id: int


class VerificationStepPublic(BaseModel):
    id: int
    verification_id: int
    stage: VerificationStage
    status: VerificationStatus
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result_data: Optional[dict[str, Any]] = None
    created_at: datetime


class VerificationPublic(BaseModel):
    id: int
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    document_type_id: Optional[int] = None
    document_type_name: Optional[str] = None
    status: VerificationStatus
    current_stage: Optional[VerificationStage] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result_data: Optional[dict[str, Any]] = None
    created_at: datetime


class VerificationListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[VerificationPublic]


class VerificationListWithStatsResponse(VerificationListResponse):
    status_counts: dict[str, int]

    class Config:
        validate_assignment = True
