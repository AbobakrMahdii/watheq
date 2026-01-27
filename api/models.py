from typing import Optional

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    name: str
    username: Optional[str] = None
    email: EmailStr
    password: str


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
    is_active: bool = True
    requires_back_image: bool = False


class DocumentTypeCreate(DocumentTypeBase):
    pass


class DocumentTypeUpdate(DocumentTypeBase):
    name: Optional[str] = None
    is_active: Optional[bool] = None
    requires_back_image: Optional[bool] = None


class DocumentTypeInDB(DocumentTypeBase):
    id: Optional[int] = None
    created_at: Optional[str] = None # Or datetime object

    class Config:
        from_attributes = True


class DocumentTypePublic(DocumentTypeBase):
    id: int
    created_at: str # Or datetime object

    class Config:
        from_attributes = True
