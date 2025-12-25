from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    name: str
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

  #عشان يطلع الكومنت41