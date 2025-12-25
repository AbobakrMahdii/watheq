from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm

from database import get_user_collection
from models import UserCreate
from security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_admin
)

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register")
async def register(user: UserCreate):
    users = get_user_collection()

    if await users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    await users.insert_one({
        "name": user.name,
        "email": user.email,
        "password": get_password_hash(user.password),
        "role": "user"
    })

    return {"message": "registered successfully"}


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    users = get_user_collection()
    user = await users.find_one({"email": form_data.username})

    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({
        "sub": str(user["_id"]),
        "email": user["email"],
        "role": user.get("role", "user")
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.get("role", "user")
    }


@router.get("/me")
def me(current_user=Depends(get_current_user)):
    return current_user


@router.get("/admin/me")
def admin_me(admin=Depends(get_current_admin)):
    return admin