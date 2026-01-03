from fastapi import APIRouter, HTTPException, Depends, Request

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
    if getattr(user, "username", None) and await users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    await users.insert_one({
        "name": user.name,
        "username": getattr(user, "username", None),
        "email": user.email,
        "password": get_password_hash(user.password),
        "role": "user"
    })

    return {"message": "registered successfully"}


@router.post("/login")
async def login(request: Request):
    users = get_user_collection()

    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        payload = await request.json()
        identifier = (payload.get("email") or payload.get("username") or "").strip()
        password = payload.get("password") or ""
    else:
        form = await request.form()
        identifier = (form.get("username") or form.get("email") or "").strip()
        password = form.get("password") or ""

    if not identifier or not password:
        raise HTTPException(
            status_code=400,
            detail={"message": "Email and password are required", "code": "VALIDATION_ERROR"},
        )

    if "@" in identifier:
        user = await users.find_one({"email": identifier})
    else:
        user = await users.find_one({"username": identifier})
        user = user or await users.find_one({"email": identifier})

    if not user or not verify_password(password, user["password"]):
        raise HTTPException(
            status_code=401,
            detail={"message": "Invalid email or password", "code": "INVALID_CREDENTIALS"},
        )

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
