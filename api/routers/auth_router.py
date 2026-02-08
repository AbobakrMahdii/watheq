from fastapi import APIRouter, HTTPException, Depends, Request
from passlib.exc import UnknownHashError

from api.database import get_user_collection
from ..models import UserCreate, ChangePasswordRequest
from ..security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_admin,
)
from api.services.audit_log_service import log_auth_event

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register")
async def register(user: UserCreate):
    users = get_user_collection()

    if await users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if user.username and await users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    await users.insert_one(
        {
            "name": user.name,
            "username": user.username,
            "email": user.email,
            "password": get_password_hash(user.password),
            "role": "user",
            "is_active": True,
            "deleted_at": None,
        }
    )

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
        await log_auth_event(
            request,
            operation_type="Login",
            status="failed",
            failure_reason="Email and password are required",
            user_identifier=identifier,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Email and password are required",
                "code": "VALIDATION_ERROR",
            },
        )

    if "@" in identifier:
        user = await users.find_one({"email": identifier})
    else:
        user = await users.find_one({"username": identifier})
        user = user or await users.find_one({"email": identifier})

    valid_password = False
    if user:
        is_active = (
            True if user.get("is_active") is None else bool(user.get("is_active"))
        )
        if user.get("deleted_at") or not is_active:
            await log_auth_event(
                request,
                operation_type="Login",
                status="failed",
                failure_reason="User is suspended or deleted",
                user_identifier=identifier,
            )
            raise HTTPException(
                status_code=403,
                detail={"message": "Account is suspended", "code": "SUSPENDED"},
            )
        try:
            valid_password = verify_password(password, user["password"])
        except UnknownHashError:
            # Fallback for legacy/plaintext passwords
            if isinstance(user.get("password"), str) and user["password"] == password:
                valid_password = True
                try:
                    new_hash = get_password_hash(password)
                    await users.update_one(
                        {"_id": user["_id"]}, {"$set": {"password": new_hash}}
                    )
                except Exception:
                    pass

    if not user or not valid_password:
        await log_auth_event(
            request,
            operation_type="Login",
            status="failed",
            failure_reason="Invalid email or password",
            user_identifier=identifier,
        )
        raise HTTPException(
            status_code=401,
            detail={
                "message": "Invalid email or password",
                "code": "INVALID_CREDENTIALS",
            },
        )

    token = create_access_token(
        {
            "sub": str(user["_id"]),
            "email": user["email"],
            "role": user.get("role", "user"),
        }
    )

    await log_auth_event(
        request,
        operation_type="Login",
        status="success",
        user_id=int(user["_id"]),
        user_name=user.get("name"),
        user_email=user.get("email"),
        user_role=user.get("role", "user"),
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.get("role", "user"),
    }


@router.get("/me")
async def me(current_user=Depends(get_current_user)):
    users = get_user_collection()
    user_doc = None
    try:
        user_id = (
            int(current_user.get("sub"))
            if str(current_user.get("sub")).isdigit()
            else None
        )
        if user_id is not None:
            user_doc = await users.find_one({"_id": user_id})
    except Exception:
        user_doc = None

    if user_doc:
        return {
            "id": user_doc.get("_id"),
            "name": user_doc.get("name"),
            "username": user_doc.get("username"),
            "email": user_doc.get("email"),
            "role": user_doc.get("role"),
        }

    return {
        "id": current_user.get("sub"),
        "email": current_user.get("email"),
        "role": current_user.get("role"),
    }


@router.get("/admin/me")
async def admin_me(admin=Depends(get_current_admin)):
    users = get_user_collection()
    user_doc = None
    try:
        user_id = int(admin.get("sub")) if str(admin.get("sub")).isdigit() else None
        if user_id is not None:
            user_doc = await users.find_one({"_id": user_id})
    except Exception:
        user_doc = None

    if user_doc:
        return {
            "id": user_doc.get("_id"),
            "name": user_doc.get("name"),
            "username": user_doc.get("username"),
            "email": user_doc.get("email"),
            "role": user_doc.get("role"),
        }

    return admin


@router.put("/change-password")
async def change_password(
    body: ChangePasswordRequest, current_user=Depends(get_current_user)
):
    users = get_user_collection()
    user_id = (
        int(current_user.get("sub")) if str(current_user.get("sub")).isdigit() else None
    )
    if user_id is None:
        raise HTTPException(400, "Invalid user")

    user_doc = await users.find_one({"_id": user_id})
    if not user_doc:
        raise HTTPException(404, "User not found")

    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters")

    if not verify_password(body.current_password, user_doc["password"]):
        raise HTTPException(400, "Current password is incorrect")

    await users.update_one(
        {"_id": user_id}, {"$set": {"password": get_password_hash(body.new_password)}}
    )
    return {"message": "Password changed successfully"}


@router.post("/logout")
async def logout(request: Request, current_user=Depends(get_current_user)):
    await log_auth_event(
        request,
        operation_type="Logout",
        status="success",
        user_id=(
            int(current_user.get("sub"))
            if str(current_user.get("sub")).isdigit()
            else None
        ),
        user_email=current_user.get("email"),
        user_role=current_user.get("role"),
    )
    return {"message": "logged out"}
