from fastapi import APIRouter, Depends, HTTPException

from database import get_user_collection
from security import get_current_admin, get_current_super_admin
from models import UserCreate
from security import get_password_hash

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin"]
)

def to_public_user(user: dict) -> dict:
    user["_id"] = str(user["_id"])
    user.pop("password", None)
    return user

# =========================
# Users list (admin + super)
# =========================
@router.get("/users")
async def get_users(admin=Depends(get_current_admin)):
    users = get_user_collection()
    return [to_public_user(u) async for u in users.find({"role": "user"})]


@router.post("/users")
async def create_user(user: UserCreate, admin=Depends(get_current_admin)):
    users = get_user_collection()

    if await users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if getattr(user, "username", None) and await users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    await users.insert_one(
        {
            "name": user.name,
            "username": getattr(user, "username", None),
            "email": user.email,
            "password": get_password_hash(user.password),
            "role": "user",
        }
    )

    return {"message": "User created"}

# =========================
# Admins list (super only)
# =========================
@router.get("/admins")
async def get_admins(super_admin=Depends(get_current_super_admin)):
    users = get_user_collection()
    return [
        to_public_user(u)
        async for u in users.find({"role": {"$in": ["admin", "super_admin"]}})
    ]


@router.post("/admins")
async def create_admin(user: UserCreate, super_admin=Depends(get_current_super_admin)):
    users = get_user_collection()

    if await users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if getattr(user, "username", None) and await users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    await users.insert_one(
        {
            "name": user.name,
            "username": getattr(user, "username", None),
            "email": user.email,
            "password": get_password_hash(user.password),
            "role": "admin",
        }
    )

    return {"message": "Admin created"}

# =========================
# Promote user → admin
# =========================
@router.put("/users/{user_id}/make-admin")
async def make_admin(
    user_id: str,
    super_admin=Depends(get_current_super_admin)
):
    users = get_user_collection()

    try:
        uid = int(user_id)
    except Exception:
        raise HTTPException(400, "Invalid user id")

    await users.update_one({"_id": uid}, {"$set": {"role": "admin"}})

    return {"message": "User promoted to admin"}

# =========================
# Demote admin → user
# =========================
@router.put("/users/{user_id}/remove-admin")
async def remove_admin(
    user_id: str,
    super_admin=Depends(get_current_super_admin)
):
    users = get_user_collection()

    try:
        uid = int(user_id)
    except Exception:
        raise HTTPException(400, "Invalid user id")

    user = await users.find_one({"_id": uid})
    if not user:
        raise HTTPException(404, "User not found")

    if user["role"] == "super_admin":
        raise HTTPException(403, "Cannot demote super admin")

    await users.update_one(
        {"_id": uid},
        {"$set": {"role": "user"}}
    )

    return {"message": "Admin removed"}
