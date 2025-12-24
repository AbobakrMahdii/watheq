from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId

from database import get_user_collection
from security import get_current_admin, get_current_super_admin

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

# =========================
# Promote user → admin
# =========================
@router.put("/users/{user_id}/make-admin")
async def make_admin(
    user_id: str,
    super_admin=Depends(get_current_super_admin)
):
    users = get_user_collection()

    if not ObjectId.is_valid(user_id):
        raise HTTPException(400, "Invalid user id")

    await users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"role": "admin"}}
    )

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

    user = await users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(404, "User not found")

    if user["role"] == "super_admin":
        raise HTTPException(403, "Cannot demote super admin")

    await users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"role": "user"}}
    )

    return {"message": "Admin removed"}