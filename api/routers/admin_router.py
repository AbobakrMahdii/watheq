from fastapi import APIRouter, Depends, HTTPException

from api.database import get_user_collection, database
from ..models import UserCreate
from ..security import get_current_admin, get_current_super_admin
from ..security import get_password_hash

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
    if user.username and await users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    await users.insert_one(
        {
            "name": user.name,
            "username": user.username,
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
    if user.username and await users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")

    await users.insert_one(
        {
            "name": user.name,
            "username": user.username,
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

    await users.update_one(
        {"_id": uid},
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

# =========================
# Analytics summary (admin + super)
# =========================
@router.get("/analytics")
async def get_analytics(admin=Depends(get_current_admin)):
    if not database.is_connected:
        await database.connect()

    async def _count(query: str, values: dict | None = None) -> int:
        row = await database.fetch_one(query, values=values)
        return int(row["total"]) if row else 0

    total_users = await _count("SELECT COUNT(*) as total FROM users WHERE role = 'user'")
    total_admins = await _count(
        "SELECT COUNT(*) as total FROM users WHERE role IN ('admin','super_admin')"
    )
    total_verifications = await _count("SELECT COUNT(*) as total FROM verifications")
    total_document_types = await _count("SELECT COUNT(*) as total FROM document_types")
    total_audit_logs = await _count("SELECT COUNT(*) as total FROM audit_logs")

    return {
        "total_users": total_users,
        "total_admins": total_admins,
        "total_verifications": total_verifications,
        "total_authentications": total_verifications,
        "total_document_types": total_document_types,
        "total_audit_logs": total_audit_logs,
    }
