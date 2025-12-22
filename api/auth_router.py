from fastapi import APIRouter, HTTPException, Depends
from database import get_user_collection
from models import UserCreate, UserLogin
from security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_admin
)

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])



# Register (User)

@router.post("/register")
async def register(user: UserCreate):
    users = get_user_collection()

    existing = await users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_dict = {
        "name": user.name,
        "email": user.email,
        "password": get_password_hash(user.password),
        "role": "user"   
    }

    await users.insert_one(user_dict)
    return {"message": "registered successfully"}



# Login (User / Admin)

@router.post("/login")
async def login(user: UserLogin):
    users = get_user_collection()

    db_user = await users.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    role = db_user.get("role", "user")

    access_token = create_access_token(
        data={
            "sub": str(db_user["_id"]),
            "email": db_user["email"],
            "role": role
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": role
    }



#

@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    return {
        "message": "user authenticated",
        "user": current_user
    }





@router.get("/admin/me")
def admin_me(admin_user: dict = Depends(get_current_admin)):
    return {
        "message": "admin authenticated",
        "admin": admin_user
    }