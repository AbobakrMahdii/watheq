import asyncio

from api.database import database, get_user_collection, init_db
from api.security import get_password_hash

SUPER_ADMIN = {
    "name": "Super Admin",
    "username": "admin",
    "email": "admin@admin.com",
    "password": "admin123",
    "role": "super_admin",
}


async def seed_super_admin() -> None:
    try:
        await init_db()
        users = get_user_collection()

        existing = await users.find_one({"email": SUPER_ADMIN["email"]})
        if existing:
            print("Super admin already exists")
            return

        await users.insert_one(
            {
                "name": SUPER_ADMIN["name"],
                "username": SUPER_ADMIN["username"],
                "email": SUPER_ADMIN["email"],
                "password": get_password_hash(SUPER_ADMIN["password"]),
                "role": SUPER_ADMIN["role"],
                "is_active": True,
                "deleted_at": None,
            }
        )

        print("Super admin created successfully")
        print("Email:", SUPER_ADMIN["email"])
        print("Password:", SUPER_ADMIN["password"])
    finally:
        await database.disconnect()


if __name__ == "__main__":
    asyncio.run(seed_super_admin())
