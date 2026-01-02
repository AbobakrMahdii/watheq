import asyncio
from api.database import database, get_user_collection, init_db
from api.security import get_password_hash

SUPER_ADMIN = {
    "name": "Super Admin",
    "email": "admin@admin.com",
    "password": "admin123",
    "role": "super_admin",
}

async def seed_super_admin():
    # 1️⃣ Initialize DB & tables
    await init_db()

    # 2️⃣ Get users collection
    users = get_user_collection()

    # 3️⃣ Check if super admin exists
    existing = await users.find_one({"email": SUPER_ADMIN["email"]})
    if existing:
        print("✅ Super admin already exists")
        await database.disconnect()
        return

    # 4️⃣ Insert super admin
    await users.insert_one({
        "name": SUPER_ADMIN["name"],
        "email": SUPER_ADMIN["email"],
        "password": get_password_hash(SUPER_ADMIN["password"]),
        "role": SUPER_ADMIN["role"],
    })

    print("🚀 Super admin created successfully")
    print("📧 Email:", SUPER_ADMIN["email"])
    print("🔑 Password:", SUPER_ADMIN["password"])

    await database.disconnect()

if __name__ == "__main__":
    asyncio.run(seed_super_admin())
