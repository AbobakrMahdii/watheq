import os
import asyncio
import aiomysql
from databases import Database
from typing import AsyncIterator, Dict, Any, Optional

# =========================
# Config
# =========================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "watheq_db")

DATABASE_URL = f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
database = Database(DATABASE_URL)

# =========================
# Users Collection
# =========================
class UsersCollection:
    def __init__(self, db: Database):
        self.db = db

    async def find_one(self, filt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not filt:
            return None
        if "_id" in filt:
            q = "SELECT id as _id, name, username, email, password, role FROM users WHERE id = :id"
            row = await self.db.fetch_one(q, values={"id": int(filt["_id"])})
            return dict(row) if row else None
        if "username" in filt:
            q = "SELECT id as _id, name, username, email, password, role FROM users WHERE username = :username"
            row = await self.db.fetch_one(q, values={"username": filt["username"]})
            return dict(row) if row else None
        if "email" in filt:
            q = "SELECT id as _id, name, username, email, password, role FROM users WHERE email = :email"
            row = await self.db.fetch_one(q, values={"email": filt["email"]})
            return dict(row) if row else None
        return None

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        username = doc.get("username")
        if username:
            q = "INSERT INTO users (name, username, email, password, role) VALUES (:name, :username, :email, :password, :role)"
        else:
            q = "INSERT INTO users (name, email, password, role) VALUES (:name, :email, :password, :role)"
        return await self.db.execute(q, values=doc)

    async def update_one(self, filt: Dict[str, Any], update: Dict[str, Any]) -> int:
        if not filt:
            return 0
        if "_id" in filt:
            user_id = int(filt["_id"])
        else:
            return 0
        if "$set" in update:
            sets = update["$set"]
            parts = []
            values = {"id": user_id}
            i = 0
            for k, v in sets.items():
                param = f"val{i}"
                parts.append(f"{k} = :{param}")
                values[param] = v
                i += 1
            q = "UPDATE users SET " + ", ".join(parts) + " WHERE id = :id"
            await self.db.execute(q, values=values)
            return 1
        return 0

    def find(self, filt: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        async def _iter():
            if filt and "role" in filt:
                if isinstance(filt["role"], dict) and "$in" in filt["role"]:
                    vals = list(filt["role"]["$in"] or [])
                    if not vals:
                        rows = []
                    else:
                        placeholders = ", ".join([f":v{i}" for i in range(len(vals))])
                        q = (
                            "SELECT id as _id, name, username, email, password, role "
                            f"FROM users WHERE role IN ({placeholders})"
                        )
                        rows = await self.db.fetch_all(
                            q,
                            values={f"v{i}": vals[i] for i in range(len(vals))},
                        )
                else:
                    q = "SELECT id as _id, name, username, email, password, role FROM users WHERE role = :role"
                    rows = await self.db.fetch_all(q, values={"role": filt["role"]})
            else:
                q = "SELECT id as _id, name, username, email, password, role FROM users"
                rows = await self.db.fetch_all(q)
            for r in rows:
                yield dict(r)
        return _iter()

# Instance
users = UsersCollection(database)

# Function to get collection
def get_user_collection():
    return users

# =========================
# Initialize DB + tables
# =========================
async def init_db():
    # Connect to MySQL server without specifying DB
    conn = await aiomysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    async with conn.cursor() as cur:
        await cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};")
    conn.close()

    # Connect using databases
    await database.connect()

    # Create users table if not exists
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        username VARCHAR(100) UNIQUE,
        email VARCHAR(100) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        role VARCHAR(20) NOT NULL
    );
    """
    await database.execute(query)

    # Try to add username column on existing deployments (ignore if already exists)
    try:
        await database.execute("ALTER TABLE users ADD COLUMN username VARCHAR(100) UNIQUE;")
    except Exception:
        pass
