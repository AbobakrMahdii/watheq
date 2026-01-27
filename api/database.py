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
# Document Types Collection
# =========================
class DocumentTypesCollection:
    def __init__(self, db: Database):
        self.db = db

    async def find_one(self, filt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not filt:
            return None
        if "_id" in filt:
            q = "SELECT id as id, name, is_active, requires_back_image, created_at FROM document_types WHERE id = :id"
            row = await self.db.fetch_one(q, values={"id": int(filt["_id"])})
            return dict(row) if row else None
        if "name" in filt:
            q = "SELECT id as id, name, is_active, requires_back_image, created_at FROM document_types WHERE name = :name"
            row = await self.db.fetch_one(q, values={"name": filt["name"]})
            return dict(row) if row else None
        return None

    async def find(self, filt: Dict[str, Any] = None) -> list[Dict[str, Any]]:
        query_parts = []
        values = {}
        if filt and "is_active" in filt:
            query_parts.append("is_active = :is_active")
            values["is_active"] = filt["is_active"]

        q = "SELECT id as id, name, is_active, requires_back_image, created_at FROM document_types"
        if query_parts:
            q += " WHERE " + " AND ".join(query_parts)
        q += " ORDER BY name"
        rows = await self.db.fetch_all(q, values=values)
        return [dict(row) for row in rows]

    async def insert_one(self, doc: Dict[str, Any]) -> int:
        q = """
            INSERT INTO document_types (name, is_active, requires_back_image, created_at)
            VALUES (:name, :is_active, :requires_back_image, :created_at)
        """
        return await self.db.execute(q, values=doc)

    async def update_one(self, doc_id: int, update_data: Dict[str, Any]) -> int:
        set_parts = []
        values = {"id": doc_id}
        for key, value in update_data.items():
            set_parts.append(f"{key} = :{key}")
            values[key] = value
        
        if not set_parts:
            return 0 # No update to perform

        q = f"UPDATE document_types SET {', '.join(set_parts)} WHERE id = :id"
        return await self.db.execute(q, values=values)

    async def delete_one(self, doc_id: int) -> int:
        q = "DELETE FROM document_types WHERE id = :id"
        return await self.db.execute(q, values={"id": doc_id})

_document_types_collection: Optional[DocumentTypesCollection] = None

def get_document_type_collection() -> DocumentTypesCollection:
    global _document_types_collection
    if _document_types_collection is None:
        _document_types_collection = DocumentTypesCollection(database)
    return _document_types_collection

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

    # Create document_types table if not exists
    query = """
    CREATE TABLE IF NOT EXISTS document_types (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) UNIQUE NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        requires_back_image BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    await database.execute(query)
