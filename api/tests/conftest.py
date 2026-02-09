"""
Pytest configuration and shared fixtures for Watheq API tests.

Key design decisions:
- Raw aiomysql connections (not the `databases` library) are used for ALL
  test-fixture DB operations.  This avoids event-loop cross-contamination
  because ``TestClient`` runs the ASGI app on its own internal loop while
  pytest-asyncio fixtures run on a separate loop.
- The app's ``Database`` object is replaced with one pointing at the test DB
  so that routes / startup use the right schema.
- After every test the tables are truncated via a raw connection and the
  ``databases`` pool is disconnected so the next ``TestClient`` gets a clean
  connection.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from typing import Generator, AsyncGenerator
from urllib.parse import quote_plus
from fastapi.testclient import TestClient
from httpx import AsyncClient
import aiomysql
from databases import Database

from api.app import app
from api import database as db_module
from api.security import create_access_token, get_password_hash

# ── constants ─────────────────────────────────────────────────────────────
TEST_DB_NAME = "watheq_test_db"


# ── raw DB helpers (event-loop-safe) ──────────────────────────────────────
async def _raw_conn(db_name: str = TEST_DB_NAME):
    """Get a raw aiomysql connection to the test DB."""
    return await aiomysql.connect(
        host=db_module.DB_HOST,
        port=db_module.DB_PORT,
        user=db_module.DB_USER,
        password=db_module.DB_PASSWORD,
        db=db_name if db_name else None,
        autocommit=True,
    )


async def raw_execute(sql: str, params: tuple = None, db_name: str = TEST_DB_NAME):
    """Execute SQL and return lastrowid."""
    conn = await _raw_conn(db_name)
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return cur.lastrowid
    finally:
        conn.close()


async def raw_fetch_one(sql: str, params: tuple = None, db_name: str = TEST_DB_NAME):
    """Fetch one row as dict."""
    conn = await _raw_conn(db_name)
    try:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, params)
            return await cur.fetchone()
    finally:
        conn.close()


# ── session fixture: create / destroy test database ───────────────────────
@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():
    """Create a fresh test DB and point the app's Database object at it."""
    # Save originals
    original_db_name = db_module.DB_NAME
    original_database = db_module.database
    original_url = db_module.DATABASE_URL

    try:
        conn = await _raw_conn(db_name=None)
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{TEST_DB_NAME}`")
            await cur.execute(
                f"CREATE DATABASE `{TEST_DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.close()
    except Exception as e:
        pytest.skip(f"Could not setup test database: {e}")

    # Point the app at the test database
    db_module.DB_NAME = TEST_DB_NAME
    new_url = (
        f"mysql+aiomysql://{quote_plus(db_module.DB_USER)}:"
        f"{quote_plus(db_module.DB_PASSWORD)}"
        f"@{db_module.DB_HOST}:{db_module.DB_PORT}/{TEST_DB_NAME}"
    )
    db_module.DATABASE_URL = new_url
    db_module.database = Database(new_url)

    # Update the eagerly-created users collection to use the new database
    db_module.users.db = db_module.database

    # Reset all lazily-cached collection singletons so they get recreated
    # with the new database object on first access.
    db_module._document_hashes_collection = None
    db_module._biometric_audit_collection = None
    db_module._document_types_collection = None
    db_module._audit_logs_collection = None
    db_module._verifications_collection = None
    db_module._verification_steps_collection = None
    db_module._citizen_records_collection = None
    db_module._verification_notes_collection = None

    yield

    # Disconnect
    try:
        if db_module.database.is_connected:
            await db_module.database.disconnect()
    except Exception:
        pass

    # Drop test DB
    try:
        conn = await _raw_conn(db_name=None)
        async with conn.cursor() as cur:
            await cur.execute(f"DROP DATABASE IF EXISTS `{TEST_DB_NAME}`")
        conn.close()
    except Exception:
        pass

    # Restore originals
    db_module.DB_NAME = original_db_name
    db_module.DATABASE_URL = original_url
    db_module.database = original_database


# ── per-test fixture: truncate after each test ────────────────────────────
TABLES_TO_CLEAN = [
    "notifications",
    "verification_steps",
    "verifications",
    "citizen_records",
    "audit_logs",
    "document_types",
    "users",
]


@pytest_asyncio.fixture(autouse=True)
async def cleanup_between_tests(setup_test_database):
    """Truncate all tables after each test via raw connection."""
    yield

    # Truncate using raw aiomysql (safe regardless of event loop)
    try:
        conn = await _raw_conn()
        async with conn.cursor() as cur:
            await cur.execute("SET FOREIGN_KEY_CHECKS = 0")
            for table in TABLES_TO_CLEAN:
                try:
                    await cur.execute(f"TRUNCATE TABLE `{table}`")
                except Exception:
                    pass  # table may not exist yet
            await cur.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.close()
    except Exception:
        pass

    # Disconnect the databases pool so the next TestClient gets a fresh one
    try:
        if db_module.database.is_connected:
            await db_module.database.disconnect()
    except Exception:
        pass

    # Reset lazy collection caches so they reconnect via the fresh pool
    db_module._document_hashes_collection = None
    db_module._biometric_audit_collection = None
    db_module._document_types_collection = None
    db_module._audit_logs_collection = None
    db_module._verifications_collection = None
    db_module._verification_steps_collection = None
    db_module._citizen_records_collection = None
    db_module._verification_notes_collection = None


# ── client fixture ────────────────────────────────────────────────────────
@pytest.fixture
def client() -> Generator:
    """Synchronous FastAPI test client — triggers app startup/shutdown."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
async def async_client() -> AsyncGenerator:
    """Async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# ── data fixtures (raw aiomysql — avoids cross-loop issues) ──────────────
@pytest_asyncio.fixture
async def test_user(cleanup_between_tests):
    """Insert a regular test user and return its data dict."""
    hashed = get_password_hash("testpass123")
    uid = await raw_execute(
        "INSERT INTO users (name, username, email, password, role, is_active) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        ("Test User", "testuser", "test@example.com", hashed, "user", True),
    )
    return {
        "id": uid,
        "name": "Test User",
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "role": "user",
        "is_active": True,
    }


@pytest_asyncio.fixture
async def test_admin(cleanup_between_tests):
    """Insert an admin test user and return its data dict."""
    hashed = get_password_hash("adminpass123")
    uid = await raw_execute(
        "INSERT INTO users (name, username, email, password, role, is_active) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        ("Test Admin", "testadmin", "admin@example.com", hashed, "admin", True),
    )
    return {
        "id": uid,
        "name": "Test Admin",
        "username": "testadmin",
        "email": "admin@example.com",
        "password": "adminpass123",
        "role": "admin",
        "is_active": True,
    }


@pytest_asyncio.fixture
async def user_token(test_user):
    """JWT token for the regular test user (matches login endpoint format)."""
    return create_access_token(
        data={
            "sub": str(test_user["id"]),
            "email": test_user["email"],
            "role": test_user["role"],
        }
    )


@pytest_asyncio.fixture
async def admin_token(test_admin):
    """JWT token for the admin test user (matches login endpoint format)."""
    return create_access_token(
        data={
            "sub": str(test_admin["id"]),
            "email": test_admin["email"],
            "role": test_admin["role"],
        }
    )


@pytest_asyncio.fixture
async def test_document_type(cleanup_between_tests):
    """Insert a test document type and return its data dict."""
    did = await raw_execute(
        "INSERT INTO document_types (name, folder_name, is_active, requires_back_image) "
        "VALUES (%s, %s, %s, %s)",
        ("National ID", "identity", True, False),
    )
    return {
        "id": did,
        "name": "National ID",
        "folder_name": "identity",
        "is_active": True,
        "requires_back_image": False,
    }


@pytest.fixture
def mock_image_file():
    """Create a mock JPEG image for upload tests."""
    from io import BytesIO
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    buf.name = "test_image.jpg"
    return buf


@pytest.fixture
def storage_cleanup():
    """Clean up storage directory after tests."""
    import shutil

    yield

    test_storage = "storage/test_verifications"
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)


@pytest.fixture
def raw_db():
    """Expose the raw_execute helper for use inside test method bodies.

    Use this instead of ``db_module.database.execute()`` because the databases
    library connection lives on TestClient's internal event loop.
    """
    return raw_execute
