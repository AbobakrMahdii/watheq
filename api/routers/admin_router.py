from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from datetime import datetime, timezone
import csv
import io

from api.database import get_user_collection, database
from ..models import UserCreate, UserUpdate
from ..security import get_current_admin, get_current_super_admin
from ..security import get_password_hash


def _find_font_path() -> str | None:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    try:
        from pathlib import Path

        for path in candidates:
            if Path(path).exists():
                return path
    except Exception:
        return None
    return None


def _safe_pdf_text(value: object) -> str:
    text = "" if value is None else str(value)
    try:
        text.encode("latin-1")
        return text
    except Exception:
        return text.encode("latin-1", errors="ignore").decode("latin-1")


def _pdf_write_line(pdf, text: str, line_height: float = 5.0) -> None:
    width = pdf.w - pdf.l_margin - pdf.r_margin
    if width <= 0:
        return
    text = (text or "").replace("\n", " ").strip()
    if text:
        text = (
            text.replace("_", " _ ")
            .replace("/", " / ")
            .replace("\\", " \\ ")
            .replace("-", " - ")
        )
    if not text:
        pdf.ln(line_height)
        return

    words = text.split(" ")
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if pdf.get_string_width(candidate) <= width:
            line = candidate
            continue
        if line:
            pdf.cell(0, line_height, line, ln=1)
            line = ""
        # If the single word is too long, split by chars
        if pdf.get_string_width(word) <= width:
            line = word
        else:
            chunk = ""
            for ch in word:
                if pdf.get_string_width(chunk + ch) <= width:
                    chunk += ch
                else:
                    if chunk:
                        pdf.cell(0, line_height, chunk, ln=1)
                    chunk = ch
            line = chunk
    if line:
        pdf.cell(0, line_height, line, ln=1)


async def _resolve_user_doc(users, user_id: str) -> dict:
    """
    Resolve a user document by id (numeric), username, or email.
    Raises HTTPException 404 if not found.
    """
    uid_val = None
    try:
        uid_val = int(str(user_id).strip())
    except Exception:
        uid_val = None

    user = None
    if uid_val is not None:
        user = await users.find_one({"_id": uid_val})
    if user is None:
        # fallback: try username/email
        if isinstance(user_id, str) and "@" in user_id:
            user = await users.find_one({"email": user_id})
        else:
            user = await users.find_one({"username": user_id})
    if not user:
        raise HTTPException(404, "User not found")
    return user


router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


def to_public_user(user: dict) -> dict:
    user["_id"] = str(user["_id"])
    user.pop("password", None)
    is_active = user.get("is_active")
    user["is_active"] = True if is_active is None else bool(is_active)
    user["deleted_at"] = user.get("deleted_at")
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
            "is_active": True,
            "deleted_at": None,
        }
    )

    return {"message": "User created"}


# =========================
# Edit user data (admin + super)
# =========================
@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    body: UserUpdate,
    admin=Depends(get_current_admin),
):
    users = get_user_collection()
    user = await _resolve_user_doc(users, user_id)

    # Permission: only the first super admin can edit super_admin users
    if user.get("role") == "super_admin":
        requester_id = str(admin.get("sub"))
        first_sa_id = await _get_first_super_admin_id()
        if requester_id != first_sa_id:
            raise HTTPException(
                403, "Only the primary super admin can edit super admins"
            )

    update_fields = {}
    if body.name is not None:
        update_fields["name"] = body.name
    if body.username is not None:
        # Check uniqueness
        existing = await users.find_one({"username": body.username})
        if existing and existing["_id"] != user["_id"]:
            raise HTTPException(400, "Username already taken")
        update_fields["username"] = body.username
    if body.email is not None:
        existing = await users.find_one({"email": body.email})
        if existing and existing["_id"] != user["_id"]:
            raise HTTPException(400, "Email already registered")
        update_fields["email"] = body.email
    if body.password is not None:
        update_fields["password"] = get_password_hash(body.password)

    if not update_fields:
        raise HTTPException(400, "No fields to update")

    await users.update_one({"_id": user["_id"]}, {"$set": update_fields})
    return {"message": "User updated"}


async def _get_first_super_admin_id() -> str | None:
    """Return the _id of the first (earliest) super_admin in the DB."""
    users = get_user_collection()
    return await users.get_first_super_admin_id()


# =========================
# Admins list (super only)
# =========================
@router.get("/admins")
async def get_admins(super_admin=Depends(get_current_super_admin)):
    users = get_user_collection()
    first_sa_id = await _get_first_super_admin_id()
    result = []
    async for u in users.find({"role": {"$in": ["admin", "super_admin"]}}):
        pub = to_public_user(u)
        pub["is_first_super_admin"] = pub["_id"] == first_sa_id
        result.append(pub)
    return result


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
            "is_active": True,
            "deleted_at": None,
        }
    )

    return {"message": "Admin created"}


# =========================
# Promote user → admin
# =========================
@router.put("/users/{user_id}/make-admin")
async def make_admin(user_id: str, super_admin=Depends(get_current_super_admin)):
    users = get_user_collection()

    user = await _resolve_user_doc(users, user_id)

    await users.update_one({"_id": user["_id"]}, {"$set": {"role": "admin"}})

    return {"message": "User promoted to admin"}


# =========================
# Demote admin → user
# =========================
@router.put("/users/{user_id}/remove-admin")
async def remove_admin(user_id: str, super_admin=Depends(get_current_super_admin)):
    users = get_user_collection()
    user = await _resolve_user_doc(users, user_id)

    if user["role"] == "super_admin":
        requester_id = str(super_admin.get("sub"))
        first_sa_id = await _get_first_super_admin_id()
        if requester_id != first_sa_id:
            raise HTTPException(
                403, "Only the primary super admin can demote other super admins"
            )
        if str(user["_id"]) == first_sa_id:
            raise HTTPException(403, "Cannot demote the primary super admin")

    await users.update_one({"_id": user["_id"]}, {"$set": {"role": "user"}})

    return {"message": "Admin removed"}


# =========================
# Suspend / Activate user
# =========================
@router.put("/users/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    admin=Depends(get_current_admin),
):
    users = get_user_collection()
    user = await _resolve_user_doc(users, user_id)
    if user.get("role") == "super_admin":
        raise HTTPException(403, "Cannot suspend super admin")

    await users.update_one({"_id": user["_id"]}, {"$set": {"is_active": False}})
    return {"message": "User suspended"}


@router.put("/users/{user_id}/activate")
async def activate_user(
    user_id: str,
    admin=Depends(get_current_admin),
):
    users = get_user_collection()
    user = await _resolve_user_doc(users, user_id)
    if user.get("role") == "super_admin":
        raise HTTPException(403, "Cannot activate super admin")

    await users.update_one({"_id": user["_id"]}, {"$set": {"is_active": True}})
    return {"message": "User activated"}


# =========================
# Soft delete user
# =========================
@router.delete("/users/{user_id}")
async def soft_delete_user(
    user_id: str,
    admin=Depends(get_current_admin),
):
    users = get_user_collection()
    user = await _resolve_user_doc(users, user_id)
    if user.get("role") == "super_admin":
        raise HTTPException(403, "Cannot delete super admin")

    await users.update_one(
        {"_id": user["_id"]},
        {"$set": {"deleted_at": datetime.now(timezone.utc), "is_active": False}},
    )
    return {"message": "User soft-deleted"}


# =========================
# Analytics summary (admin + super)
# =========================
@router.get("/analytics")
async def get_analytics(
    admin=Depends(get_current_admin),
    date_from: str | None = None,
    date_to: str | None = None,
):
    return await _compute_analytics(date_from=date_from, date_to=date_to)


async def _compute_analytics(date_from: str | None = None, date_to: str | None = None) -> dict:
    if not database.is_connected:
        await database.connect()

    # Build optional date filter for verifications
    date_clause = ""
    values: dict = {}
    if date_from:
        date_clause += " AND v.created_at >= :date_from"
        values["date_from"] = date_from
    if date_to:
        date_clause += " AND v.created_at <= :date_to"
        values["date_to"] = date_to

    async def _count(query: str, vals: dict | None = None) -> int:
        row = await database.fetch_one(query, values=vals)
        return int(row["total"]) if row else 0

    total_users = await _count(
        "SELECT COUNT(*) as total FROM users WHERE role = 'user' AND deleted_at IS NULL"
    )
    total_admins = await _count(
        "SELECT COUNT(*) as total FROM users WHERE role IN ('admin','super_admin') AND deleted_at IS NULL"
    )
    total_verifications = await _count(
        f"SELECT COUNT(*) as total FROM verifications v WHERE 1=1{date_clause}", values
    )
    total_document_types = await _count("SELECT COUNT(*) as total FROM document_types")
    total_audit_logs = await _count("SELECT COUNT(*) as total FROM audit_logs")

    # Status breakdown
    status_rows = await database.fetch_all(
        f"SELECT v.status, COUNT(*) as cnt FROM verifications v WHERE 1=1{date_clause} GROUP BY v.status",
        values=values,
    )
    status_breakdown = {r["status"]: int(r["cnt"]) for r in status_rows}

    # By document type
    type_rows = await database.fetch_all(
        f"""SELECT dt.name as doc_type, COUNT(*) as cnt
            FROM verifications v
            LEFT JOIN document_types dt ON v.document_type_id = dt.id
            WHERE 1=1{date_clause}
            GROUP BY v.document_type_id, dt.name""",
        values=values,
    )
    by_document_type = [
        {"type": r["doc_type"] or "Unknown", "count": int(r["cnt"])} for r in type_rows
    ]

    # Daily time-series (last 30 days if no range)
    ts_clause = (
        date_clause
        if date_clause
        else " AND v.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)"
    )
    ts_values = values if date_clause else {}
    time_series = await database.fetch_all(
        f"""SELECT DATE(v.created_at) as day, v.status, COUNT(*) as cnt
            FROM verifications v WHERE 1=1{ts_clause}
            GROUP BY DATE(v.created_at), v.status
            ORDER BY day""",
        values=ts_values,
    )
    daily_data: dict = {}
    for r in time_series:
        day = str(r["day"])
        if day not in daily_data:
            daily_data[day] = {
                "date": day,
                "SUCCESS": 0,
                "FAILED": 0,
                "RUNNING": 0,
                "PENDING": 0,
            }
        daily_data[day][r["status"]] = int(r["cnt"])
    time_series_list = list(daily_data.values())

    # Failure reasons (top 10)
    failure_rows = await database.fetch_all(
        f"""SELECT JSON_EXTRACT(v.result_data, '$.failure_reason_code') as reason, COUNT(*) as cnt
            FROM verifications v
            WHERE v.status = 'FAILED'{date_clause}
            GROUP BY reason ORDER BY cnt DESC LIMIT 10""",
        values=values,
    )
    failure_reasons = [
        {"reason": (r["reason"] or "UNKNOWN").strip('"'), "count": int(r["cnt"])}
        for r in failure_rows
    ]

    # Average processing time (seconds)
    avg_row = await database.fetch_one(
        f"""SELECT AVG(TIMESTAMPDIFF(SECOND, v.start_time, v.end_time)) as avg_sec
            FROM verifications v
            WHERE v.end_time IS NOT NULL AND v.start_time IS NOT NULL{date_clause}""",
        values=values,
    )
    avg_processing_time = round(float(avg_row["avg_sec"] or 0), 1)

    return {
        "total_users": total_users,
        "total_admins": total_admins,
        "total_verifications": total_verifications,
        "total_authentications": total_verifications,
        "total_document_types": total_document_types,
        "total_audit_logs": total_audit_logs,
        "status_breakdown": status_breakdown,
        "by_document_type": by_document_type,
        "time_series": time_series_list,
        "failure_reasons": failure_reasons,
        "avg_processing_time_sec": avg_processing_time,
    }


@router.get("/analytics/export")
async def export_analytics(
    format: str = Query("csv", description="Export format: csv or pdf"),
    date_from: str | None = None,
    date_to: str | None = None,
    admin=Depends(get_current_admin),
):
    fmt = (format or "csv").lower()
    if fmt not in ("csv", "pdf"):
        raise HTTPException(status_code=400, detail="Unsupported export format")

    data = await _compute_analytics(date_from=date_from, date_to=date_to)

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["section", "key", "value"])

        for key in (
            "total_users",
            "total_admins",
            "total_verifications",
            "total_authentications",
            "total_document_types",
            "total_audit_logs",
            "avg_processing_time_sec",
        ):
            writer.writerow(["summary", key, data.get(key)])

        for status, count in (data.get("status_breakdown") or {}).items():
            writer.writerow(["status_breakdown", status, count])

        for row in data.get("by_document_type") or []:
            writer.writerow(["by_document_type", row.get("type"), row.get("count")])

        for row in data.get("failure_reasons") or []:
            writer.writerow(["failure_reasons", row.get("reason"), row.get("count")])

        for row in data.get("time_series") or []:
            writer.writerow(
                [
                    "time_series",
                    row.get("date"),
                    f"SUCCESS={row.get('SUCCESS',0)};FAILED={row.get('FAILED',0)};RUNNING={row.get('RUNNING',0)};PENDING={row.get('PENDING',0)}",
                ]
            )

        filename = f'analytics_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        return Response(
            output.getvalue().encode("utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # PDF
    from fpdf import FPDF

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    font_path = _find_font_path()
    if font_path:
        pdf.add_font("ReportFont", "", font_path, uni=True)
        pdf.set_font("ReportFont", size=10)
        safe = lambda s: "" if s is None else str(s)
    else:
        pdf.set_font("Helvetica", size=10)
        safe = _safe_pdf_text

    pdf.set_font_size(12)
    pdf.cell(0, 8, safe("Analytics Export"), ln=1)
    pdf.set_font_size(10)

    pdf.cell(0, 6, safe("Summary"), ln=1)
    for key in (
        "total_users",
        "total_admins",
        "total_verifications",
        "total_authentications",
        "total_document_types",
        "total_audit_logs",
        "avg_processing_time_sec",
    ):
        _pdf_write_line(pdf, safe(f"{key}: {data.get(key)}"))

    pdf.ln(2)
    pdf.cell(0, 6, safe("Status Breakdown"), ln=1)
    for status, count in (data.get("status_breakdown") or {}).items():
        _pdf_write_line(pdf, safe(f"{status}: {count}"))

    pdf.ln(2)
    pdf.cell(0, 6, safe("By Document Type"), ln=1)
    for row in data.get("by_document_type") or []:
        _pdf_write_line(pdf, safe(f"{row.get('type')}: {row.get('count')}"))

    pdf.ln(2)
    pdf.cell(0, 6, safe("Failure Reasons"), ln=1)
    for row in data.get("failure_reasons") or []:
        _pdf_write_line(pdf, safe(f"{row.get('reason')}: {row.get('count')}"))

    pdf.ln(2)
    pdf.cell(0, 6, safe("Time Series"), ln=1)
    for row in data.get("time_series") or []:
        _pdf_write_line(pdf, safe(f"Date: {row.get('date')}"))
        pdf.set_font_size(9)
        _pdf_write_line(pdf, safe(f"SUCCESS: {row.get('SUCCESS',0)}"))
        _pdf_write_line(pdf, safe(f"FAILED: {row.get('FAILED',0)}"))
        _pdf_write_line(pdf, safe(f"RUNNING: {row.get('RUNNING',0)}"))
        _pdf_write_line(pdf, safe(f"PENDING: {row.get('PENDING',0)}"))
        pdf.set_font_size(10)
        pdf.ln(1)

    filename = f'analytics_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf'
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1", errors="ignore")
    elif isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)

    return Response(
        pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
