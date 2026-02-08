"""Database Backup Script — سكربت النسخ الاحتياطي لقاعدة البيانات

Uses mysqldump to create a timestamped backup of the watheq_db MySQL database.
Reads connection settings from environment variables (same as api/database.py).

Usage:
  python scripts/backup_db.py                  # backup to scripts/backups/
  python scripts/backup_db.py --output /path   # backup to custom directory
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main() -> None:
    # Read DB config from env (same defaults as api/database.py)
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "watheq_db")

    # Output directory
    output_dir = Path("scripts/backups")
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_dir = Path(sys.argv[idx + 1])

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"watheq_backup_{timestamp}.sql"
    output_path = output_dir / filename

    # Build mysqldump command
    cmd = [
        "mysqldump",
        f"--host={db_host}",
        f"--port={db_port}",
        f"--user={db_user}",
        "--single-transaction",
        "--routines",
        "--triggers",
        db_name,
    ]

    env = os.environ.copy()
    if db_password:
        env["MYSQL_PWD"] = db_password

    print(f"⏳ جاري إنشاء نسخة احتياطية من {db_name}@{db_host}:{db_port} ...")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                check=True,
            )

        size_kb = output_path.stat().st_size / 1024
        print(f"✅ تم النسخ الاحتياطي بنجاح: {output_path} ({size_kb:.1f} KB)")

    except FileNotFoundError:
        print("❌ mysqldump غير موجود. تأكد من تثبيت MySQL client.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ فشل النسخ الاحتياطي: {e.stderr}")
        if output_path.exists():
            output_path.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
