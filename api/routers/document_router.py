import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

router = APIRouter(prefix="/document", tags=["document"])


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _compute_percent(report: Dict[str, Any]) -> Optional[float]:
    """
    Compute an interpretable percent score from LogoAndStamp JSON report(s).
    """
    try:
        verification_results = report.get("verification_results") or {}
        ssim = float(verification_results.get("ssim_score", 0.0))
        orb = float(verification_results.get("orb_match_ratio", 0.0))
        resnet_conf = float(verification_results.get("resnet_confidence", 0.0))

        base = _clamp01((ssim + orb + resnet_conf) / 3.0) * 100.0
        decision = (report.get("decision") or "").upper()

        # Conservative mapping: keep suspicious in mid-range.
        if decision == "FORGED":
            return float(min(base, 30.0))
        if decision == "SUSPICIOUS":
            return float(min(max(base, 40.0), 75.0))
        if decision == "AUTHENTIC":
            return float(min(max(base, 80.0), 99.0))
        return float(base)
    except Exception:
        return None


def _run_logo_and_stamp_unified(input_path: Path) -> Dict[str, Any]:
    """
    Run `ai/LogoAndStamp/main_unified.py` as a subprocess and read the JSON report.
    This avoids import-path issues inside that module.
    """
    repo_root = Path(__file__).resolve().parents[2]
    module_dir = repo_root / "ai" / "LogoAndStamp"
    if not module_dir.exists():
        raise RuntimeError("ai/LogoAndStamp not found")

    config_path = module_dir / "config.yaml"
    if not config_path.exists():
        raise RuntimeError("ai/LogoAndStamp/config.yaml not found")

    with tempfile.TemporaryDirectory(prefix="watheq_doc_verify_") as tmpdir:
        out_dir = Path(tmpdir) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run with cwd so `import main` works inside `main_unified.py`.
        cmd = [
            sys.executable,
            str(module_dir / "main_unified.py"),
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(out_dir),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(module_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LogoAndStamp failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )

        report_path = out_dir / f"{input_path.stem}_unified_report.json"
        if not report_path.exists():
            raise RuntimeError("Unified report not generated")

        return json.loads(report_path.read_text(encoding="utf-8"))


@router.post("/verify")
async def verify_document(
    file: UploadFile = File(...),
    _: Any = Depends(lambda: True),
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file upload")

        suffix = Path(file.filename or "").suffix or ".jpg"
        with tempfile.TemporaryDirectory(prefix="watheq_doc_") as tmpdir:
            input_path = Path(tmpdir) / f"document{suffix}"
            input_path.write_bytes(data)

            unified = _run_logo_and_stamp_unified(input_path)

            # Extract best-effort score from per-module reports when available.
            logo_report = (
                (unified.get("verifications") or {}).get("logo") or {}
            )
            stamp_report = (
                (unified.get("verifications") or {}).get("stamp") or {}
            )

            logo_percent = _compute_percent(logo_report)
            stamp_percent = _compute_percent(stamp_report)

            available = [p for p in [logo_percent, stamp_percent] if p is not None]
            authenticity_percent = float(sum(available) / len(available)) if available else None

            return {
                "final_decision": unified.get("final_decision"),
                "authenticity_percent": authenticity_percent,
                "logo": {"decision": logo_report.get("decision"), "percent": logo_percent, "report": logo_report},
                "stamp": {"decision": stamp_report.get("decision"), "percent": stamp_percent, "report": stamp_report},
                "raw": unified,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

