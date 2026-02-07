import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Request
from api.services.audit_log_service import log_file_event

router = APIRouter(prefix="/document", tags=["document"])


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _compute_percent(element_results: Dict[str, Any]) -> Optional[float]:
    """
    Compute authenticity percent from element verification scores.
    """
    try:
        scores = [r.get("score", 0) for r in element_results.values() if r.get("status") != "ERROR"]
        return (sum(scores) / len(scores) * 100) if scores else None
    except Exception:
        return None


def _run_verify_document(input_path: Path, doc_type_folder: str = "identity") -> Dict[str, Any]:
    """
    Run `ai/verify_document.py` as a subprocess and return the JSON result.
    """
    repo_root = Path(__file__).resolve().parents[2]
    verify_script = repo_root / "ai" / "verify_document.py"
    
    if not verify_script.exists():
        raise RuntimeError("ai/verify_document.py not found")

    cmd = [
        sys.executable,
        str(verify_script),
        "--image", str(input_path),
        "--type", doc_type_folder,
        "--json",
    ]
    
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    if proc.returncode != 0:
        raise RuntimeError(
            f"AI verification failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    return json.loads(proc.stdout)


@router.post("/verify")
async def verify_document(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = "identity",
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

            result = _run_verify_document(input_path, doc_type)

            element_results = result.get("element_results", {})
            authenticity_percent = _compute_percent(element_results)

            request.state.audit_logged = True
            await log_file_event(
                request,
                status="success",
                operation_type="Verify",
                module="document",
                file_name=file.filename,
                file_size=len(data) if data is not None else None,
            )
            return {
                "final_decision": result.get("decision"),
                "authenticity_percent": authenticity_percent,
                "failed_elements": result.get("failed_elements", []),
                "element_results": element_results,
            }
    except HTTPException:
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason="Validation error",
            operation_type="Verify",
            module="document",
            file_name=file.filename,
            file_size=len(data) if "data" in locals() and data is not None else None,
        )
        raise
    except Exception as e:
        request.state.audit_logged = True
        await log_file_event(
            request,
            status="failed",
            failure_reason=f"{type(e).__name__}: {e}",
            operation_type="Verify",
            module="document",
            file_name=file.filename,
            file_size=len(data) if "data" in locals() and data is not None else None,
        )
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
