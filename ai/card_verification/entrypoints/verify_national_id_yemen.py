#!/usr/bin/env python3
"""Orchestrate national_id_yemen_v1 verification with deterministic steps only."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout.strip()


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _artifact_if_exists(path: Path) -> str | None:
    return str(path) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Yemen National ID (deterministic wiring).")
    parser.add_argument("--image", required=True, help="Path to input card image.")
    parser.add_argument("--template", required=True, help="Path to layout.yaml.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--rectified", help="Path to a pre-rectified card image.")
    parser.add_argument("--selfie", help="Path to a selfie image (optional).")
    args = parser.parse_args()

    image_path = Path(args.image)
    template_path = Path(args.template)
    out_dir = Path(args.out_dir)
    rectified_arg = Path(args.rectified) if args.rectified else None
    selfie_path = Path(args.selfie) if args.selfie else None

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 0
    if not template_path.exists():
        print(f"Template not found: {template_path}")
        return 0
    if rectified_arg and not rectified_arg.exists():
        print(f"Rectified image not found: {rectified_arg}")
        return 0
    if selfie_path and not selfie_path.exists():
        print(f"Selfie image not found: {selfie_path}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).resolve().parents[1]
    rectifier = base_dir / "detect_and_rectify_card.py"
    layout_script = base_dir / "layout_verify.py"
    stamp_script = base_dir / "stamp_verify.py"
    emblem_script = base_dir / "emblem_verify.py"
    face_script = base_dir / "extract_photo_and_verify_face.py"

    rectified_path = rectified_arg
    rect_report_path = out_dir / "rectification_report.json"
    if rectified_path is None:
        if not rectifier.exists():
            print(f"Rectification script not found: {rectifier}")
            return 0
        code, out = _run(
            [
                sys.executable,
                str(rectifier),
                "--image",
                str(image_path),
                "--out_dir",
                str(out_dir),
            ]
        )
        if code != 0:
            print(f"Rectification failed: {out}")
            return 0
        rectified_path = out_dir / "card_rectified.png"
        if not rectified_path.exists():
            print("Rectification did not produce card_rectified.png")
            return 0

    # Layout verification (must pass to proceed)
    code, out = _run(
        [
            sys.executable,
            str(layout_script),
            "--image",
            str(image_path),
            "--rectified",
            str(rectified_path),
            "--template",
            str(template_path),
            "--out_dir",
            str(out_dir),
        ]
    )
    if code != 0:
        print(f"Layout verification failed: {out}")
        return 0

    layout_report_path = out_dir / "report.json"
    layout_report = _read_json(layout_report_path)
    layout_status = (layout_report.get("layout_status") or "").upper()

    rect_report = _read_json(rect_report_path)

    final_report: dict = {
        "input_image": str(image_path),
        "template": str(template_path),
        "rectified_image": str(rectified_path),
        "rectification": rect_report,
        "layout": layout_report,
        "stamp": None,
        "emblem": None,
        "face": None,
        "final_decision": None,
        "reasons": [],
        "artifacts": {
            "rectified_image": _artifact_if_exists(rectified_path),
            "rectification_report": _artifact_if_exists(rect_report_path),
            "layout_report": _artifact_if_exists(layout_report_path),
            "layout_overlay": _artifact_if_exists(out_dir / "overlay_layout_verify.png"),
            "layout_stamp_mask": _artifact_if_exists(out_dir / "stamp_mask.png"),
            "photo_crop": (layout_report.get("artifacts") or {}).get("photo_crop"),
        },
    }

    if layout_status != "PASS":
        final_report["final_decision"] = "LAYOUT_FAIL"
        reason = layout_report.get("reason") or "LAYOUT_MISMATCH"
        final_report["reasons"] = [reason]
        (out_dir / "final_report.json").write_text(
            json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Final decision: LAYOUT_FAIL")
        return 0

    # Stamp verification
    code, out = _run(
        [
            sys.executable,
            str(stamp_script),
            "--image",
            str(image_path),
            "--rectified",
            str(rectified_path),
            "--template",
            str(template_path),
            "--out_dir",
            str(out_dir),
        ]
    )
    if code != 0:
        print(f"Stamp verification failed: {out}")
        return 0
    stamp_report = _read_json(out_dir / "stamp_auth_report.json")
    final_report["stamp"] = stamp_report
    final_report["artifacts"].update(
        {
            "stamp_report": _artifact_if_exists(out_dir / "stamp_auth_report.json"),
            "stamp_crop": _artifact_if_exists(out_dir / "stamp_crop.png"),
            "stamp_ref": _artifact_if_exists(out_dir / "stamp_ref.png"),
            "stamp_mask": _artifact_if_exists(out_dir / "stamp_mask.png"),
            "stamp_match_vis": _artifact_if_exists(out_dir / "stamp_match_vis.png"),
        }
    )

    # Emblem verification
    code, out = _run(
        [
            sys.executable,
            str(emblem_script),
            "--image",
            str(image_path),
            "--rectified",
            str(rectified_path),
            "--template",
            str(template_path),
            "--out_dir",
            str(out_dir),
        ]
    )
    if code != 0:
        print(f"Emblem verification failed: {out}")
        return 0
    emblem_report = _read_json(out_dir / "emblem_auth_report.json")
    final_report["emblem"] = emblem_report
    final_report["artifacts"].update(
        {
            "emblem_report": _artifact_if_exists(out_dir / "emblem_auth_report.json"),
            "emblem_crop": _artifact_if_exists(out_dir / "emblem_crop.png"),
            "emblem_ref": _artifact_if_exists(out_dir / "emblem_ref.png"),
            "emblem_match_vis": _artifact_if_exists(out_dir / "emblem_match_vis.png"),
            "emblem_overlay": _artifact_if_exists(out_dir / "overlay_emblem.png"),
        }
    )

    face_report = None
    if selfie_path:
        code, out = _run(
            [
                sys.executable,
                str(face_script),
                "--image",
                str(image_path),
                "--template",
                str(template_path),
                "--selfie",
                str(selfie_path),
                "--out_dir",
                str(out_dir),
            ]
        )
        if code != 0:
            print(f"Face verification failed: {out}")
            return 0
        face_report = _read_json(out_dir / "face_verification_report.json")
        final_report["face"] = face_report
        final_report["artifacts"].update(
            {
                "face_report": _artifact_if_exists(out_dir / "face_verification_report.json"),
                "photo_crop": _artifact_if_exists(out_dir / "photo_crop.png"),
                "photo_overlay": _artifact_if_exists(out_dir / "overlay_photo.png"),
            }
        )

    # Fuse decision
    reasons: list[str] = []
    element_forged = False
    element_inconclusive = False

    stamp_decision = (stamp_report.get("decision") or "").upper()
    emblem_decision = (emblem_report.get("decision") or "").upper()

    if stamp_decision == "STAMP_FORGED" or emblem_decision == "EMBLEM_FORGED":
        element_forged = True
        if stamp_decision == "STAMP_FORGED":
            reasons.append("STAMP_FORGED")
        if emblem_decision == "EMBLEM_FORGED":
            reasons.append("EMBLEM_FORGED")

    if stamp_decision in ("STAMP_SUSPICIOUS", "ELEMENT_MISSING") or emblem_decision in (
        "EMBLEM_SUSPICIOUS",
        "ELEMENT_MISSING",
    ):
        element_inconclusive = True
        if stamp_decision in ("STAMP_SUSPICIOUS", "ELEMENT_MISSING"):
            reasons.append("STAMP_INCONCLUSIVE")
        if emblem_decision in ("EMBLEM_SUSPICIOUS", "ELEMENT_MISSING"):
            reasons.append("EMBLEM_INCONCLUSIVE")

    if face_report:
        face_decision = (face_report.get("decision") or "").upper()
        if face_decision == "FACE_MISMATCH":
            final_report["final_decision"] = "FACE_MISMATCH"
            reasons.append("FACE_MISMATCH")
        elif face_decision == "INCONCLUSIVE":
            element_inconclusive = True
            reasons.append("FACE_INCONCLUSIVE")

    if final_report["final_decision"] is None:
        if element_forged:
            final_report["final_decision"] = "ELEMENT_FORGED"
        elif element_inconclusive:
            final_report["final_decision"] = "ELEMENT_INCONCLUSIVE"
        else:
            final_report["final_decision"] = "VERIFIED_OK"

    final_report["reasons"] = reasons
    (out_dir / "final_report.json").write_text(
        json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Final decision: {final_report['final_decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
