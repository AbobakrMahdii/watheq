#!/usr/bin/env python3
"""
Watheq AI Training Script (v3 — Binary Classifiers + Font Profiles)

خط أنابيب التدريب:
1. قراءة layout_config.yaml لكل نوع وثيقة
2. ربط الصور المرجعية بأسماء العناصر الصحيحة
3. توليد بيانات تدريب معززة (augmented + synthetic forgeries) بالألوان
4. تدريب مصنف ثنائي حقيقي (EfficientNet-B0) لكل عنصر — أصلي/مزور
5. تعلم خصائص الخطوط للمناطق النصية
6. حفظ الأوزان المدربة + ملفات الخطوط + التكوين

The trained models make verification decisions from what they LEARNED,
not by comparing against reference images at runtime.

Usage:
    python ai/train_ai.py --list
    python ai/train_ai.py --all
    python ai/train_ai.py --all --force
    python ai/train_ai.py --type identity
    python ai/train_ai.py --type identity --element logo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AI_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = AI_DIR.parent
REFERENCES_DIR = AI_DIR / "data" / "refrences"
TRAINING_DIR = AI_DIR / "data" / "training"
MODELS_DIR = AI_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
FONTS_DIR = MODELS_DIR / "fonts"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def to_relative_path(path: Path) -> str:
    """Convert absolute path to relative path from project root with forward slashes."""
    try:
        rel_path = path.resolve().relative_to(PROJECT_ROOT)
        # Use forward slashes for cross-platform compatibility
        return str(rel_path).replace("\\", "/")
    except ValueError:
        # If path is outside project root, return as-is
        return str(path).replace("\\", "/")


# Ensure imports
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR.parent))


def _detect_device() -> str:
    """Auto-detect CUDA GPU if available."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(
                f"\u2713 GPU detected: {gpu_name} ({vram:.1f} GB VRAM) \u2014 using CUDA"
            )
            return "cuda"
        else:
            logger.info("PyTorch installed but CUDA not available \u2014 using CPU")
            logger.info(
                "  Tip: reinstall with GPU support:  scripts\\setup_python.bat --gpu"
            )
    except ImportError:
        logger.error("PyTorch is not installed! Run: scripts\\setup_python.bat")
    return "cpu"


# ───────────────────────── Layout Config ─────────────────────────


def load_layout_config(doc_type: str) -> Optional[Dict[str, Any]]:
    """Load layout_config.yaml for a document type."""
    config_path = REFERENCES_DIR / doc_type / "layout_config.yaml"
    if not config_path.exists():
        logger.warning(f"No layout_config.yaml for {doc_type}")
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_elements_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all element definitions from layout config (visual + text)."""
    elements = []
    for ref_stem, elem_data in config.get("elements", {}).items():
        elements.append(
            {
                "ref_stem": ref_stem,
                "class_name": elem_data["class_name"],
                "ref_file": elem_data.get("ref_file"),
                "roi": elem_data.get("roi", {}),
                "tolerance": elem_data.get("tolerance", 0.10),
                "weight": elem_data.get("weight", 1.0),
                "critical": elem_data.get("critical", False),
                "type": elem_data.get("type", "visual"),
            }
        )
    for text_name, text_data in config.get("text_regions", {}).items():
        elements.append(
            {
                "ref_stem": text_name,
                "class_name": text_data["class_name"],
                "ref_file": None,
                "roi": text_data.get("roi", {}),
                "tolerance": text_data.get("tolerance", 0.12),
                "weight": text_data.get("weight", 1.0),
                "critical": text_data.get("critical", False),
                "type": "text",
            }
        )
    return elements


# ───────────────────────── Discovery ─────────────────────────


def discover_doc_types() -> List[str]:
    """Discover all document types from reference images directory."""
    if not REFERENCES_DIR.exists():
        return []
    return sorted(
        [
            item.name
            for item in REFERENCES_DIR.iterdir()
            if item.is_dir() and (item / "layout_config.yaml").exists()
        ]
    )


def get_reference_path(doc_type: str, ref_file: str) -> Optional[Path]:
    """Get the full path to a reference image file."""
    ref_path = REFERENCES_DIR / doc_type / ref_file
    if ref_path.exists():
        return ref_path
    # Try other extensions
    stem = Path(ref_file).stem
    doc_dir = REFERENCES_DIR / doc_type
    for ext in IMAGE_EXTENSIONS:
        p = doc_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ───────────────────────── Training Config ─────────────────────────


def load_training_config(doc_type: str) -> Optional[Dict[str, Any]]:
    """Load existing training config for a document type."""
    config_path = TRAINING_DIR / doc_type / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def save_training_config(doc_type: str, config: Dict[str, Any]) -> None:
    """Save training config for a document type."""
    output_dir = TRAINING_DIR / doc_type
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


# ───────────────────────── Data Generation ─────────────────────────


def generate_augmented_data(
    doc_type: str, ref_stem: str, ref_file: str, force: bool = False
) -> Dict[str, Any]:
    """Generate augmented training data for a visual element."""
    ref_path = get_reference_path(doc_type, ref_file)
    if ref_path is None:
        return {
            "status": "error",
            "message": f"Reference image not found: {ref_file}",
            "element": ref_stem,
        }

    output_dir = TRAINING_DIR / doc_type / ref_stem
    if output_dir.exists() and (output_dir / "genuine.txt").exists() and not force:
        # Count existing files
        gen_count = (
            len(list((output_dir / "out_genuine").glob("*.png")))
            if (output_dir / "out_genuine").exists()
            else 0
        )
        forg_count = (
            len(list((output_dir / "out_forged").glob("*.png")))
            if (output_dir / "out_forged").exists()
            else 0
        )
        return {
            "status": "skipped",
            "message": f"Data exists ({gen_count}g + {forg_count}f). Use --force to regenerate.",
            "element": ref_stem,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Generating augmented data for {ref_stem} (from {ref_file})...")

    try:
        from generate_synthetic_data import generate_separate

        generate_separate(ref_path, output_dir, n_genuine=400, n_forged=400)
        return {
            "status": "success",
            "element": ref_stem,
            "reference_path": to_relative_path(ref_path),
            "output_dir": to_relative_path(output_dir),
            "generated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to generate data for {ref_stem}: {e}")
        return {"status": "error", "message": str(e), "element": ref_stem}


# ───────────────────────── Classifier Training ─────────────────────────


def train_element_classifier(
    doc_type: str, ref_stem: str, class_name: str, layout_config: Dict
) -> Dict[str, Any]:
    """Train a binary classifier for one visual element."""
    from ai.models.element_classifier import ElementClassifier

    genuine_dir = TRAINING_DIR / doc_type / ref_stem / "out_genuine"
    forged_dir = TRAINING_DIR / doc_type / ref_stem / "out_forged"

    if not genuine_dir.exists() or not forged_dir.exists():
        return {
            "status": "error",
            "message": f"Training data not found for {ref_stem}",
            "class_name": class_name,
        }

    save_path = WEIGHTS_DIR / f"{doc_type}_{class_name}.pt"
    training_params = layout_config.get("training", {})

    device = _detect_device()
    logger.info(
        f"  Training classifier for {class_name} (from {ref_stem}) on {device.upper()}..."
    )

    classifier = ElementClassifier(device=device)
    result = classifier.train_from_dirs(
        genuine_dir=genuine_dir,
        forged_dir=forged_dir,
        save_path=save_path,
        epochs=training_params.get("epochs", 20),
        batch_size=training_params.get("batch_size", 32),
        lr=training_params.get("learning_rate", 1e-4),
        val_split=training_params.get("val_split", 0.2),
        patience=training_params.get("early_stopping_patience", 5),
    )

    if result["status"] == "success":
        logger.info(
            f"  ✓ {class_name}: val_acc={result['best_val_acc']:.1%} "
            f"({result['epochs_trained']} epochs)"
        )
    else:
        logger.error(f"  ✗ {class_name}: {result.get('message', 'unknown error')}")

    result["class_name"] = class_name
    result["ref_stem"] = ref_stem
    result["weight_path"] = to_relative_path(save_path)
    return result


# ───────────────────────── Font Profile Learning ─────────────────────────


def learn_text_font_profile(
    doc_type: str, class_name: str, roi: Dict[str, float]
) -> Dict[str, Any]:
    """Learn font profile for a text region from the full reference document."""
    import cv2
    import numpy as np
    from ai.models.font_analyzer import FontAnalyzer

    # Find the full document reference image
    ref_dir = REFERENCES_DIR / doc_type
    full_ref = None
    for ext in IMAGE_EXTENSIONS:
        for name in ["full", "document", "front"]:
            p = ref_dir / f"{name}{ext}"
            if p.exists():
                full_ref = p
                break
        if full_ref:
            break

    if full_ref is None:
        return {
            "status": "error",
            "class_name": class_name,
            "message": "No full document reference image found",
        }

    # Load and crop text region
    img = cv2.imdecode(np.fromfile(str(full_ref), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {
            "status": "error",
            "class_name": class_name,
            "message": f"Cannot read: {full_ref}",
        }

    h, w = img.shape[:2]
    x0 = int(roi.get("x", 0) * w)
    y0 = int(roi.get("y", 0) * h)
    rw = int(roi.get("w", 0.1) * w)
    rh = int(roi.get("h", 0.1) * h)
    x1 = min(w, x0 + rw)
    y1 = min(h, y0 + rh)

    text_crop = img[y0:y1, x0:x1]
    if text_crop.size == 0:
        return {
            "status": "error",
            "class_name": class_name,
            "message": "Empty text region crop",
        }

    # Learn font profile
    analyzer = FontAnalyzer()
    profile = analyzer.learn_font_profile(text_crop, class_name, doc_type)

    # Save profile
    profile_path = FONTS_DIR / f"{doc_type}_{class_name}.json"
    FontAnalyzer.save_profile(profile, profile_path)

    return {
        "status": "success",
        "class_name": class_name,
        "profile_path": to_relative_path(profile_path),
        "ink_density": profile.ink_density,
        "stroke_width": profile.stroke_width_mean,
    }


# ───────────────────────── Main Training Orchestrator ─────────────────────────


def train_doc_type(
    doc_type: str,
    specific_element: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Train all models for a document type.

    Steps:
    1. Load layout_config.yaml
    2. Map element names correctly
    3. Generate augmented training data (RGB)
    4. Train binary classifiers per visual element
    5. Learn font profiles for text regions
    6. Save training config with learned layout positions
    """
    layout_config = load_layout_config(doc_type)
    if layout_config is None:
        return {
            "status": "error",
            "doc_type": doc_type,
            "message": "No layout_config.yaml found. Create one in ai/data/refrences/{doc_type}/",
        }

    elements = get_elements_from_config(layout_config)
    if not elements:
        return {
            "status": "error",
            "doc_type": doc_type,
            "message": "No elements defined in layout_config.yaml",
        }

    if specific_element:
        elements = [
            e
            for e in elements
            if e["ref_stem"] == specific_element or e["class_name"] == specific_element
        ]
        if not elements:
            all_names = [e["ref_stem"] for e in get_elements_from_config(layout_config)]
            return {
                "status": "error",
                "doc_type": doc_type,
                "message": f"Element '{specific_element}' not found. Available: {all_names}",
            }

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {doc_type} ({len(elements)} elements)")
    logger.info(f"{'='*60}")

    augmentation_results = {}
    classifier_results = {}
    font_results = {}

    for elem in elements:
        ref_stem = elem["ref_stem"]
        class_name = elem["class_name"]
        elem_type = elem["type"]

        if elem_type == "visual" and elem.get("ref_file"):
            # Check if trained weight already exists → skip entirely
            pt_path = WEIGHTS_DIR / f"{doc_type}_{class_name}.pt"
            if pt_path.exists() and not force:
                logger.info(
                    f"\n  [{ref_stem} → {class_name}] ✓ Weight exists — skipped"
                )
                classifier_results[class_name] = {
                    "status": "skipped",
                    "message": f"{pt_path.name} already exists",
                    "class_name": class_name,
                    "ref_stem": ref_stem,
                    "weight_path": to_relative_path(pt_path),
                }
                continue

            # Weight missing → ensure augmented data exists, then train
            logger.info(f"\n  [{ref_stem} → {class_name}] Visual element")

            # Step 1: Generate augmented data (skips if already present)
            aug_result = generate_augmented_data(
                doc_type, ref_stem, elem["ref_file"], force=force
            )
            augmentation_results[ref_stem] = aug_result
            logger.info(
                f"    Data: {aug_result['status']} — {aug_result.get('message', 'OK')}"
            )

            # Step 2: Train binary classifier
            cls_result = train_element_classifier(
                doc_type, ref_stem, class_name, layout_config
            )
            classifier_results[class_name] = cls_result

        elif elem_type == "text":
            # Step 3: Learn font profile from the full reference document
            logger.info(f"\n  [{class_name}] Text region — learning font profile")
            font_result = learn_text_font_profile(doc_type, class_name, elem["roi"])
            font_results[class_name] = font_result
            logger.info(f"    Font: {font_result['status']}")

    # Step 4: Save training config with layout positions
    learned_layout = {}
    for elem in elements:
        learned_layout[elem["class_name"]] = {
            "roi": elem["roi"],
            "tolerance": elem["tolerance"],
            "weight": elem["weight"],
            "critical": elem["critical"],
            "type": elem["type"],
            "ref_stem": elem["ref_stem"],
        }

    config = {
        "doc_type": doc_type,
        "version": "3.0",
        "model": "ElementClassifier (EfficientNet-B0 binary)",
        "trained_at": datetime.now().isoformat(),
        "layout": learned_layout,
        "thresholds": {
            "pass_score": layout_config.get("thresholds", {}).get("pass_score", 0.95),
            "suspicious_score": layout_config.get("thresholds", {}).get(
                "suspicious_score", 0.70
            ),
        },
        "augmentation_results": augmentation_results,
        "classifier_results": {
            k: {sk: sv for sk, sv in v.items() if sk != "history"}
            for k, v in classifier_results.items()
        },
        "font_results": font_results,
    }
    save_training_config(doc_type, config)

    # Summary
    cls_success = sum(
        1 for r in classifier_results.values() if r.get("status") == "success"
    )
    font_success = sum(1 for r in font_results.values() if r.get("status") == "success")
    total = len(elements)

    logger.info(f"\n{'─'*60}")
    logger.info(
        f"  {doc_type}: {cls_success} classifiers trained, {font_success} font profiles learned"
    )
    logger.info(f"{'─'*60}")

    return {
        "status": "success" if (cls_success + font_success) == total else "partial",
        "doc_type": doc_type,
        "classifiers_trained": cls_success,
        "font_profiles_learned": font_success,
        "elements_total": total,
        "classifier_results": classifier_results,
        "font_results": font_results,
    }


def _all_weights_exist(doc_type: str) -> bool:
    """Return True if every visual element already has a trained .pt file."""
    config = load_layout_config(doc_type)
    if config is None:
        return False
    elements = get_elements_from_config(config)
    for elem in elements:
        if elem["type"] == "visual" and elem.get("ref_file"):
            pt_path = WEIGHTS_DIR / f"{doc_type}_{elem['class_name']}.pt"
            if not pt_path.exists():
                return False
    return True


def train_all(force: bool = False) -> List[Dict]:
    """Train all discovered document types.

    Skip logic (when force=False):
      – If ALL .pt weight files already exist for a doc type → skip entirely.
      – Otherwise, train only the missing elements (see train_doc_type).
    """
    doc_types = discover_doc_types()
    if not doc_types:
        logger.warning("No document types found with layout_config.yaml")
        return []

    results = []
    for doc_type in doc_types:
        if not force and _all_weights_exist(doc_type):
            logger.info(
                f"Skipping {doc_type} (all classifier weights already exist, "
                f"use --force to retrain)"
            )
            results.append(
                {
                    "status": "skipped",
                    "doc_type": doc_type,
                    "message": "All .pt weights present — nothing to train",
                }
            )
            continue

        result = train_doc_type(doc_type, force=force)
        results.append(result)

    return results


# ───────────────────────── CLI ─────────────────────────


def list_doc_types():
    """List all document types, elements, and training status."""
    doc_types = discover_doc_types()
    if not doc_types:
        # Also check for dirs without config
        all_dirs = (
            [d.name for d in REFERENCES_DIR.iterdir() if d.is_dir()]
            if REFERENCES_DIR.exists()
            else []
        )
        if all_dirs:
            print(f"\nDocument type folders found: {all_dirs}")
            print("But none have layout_config.yaml. Create one to enable training.")
        else:
            print("No document types found in ai/data/refrences/")
        return

    print(f"\n{'='*60}")
    print(f"  Watheq AI — Document Types (v3)")
    print(f"{'='*60}\n")

    for dt in doc_types:
        config = load_training_config(dt)
        layout = load_layout_config(dt)
        trained = "✓" if config and config.get("version") == "3.0" else "✗"
        version = config.get("version", "—") if config else "—"
        print(f"  {trained} {dt} (v{version})")

        if layout:
            elements = get_elements_from_config(layout)
            for elem in elements:
                class_name = elem["class_name"]
                elem_type = elem["type"]
                weight_path = WEIGHTS_DIR / f"{dt}_{class_name}.pt"
                font_path = FONTS_DIR / f"{dt}_{class_name}.json"

                if elem_type == "visual":
                    status = "✓" if weight_path.exists() else "✗"
                    print(
                        f"      {status} {elem['ref_stem']} → {class_name} [classifier]"
                    )
                else:
                    status = "✓" if font_path.exists() else "✗"
                    print(f"      {status} {class_name} [font profile]")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Watheq AI Training (v3 — Binary Classifiers + Font Profiles)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List document types")
    parser.add_argument(
        "--all", "-a", action="store_true", help="Train all document types"
    )
    parser.add_argument("--force", "-f", action="store_true", help="Force retrain")
    parser.add_argument("--type", "-t", type=str, help="Train specific document type")
    parser.add_argument("--element", "-e", type=str, help="Train specific element")
    args = parser.parse_args()

    if args.list:
        list_doc_types()
        return

    if args.all:
        results = train_all(force=args.force)
        print(f"\n{'─'*60}")
        print(f"  Training complete: {len(results)} document types processed")
        for r in results:
            status = r["status"]
            dt = r.get("doc_type", "?")
            if status == "success" or status == "partial":
                cls = r.get("classifiers_trained", 0)
                font = r.get("font_profiles_learned", 0)
                print(f"    {status}: {dt} ({cls} classifiers, {font} font profiles)")
            else:
                print(f"    {status}: {dt}")
        return

    if args.type:
        result = train_doc_type(
            args.type,
            specific_element=args.element,
            force=args.force,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
