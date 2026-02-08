#!/usr/bin/env python3
"""
Watheq AI Training Script (v2 — YOLOv8 + Siamese Network)

خط أنابيب التدريب:
1. تحميل الصور المرجعية من ai/data/refrences/{doc_type}/
2. توليد بيانات التدريب المعززة (augmented + synthetic forgeries)
3. تدريب YOLOv8 لكشف العناصر (اختياري — يتطلب YOLO annotations)
4. تدريب شبكة سيامية للتحقق من الأصالة
5. توليد تضمينات مرجعية لكل عنصر

Usage:
    python ai/train_ai.py --list
    python ai/train_ai.py --all
    python ai/train_ai.py --all --force
    python ai/train_ai.py --type identity
    python ai/train_ai.py --type identity --element logo_main
    python ai/train_ai.py --type identity --embeddings-only
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AI_DIR = Path(__file__).parent.resolve()
REFERENCES_DIR = AI_DIR / "data" / "refrences"
TRAINING_DIR = AI_DIR / "data" / "training"
MODELS_DIR = AI_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
EMBEDDINGS_DIR = MODELS_DIR / "embeddings"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MIN_SAMPLES_RECOMMENDED = 400

# Ensure imports
sys.path.insert(0, str(AI_DIR))
sys.path.insert(0, str(AI_DIR.parent))


def discover_doc_types() -> List[str]:
    """Discover all document types from reference images directory."""
    if not REFERENCES_DIR.exists():
        return []
    return sorted([item.name for item in REFERENCES_DIR.iterdir() if item.is_dir()])


def discover_elements(doc_type: str) -> List[str]:
    """Discover all reference elements for a document type."""
    doc_dir = REFERENCES_DIR / doc_type
    if not doc_dir.exists():
        return []
    return sorted(
        [
            item.stem
            for item in doc_dir.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_reference_path(doc_type: str, element: str) -> Optional[Path]:
    """Get the full path to reference image for an element."""
    doc_dir = REFERENCES_DIR / doc_type
    for ext in IMAGE_EXTENSIONS:
        ref_path = doc_dir / f"{element}{ext}"
        if ref_path.exists():
            return ref_path
    return None


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


def generate_augmented_data(
    doc_type: str, element: str, force: bool = False
) -> Dict[str, Any]:
    """
    Generate augmented training data for an element.

    Step 1: Creates genuine augmentations from reference
    Step 2: Creates synthetic forgeries
    """
    ref_path = get_reference_path(doc_type, element)
    if ref_path is None:
        return {
            "status": "error",
            "message": f"Reference image not found for {element}",
            "element": element,
        }

    output_dir = TRAINING_DIR / doc_type / element
    if output_dir.exists() and (output_dir / "genuine.txt").exists() and not force:
        return {
            "status": "success",
            "message": "Data already generated (use --force to overwrite)",
            "element": element,
            "reference_path": str(ref_path),
            "output_dir": str(output_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  Generating augmented data for %s...", element)

    try:
        from generate_synthetic_data import generate_separate

        generate_separate(ref_path, output_dir, n_genuine=400, n_forged=400)
        return {
            "status": "success",
            "element": element,
            "reference_path": str(ref_path),
            "output_dir": str(output_dir),
            "generated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to generate data for {element}: {e}")
        return {"status": "error", "message": str(e), "element": element}


def generate_reference_embeddings(doc_type: str, elements: List[str]) -> Dict[str, Any]:
    """
    Generate reference embeddings for each element using the Siamese verifier.

    Step 5: For each element, load genuine augmented samples, compute mean embedding,
    and save to ai/models/embeddings/{doc_type}/{element}.npy
    """
    import cv2
    import numpy as np

    from ai.models.siamese_verifier import SiameseVerifier

    # Load verifier (fallback mode if no trained model yet)
    model_path = WEIGHTS_DIR / f"siamese_{doc_type}.pt"
    verifier = SiameseVerifier(
        model_path=model_path if model_path.exists() else None,
        embeddings_dir=None,  # Don't load existing embeddings
    )

    results = {}
    for element in elements:
        ref_path = get_reference_path(doc_type, element)
        if ref_path is None:
            results[element] = {"status": "error", "message": "No reference image"}
            continue

        # Collect images: original reference + genuine augmentations
        images = []

        # Load original reference
        img = cv2.imread(str(ref_path))
        if img is not None:
            images.append(img)

        # Load genuine augmented samples (up to 50)
        genuine_dir = TRAINING_DIR / doc_type / element / "out_genuine"
        if genuine_dir.exists():
            for i, p in enumerate(sorted(genuine_dir.glob("*.png"))[:50]):
                aug_img = cv2.imread(str(p))
                if aug_img is not None:
                    # Convert grayscale to BGR if needed
                    if len(aug_img.shape) == 2:
                        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR)
                    images.append(aug_img)

        if not images:
            results[element] = {"status": "error", "message": "No images to embed"}
            continue

        try:
            embedding = verifier.generate_reference_embedding(
                images, element, doc_type, output_dir=EMBEDDINGS_DIR
            )
            results[element] = {
                "status": "success",
                "images_used": len(images),
                "embedding_shape": list(embedding.shape),
            }
        except Exception as e:
            results[element] = {"status": "error", "message": str(e)}

    return results


def train_doc_type(
    doc_type: str,
    specific_element: Optional[str] = None,
    force: bool = False,
    embeddings_only: bool = False,
) -> Dict[str, Any]:
    """
    Train all models for a document type.

    Steps:
    1. Discover elements from reference images
    2. Generate augmented training data (genuine + forged)
    3. Generate reference embeddings using Siamese verifier
    4. Save training config
    """
    elements = discover_elements(doc_type)
    if not elements:
        return {
            "status": "error",
            "doc_type": doc_type,
            "message": "No reference elements found",
        }

    if specific_element:
        if specific_element not in elements:
            return {
                "status": "error",
                "doc_type": doc_type,
                "message": f"Element '{specific_element}' not found. Available: {elements}",
            }
        elements = [specific_element]

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {doc_type} ({len(elements)} elements)")
    logger.info(f"{'='*60}")

    # Step 1 & 2: Generate augmented data
    augmentation_results = {}
    if not embeddings_only:
        for element in elements:
            logger.info(f"  [{element}] Generating augmented data...")
            result = generate_augmented_data(doc_type, element, force=force)
            augmentation_results[element] = result
            status = result["status"]
            logger.info(f"  [{element}] {status}: {result.get('message', 'OK')}")

    # Step 3: Generate reference embeddings
    logger.info(f"\n  Generating reference embeddings...")
    embedding_results = generate_reference_embeddings(doc_type, elements)

    for elem, res in embedding_results.items():
        status = res["status"]
        msg = res.get("message", f"{res.get('images_used', 0)} images")
        logger.info(f"  [{elem}] Embedding: {status} ({msg})")

    # Step 4: Save training config
    thresholds = {elem: 0.50 for elem in elements}
    config = {
        "doc_type": doc_type,
        "elements": elements,
        "thresholds": thresholds,
        "trained_at": datetime.now().isoformat(),
        "version": "2.0",
        "model": "YOLOv8 + Siamese (EfficientNet-B0)",
        "augmentation_results": augmentation_results,
        "embedding_results": embedding_results,
    }
    save_training_config(doc_type, config)

    success_count = sum(
        1 for r in embedding_results.values() if r["status"] == "success"
    )

    return {
        "status": "success" if success_count == len(elements) else "partial",
        "doc_type": doc_type,
        "elements_trained": success_count,
        "elements_total": len(elements),
        "embedding_results": embedding_results,
    }


def train_all(force: bool = False, embeddings_only: bool = False) -> List[Dict]:
    """Train all discovered document types."""
    doc_types = discover_doc_types()
    if not doc_types:
        logger.warning("No document types found in references directory")
        return []

    results = []
    for doc_type in doc_types:
        existing = load_training_config(doc_type)
        if (
            existing
            and not force
            and not embeddings_only
            and existing.get("version") == "2.0"
        ):
            logger.info(f"Skipping {doc_type} (already trained v2.0, use --force)")
            results.append(
                {
                    "status": "skipped",
                    "doc_type": doc_type,
                    "message": "Already trained v2.0",
                    "trained_at": existing.get("trained_at"),
                }
            )
            continue

        result = train_doc_type(doc_type, force=force, embeddings_only=embeddings_only)
        results.append(result)

    return results


def list_doc_types():
    """List all document types and their elements."""
    doc_types = discover_doc_types()
    if not doc_types:
        print("No document types found in ai/data/refrences/")
        return

    print(f"\n{'='*60}")
    print(f"  Watheq AI — Document Types")
    print(f"{'='*60}\n")

    for dt in doc_types:
        elements = discover_elements(dt)
        config = load_training_config(dt)
        trained = "✓" if config else "✗"
        version = config.get("version", "?") if config else "—"
        print(f"  {trained} {dt} (v{version})")
        for elem in elements:
            ref = get_reference_path(dt, elem)
            emb_path = EMBEDDINGS_DIR / dt / f"{elem}.npy"
            emb_status = "✓" if emb_path.exists() else "✗"
            print(f"      {emb_status} {elem} ({ref.suffix if ref else '?'})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Watheq AI Training (v2 — YOLOv8 + Siamese)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List document types")
    parser.add_argument(
        "--all", "-a", action="store_true", help="Train all document types"
    )
    parser.add_argument("--force", "-f", action="store_true", help="Force retrain")
    parser.add_argument("--type", "-t", type=str, help="Train specific document type")
    parser.add_argument("--element", "-e", type=str, help="Train specific element")
    parser.add_argument(
        "--embeddings-only", action="store_true", help="Only generate embeddings"
    )
    args = parser.parse_args()

    if args.list:
        list_doc_types()
        return

    if args.all:
        results = train_all(force=args.force, embeddings_only=args.embeddings_only)
        print(f"\n{'─'*60}")
        print(f"  Training complete: {len(results)} document types processed")
        for r in results:
            status = r["status"]
            dt = r.get("doc_type", "?")
            print(f"    {status}: {dt}")
        return

    if args.type:
        result = train_doc_type(
            args.type,
            specific_element=args.element,
            force=args.force,
            embeddings_only=args.embeddings_only,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
