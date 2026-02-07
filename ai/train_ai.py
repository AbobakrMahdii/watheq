#!/usr/bin/env python3
"""
Watheq AI Training Script

Unified training for all document types in ai/data/refrences/.
Each document type can have different elements (logo, seal, barcode, full, signature, etc.)

Usage:
    python ai/train_ai.py --list              # List available doc types
    python ai/train_ai.py --all               # Train all doc types
    python ai/train_ai.py --all --force       # Force retraining
    python ai/train_ai.py --type identity     # Train specific type
    python ai/train_ai.py --type identity --element logo  # Train specific element
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from generate_synthetic_data import generate_separate
except ImportError:
    # If running from root, ensure ai module is in path or run as module
    sys.path.append(str(Path(__file__).parent))
    from generate_synthetic_data import generate_separate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================
# Configuration
# ===========================
AI_DIR = Path(__file__).parent.resolve()
REFERENCES_DIR = AI_DIR / "data" / "refrences"
TRAINING_DIR = AI_DIR / "data" / "training"

# Minimum recommended training samples
MIN_SAMPLES_RECOMMENDED = 400

# Image extensions to look for
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def discover_doc_types() -> List[str]:
    """
    List all available document types in the references folder.
    Each subfolder in refrences/ is a document type.
    """
    if not REFERENCES_DIR.exists():
        logger.error(f"References directory not found: {REFERENCES_DIR}")
        return []
    
    doc_types = []
    for item in REFERENCES_DIR.iterdir():
        if item.is_dir():
            doc_types.append(item.name)
    
    return sorted(doc_types)


def discover_elements(doc_type: str) -> List[str]:
    """
    List all elements for a document type.
    Elements are reference images (logo.jpeg, seal.png, etc.)
    Returns element names without extensions.
    """
    doc_dir = REFERENCES_DIR / doc_type
    if not doc_dir.exists():
        logger.error(f"Document type directory not found: {doc_dir}")
        return []
    
    elements = []
    for item in doc_dir.iterdir():
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            elements.append(item.stem)  # Name without extension
    
    return sorted(elements)


def get_reference_path(doc_type: str, element: str) -> Optional[Path]:
    """
    Get the full path to reference image for an element.
    Searches for any valid image extension.
    """
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
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def save_training_config(doc_type: str, config: Dict[str, Any]) -> None:
    """Save training configuration for a document type."""
    output_dir = TRAINING_DIR / doc_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Saved config to {config_path}")


def train_element(doc_type: str, element: str, force: bool = False) -> Dict[str, Any]:
    """
    Train/prepare model for a specific element of a document type.
    Generates synthetic data (genuine/forged) for the element.
    """
    ref_path = get_reference_path(doc_type, element)
    if ref_path is None:
        return {
            "status": "error",
            "message": f"Reference image not found for {element}",
            "element": element
        }
    
    output_dir = TRAINING_DIR / doc_type / element
    
    # Check if already generated
    if output_dir.exists() and (output_dir / "genuine.txt").exists() and not force:
         return {
            "status": "success",
            "message": "Data already generated (use --force to overwrite)",
            "element": element,
            "reference_path": str(ref_path),
            "output_dir": str(output_dir),
            "threshold": 0.80
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  generating data for {element}...")
    try:
        # Generate synthetic data (400 genuine, 400 forged for now)
        # We redirect stdout/stderr to avoid cluttering the output unless error
        generate_separate(ref_path, output_dir, n_genuine=400, n_forged=400)
        
        result = {
            "status": "success",
            "element": element,
            "reference_path": str(ref_path),
            "output_dir": str(output_dir),
            "trained_at": datetime.now().isoformat(),
            "threshold": 0.80
        }
        logger.info(f"  ✓ Processed {element}")
        
        # TODO: Here we would trigger the actual model training
        # e.g., train_resnet(output_dir)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate data for {element}: {e}")
        return {
            "status": "error",
            "message": str(e),
            "element": element
        }


def train_doc_type(doc_type: str, specific_element: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    """
    Train all elements for a document type.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Training: {doc_type}")
    logger.info(f"{'='*50}")
    
    elements = discover_elements(doc_type)
    if not elements:
        return {
            "status": "error",
            "doc_type": doc_type,
            "message": "No reference elements found"
        }
    
    if specific_element:
        if specific_element not in elements:
            return {
                "status": "error",
                "doc_type": doc_type,
                "message": f"Element '{specific_element}' not found. Available: {elements}"
            }
        elements = [specific_element]
    
    logger.info(f"Elements to train: {elements}")
    
    results = {}
    thresholds = {}
    
    for element in elements:
        result = train_element(doc_type, element, force=force)
        results[element] = result
        if result["status"] == "success":
            thresholds[element] = result.get("threshold", 0.80)
    
    # Save training configuration
    config = {
        "doc_type": doc_type,
        "elements": elements,
        "thresholds": thresholds,
        "trained_at": datetime.now().isoformat(),
        "version": "1.0",
        "results": results
    }
    save_training_config(doc_type, config)
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    
    return {
        "status": "success" if success_count == len(elements) else "partial",
        "doc_type": doc_type,
        "elements_trained": success_count,
        "elements_total": len(elements),
        "results": results
    }


def train_all(force: bool = False) -> List[Dict[str, Any]]:
    """Train models for all available document types. Skips already trained types unless force=True."""
    doc_types = discover_doc_types()
    
    if not doc_types:
        logger.error("No document types found in references folder!")
        return []
    
    logger.info(f"Found {len(doc_types)} document types: {doc_types}")
    
    results = []
    skipped = 0
    
    for doc_type in doc_types:
        # Check if already trained
        existing_config = load_training_config(doc_type)
        if existing_config and not force:
            logger.info(f"  ⏭ Skipping {doc_type} (already trained on {existing_config.get('trained_at', 'unknown')})")
            results.append({
                "status": "skipped",
                "doc_type": doc_type,
                "message": "Already trained",
                "trained_at": existing_config.get('trained_at')
            })
            skipped += 1
            continue
        
        result = train_doc_type(doc_type, force=force)
        results.append(result)
    
    if skipped == len(doc_types):
        logger.info("All document types already trained. Use --force to re-train.")
    
    return results


def list_doc_types() -> None:
    """Print available document types and their elements."""
    doc_types = discover_doc_types()
    
    if not doc_types:
        print("No document types found in:", REFERENCES_DIR)
        return
    
    print(f"\n{'='*50}")
    print("Available Document Types")
    print(f"{'='*50}\n")
    
    for doc_type in doc_types:
        elements = discover_elements(doc_type)
        config = load_training_config(doc_type)
        
        status = "✓ Trained" if config else "○ Not trained"
        print(f"  {doc_type}/ {status}")
        
        for element in elements:
            ref_path = get_reference_path(doc_type, element)
            print(f"    - {element} ({ref_path.name if ref_path else 'missing'})")
        
        if config:
            print(f"    Last trained: {config.get('trained_at', 'unknown')}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Watheq AI Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_ai.py --list
    python train_ai.py --all
    python train_ai.py --all --force
    python train_ai.py --type identity
    python train_ai.py --type identity --element logo
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available document types and their elements"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Train all document types"
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-training even if already exists"
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        help="Train specific document type"
    )
    
    parser.add_argument(
        "--element", "-e",
        type=str,
        help="Train specific element (requires --type)"
    )
    
    args = parser.parse_args()
    
    # Ensure at least one action is specified
    if not any([args.list, args.all, args.type]):
        parser.print_help()
        sys.exit(1)
    
    # List mode
    if args.list:
        list_doc_types()
        return
    
    # Train all
    if args.all:
        results = train_all(force=args.force)
        
        print(f"\n{'='*50}")
        print("Training Complete")
        print(f"{'='*50}\n")
        
        for r in results:
            status = "✓" if r["status"] == "success" else "△" if r["status"] == "partial" else "⏭" if r["status"] == "skipped" else "✗"
            print(f"  {status} {r['doc_type']}: {r.get('elements_trained', 0)}/{r.get('elements_total', 0)} elements {r.get('message', '')}")
        
        return
    
    # Train specific type
    if args.type:
        result = train_doc_type(args.type, args.element, force=args.force)
        
        print(f"\n{'='*50}")
        print("Training Complete")
        print(f"{'='*50}\n")
        
        if result["status"] == "error":
            print(f"  ✗ Error: {result.get('message')}")
        else:
            print(f"  ✓ {result['doc_type']}: {result['elements_trained']}/{result['elements_total']} elements trained")
        
        return


if __name__ == "__main__":
    main()
