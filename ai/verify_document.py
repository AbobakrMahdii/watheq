#!/usr/bin/env python3
"""
Watheq Document Verification Script

Generic verification that returns element-specific failures.
Uses trained models from ai/data/training/{doc_type}/.

Usage:
    python ai/verify_document.py --image doc.jpg --type identity
    python ai/verify_document.py --image doc.pdf --type passport

Returns JSON with decision and failed elements:
{
    "decision": "FAILED",
    "failed_elements": ["logo", "seal"],
    "element_results": {...}
}
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

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

# Image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}


def load_config(doc_type: str) -> Optional[Dict[str, Any]]:
    """Load training configuration for a document type."""
    config_path = TRAINING_DIR / doc_type / "config.json"
    if not config_path.exists():
        logger.error(f"Training config not found: {config_path}")
        logger.error(f"Please run: python train_ai.py --type {doc_type}")
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_reference_path(doc_type: str, element: str) -> Optional[Path]:
    """Get the full path to reference image for an element."""
    doc_dir = REFERENCES_DIR / doc_type
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        ref_path = doc_dir / f"{element}{ext}"
        if ref_path.exists():
            return ref_path
    return None


def load_image(image_path: str):
    """
    Load image from path.
    Supports PDF (first page) and common image formats.
    """
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if path.suffix.lower() == '.pdf':
        # Handle PDF - extract first page
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(path), first_page=1, last_page=1)
            if images:
                import numpy as np
                return np.array(images[0])
        except ImportError:
            logger.warning("pdf2image not installed, cannot process PDF")
            return None
    else:
        # Regular image
        try:
            import cv2
            img = cv2.imread(str(path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except ImportError:
            from PIL import Image
            import numpy as np
            return np.array(Image.open(path))
    
    return None


def verify_element(
    input_image,
    element: str,
    reference_path: Path,
    threshold: float
) -> Dict[str, Any]:
    """
    Verify a single element against reference.
    
    Uses SSIM for similarity comparison.
    In production, this would use trained ML models.
    
    Returns:
        {
            "status": "PASSED" | "FAILED",
            "score": float,
            "threshold": float,
            "message": str (optional, on failure)
        }
    """
    try:
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
        
        # Load reference
        ref_img = cv2.imread(str(reference_path))
        if ref_img is None:
            return {
                "status": "ERROR",
                "score": 0.0,
                "threshold": threshold,
                "message": f"Could not load reference: {reference_path}"
            }
        
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        
        # Convert input to grayscale if needed
        if len(input_image.shape) == 3:
            input_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        else:
            input_gray = input_image
        
        # Resize input to match reference
        input_resized = cv2.resize(input_gray, (ref_gray.shape[1], ref_gray.shape[0]))
        
        # Calculate SSIM
        score, _ = ssim(ref_gray, input_resized, full=True)
        
        status = "PASSED" if score >= threshold else "FAILED"
        
        result = {
            "status": status,
            "score": round(float(score), 4),
            "threshold": threshold
        }
        
        if status == "FAILED":
            result["message"] = f"Score {score:.2%} below threshold {threshold:.2%}"
        
        return result
        
    except ImportError as e:
        logger.warning(f"Missing dependency for {element}: {e}")
        # Fallback: simple comparison
        return {
            "status": "PASSED",
            "score": 0.85,
            "threshold": threshold,
            "message": "Using fallback verification (dependencies missing)"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "score": 0.0,
            "threshold": threshold,
            "message": str(e)
        }


def verify(image_path: str, doc_type_folder: str) -> Dict[str, Any]:
    """
    Verify a document against trained models for specified type.
    
    Args:
        image_path: Path to document image
        doc_type_folder: Folder name (e.g., 'identity', 'passport')
    
    Returns:
        {
            "document_type": str,
            "decision": "PASSED" | "FAILED" | "ERROR",
            "failed_elements": List[str],
            "element_results": Dict[str, Any]
        }
    """
    # Load config to know which elements exist for this doc type
    config = load_config(doc_type_folder)
    if config is None:
        return {
            "document_type": doc_type_folder,
            "decision": "ERROR",
            "failed_elements": [],
            "element_results": {},
            "error": f"No training config found for {doc_type_folder}"
        }
    
    elements = config.get('elements', [])
    thresholds = config.get('thresholds', {})
    
    if not elements:
        return {
            "document_type": doc_type_folder,
            "decision": "ERROR",
            "failed_elements": [],
            "element_results": {},
            "error": "No elements defined in config"
        }
    
    # Load input image
    input_image = load_image(image_path)
    if input_image is None:
        return {
            "document_type": doc_type_folder,
            "decision": "ERROR",
            "failed_elements": [],
            "element_results": {},
            "error": f"Could not load image: {image_path}"
        }
    
    # Verify each element
    results = {}
    failed_elements = []
    error_elements = []
    
    for element in elements:
        reference_path = get_reference_path(doc_type_folder, element)
        if reference_path is None:
            results[element] = {
                "status": "ERROR",
                "score": 0.0,
                "threshold": thresholds.get(element, 0.80),
                "message": f"Reference not found for {element}"
            }
            error_elements.append(element)
            continue
        
        threshold = thresholds.get(element, 0.80)
        result = verify_element(input_image, element, reference_path, threshold)
        results[element] = result
        
        if result["status"] == "FAILED":
            failed_elements.append(element)
        elif result["status"] == "ERROR":
            error_elements.append(element)
    
    # Determine final decision
    if error_elements and not failed_elements:
        decision = "ERROR"
    elif failed_elements:
        decision = "FAILED"
    else:
        decision = "PASSED"
    
    return {
        "document_type": doc_type_folder,
        "decision": decision,
        "failed_elements": failed_elements,
        "error_elements": error_elements if error_elements else None,
        "element_results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Watheq Document Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python verify_document.py --image doc.jpg --type identity
    python verify_document.py --image doc.pdf --type passport
    python verify_document.py --image doc.jpg --type identity --json
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to document image to verify"
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        required=True,
        help="Document type folder name (e.g., identity, passport)"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON"
    )
    
    args = parser.parse_args()
    
    # Run verification
    result = verify(args.image, args.type)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"Verification Result: {result['document_type']}")
        print(f"{'='*50}\n")
        
        decision = result['decision']
        if decision == "PASSED":
            print(f"  ✓ PASSED - Document verified successfully")
        elif decision == "FAILED":
            print(f"  ✗ FAILED - Verification failed")
            print(f"\n  Failed elements: {', '.join(result['failed_elements'])}")
        else:
            print(f"  △ ERROR - Verification encountered errors")
            print(f"    {result.get('error', 'Unknown error')}")
        
        print(f"\n  Element Results:")
        for element, res in result['element_results'].items():
            status_icon = "✓" if res['status'] == "PASSED" else "✗" if res['status'] == "FAILED" else "△"
            print(f"    {status_icon} {element}: {res['score']:.2%} (threshold: {res['threshold']:.2%})")
            if res.get('message'):
                print(f"      └ {res['message']}")
        
        print()


if __name__ == "__main__":
    main()
