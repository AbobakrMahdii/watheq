"""
SSIM (Structural Similarity Index) matcher for logo verification.

This module compares the extracted logo patch against the reference logo
using SSIM with optional CLAHE preprocessing for improved robustness.
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8),
    data_range: Optional[float] = None
) -> float:
    """
    Compute SSIM between two images.
    
    Automatically handles:
    - Size normalization (resizes second image to match first)
    - Grayscale conversion
    - Optional CLAHE contrast normalization
    
    Args:
        img1: First image (reference)
        img2: Second image (query)
        use_clahe: Whether to apply CLAHE before comparison
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid: CLAHE tile grid size
        data_range: Data range for SSIM (default: auto-detect)
        
    Returns:
        SSIM score in range [-1, 1], where 1 = identical
    """
    # Convert to grayscale if needed
    gray1 = _to_grayscale(img1)
    gray2 = _to_grayscale(img2)
    
    # Ensure same size
    gray1, gray2 = _ensure_same_size(gray1, gray2)
    
    # Apply CLAHE if requested
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid
        )
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)
    
    # Determine data range
    if data_range is None:
        data_range = gray1.max() - gray1.min()
        if data_range == 0:
            data_range = 255.0
    
    # Compute SSIM
    score = ssim(gray1, gray2, data_range=data_range)
    
    return float(score)


def compute_ssim_with_map(
    img1: np.ndarray,
    img2: np.ndarray,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8)
) -> Tuple[float, np.ndarray]:
    """
    Compute SSIM and return the similarity map.
    
    The similarity map shows local similarity values, useful for
    identifying regions of difference.
    
    Args:
        img1: First image (reference)
        img2: Second image (query)
        use_clahe: Whether to apply CLAHE before comparison
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid: CLAHE tile grid size
        
    Returns:
        Tuple of (overall_ssim_score, similarity_map)
    """
    # Convert to grayscale if needed
    gray1 = _to_grayscale(img1)
    gray2 = _to_grayscale(img2)
    
    # Ensure same size
    gray1, gray2 = _ensure_same_size(gray1, gray2)
    
    # Apply CLAHE if requested
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid
        )
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)
    
    # Compute SSIM with full output
    score, ssim_map = ssim(
        gray1, gray2, 
        data_range=255,
        full=True
    )
    
    return float(score), ssim_map


def classify_ssim_signal(
    ssim_score: float,
    thresholds: Dict[str, float]
) -> str:
    """
    Classify SSIM score into signal categories.
    
    Args:
        ssim_score: SSIM score
        thresholds: Dict with 'strong_genuine' and 'suspicious' thresholds
        
    Returns:
        'strong_genuine', 'suspicious', or 'forged'
    """
    strong_thresh = thresholds.get('strong_genuine', 0.90)
    suspicious_thresh = thresholds.get('suspicious', 0.70)
    
    if ssim_score >= strong_thresh:
        return 'strong_genuine'
    elif ssim_score >= suspicious_thresh:
        return 'suspicious'
    else:
        return 'forged'


def get_ssim_difference_regions(
    ssim_map: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Get a binary mask of regions with low similarity.
    
    Useful for visualizing where the images differ.
    
    Args:
        ssim_map: Similarity map from compute_ssim_with_map
        threshold: Threshold below which to mark as different
        
    Returns:
        Binary mask where True indicates low-similarity regions
    """
    # Normalize map to [0, 1] if needed
    if ssim_map.min() < 0:
        ssim_map = (ssim_map + 1) / 2
    
    return ssim_map < threshold


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if not already."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_same_size(
    img1: np.ndarray, 
    img2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure two images have the same dimensions."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if (h1, w1) == (h2, w2):
        return img1, img2
    
    # Resize img2 to match img1
    img2_resized = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
    return img1, img2_resized


def batch_compute_ssim(
    reference: np.ndarray,
    queries: list,
    use_clahe: bool = True
) -> list:
    """
    Compute SSIM for multiple query images against a reference.
    
    Args:
        reference: Reference image
        queries: List of query images
        use_clahe: Whether to apply CLAHE
        
    Returns:
        List of SSIM scores
    """
    scores = []
    for query in queries:
        score = compute_ssim(reference, query, use_clahe=use_clahe)
        scores.append(score)
    return scores
