"""
Image loading utilities for the Wathiq verification pipeline.

Supports loading images from various formats (JPG, PNG, PDF) and
provides preprocessing functions for consistent input handling.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Optional PDF support - graceful fallback if not installed
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


def load_image(
    path: Union[str, Path],
    grayscale: bool = False,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to image file (JPG, PNG, or first page of PDF)
        grayscale: If True, convert to grayscale
        target_size: Optional (width, height) to resize to
        
    Returns:
        Loaded image as numpy array (BGR or grayscale)
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file format is unsupported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    suffix = path.suffix.lower()
    
    # Handle PDF files
    if suffix == '.pdf':
        if not PDF_SUPPORT:
            raise ValueError("PDF support requires pdf2image package")
        images = pdf_to_images(str(path))
        if not images:
            raise ValueError(f"Could not extract images from PDF: {path}")
        img = images[0]  # Use first page
    else:
        # Standard image formats
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
    
    # Convert to grayscale if requested
    if grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize if target size specified
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    return img


def pdf_to_images(
    pdf_path: str,
    dpi: int = 200,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None
) -> List[np.ndarray]:
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering (default 200)
        first_page: First page to convert (1-indexed, default: first)
        last_page: Last page to convert (1-indexed, default: last)
        
    Returns:
        List of images as numpy arrays (BGR format)
        
    Raises:
        ValueError: If PDF support is not available
    """
    if not PDF_SUPPORT:
        raise ValueError("PDF support requires pdf2image package. Install with: pip install pdf2image")
    
    pil_images = convert_from_path(
        pdf_path, 
        dpi=dpi,
        first_page=first_page,
        last_page=last_page
    )
    
    # Convert PIL images to numpy BGR format
    images = []
    for pil_img in pil_images:
        # PIL is RGB, OpenCV uses BGR
        rgb_array = np.array(pil_img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        images.append(bgr_array)
    
    return images


def preprocess_image(
    img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Preprocess an image for verification.
    
    Applies resizing and optional CLAHE (Contrast Limited Adaptive 
    Histogram Equalization) for better contrast normalization.
    
    Args:
        img: Input image (BGR or grayscale)
        target_size: (width, height) to resize to
        use_clahe: Whether to apply CLAHE normalization
        clahe_clip_limit: CLAHE clip limit parameter
        clahe_tile_grid: CLAHE tile grid size
        
    Returns:
        Preprocessed image (grayscale)
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Resize to target size
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE for contrast normalization
    if use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid
        )
        resized = clahe.apply(resized)
    
    return resized


def ensure_same_size(
    img1: np.ndarray,
    img2: np.ndarray,
    reference: str = 'first'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure two images have the same dimensions.
    
    Args:
        img1: First image
        img2: Second image
        reference: Which image to use as reference size
                   ('first', 'second', 'smaller', 'larger')
    
    Returns:
        Tuple of (resized_img1, resized_img2)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if (h1, w1) == (h2, w2):
        return img1, img2
    
    if reference == 'first':
        target_size = (w1, h1)
    elif reference == 'second':
        target_size = (w2, h2)
    elif reference == 'smaller':
        target_size = (min(w1, w2), min(h1, h2))
    else:  # 'larger'
        target_size = (max(w1, w2), max(h1, h2))
    
    resized1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)
    
    return resized1, resized2


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if not already.
    
    Args:
        img: Input image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_for_model(
    img: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Normalize image for deep learning model input.
    
    Applies ImageNet normalization (mean subtraction, std division).
    
    Args:
        img: Input image (BGR)
        target_size: Target size for model
        
    Returns:
        Normalized image as float32 array, shape (C, H, W)
    """
    # Resize
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    
    # Transpose to (C, H, W) for PyTorch
    normalized = normalized.transpose(2, 0, 1)
    
    return normalized
