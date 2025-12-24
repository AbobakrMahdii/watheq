"""
SHA-256 hashing utilities for image integrity and duplicate detection.

This module provides functions for computing cryptographic hashes of
images and files for audit trails and integrity verification.
"""

import hashlib
from pathlib import Path
from typing import Union

import cv2
import numpy as np


def compute_hash(image: np.ndarray) -> str:
    """
    Compute SHA-256 hash of an image array.
    
    The hash is computed on the raw bytes of the numpy array,
    making it sensitive to any pixel-level changes.
    
    Args:
        image: Image as numpy array
        
    Returns:
        SHA-256 hash as hexadecimal string (prefixed with 'sha256:')
    """
    # Ensure consistent byte representation
    image_bytes = image.tobytes()
    hash_value = hashlib.sha256(image_bytes).hexdigest()
    return f"sha256:{hash_value}"


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Reads the file in chunks to handle large files efficiently.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash as hexadecimal string (prefixed with 'sha256:')
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sha256_hash = hashlib.sha256()
    
    with open(path, 'rb') as f:
        # Read in 64KB chunks
        for chunk in iter(lambda: f.read(65536), b''):
            sha256_hash.update(chunk)
    
    return f"sha256:{sha256_hash.hexdigest()}"


def compute_normalized_hash(
    image: np.ndarray,
    size: tuple = (64, 64)
) -> str:
    """
    Compute a size-normalized hash for scale-invariant comparison.
    
    This hash is useful for detecting near-duplicate images that
    may have been resized.
    
    Args:
        image: Image as numpy array
        size: Target size for normalization (default 64x64)
        
    Returns:
        SHA-256 hash of normalized image
    """
    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to standard size
    normalized = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    
    return compute_hash(normalized)


def verify_hash(
    image: np.ndarray,
    expected_hash: str
) -> bool:
    """
    Verify that an image matches an expected hash.
    
    Args:
        image: Image to verify
        expected_hash: Expected SHA-256 hash (with or without prefix)
        
    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = compute_hash(image)
    
    # Handle both prefixed and non-prefixed hashes
    expected_clean = expected_hash.replace('sha256:', '')
    actual_clean = actual_hash.replace('sha256:', '')
    
    return expected_clean == actual_clean


def is_duplicate(
    image: np.ndarray,
    known_hashes: set
) -> bool:
    """
    Check if an image is a duplicate based on known hashes.
    
    Args:
        image: Image to check
        known_hashes: Set of known hash values
        
    Returns:
        True if image hash exists in known_hashes
    """
    image_hash = compute_hash(image)
    
    # Check both prefixed and non-prefixed versions
    hash_clean = image_hash.replace('sha256:', '')
    
    return (image_hash in known_hashes or 
            hash_clean in known_hashes or
            f"sha256:{hash_clean}" in known_hashes)


def compute_perceptual_hash(
    image: np.ndarray,
    hash_size: int = 8
) -> str:
    """
    Compute a perceptual hash (pHash) for similarity detection.
    
    Perceptual hashes are more robust to minor visual changes
    compared to cryptographic hashes.
    
    Args:
        image: Image as numpy array
        hash_size: Size of the hash (default 8 produces 64-bit hash)
        
    Returns:
        Perceptual hash as hexadecimal string (prefixed with 'phash:')
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize to hash_size + 1 for DCT
    resized = cv2.resize(gray, (hash_size + 1, hash_size), 
                         interpolation=cv2.INTER_AREA)
    
    # Compute difference (gradient-based)
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Convert to hash
    hash_int = sum([2**i for i, v in enumerate(diff.flatten()) if v])
    hash_hex = format(hash_int, f'0{hash_size * hash_size // 4}x')
    
    return f"phash:{hash_hex}"


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two perceptual hashes.
    
    Lower distance means more similar images.
    
    Args:
        hash1: First perceptual hash
        hash2: Second perceptual hash
        
    Returns:
        Hamming distance (number of different bits)
    """
    # Remove prefixes
    h1 = hash1.replace('phash:', '').replace('sha256:', '')
    h2 = hash2.replace('phash:', '').replace('sha256:', '')
    
    if len(h1) != len(h2):
        raise ValueError("Hashes must have same length for comparison")
    
    # Convert hex to binary and count differences
    b1 = bin(int(h1, 16))[2:].zfill(len(h1) * 4)
    b2 = bin(int(h2, 16))[2:].zfill(len(h2) * 4)
    
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))
