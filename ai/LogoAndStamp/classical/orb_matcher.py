"""
ORB (Oriented FAST and Rotated BRIEF) feature matcher for logo verification.

This module performs keypoint-based matching between the extracted logo
and the reference logo using ORB features with Lowe's ratio test.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ORBResult:
    """Result of ORB matching."""
    match_ratio: float          # Ratio of good matches to total keypoints
    num_good_matches: int       # Number of matches passing ratio test
    total_keypoints_img1: int   # Keypoints detected in first image
    total_keypoints_img2: int   # Keypoints detected in second image
    total_matches: int          # Total matches before ratio test
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'match_ratio': self.match_ratio,
            'num_good_matches': self.num_good_matches,
            'total_keypoints_img1': self.total_keypoints_img1,
            'total_keypoints_img2': self.total_keypoints_img2,
            'total_matches': self.total_matches
        }


def match_orb(
    img1: np.ndarray,
    img2: np.ndarray,
    n_features: int = 500,
    ratio_thresh: float = 0.75,
    cross_check: bool = False
) -> ORBResult:
    """
    Perform ORB feature matching between two images.
    
    Uses Lowe's ratio test to filter matches for robustness.
    
    Args:
        img1: First image (reference)
        img2: Second image (query)
        n_features: Maximum number of ORB features to detect
        ratio_thresh: Lowe's ratio test threshold (lower = stricter)
        cross_check: Whether to use cross-check matching (slower but stricter)
        
    Returns:
        ORBResult with matching statistics
        
    Example:
        >>> result = match_orb(reference_logo, query_logo)
        >>> print(f"Match ratio: {result.match_ratio:.3f}")
        >>> print(f"Good matches: {result.num_good_matches}")
    """
    # Convert to grayscale if needed
    gray1 = _to_grayscale(img1)
    gray2 = _to_grayscale(img2)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=n_features)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # Handle cases with no keypoints
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return ORBResult(
            match_ratio=0.0,
            num_good_matches=0,
            total_keypoints_img1=len(kp1) if kp1 else 0,
            total_keypoints_img2=len(kp2) if kp2 else 0,
            total_matches=0
        )
    
    # Create matcher
    if cross_check:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:50]
        total_matches = len(matches)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        total_matches = len(matches)
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
    
    # Calculate match ratio
    min_keypoints = min(len(kp1), len(kp2))
    match_ratio = len(good_matches) / min_keypoints if min_keypoints > 0 else 0.0
    
    # Clamp ratio to [0, 1]
    match_ratio = min(max(match_ratio, 0.0), 1.0)
    
    return ORBResult(
        match_ratio=match_ratio,
        num_good_matches=len(good_matches),
        total_keypoints_img1=len(kp1),
        total_keypoints_img2=len(kp2),
        total_matches=total_matches
    )


def classify_orb_signal(
    orb_result: ORBResult,
    thresholds: Dict[str, float]
) -> str:
    """
    Classify ORB matching result into signal categories.
    
    Args:
        orb_result: Result from match_orb
        thresholds: Dict with 'strong_genuine', 'suspicious', 'min_good_matches'
        
    Returns:
        'strong_genuine', 'suspicious', or 'forged'
    """
    strong_thresh = thresholds.get('strong_genuine', 0.35)
    suspicious_thresh = thresholds.get('suspicious', 0.15)
    min_matches = thresholds.get('min_good_matches', 10)
    
    if (orb_result.match_ratio >= strong_thresh and 
        orb_result.num_good_matches >= min_matches):
        return 'strong_genuine'
    elif orb_result.match_ratio >= suspicious_thresh:
        return 'suspicious'
    else:
        return 'forged'


def get_matching_keypoints(
    img1: np.ndarray,
    img2: np.ndarray,
    n_features: int = 500,
    ratio_thresh: float = 0.75
) -> Tuple[List, List, List]:
    """
    Get the actual matched keypoint locations.
    
    Useful for visualization and geometric verification.
    
    Args:
        img1: First image
        img2: Second image
        n_features: Maximum ORB features
        ratio_thresh: Lowe's ratio test threshold
        
    Returns:
        Tuple of (keypoints1, keypoints2, good_matches)
    """
    gray1 = _to_grayscale(img1)
    gray2 = _to_grayscale(img2)
    
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return [], [], []
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    return list(kp1), list(kp2), good_matches


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    max_matches: int = 50,
    ratio_thresh: float = 0.75
) -> np.ndarray:
    """
    Create a visualization of ORB matches between two images.
    
    Args:
        img1: First image
        img2: Second image
        max_matches: Maximum matches to draw
        ratio_thresh: Lowe's ratio test threshold
        
    Returns:
        Visualization image with matches drawn
    """
    kp1, kp2, good_matches = get_matching_keypoints(
        img1, img2, ratio_thresh=ratio_thresh
    )
    
    # Sort by distance and take top matches
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
    
    # Ensure both images are color for visualization
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1.copy()
        
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2.copy()
    
    # Draw matches
    vis = cv2.drawMatches(
        img1_color, kp1,
        img2_color, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return vis


def compute_homography_inliers(
    img1: np.ndarray,
    img2: np.ndarray,
    n_features: int = 500,
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 5.0
) -> Tuple[int, Optional[np.ndarray]]:
    """
    Compute homography and count inliers for geometric verification.
    
    A high inlier count indicates geometric consistency between images,
    which is a strong indicator of genuine matching.
    
    Args:
        img1: First image
        img2: Second image
        n_features: Maximum ORB features
        ratio_thresh: Lowe's ratio test threshold
        ransac_thresh: RANSAC reprojection threshold
        
    Returns:
        Tuple of (num_inliers, homography_matrix or None)
    """
    kp1, kp2, good_matches = get_matching_keypoints(
        img1, img2, n_features, ratio_thresh
    )
    
    if len(good_matches) < 4:
        return 0, None
    
    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    
    if mask is None:
        return 0, None
    
    num_inliers = int(mask.sum())
    return num_inliers, H


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if not already."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
