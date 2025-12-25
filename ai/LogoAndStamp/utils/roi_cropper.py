"""
ROI (Region of Interest) cropping utilities for logo and stamp extraction.

This module handles cropping regions from ID card images using configurable 
ratio-based coordinates, with support for robust jitter-based search for stamps.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np


@dataclass
class ROIConfig:
    """Configuration for ROI cropping (typically top-center logo)."""
    x_start_ratio: float = 0.35
    x_end_ratio: float = 0.65
    y_start_ratio: float = 0.0
    y_end_ratio: float = 0.20
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'ROIConfig':
        """Create ROIConfig from dictionary."""
        roi_config = config.get('roi', {})
        return cls(
            x_start_ratio=roi_config.get('x_start_ratio', 0.35),
            x_end_ratio=roi_config.get('x_end_ratio', 0.65),
            y_start_ratio=roi_config.get('y_start_ratio', 0.0),
            y_end_ratio=roi_config.get('y_end_ratio', 0.20)
        )
        
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x_start_ratio': self.x_start_ratio,
            'x_end_ratio': self.x_end_ratio,
            'y_start_ratio': self.y_start_ratio,
            'y_end_ratio': self.y_end_ratio
        }


@dataclass
class StampROIConfig:
    """Configuration for stamp ROI cropping (bottom-center)."""
    x_start_ratio: float = 0.35
    x_end_ratio: float = 0.65
    y_start_ratio: float = 0.75
    y_end_ratio: float = 1.0
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'StampROIConfig':
        """Create StampROIConfig from dictionary."""
        stamp_roi = config.get('stamp_roi', {})
        return cls(
            x_start_ratio=stamp_roi.get('x_start_ratio', 0.35),
            x_end_ratio=stamp_roi.get('x_end_ratio', 0.65),
            y_start_ratio=stamp_roi.get('y_start_ratio', 0.75),
            y_end_ratio=stamp_roi.get('y_end_ratio', 1.0)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x_start_ratio': self.x_start_ratio,
            'x_end_ratio': self.x_end_ratio,
            'y_start_ratio': self.y_start_ratio,
            'y_end_ratio': self.y_end_ratio
        }


def calculate_roi_bounds(
    image_shape: Tuple[int, ...],
    config: Union[ROIConfig, StampROIConfig, Dict]
) -> Tuple[int, int, int, int]:
    """
    Calculate the pixel bounds for the ROI based on image dimensions.
    
    Args:
        image_shape: Shape of the input image (H, W, ...)
        config: ROI Configuration
        
    Returns:
        Tuple of (x_start, y_start, x_end, y_end) in pixels
    """
    height, width = image_shape[:2]
    
    # Validate ratios
    if not (0 <= config.x_start_ratio < config.x_end_ratio <= 1):
        raise ValueError(f"Invalid x ratios: {config.x_start_ratio} to {config.x_end_ratio}")
    if not (0 <= config.y_start_ratio < config.y_end_ratio <= 1):
        raise ValueError(f"Invalid y ratios: {config.y_start_ratio} to {config.y_end_ratio}")
    
    x_start = int(width * config.x_start_ratio)
    x_end = int(width * config.x_end_ratio)
    y_start = int(height * config.y_start_ratio)
    y_end = int(height * config.y_end_ratio)
    
    return x_start, y_start, x_end, y_end


def crop_logo_roi(
    image: np.ndarray,
    config: Union[ROIConfig, Dict]
) -> np.ndarray:
    """Crop the logo region (top-center)."""
    if isinstance(config, dict):
        config = ROIConfig.from_dict(config)
    
    x_start, y_start, x_end, y_end = calculate_roi_bounds(image.shape, config)
    
    if len(image.shape) == 2:
        return image[y_start:y_end, x_start:x_end]
    return image[y_start:y_end, x_start:x_end, :]


def crop_stamp_roi(
    image: np.ndarray,
    config: Union[StampROIConfig, Dict]
) -> np.ndarray:
    """Crop the stamp region (bottom-center)."""
    if isinstance(config, dict):
        config = StampROIConfig.from_dict(config)
        
    x_start, y_start, x_end, y_end = calculate_roi_bounds(image.shape, config)
    
    if len(image.shape) == 2:
        return image[y_start:y_end, x_start:x_end]
    return image[y_start:y_end, x_start:x_end, :]


def get_candidate_rois(
    image: np.ndarray,
    base_config: StampROIConfig,
    h_jitter: float = 0.08,
    v_jitter: float = 0.05,
    h_step: float = 0.02,
    v_step: float = 0.01
) -> List[StampROIConfig]:
    """
    Generate a grid of candidate StampROIConfigs around the default.
    """
    candidates = []
    
    h_offsets = np.arange(-h_jitter, h_jitter + h_step/2, h_step)
    v_offsets = np.arange(-v_jitter, v_jitter + v_step/2, v_step)
    
    for h_off in h_offsets:
        for v_off in v_offsets:
            x_start = max(0.0, min(1.0, base_config.x_start_ratio + h_off))
            x_end = max(0.0, min(1.0, base_config.x_end_ratio + h_off))
            y_start = max(0.0, min(1.0, base_config.y_start_ratio + v_off))
            y_end = max(0.0, min(1.0, base_config.y_end_ratio + v_off))
            
            if x_end > x_start and y_end > y_start:
                config = StampROIConfig(x_start, x_end, y_start, y_end)
                candidates.append(config)
                
    return candidates


def crop_with_padding(
    image: np.ndarray,
    config: Union[ROIConfig, StampROIConfig, Dict],
    padding_ratio: float = 0.05
) -> np.ndarray:
    """Crop region with additional padding."""
    height, width = image.shape[:2]
    x_start, y_start, x_end, y_end = calculate_roi_bounds(image.shape, config)
    
    roi_width = x_end - x_start
    roi_height = y_end - y_start
    pad_x = int(roi_width * padding_ratio)
    pad_y = int(roi_height * padding_ratio)
    
    x_start = max(0, x_start - pad_x)
    x_end = min(width, x_end + pad_x)
    y_start = max(0, y_start - pad_y)
    y_end = min(height, y_end + pad_y)
    
    if len(image.shape) == 2:
        return image[y_start:y_end, x_start:x_end]
    return image[y_start:y_end, x_start:x_end, :]


def visualize_roi(
    image: np.ndarray,
    config: Union[ROIConfig, StampROIConfig, Dict],
    label: str = "ROI",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw ROI bounding box for visualization."""
    vis_img = image.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    
    x_start, y_start, x_end, y_end = calculate_roi_bounds(image.shape, config)
    cv2.rectangle(vis_img, (x_start, y_start), (x_end, y_end), color, thickness)
    cv2.putText(vis_img, label, (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis_img


def evaluate_roi_candidate(
    candidate_roi: np.ndarray,
    reference_logo: np.ndarray,
    verifier_siamese=None,
    verifier_orb=None
) -> float:
    """
    Compute a composite similarity score for a candidate ROI.
    """
    # Resize reference to match candidate
    ref_resized = cv2.resize(reference_logo, (candidate_roi.shape[1], candidate_roi.shape[0]))
    
    # 1. SSIM
    from classical.ssim_matcher import compute_ssim
    ssim_score = compute_ssim(candidate_roi, ref_resized)
    
    # 2. ORB (if provided)
    orb_score = 0.0
    if verifier_orb:
        orb_res = verifier_orb(candidate_roi, ref_resized)
        orb_score = orb_res.match_ratio
        
    # 3. Siamese (if provided)
    siamese_score = 0.0
    if verifier_siamese:
        siamese_score = verifier_siamese.compute_similarity(candidate_roi, ref_resized)
        
    # Composite score (weighted)
    if verifier_siamese:
        return 0.2 * ssim_score + 0.2 * orb_score + 0.6 * siamese_score
    return 0.5 * ssim_score + 0.5 * orb_score
