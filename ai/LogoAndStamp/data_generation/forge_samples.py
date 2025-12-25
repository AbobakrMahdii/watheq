"""
Synthetic forged sample generator with multi-level tampering.

This module generates forged samples at three difficulty levels:
- Level 1 (Obvious): Visible tampering, easy to detect
- Level 2 (Subtle): Pixel-level manipulations, harder to detect  
- Level 3 (Near-Authentic): Hard negatives, >95% visually similar

Each sample is logged with its synthesis type and seed for reproducibility.
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import argparse

import cv2
import numpy as np


class ForgeryLevel(Enum):
    """Forgery difficulty levels."""
    OBVIOUS = 1        # Level 1: Easy to detect
    SUBTLE = 2         # Level 2: Pixel-level, harder
    NEAR_AUTHENTIC = 3 # Level 3: Hard negatives


def _convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@dataclass
class ForgeryLog:
    """Log entry for a generated forged sample."""
    filename: str
    class_label: str
    level: int
    level_name: str
    forgery_type: str
    parameters: Dict
    seed: int
    timestamp: str
    
    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert numpy types to native Python types
        d['parameters'] = _convert_to_native(d['parameters'])
        return d


class ForgerySynthesizer:
    """
    Generate synthetic forged samples at multiple difficulty levels.
    
    Implements realistic tampering simulations from obvious to
    near-authentic (hard negatives) for training robust detection.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the synthesizer.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.base_seed = seed
        
        # Register forgery methods by level
        self.level_1_methods = [
            ('partial_occlusion', self.partial_occlusion),
            ('copy_paste_shift', self.copy_paste_shift),
            ('strong_warp', self.strong_warp),
            ('erasure_blur', self.erasure_blur),
            ('scan_degradation', self.scan_degradation),
            ('color_manipulation', self.color_manipulation),
            ('double_stamp_offset', self.double_stamp_offset),
            ('partial_stamping', self.partial_stamping),
        ]
        
        self.level_2_methods = [
            ('pixel_noise_injection', self.pixel_noise_injection),
            ('micro_edge_edits', self.micro_edge_edits),
            ('compression_inconsistency', self.compression_inconsistency),
            ('resampling_artifacts', self.resampling_artifacts),
            ('partial_channel_manipulation', self.partial_channel_manipulation),
            ('low_opacity_overlay', self.low_opacity_overlay),
        ]
        
        self.level_3_methods = [
            ('micro_blur_sharpen', self.micro_blur_sharpen),
            ('subtle_hue_shift', self.subtle_hue_shift),
            ('tiny_geometric_distortion', self.tiny_geometric_distortion),
            ('sharpening_halos', self.sharpening_halos),
            ('clone_stamp_edit', self.clone_stamp_edit),
            ('micro_pattern_injection', self.micro_pattern_injection),
            ('ink_bleeding_fade', self.ink_bleeding_fade),
            ('stamp_smear', self.stamp_smear),
            ('uneven_pressure', self.uneven_pressure),
        ]
    
    # ==================== LEVEL 1: OBVIOUS FORGERIES ====================
    
    def partial_occlusion(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Cover 10-30% of the logo with random shapes."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Random occlusion size (10-30% of area)
        coverage = self.rng.uniform(0.10, 0.30)
        num_shapes = self.rng.randint(1, 4)
        
        for _ in range(num_shapes):
            shape_type = self.rng.choice(['rect', 'circle'])
            
            if shape_type == 'rect':
                rect_w = int(w * np.sqrt(coverage / num_shapes))
                rect_h = int(h * np.sqrt(coverage / num_shapes))
                x = self.rng.randint(0, max(1, w - rect_w))
                y = self.rng.randint(0, max(1, h - rect_h))
                color = tuple(int(c) for c in self.rng.randint(0, 255, 3))
                cv2.rectangle(result, (x, y), (x + rect_w, y + rect_h), color, -1)
            else:
                radius = int(min(w, h) * np.sqrt(coverage / num_shapes) / 2)
                cx = self.rng.randint(radius, max(radius + 1, w - radius))
                cy = self.rng.randint(radius, max(radius + 1, h - radius))
                color = tuple(int(c) for c in self.rng.randint(0, 255, 3))
                cv2.circle(result, (cx, cy), radius, color, -1)
        
        return result, {'coverage': coverage, 'num_shapes': num_shapes}
    
    def copy_paste_shift(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Duplicate a region and paste with visible shift."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Select region to copy (15-30% of image)
        region_size = self.rng.uniform(0.15, 0.30)
        rw = int(w * np.sqrt(region_size))
        rh = int(h * np.sqrt(region_size))
        
        src_x = self.rng.randint(0, max(1, w - rw))
        src_y = self.rng.randint(0, max(1, h - rh))
        
        # Shift by 10-25% of dimension
        shift_x = int(self.rng.uniform(0.10, 0.25) * w) * self.rng.choice([-1, 1])
        shift_y = int(self.rng.uniform(0.10, 0.25) * h) * self.rng.choice([-1, 1])
        
        dst_x = np.clip(src_x + shift_x, 0, w - rw)
        dst_y = np.clip(src_y + shift_y, 0, h - rh)
        
        # Copy region
        region = img[src_y:src_y+rh, src_x:src_x+rw].copy()
        result[dst_y:dst_y+rh, dst_x:dst_x+rw] = region
        
        return result, {'shift': (shift_x, shift_y), 'region_size': region_size}
    
    def strong_warp(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply strong perspective/barrel distortion."""
        h, w = img.shape[:2]
        
        # Perspective warp
        strength = self.rng.uniform(0.05, 0.15)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        # Random corner shifts
        pts2 = np.float32([
            [self.rng.uniform(0, strength * w), self.rng.uniform(0, strength * h)],
            [w - self.rng.uniform(0, strength * w), self.rng.uniform(0, strength * h)],
            [self.rng.uniform(0, strength * w), h - self.rng.uniform(0, strength * h)],
            [w - self.rng.uniform(0, strength * w), h - self.rng.uniform(0, strength * h)]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, M, (w, h))
        
        return warped, {'warp_strength': strength}
    
    def erasure_blur(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Erase a region and apply heavy blur to simulate tampering."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Select region to erase (10-20%)
        erase_size = self.rng.uniform(0.10, 0.20)
        ew = int(w * np.sqrt(erase_size))
        eh = int(h * np.sqrt(erase_size))
        
        x = self.rng.randint(0, max(1, w - ew))
        y = self.rng.randint(0, max(1, h - eh))
        
        # Erase and blur
        result[y:y+eh, x:x+ew] = cv2.GaussianBlur(
            result[y:y+eh, x:x+ew], 
            (21, 21), 
            10
        )
        
        return result, {'erase_position': (x, y), 'erase_size': erase_size}
    
    def scan_degradation(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Simulate heavy scan degradation."""
        # Heavy blur
        blurred = cv2.GaussianBlur(img, (7, 7), 3)
        
        # Heavy noise
        noise = self.rng.randn(*blurred.shape) * 30
        noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Reduce contrast
        low_contrast = cv2.convertScaleAbs(noisy, alpha=0.6, beta=40)
        
        # Low quality JPEG
        _, encoded = cv2.imencode('.jpg', low_contrast, [cv2.IMWRITE_JPEG_QUALITY, 40])
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if len(img.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        
        return degraded, {'blur_sigma': 3, 'noise_sigma': 30, 'jpeg_quality': 40}
    
    def color_manipulation(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply visible color changes."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Strong hue shift
        hue_shift = self.rng.uniform(20, 60)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Saturation change
        sat_mult = self.rng.uniform(0.5, 1.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result, {'hue_shift': hue_shift, 'saturation_mult': sat_mult}
    
    def double_stamp_offset(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Simulate a double-stamping artifact with a visible offset."""
        h, w = img.shape[:2]
        
        # Random offset (5-15% of dimension)
        offset_x = int(w * self.rng.uniform(0.05, 0.15)) * self.rng.choice([-1, 1])
        offset_y = int(h * self.rng.uniform(0.05, 0.15)) * self.rng.choice([-1, 1])
        
        # Create second stamp image with low opacity
        overlay = np.ones_like(img) * 255
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        shift_img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        
        # Blend (60% original, 40% shifted)
        alpha = 0.6
        result = cv2.addWeighted(img, alpha, shift_img, 1 - alpha, 0)
        
        return result, {'offset': (offset_x, offset_y), 'alpha': alpha}

    def partial_stamping(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Simulate incomplete or partial stamping."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Mask 30-50% of the region
        coverage = self.rng.uniform(0.3, 0.5)
        
        # Random mask orientation (top, bottom, left, right)
        side = self.rng.choice(['top', 'bottom', 'left', 'right'])
        
        mask = np.ones((h, w), dtype=np.float32)
        if side == 'top':
            mask[:int(h * coverage), :] = 0
        elif side == 'bottom':
            mask[int(h * (1-coverage)):, :] = 0
        elif side == 'left':
            mask[:, :int(w * coverage)] = 0
        else:
            mask[:, int(w * (1-coverage)):] = 0
            
        # Apply mask with some blur for soft edge
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        if len(img.shape) == 3:
            mask = mask[:, :, np.newaxis]
            
        # Blend with white background
        result = (img.astype(np.float32) * mask + 255 * (1 - mask)).astype(np.uint8)
        
        return result, {'coverage': coverage, 'side': side}
    
    # ==================== LEVEL 2: SUBTLE FORGERIES ====================
    
    def pixel_noise_injection(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Inject low-amplitude localized noise."""
        h, w = img.shape[:2]
        result = img.copy().astype(np.float32)
        
        # Low amplitude noise (1-5 intensity)
        amplitude = self.rng.uniform(1, 5)
        
        # Localized region (30-60% of image)
        region_ratio = self.rng.uniform(0.30, 0.60)
        rh = int(h * np.sqrt(region_ratio))
        rw = int(w * np.sqrt(region_ratio))
        rx = self.rng.randint(0, max(1, w - rw))
        ry = self.rng.randint(0, max(1, h - rh))
        
        noise = self.rng.randn(rh, rw, img.shape[2] if len(img.shape) == 3 else 1) * amplitude
        if len(img.shape) == 2:
            noise = noise.squeeze()
        
        result[ry:ry+rh, rx:rx+rw] += noise
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, {'amplitude': amplitude, 'region_ratio': region_ratio}
    
    def micro_edge_edits(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply tiny edge erosion/dilation and anti-aliasing changes."""
        # Detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges slightly (1-2 px)
        kernel_size = self.rng.choice([1, 2])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if self.rng.random() > 0.5:
            modified_edges = cv2.dilate(edges, kernel, iterations=1)
            operation = 'dilate'
        else:
            modified_edges = cv2.erode(edges, kernel, iterations=1)
            operation = 'erode'
        
        # Create subtle overlay
        result = img.copy()
        edge_mask = modified_edges > 0
        
        if len(result.shape) == 3:
            for c in range(3):
                channel = result[:, :, c].astype(np.float32)
                channel[edge_mask] = np.clip(channel[edge_mask] + self.rng.uniform(-5, 5), 0, 255)
                result[:, :, c] = channel.astype(np.uint8)
        else:
            result[edge_mask] = np.clip(result[edge_mask].astype(np.float32) + self.rng.uniform(-5, 5), 0, 255).astype(np.uint8)
        
        return result, {'operation': operation, 'kernel_size': kernel_size}
    
    def compression_inconsistency(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Create double-JPEG or inconsistent compression artifacts."""
        h, w = img.shape[:2]
        
        # First compression pass
        q1 = self.rng.randint(60, 80)
        _, enc1 = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q1])
        pass1 = cv2.imdecode(enc1, cv2.IMREAD_COLOR if len(img.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        
        # Select a region to compress differently
        region_size = self.rng.uniform(0.20, 0.40)
        rh = int(h * np.sqrt(region_size))
        rw = int(w * np.sqrt(region_size))
        rx = self.rng.randint(0, max(1, w - rw))
        ry = self.rng.randint(0, max(1, h - rh))
        
        # Different quality for the region
        q2 = self.rng.randint(30, 50)
        region = pass1[ry:ry+rh, rx:rx+rw]
        _, enc2 = cv2.imencode('.jpg', region, [cv2.IMWRITE_JPEG_QUALITY, q2])
        region_recomp = cv2.imdecode(enc2, cv2.IMREAD_COLOR if len(region.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        
        result = pass1.copy()
        result[ry:ry+rh, rx:rx+rw] = region_recomp
        
        return result, {'quality_1': q1, 'quality_2': q2, 'region_size': region_size}
    
    def resampling_artifacts(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply fractional scaling + rescaling artifacts."""
        h, w = img.shape[:2]
        
        # Fractional scale
        scale = self.rng.uniform(0.97, 1.03)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Scale down/up
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Scale back
        restored = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Add slight rotation then undo
        angle = self.rng.uniform(-1, 1)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(restored, M, (w, h))
        M_inv = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1.0)
        result = cv2.warpAffine(rotated, M_inv, (w, h))
        
        return result, {'scale_factor': scale, 'rotation_angle': angle}
    
    def partial_channel_manipulation(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Modify one color channel by a small amount."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        result = img.copy().astype(np.float32)
        
        # Select channel
        channel = self.rng.randint(0, 3)
        channel_names = ['B', 'G', 'R']
        
        # Small intensity shift (2-8)
        shift = self.rng.uniform(2, 8) * self.rng.choice([-1, 1])
        
        result[:, :, channel] = np.clip(result[:, :, channel] + shift, 0, 255)
        
        return result.astype(np.uint8), {'channel': channel_names[channel], 'shift': shift}
    
    def low_opacity_overlay(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Add faint watermark-like overlay or overprint lines."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Create overlay
        overlay = np.zeros((h, w, 3 if len(img.shape) == 3 else 1), dtype=np.float32)
        
        overlay_type = self.rng.choice(['lines', 'pattern'])
        
        if overlay_type == 'lines':
            # Horizontal lines
            spacing = self.rng.randint(10, 30)
            for y in range(0, h, spacing):
                cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
        else:
            # Grid pattern
            spacing = self.rng.randint(15, 40)
            for x in range(0, w, spacing):
                cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
            for y in range(0, h, spacing):
                cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
        
        # Very low opacity (2-6%)
        opacity = self.rng.uniform(0.02, 0.06)
        
        if len(result.shape) == 2:
            overlay = overlay[:, :, 0]
        
        blended = result.astype(np.float32) + overlay * opacity
        result = np.clip(blended, 0, 255).astype(np.uint8)
        
        return result, {'overlay_type': overlay_type, 'opacity': opacity}
    
    # ==================== LEVEL 3: NEAR-AUTHENTIC (HARD NEGATIVES) ====================
    
    def micro_blur_sharpen(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply micro-blur or sharpen to 10-20% of logo area only."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Small region (10-20% of area)
        region_ratio = self.rng.uniform(0.10, 0.20)
        rh = int(h * np.sqrt(region_ratio))
        rw = int(w * np.sqrt(region_ratio))
        rx = self.rng.randint(0, max(1, w - rw))
        ry = self.rng.randint(0, max(1, h - rh))
        
        region = result[ry:ry+rh, rx:rx+rw]
        
        operation = self.rng.choice(['blur', 'sharpen'])
        
        if operation == 'blur':
            # Very subtle blur
            region = cv2.GaussianBlur(region, (3, 3), 0.5)
        else:
            # Subtle sharpening
            kernel = np.array([[-0.5, -0.5, -0.5],
                               [-0.5,  5.0, -0.5],
                               [-0.5, -0.5, -0.5]]) * 0.2 + np.eye(3) * 0.8
            region = cv2.filter2D(region, -1, kernel)
        
        result[ry:ry+rh, rx:rx+rw] = region
        
        return result, {'operation': operation, 'region_ratio': region_ratio}
    
    def subtle_hue_shift(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply very subtle per-region hue/tonal drift (ΔRB 1-4)."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        result = img.copy().astype(np.float32)
        h, w = img.shape[:2]
        
        # Divide into regions
        num_regions = self.rng.randint(4, 9)
        grid_size = int(np.sqrt(num_regions))
        
        region_h = h // grid_size
        region_w = w // grid_size
        
        shifts = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                y1, y2 = gy * region_h, (gy + 1) * region_h
                x1, x2 = gx * region_w, (gx + 1) * region_w
                
                # Very small shift per channel (1-4 intensity)
                shift = self.rng.uniform(-4, 4, 3)
                result[y1:y2, x1:x2] += shift
                shifts.append(shift.tolist())
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, {'num_regions': num_regions, 'max_shift': 4}
    
    def tiny_geometric_distortion(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Apply 1-3 pixel displacement that keeps appearance normal."""
        h, w = img.shape[:2]
        
        # Very subtle perspective (1-3 px corners)
        max_disp = self.rng.randint(1, 4)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = pts1.copy()
        
        # Tiny random displacements
        for i in range(4):
            pts2[i, 0] += self.rng.randint(-max_disp, max_disp + 1)
            pts2[i, 1] += self.rng.randint(-max_disp, max_disp + 1)
        
        # Clip coordinates properly (maintain 2D shape)
        pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
        pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return result, {'max_displacement': max_disp}
    
    def sharpening_halos(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Add slight edge sharpening artifacts (halos)."""
        # Unsharp masking with very subtle overshoot
        blurred = cv2.GaussianBlur(img, (0, 0), 2)
        
        # Very low strength
        strength = self.rng.uniform(0.1, 0.3)
        
        sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
        
        return sharpened, {'strength': strength}
    
    def clone_stamp_edit(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Duplicate tiny patch and blend softly."""
        h, w = img.shape[:2]
        result = img.copy()
        
        # Tiny patch (3-6% of area)
        patch_ratio = self.rng.uniform(0.03, 0.06)
        ph = int(h * np.sqrt(patch_ratio))
        pw = int(w * np.sqrt(patch_ratio))
        
        # Source position
        sx = self.rng.randint(0, max(1, w - pw))
        sy = self.rng.randint(0, max(1, h - ph))
        
        # Shift (5-15 px)
        shift = self.rng.randint(5, 16)
        dx = sx + shift * self.rng.choice([-1, 1])
        dy = sy + shift * self.rng.choice([-1, 1])
        
        dx = np.clip(dx, 0, w - pw)
        dy = np.clip(dy, 0, h - ph)
        
        # Get patch
        patch = img[sy:sy+ph, sx:sx+pw].copy()
        
        # Soft blend
        alpha = self.rng.uniform(0.6, 0.9)
        existing = result[dy:dy+ph, dx:dx+pw]
        blended = cv2.addWeighted(patch, alpha, existing, 1 - alpha, 0)
        result[dy:dy+ph, dx:dx+pw] = blended
        
        return result, {'patch_size': patch_ratio, 'shift': shift, 'alpha': alpha}
    
    def micro_pattern_injection(
        self, 
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Inject sub-pixel noise patterns in specific regions."""
        h, w = img.shape[:2]
        result = img.copy().astype(np.float32)
        
        # Very low amplitude pattern (0.5-2 intensity)
        amplitude = self.rng.uniform(0.5, 2.0)
        
        # Generate high-frequency pattern
        pattern = self.rng.randn(h, w) * amplitude
        
        # Apply only to specific regions (checkerboard-like)
        mask = np.zeros((h, w), dtype=bool)
        block_size = self.rng.randint(8, 16)
        
        for y in range(0, h, block_size * 2):
            for x in range(0, w, block_size * 2):
                mask[y:y+block_size, x:x+block_size] = True
        
        if len(result.shape) == 3:
            pattern = pattern[:, :, np.newaxis]
            mask = mask[:, :, np.newaxis]
        
        result = np.where(mask, result + pattern, result)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, {'amplitude': amplitude, 'block_size': block_size}
    
    def ink_bleeding_fade(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Simulate localized ink bleeding or fading."""
        h, w = img.shape[:2]
        result = img.copy().astype(np.float32)
        
        # Create random region mask
        num_regions = self.rng.randint(2, 5)
        total_mask = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(num_regions):
            cx, cy = self.rng.randint(0, w), self.rng.randint(0, h)
            radius = self.rng.randint(min(h, w) // 10, min(h, w) // 4)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (31, 31), 10)
            total_mask = np.maximum(total_mask, mask)
            
        if len(img.shape) == 3:
            total_mask = total_mask[:, :, np.newaxis]
            
        mode = self.rng.choice(['bleed', 'fade'])
        if mode == 'bleed':
            # Bleed: dilate/blur the region
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(img, kernel, iterations=1)
            blurred = cv2.GaussianBlur(dilated, (7, 7), 2)
            result = img.astype(np.float32) * (1 - total_mask) + blurred.astype(np.float32) * total_mask
        else:
            # Fade: increase brightness / reduce contrast
            faded = np.clip(img.astype(np.float32) * 1.3 + 30, 0, 255)
            result = img.astype(np.float32) * (1 - total_mask) + faded * total_mask
            
        return result.astype(np.uint8), {'mode': mode, 'num_regions': num_regions}

    def stamp_smear(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Simulate a hand-induced directional motion blur smear."""
        angle = self.rng.uniform(0, 360)
        length = self.rng.randint(3, 8)
        
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        center = length // 2
        kernel[center, :] = 1.0
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        kernel = kernel / np.sum(kernel)
        
        # Apply filter
        result = cv2.filter2D(img, -1, kernel)
        
        return result, {'angle': angle, 'length': length}

    def uneven_pressure(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Simulate uneven stamping pressure (gradient of intensity)."""
        h, w = img.shape[:2]
        
        # Create linear gradient mask
        angle = self.rng.uniform(0, 360)
        rad = np.deg2rad(angle)
        
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Directional gradient
        grad = X * np.cos(rad) + Y * np.sin(rad)
        grad = (grad - grad.min()) / (grad.max() - grad.min())
        
        # Scale to 0.7 - 1.0 range (subtle)
        intensity_map = 0.7 + 0.3 * grad
        
        if len(img.shape) == 3:
            intensity_map = intensity_map[:, :, np.newaxis]
            
        # Blend with white background (255 is paper)
        # Low pressure -> closer to 255
        result = (img.astype(np.float32) * intensity_map + 255 * (1 - intensity_map)).astype(np.uint8)
        
        return result, {'angle': angle}
    
    def generate_sample(
        self,
        reference: np.ndarray,
        level: ForgeryLevel
    ) -> Tuple[np.ndarray, str, Dict]:
        """
        Generate a single forged sample at specified level.
        
        Args:
            reference: Reference logo image
            level: Forgery difficulty level
            
        Returns:
            Tuple of (forged_image, forgery_type, parameters)
        """
        if level == ForgeryLevel.OBVIOUS:
            methods = self.level_1_methods
        elif level == ForgeryLevel.SUBTLE:
            methods = self.level_2_methods
        else:
            methods = self.level_3_methods
        
        # Select random method
        name, func = random.choice(methods)
        
        # Apply forgery
        forged, params = func(reference)
        
        return forged, name, params
    
    def generate_dataset(
        self,
        reference_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_samples: int = 600,
        train_ratio: float = 0.8,
        level_distribution: Dict[str, float] = None,
        log_path: Optional[Union[str, Path]] = None
    ) -> List[ForgeryLog]:
        """
        Generate complete forged sample dataset with all levels.
        
        Args:
            reference_path: Path to reference logo image
            output_dir: Base output directory
            num_samples: Total number of samples
            train_ratio: Ratio for training set
            level_distribution: Distribution of levels (default: equal)
            log_path: Path to save generation log
            
        Returns:
            List of forgery log entries
        """
        # Default distribution
        if level_distribution is None:
            level_distribution = {
                'obvious': 0.33,
                'subtle': 0.33,
                'near_authentic': 0.34
            }
        
        # Load reference
        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise ValueError(f"Could not load reference: {reference_path}")
        
        output_dir = Path(output_dir)
        train_dir = output_dir / 'train' / 'forged'
        val_dir = output_dir / 'val' / 'forged'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate samples per level
        level_counts = {
            ForgeryLevel.OBVIOUS: int(num_samples * level_distribution['obvious']),
            ForgeryLevel.SUBTLE: int(num_samples * level_distribution['subtle']),
            ForgeryLevel.NEAR_AUTHENTIC: int(num_samples * level_distribution['near_authentic'])
        }
        
        logs = []
        sample_idx = 0
        
        for level, count in level_counts.items():
            num_train = int(count * train_ratio)
            num_val = count - num_train
            
            # Generate training samples for this level
            for i in range(num_train):
                seed = self.base_seed + sample_idx
                self.rng = np.random.RandomState(seed)
                random.seed(seed)
                
                sample, forgery_type, params = self.generate_sample(reference, level)
                filename = f"forged_L{level.value}_{forgery_type}_train_{i:04d}.png"
                filepath = train_dir / filename
                
                cv2.imwrite(str(filepath), sample)
                
                logs.append(ForgeryLog(
                    filename=filename,
                    class_label='forged',
                    level=level.value,
                    level_name=level.name.lower(),
                    forgery_type=forgery_type,
                    parameters=params,
                    seed=seed,
                    timestamp=datetime.now().isoformat()
                ))
                
                sample_idx += 1
            
            # Generate validation samples for this level
            for i in range(num_val):
                seed = self.base_seed + sample_idx
                self.rng = np.random.RandomState(seed)
                random.seed(seed)
                
                sample, forgery_type, params = self.generate_sample(reference, level)
                filename = f"forged_L{level.value}_{forgery_type}_val_{i:04d}.png"
                filepath = val_dir / filename
                
                cv2.imwrite(str(filepath), sample)
                
                logs.append(ForgeryLog(
                    filename=filename,
                    class_label='forged',
                    level=level.value,
                    level_name=level.name.lower(),
                    forgery_type=forgery_type,
                    parameters=params,
                    seed=seed,
                    timestamp=datetime.now().isoformat()
                ))
                
                sample_idx += 1
        
        # Save log
        if log_path:
            with open(log_path, 'w') as f:
                json.dump([log.to_dict() for log in logs], f, indent=2)
        
        return logs


def main():
    """CLI entry point for generating forged samples."""
    parser = argparse.ArgumentParser(description='Generate forged samples')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference image')
    parser.add_argument('--output', type=str, default='data/logo',
                        help='Output directory')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of samples for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to save generation log')
    parser.add_argument('--element', type=str, default='logo',
                        choices=['logo', 'stamp'], help='Element type')
    
    args = parser.parse_args()
    
    # Adjust output if element is stamp
    output_dir = args.output
    if args.element == 'stamp' and output_dir == 'data/logo':
        output_dir = 'data/stamp'
    
    synthesizer = ForgerySynthesizer(seed=args.seed)
    logs = synthesizer.generate_dataset(
        reference_path=args.reference,
        output_dir=output_dir,
        num_samples=args.count,
        train_ratio=args.train_ratio,
        log_path=args.log
    )
    
    print(f"Generated {len(logs)} forged samples")
    
    # Summary by level
    level_counts = {}
    for log in logs:
        level_counts[log.level_name] = level_counts.get(log.level_name, 0) + 1
    
    for level, count in level_counts.items():
        print(f"  Level {level}: {count} samples")


if __name__ == '__main__':
    main()
