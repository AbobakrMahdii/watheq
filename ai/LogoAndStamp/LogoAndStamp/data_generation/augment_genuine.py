"""
Genuine sample augmentation for training data generation.

This module generates realistic genuine samples from a single reference logo
by applying capture-variation transforms (NOT content-altering edits).
These simulate real-world scanning/photography variations.
"""

import io
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse

import cv2
import numpy as np
from PIL import Image


@dataclass
class AugmentationLog:
    """Log entry for a generated sample."""
    filename: str
    class_label: str
    transforms_applied: List[str]
    parameters: Dict
    seed: int
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)


class GenuineAugmenter:
    """
    Generate synthetic genuine samples with capture variations.
    
    These transforms simulate real-world capture conditions:
    - Scanner vs camera
    - Lighting variations
    - Minor alignment issues
    - Compression artifacts
    
    IMPORTANT: These are content-PRESERVING transforms only.
    No content-altering edits are applied; those belong to forged samples.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the augmenter.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.base_seed = seed
    
    def apply_rotation(
        self, 
        img: np.ndarray, 
        max_angle: float = 5.0
    ) -> Tuple[np.ndarray, Dict]:
        """Apply slight rotation to simulate document alignment variation."""
        angle = self.rng.uniform(-max_angle, max_angle)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return rotated, {'rotation_angle': angle}
    
    def apply_scale(
        self, 
        img: np.ndarray, 
        scale_range: Tuple[float, float] = (0.95, 1.05)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply slight scaling to simulate distance variations."""
        scale = self.rng.uniform(*scale_range)
        h, w = img.shape[:2]
        
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if scale > 1:
            # Crop center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result = resized[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad to original size
            result = np.zeros_like(img)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return result, {'scale_factor': scale}
    
    def apply_blur(
        self, 
        img: np.ndarray, 
        sigma_range: Tuple[float, float] = (0.5, 1.5)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply Gaussian blur to simulate focus variation."""
        sigma = self.rng.uniform(*sigma_range)
        # Kernel size must be odd
        ksize = int(sigma * 4) | 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        
        return blurred, {'blur_sigma': sigma}
    
    def apply_noise(
        self, 
        img: np.ndarray, 
        sigma_range: Tuple[float, float] = (5, 15)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply Gaussian noise to simulate sensor noise."""
        sigma = self.rng.uniform(*sigma_range)
        noise = self.rng.randn(*img.shape) * sigma
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return noisy, {'noise_sigma': sigma}
    
    def apply_brightness_contrast(
        self, 
        img: np.ndarray, 
        brightness_range: Tuple[float, float] = (-25, 25),
        contrast_range: Tuple[float, float] = (0.9, 1.1)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply brightness/contrast changes to simulate lighting variation."""
        brightness = self.rng.uniform(*brightness_range)
        contrast = self.rng.uniform(*contrast_range)
        
        adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        
        return adjusted, {'brightness': brightness, 'contrast': contrast}
    
    def apply_gamma(
        self, 
        img: np.ndarray, 
        gamma_range: Tuple[float, float] = (0.9, 1.1)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply gamma correction to simulate exposure variation."""
        gamma = self.rng.uniform(*gamma_range)
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        corrected = cv2.LUT(img, table)
        
        return corrected, {'gamma': gamma}
    
    def apply_jpeg_compression(
        self, 
        img: np.ndarray, 
        quality_range: Tuple[int, int] = (70, 95)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply JPEG compression artifacts."""
        quality = int(self.rng.uniform(*quality_range))
        
        # Encode and decode via buffer
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR if len(img.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        
        return compressed, {'jpeg_quality': quality}
    
    def apply_vignetting(
        self, 
        img: np.ndarray, 
        strength_range: Tuple[float, float] = (0.1, 0.3)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply subtle vignetting to simulate camera lens effect."""
        h, w = img.shape[:2]
        strength = self.rng.uniform(*strength_range)
        
        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # Distance from center, normalized
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist
        
        # Vignette mask
        vignette = 1 - strength * (dist_norm ** 2)
        
        if len(img.shape) == 3:
            vignette = vignette[:, :, np.newaxis]
        
        vignetted = np.clip(img * vignette, 0, 255).astype(np.uint8)
        
        return vignetted, {'vignette_strength': strength}
    
    def apply_paper_texture(
        self, 
        img: np.ndarray, 
        opacity_range: Tuple[float, float] = (0.02, 0.08)
    ) -> Tuple[np.ndarray, Dict]:
        """Apply subtle paper texture overlay."""
        h, w = img.shape[:2]
        opacity = self.rng.uniform(*opacity_range)
        
        # Generate Perlin-like noise for paper texture
        scale = self.rng.randint(20, 50)
        noise = self.rng.randn(h // scale + 1, w // scale + 1)
        noise = cv2.resize(noise.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize to [0, 1]
        
        # Convert to texture
        texture = ((noise - 0.5) * 255 * opacity).astype(np.float32)
        
        if len(img.shape) == 3:
            texture = texture[:, :, np.newaxis]
        
        textured = np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)
        
        return textured, {'paper_texture_opacity': opacity}
    
    def generate_sample(
        self,
        reference: np.ndarray,
        num_transforms: int = 4
    ) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Generate a single augmented genuine sample.
        
        Args:
            reference: Reference logo image
            num_transforms: Number of transforms to apply (random subset)
            
        Returns:
            Tuple of (augmented_image, transform_names, parameters)
        """
        # Available transforms
        transforms = [
            ('rotation', self.apply_rotation),
            ('scale', self.apply_scale),
            ('blur', self.apply_blur),
            ('noise', self.apply_noise),
            ('brightness_contrast', self.apply_brightness_contrast),
            ('gamma', self.apply_gamma),
            ('jpeg_compression', self.apply_jpeg_compression),
            ('vignetting', self.apply_vignetting),
            ('paper_texture', self.apply_paper_texture)
        ]
        
        # Select random subset
        num_to_apply = min(num_transforms, len(transforms))
        selected = random.sample(transforms, num_to_apply)
        
        # Apply transforms
        result = reference.copy()
        applied_names = []
        all_params = {}
        
        for name, func in selected:
            result, params = func(result)
            applied_names.append(name)
            all_params[name] = params
        
        return result, applied_names, all_params
    
    def generate_dataset(
        self,
        reference_path: Union[str, Path],
        output_dir: Union[str, Path],
        num_samples: int = 600,
        train_ratio: float = 0.8,
        log_path: Optional[Union[str, Path]] = None
    ) -> List[AugmentationLog]:
        """
        Generate a complete genuine sample dataset.
        
        Args:
            reference_path: Path to reference logo image
            output_dir: Base output directory
            num_samples: Total number of samples to generate
            train_ratio: Ratio of samples for training set
            log_path: Optional path to save generation log
            
        Returns:
            List of augmentation log entries
        """
        # Load reference
        reference = cv2.imread(str(reference_path))
        if reference is None:
            raise ValueError(f"Could not load reference image: {reference_path}")
        
        output_dir = Path(output_dir)
        train_dir = output_dir / 'train' / 'genuine'
        val_dir = output_dir / 'val' / 'genuine'
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Split counts
        num_train = int(num_samples * train_ratio)
        num_val = num_samples - num_train
        
        logs = []
        
        # Generate training samples
        for i in range(num_train):
            seed = self.base_seed + i
            self.rng = np.random.RandomState(seed)
            random.seed(seed)
            
            sample, transforms, params = self.generate_sample(reference)
            filename = f"genuine_train_{i:04d}.png"
            filepath = train_dir / filename
            
            cv2.imwrite(str(filepath), sample)
            
            logs.append(AugmentationLog(
                filename=filename,
                class_label='genuine',
                transforms_applied=transforms,
                parameters=params,
                seed=seed,
                timestamp=datetime.now().isoformat()
            ))
        
        # Generate validation samples
        for i in range(num_val):
            seed = self.base_seed + num_train + i
            self.rng = np.random.RandomState(seed)
            random.seed(seed)
            
            sample, transforms, params = self.generate_sample(reference)
            filename = f"genuine_val_{i:04d}.png"
            filepath = val_dir / filename
            
            cv2.imwrite(str(filepath), sample)
            
            logs.append(AugmentationLog(
                filename=filename,
                class_label='genuine',
                transforms_applied=transforms,
                parameters=params,
                seed=seed,
                timestamp=datetime.now().isoformat()
            ))
        
        # Save log
        if log_path:
            with open(log_path, 'w') as f:
                json.dump([log.to_dict() for log in logs], f, indent=2)
        
        return logs


def main():
    """CLI entry point for generating genuine samples."""
    parser = argparse.ArgumentParser(description='Generate genuine samples')
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
    
    augmenter = GenuineAugmenter(seed=args.seed)
    logs = augmenter.generate_dataset(
        reference_path=args.reference,
        output_dir=output_dir,
        num_samples=args.count,
        train_ratio=args.train_ratio,
        log_path=args.log
    )
    
    print(f"Generated {len(logs)} genuine samples")


if __name__ == '__main__':
    main()
