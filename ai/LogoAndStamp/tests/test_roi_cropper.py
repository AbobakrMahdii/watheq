"""
Unit tests for ROI cropper module.

Tests:
- Crop dimensions are correct
- Boundary handling works
- Configuration loading works
"""

import numpy as np
import pytest

from utils.roi_cropper import (
    ROIConfig,
    calculate_roi_bounds,
    crop_logo_roi,
    get_roi_dimensions
)


class TestROIConfig:
    """Tests for ROIConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ROIConfig()
        assert config.x_start_ratio == 0.35
        assert config.x_end_ratio == 0.65
        assert config.y_start_ratio == 0.0
        assert config.y_end_ratio == 0.20
    
    def test_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {
            'roi': {
                'x_start_ratio': 0.30,
                'x_end_ratio': 0.70,
                'y_start_ratio': 0.05,
                'y_end_ratio': 0.25
            }
        }
        config = ROIConfig.from_dict(config_dict)
        assert config.x_start_ratio == 0.30
        assert config.x_end_ratio == 0.70
        assert config.y_start_ratio == 0.05
        assert config.y_end_ratio == 0.25
    
    def test_from_empty_dict(self):
        """Test default values when dict is empty."""
        config = ROIConfig.from_dict({})
        assert config.x_start_ratio == 0.35


class TestCalculateROIBounds:
    """Tests for calculate_roi_bounds function."""
    
    def test_standard_image(self):
        """Test bounds calculation on standard image."""
        image_shape = (500, 1000, 3)  # H, W, C
        config = ROIConfig(
            x_start_ratio=0.35,
            x_end_ratio=0.65,
            y_start_ratio=0.0,
            y_end_ratio=0.20
        )
        
        x_start, y_start, x_end, y_end = calculate_roi_bounds(image_shape, config)
        
        # Expected: x from 350 to 650, y from 0 to 100
        assert x_start == 350
        assert x_end == 650
        assert y_start == 0
        assert y_end == 100
    
    def test_small_image(self):
        """Test bounds on small image."""
        image_shape = (100, 200)  # Grayscale
        config = ROIConfig()
        
        x_start, y_start, x_end, y_end = calculate_roi_bounds(image_shape, config)
        
        assert x_start == 70   # 200 * 0.35
        assert x_end == 130    # 200 * 0.65
        assert y_start == 0
        assert y_end == 20     # 100 * 0.20
    
    def test_invalid_ratios(self):
        """Test error on invalid ratios."""
        image_shape = (500, 1000)
        
        # x_start >= x_end should fail
        config = ROIConfig(x_start_ratio=0.7, x_end_ratio=0.3)
        with pytest.raises(ValueError):
            calculate_roi_bounds(image_shape, config)
    
    def test_out_of_range_ratios(self):
        """Test error on out-of-range ratios."""
        image_shape = (500, 1000)
        
        # Ratio > 1 should fail
        config = ROIConfig(x_end_ratio=1.5)
        with pytest.raises(ValueError):
            calculate_roi_bounds(image_shape, config)


class TestCropLogoROI:
    """Tests for crop_logo_roi function."""
    
    def test_crop_color_image(self):
        """Test cropping a color image."""
        # Create test image (1000x500, BGR)
        image = np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8)
        config = ROIConfig()
        
        cropped = crop_logo_roi(image, config)
        
        # Expected dimensions (use int for comparison)
        expected_w = int(1000 * (0.65 - 0.35))  # 300
        expected_h = int(500 * (0.20 - 0.0))    # 100
        
        assert cropped.shape[1] == expected_w
        assert cropped.shape[0] == expected_h
        assert cropped.shape[2] == 3  # Still color
    
    def test_crop_grayscale_image(self):
        """Test cropping a grayscale image."""
        image = np.random.randint(0, 255, (500, 1000), dtype=np.uint8)
        config = ROIConfig()
        
        cropped = crop_logo_roi(image, config)
        
        assert len(cropped.shape) == 2  # Still grayscale
        assert cropped.shape == (100, 300)
    
    def test_crop_preserves_content(self):
        """Test that crop preserves correct region."""
        # Create image with known pattern
        image = np.zeros((500, 1000, 3), dtype=np.uint8)
        # Mark the ROI region with white
        image[0:100, 350:650] = 255
        
        config = ROIConfig()
        cropped = crop_logo_roi(image, config)
        
        # Cropped region should be all white
        assert np.all(cropped == 255)
    
    def test_crop_with_dict_config(self):
        """Test cropping with dict config."""
        image = np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8)
        config_dict = {'roi': {'x_start_ratio': 0.4, 'x_end_ratio': 0.6}}
        
        cropped = crop_logo_roi(image, config_dict)
        
        # Allow 1px tolerance for rounding
        expected_w = int(1000 * (0.6 - 0.4))  # 200
        assert abs(cropped.shape[1] - expected_w) <= 1


class TestGetROIDimensions:
    """Tests for get_roi_dimensions function."""
    
    def test_dimensions(self):
        """Test dimension calculation."""
        image_shape = (500, 1000)
        config = ROIConfig()
        
        width, height = get_roi_dimensions(image_shape, config)
        
        assert width == 300   # 1000 * 0.3
        assert height == 100  # 500 * 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
