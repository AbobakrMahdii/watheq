"""
Unit tests for ORB matcher module.

Tests:
- Output ranges are valid
- Identical images produce high match ratio
- Different images produce low match ratio
- Result dataclass works correctly
"""

import numpy as np
import pytest
import cv2

from classical.orb_matcher import (
    ORBResult,
    match_orb,
    classify_orb_signal,
    get_matching_keypoints
)


class TestORBResult:
    """Tests for ORBResult dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ORBResult(
            match_ratio=0.45,
            num_good_matches=25,
            total_keypoints_img1=100,
            total_keypoints_img2=80,
            total_matches=60
        )
        
        d = result.to_dict()
        
        assert d['match_ratio'] == 0.45
        assert d['num_good_matches'] == 25
        assert d['total_keypoints_img1'] == 100
        assert d['total_keypoints_img2'] == 80
        assert d['total_matches'] == 60


class TestMatchORB:
    """Tests for match_orb function."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image with features."""
        # Create image with some patterns for ORB to detect
        img = np.zeros((200, 200), dtype=np.uint8)
        
        # Add some shapes
        cv2.rectangle(img, (20, 20), (60, 60), 255, 2)
        cv2.circle(img, (120, 50), 30, 255, 2)
        cv2.rectangle(img, (100, 100), (180, 180), 255, 2)
        
        # Add noise for texture
        noise = np.random.randint(0, 50, (200, 200), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def test_identical_images(self, sample_image):
        """Test that identical images have high match ratio."""
        result = match_orb(sample_image, sample_image.copy())
        
        # Identical images should have very high match ratio
        assert result.match_ratio >= 0.8
        assert result.num_good_matches > 0
    
    def test_different_images(self, sample_image):
        """Test that different images have low match ratio."""
        # Create a completely different image
        different = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(different, (80, 80), (150, 150), 255, -1)
        
        result = match_orb(sample_image, different)
        
        # Very different images should have low match ratio
        assert result.match_ratio < 0.3
    
    def test_output_ranges(self, sample_image):
        """Test that output values are in valid ranges."""
        # Create slightly modified image
        modified = sample_image.copy()
        modified = cv2.GaussianBlur(modified, (3, 3), 1)
        
        result = match_orb(sample_image, modified)
        
        # match_ratio should be in [0, 1]
        assert 0 <= result.match_ratio <= 1
        
        # Counts should be non-negative
        assert result.num_good_matches >= 0
        assert result.total_keypoints_img1 >= 0
        assert result.total_keypoints_img2 >= 0
        assert result.total_matches >= 0
    
    def test_empty_image(self):
        """Test behavior with blank images (no features)."""
        blank = np.zeros((100, 100), dtype=np.uint8)
        
        result = match_orb(blank, blank)
        
        # Should handle gracefully
        assert result.match_ratio == 0.0
        assert result.num_good_matches == 0
    
    def test_color_images(self):
        """Test with color images."""
        # Create color test image
        color_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(color_img, (20, 20), (100, 100), (255, 255, 255), 2)
        
        result = match_orb(color_img, color_img.copy())
        
        # Should work with color images
        assert result.match_ratio >= 0
    
    def test_different_sizes(self):
        """Test with different sized images."""
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
        
        # Should not crash
        result = match_orb(img1, img2)
        assert 0 <= result.match_ratio <= 1
    
    def test_ratio_threshold_effect(self, sample_image):
        """Test that stricter ratio threshold produces fewer matches."""
        modified = sample_image.copy()
        modified = cv2.GaussianBlur(modified, (5, 5), 2)
        
        # Stricter threshold
        result_strict = match_orb(sample_image, modified, ratio_thresh=0.5)
        
        # More lenient threshold
        result_lenient = match_orb(sample_image, modified, ratio_thresh=0.9)
        
        # Stricter should have fewer or equal matches
        assert result_strict.num_good_matches <= result_lenient.num_good_matches


class TestClassifyORBSignal:
    """Tests for classify_orb_signal function."""
    
    def test_strong_genuine(self):
        """Test strong genuine classification."""
        result = ORBResult(
            match_ratio=0.45,
            num_good_matches=15,
            total_keypoints_img1=50,
            total_keypoints_img2=50,
            total_matches=30
        )
        
        thresholds = {
            'strong_genuine': 0.35,
            'suspicious': 0.15,
            'min_good_matches': 10
        }
        
        signal = classify_orb_signal(result, thresholds)
        assert signal == 'strong_genuine'
    
    def test_suspicious(self):
        """Test suspicious classification."""
        result = ORBResult(
            match_ratio=0.20,
            num_good_matches=8,
            total_keypoints_img1=50,
            total_keypoints_img2=50,
            total_matches=20
        )
        
        thresholds = {
            'strong_genuine': 0.35,
            'suspicious': 0.15,
            'min_good_matches': 10
        }
        
        signal = classify_orb_signal(result, thresholds)
        assert signal == 'suspicious'
    
    def test_forged(self):
        """Test forged classification."""
        result = ORBResult(
            match_ratio=0.10,
            num_good_matches=3,
            total_keypoints_img1=50,
            total_keypoints_img2=50,
            total_matches=10
        )
        
        thresholds = {
            'strong_genuine': 0.35,
            'suspicious': 0.15,
            'min_good_matches': 10
        }
        
        signal = classify_orb_signal(result, thresholds)
        assert signal == 'forged'
    
    def test_insufficient_matches(self):
        """Test that insufficient matches prevents strong_genuine."""
        result = ORBResult(
            match_ratio=0.50,  # High ratio
            num_good_matches=5,  # But too few matches
            total_keypoints_img1=10,
            total_keypoints_img2=10,
            total_matches=8
        )
        
        thresholds = {
            'strong_genuine': 0.35,
            'suspicious': 0.15,
            'min_good_matches': 10
        }
        
        signal = classify_orb_signal(result, thresholds)
        # Should not be strong_genuine due to low match count
        assert signal == 'suspicious'


class TestGetMatchingKeypoints:
    """Tests for get_matching_keypoints function."""
    
    def test_returns_keypoints(self):
        """Test that function returns keypoints and matches."""
        img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        # Add some structure
        cv2.rectangle(img, (20, 20), (80, 80), 255, 2)
        
        kp1, kp2, matches = get_matching_keypoints(img, img.copy())
        
        assert len(kp1) > 0
        assert len(kp2) > 0
        # For identical images, should have matches
        assert len(matches) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
