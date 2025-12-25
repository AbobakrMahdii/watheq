import unittest
import numpy as np
import cv2
from pathlib import Path
from utils.roi_cropper import StampROIConfig, get_candidate_rois, crop_stamp_roi, evaluate_roi_candidate

class TestStampROI(unittest.TestCase):
    def setUp(self):
        # Create a dummy image with a white square (representing a stamp)
        self.image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # Put a "stamp" at bottom center area
        # Bottom center is roughly (500, 850)
        cv2.rectangle(self.image, (450, 800), (550, 900), (255, 255, 255), -1)
        
        self.reference = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.reference, (0, 0), (100, 100), (255, 255, 255), -1)
        
        self.config = StampROIConfig(
            x_ratio=0.5,
            y_ratio=0.85,
            width_ratio=0.1,
            height_ratio=0.1
        )

    def test_jitter_generation(self):
        candidates = get_candidate_rois(self.image, self.config, h_jitter=0.04, v_jitter=0.02, h_step=0.02, v_step=0.01)
        # H offsets: -0.04, -0.02, 0, 0.02, 0.04 (5)
        # V offsets: -0.02, -0.01, 0, 0.01, 0.02 (5)
        # Total: 25
        self.assertEqual(len(candidates), 25)

    def test_roi_selection_precision(self):
        # Slightly shift the image stamp
        shifted_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # Shifted by 2% right (500 -> 520)
        cv2.rectangle(shifted_image, (470, 800), (570, 900), (255, 255, 255), -1)
        
        candidates = get_candidate_rois(shifted_image, self.config, h_jitter=0.08, v_jitter=0.05, h_step=0.02, v_step=0.01)
        
        best_score = -1
        best_cfg = None
        
        for cfg in candidates:
            patch = crop_stamp_roi(shifted_image, cfg)
            score = evaluate_roi_candidate(patch, self.reference)
            if score > best_score:
                best_score = score
                best_cfg = cfg
                
        # Best x_ratio should be 0.52 (original 0.5 + 0.02 shift)
        self.assertAlmostEqual(best_cfg.x_ratio, 0.52, places=2)
        self.assertGreater(best_score, 0.9)

if __name__ == '__main__':
    unittest.main()
