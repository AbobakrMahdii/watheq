import unittest
import torch
import numpy as np
from models.siamese_net import SiameseVerifier

class TestSiameseSimilarity(unittest.TestCase):
    def setUp(self):
        self.verifier = SiameseVerifier()
        
    def test_identity_similarity(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        score = self.verifier.compute_similarity(img, img)
        self.assertAlmostEqual(score, 1.0, places=4)
        
    def test_different_images_similarity(self):
        img1 = np.zeros((224, 224, 3), dtype=np.uint8)
        img2 = np.ones((224, 224, 3), dtype=np.uint8) * 255
        score = self.verifier.compute_similarity(img1, img2)
        self.assertLess(score, 1.0) 
        self.assertGreaterEqual(score, -1.0)
        
    def test_classification_logic(self):
        thresh = {'strong_genuine': 0.8, 'suspicious': 0.6}
        self.assertEqual(self.verifier.classify_signal(0.9, thresh), 'strong_genuine')
        self.assertEqual(self.verifier.classify_signal(0.7, thresh), 'suspicious')
        self.assertEqual(self.verifier.classify_signal(0.5, thresh), 'forged')

if __name__ == '__main__':
    unittest.main()
