"""
Siamese Network for one-shot stamp similarity verification.

This module provides a Siamese architecture using a shared ResNet backbone
to compute similarity between a cropped stamp patch and the official reference.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseNet(nn.Module):
    """
    Siamese Network for computing similarity between two images.
    
    Uses a shared ResNet50 backbone to extract features.
    """
    
    def __init__(self, feature_dim: int = 2048):
        super().__init__()
        # Load pretrained ResNet50 without the classification head
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Output is (B, 2048, 1, 1) -> flatten to (B, 2048)
        self.feature_dim = feature_dim

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for a single input."""
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two inputs."""
        feat1 = self.forward_once(input1)
        feat2 = self.forward_once(input2)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(feat1, feat2)
        return similarity


class SiameseVerifier:
    """
    High-level wrapper for Siamese-based similarity verification.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = SiameseNet()
        
        # If we have a custom trained Siamese model, load it
        # Otherwise, use the pretrained ImageNet features (zero-shot/one-shot)
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Preprocess image for ResNet."""
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        return tensor.unsqueeze(0).to(self.device)

    def compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute similarity between two images."""
        t1 = self.preprocess(img1)
        t2 = self.preprocess(img2)
        
        with torch.no_grad():
            similarity = self.model(t1, t2)
            
        return float(similarity.cpu().item())

    def classify_signal(self, score: float, thresholds: Dict[str, float]) -> str:
        """Classify similarity score into signal strength."""
        strong = thresholds.get('strong_genuine', 0.80)
        suspicious = thresholds.get('suspicious', 0.60)
        
        if score >= strong:
            return 'strong_genuine'
        elif score >= suspicious:
            return 'suspicious'
        else:
            return 'forged'
