"""
ResNet50-based logo classifier for genuine/forged detection.

This module provides a transfer learning approach using pretrained ResNet50
with a custom classification head for binary logo verification.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class LogoClassifier(nn.Module):
    """
    ResNet50-based binary classifier for logo verification.
    
    Architecture:
    - Pretrained ResNet50 backbone (optionally frozen)
    - Last residual block (layer4) can be unfrozen for fine-tuning
    - Custom classification head with dropout for regularization
    """
    
    def __init__(
        self,
        freeze_backbone: bool = True,
        unfreeze_last_block: bool = True,
        dropout: float = 0.5,
        num_classes: int = 2
    ):
        """
        Initialize the logo classifier.
        
        Args:
            freeze_backbone: Whether to freeze all backbone layers
            unfreeze_last_block: Whether to unfreeze layer4 (last block)
            dropout: Dropout probability for regularization
            num_classes: Number of output classes (default 2: genuine/forged)
        """
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Unfreeze last residual block if requested
        if freeze_backbone and unfreeze_last_block:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
        
        # Get the number of features from backbone
        num_features = self.backbone.fc.in_features
        
        # Replace classifier head with custom head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_trainable_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class LogoVerifier:
    """
    High-level wrapper for logo verification using the classifier.
    
    Handles image preprocessing, model loading, and prediction.
    """
    
    CLASS_NAMES = ['forged', 'genuine']
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the verifier.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Auto-select device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = LogoClassifier()
        
        # Load weights if provided
        if model_path is not None:
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Preprocess an image for model input.
        
        Args:
            image: Input image (BGR numpy array)
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Resize
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def predict(
        self,
        image: np.ndarray
    ) -> Tuple[str, float]:
        """
        Predict whether a logo image is genuine or forged.
        
        Args:
            image: Input logo image (BGR numpy array)
            
        Returns:
            Tuple of (prediction_label, confidence)
            - prediction_label: 'genuine' or 'forged'
            - confidence: Probability of predicted class (0-1)
        """
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
        # Get prediction
        prob_values = probabilities.cpu().numpy()[0]
        predicted_class = int(prob_values.argmax())
        confidence = float(prob_values[predicted_class])
        
        label = self.CLASS_NAMES[predicted_class]
        
        return label, confidence
    
    def predict_with_details(
        self,
        image: np.ndarray
    ) -> Dict:
        """
        Get detailed prediction results.
        
        Args:
            image: Input logo image
            
        Returns:
            Dictionary with prediction details
        """
        input_tensor = self.preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        prob_values = probabilities.cpu().numpy()[0]
        predicted_class = int(prob_values.argmax())
        
        return {
            'prediction': self.CLASS_NAMES[predicted_class],
            'confidence': float(prob_values[predicted_class]),
            'probabilities': {
                'forged': float(prob_values[0]),
                'genuine': float(prob_values[1])
            },
            'device': str(self.device)
        }
    
    def classify_signal(
        self,
        prediction: str,
        confidence: float,
        thresholds: Dict[str, float]
    ) -> str:
        """
        Classify prediction into signal categories.
        
        Args:
            prediction: 'genuine' or 'forged'
            confidence: Prediction confidence
            thresholds: Dict with 'strong_genuine' and 'suspicious'
            
        Returns:
            'strong_genuine', 'suspicious', or 'forged'
        """
        strong_thresh = thresholds.get('strong_genuine', 0.80)
        suspicious_thresh = thresholds.get('suspicious', 0.50)
        
        if prediction == 'genuine' and confidence >= strong_thresh:
            return 'strong_genuine'
        elif confidence >= suspicious_thresh:
            return 'suspicious'
        else:
            return 'forged' if prediction == 'forged' else 'suspicious'


class StampVerifier(LogoVerifier):
    """
    High-level wrapper for stamp verification using the classifier.
    
    Inherits preprocessing and prediction logic from LogoVerifier
    but typically uses a different checkpoint.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the stamp verifier.
        
        Args:
            model_path: Path to trained stamp model checkpoint
            device: Device to use
        """
        # Default stamp model path if not provided
        if model_path is None:
            model_path = Path('models/stamp_resnet50.pt')
            
        super().__init__(model_path=model_path, device=device)


def create_model(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    unfreeze_last_block: bool = True,
    dropout: float = 0.5
) -> LogoClassifier:
    """
    Factory function to create a LogoClassifier model.
    
    Args:
        pretrained: Not used (always uses pretrained weights)
        freeze_backbone: Whether to freeze backbone layers
        unfreeze_last_block: Whether to unfreeze layer4
        dropout: Dropout probability
        
    Returns:
        Initialized LogoClassifier
    """
    return LogoClassifier(
        freeze_backbone=freeze_backbone,
        unfreeze_last_block=unfreeze_last_block,
        dropout=dropout
    )


def save_model(
    model: LogoClassifier,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        metrics: Optional training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)
