"""Tests for AI ElementClassifier model.

ElementClassifier is a wrapper class around a PyTorch _Classifier model.
It exposes: predict(np_image)->float, load(path)->cls, _build_model(), .model, .device
"""

import pytest

# Skip all AI tests if PyTorch is not available
pytest.importorskip("torch")

import torch
import numpy as np
from PIL import Image


@pytest.mark.unit
class TestElementClassifier:
    """Test suite for ElementClassifier model."""

    @pytest.fixture
    def sample_np_image(self):
        """Create a sample BGR numpy image for testing."""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_model_import(self):
        """Test that ElementClassifier can be imported."""
        from ai.models.element_classifier import ElementClassifier

        assert ElementClassifier is not None

    def test_model_initialization(self):
        """Test model initialization — wrapper attributes."""
        from ai.models.element_classifier import ElementClassifier

        model = ElementClassifier()
        assert model is not None
        assert hasattr(model, "device")
        assert hasattr(model, "model")
        assert model.device == "cpu"
        # model is None until _build_model is called
        assert model.model is None

        # After building, the internal nn.Module should exist
        model._build_model()
        assert model.model is not None

    def test_model_forward_pass(self, sample_np_image):
        """Test model forward pass via predict()."""
        from ai.models.element_classifier import ElementClassifier

        model = ElementClassifier()
        model._build_model()

        # predict() takes a BGR numpy array and returns a float
        result = model.predict(sample_np_image)

        assert isinstance(result, float)

    def test_model_output_range(self, sample_np_image):
        """Test that model output is in valid range [0, 1]."""
        from ai.models.element_classifier import ElementClassifier

        model = ElementClassifier()
        model._build_model()

        result = model.predict(sample_np_image)

        # predict() applies sigmoid, so output should be between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_model_device_compatibility(self):
        """Test that model works on CPU (and GPU if available)."""
        from ai.models.element_classifier import ElementClassifier

        # Test CPU explicitly
        model = ElementClassifier(device="cpu")
        model._build_model()
        assert model.device == "cpu"
        assert model.model is not None

        # Verify the internal model's parameters are on CPU
        param = next(model.model.parameters())
        assert param.device == torch.device("cpu")

        # Test GPU if available
        if torch.cuda.is_available():
            gpu_model = ElementClassifier(device="cuda")
            gpu_model._build_model()
            param = next(gpu_model.model.parameters())
            assert param.device.type == "cuda"

    def test_model_save_load(self, tmp_path):
        """Test saving and loading model weights via the class API."""
        from ai.models.element_classifier import ElementClassifier

        model = ElementClassifier()
        model._build_model()

        # Save the internal model's state_dict
        save_path = tmp_path / "test_model.pt"
        torch.save(model.model.state_dict(), save_path)

        # Load using the classmethod
        loaded_model = ElementClassifier.load(save_path, device="cpu")
        assert loaded_model.model is not None

        # Verify parameters match
        for p1, p2 in zip(model.model.parameters(), loaded_model.model.parameters()):
            assert torch.equal(p1, p2)


@pytest.mark.slow
class TestElementClassifierTraining:
    """Test training-related functionality."""

    def test_model_training_mode(self):
        """Test switching between train and eval modes on the internal model."""
        from ai.models.element_classifier import ElementClassifier

        model = ElementClassifier()
        model._build_model()

        # Test training mode on the internal nn.Module
        model.model.train()
        assert model.model.training is True

        # Test eval mode
        model.model.eval()
        assert model.model.training is False

    def test_gradient_computation(self):
        """Test that gradients are computed during training."""
        from ai.models.element_classifier import ElementClassifier
        import torchvision.transforms as T

        sample_image = Image.new("RGB", (224, 224), color="blue")

        model = ElementClassifier()
        model._build_model()
        model.model.train()

        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img_tensor = transform(sample_image).unsqueeze(0)
        target = torch.tensor([[1.0]])

        # Forward pass through the internal model
        output = model.model(img_tensor)

        # Compute loss
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in model.model.parameters():
            if param.requires_grad:
                assert param.grad is not None
