"""
Element Binary Classifier for Document Verification.

مصنف ثنائي لعناصر الوثيقة — أصلي أم مزور:
- يستخدم EfficientNet-B0 كعمود فقري مع رأس تصنيف ثنائي
- يتعلم من البيانات المولدة (400 أصلي + 400 مزور)
- يخرج احتمال مباشر: 0.0 (مزور) إلى 1.0 (أصلي)
- لا يحتاج مقارنة مرجعية أثناء التحقق — القرار من التعلم فقط

Usage:
    # Training
    classifier = ElementClassifier()
    classifier.train_from_dirs(genuine_dir, forged_dir, save_path="weights/identity_logo_main.pt")

    # Inference
    classifier = ElementClassifier.load("weights/identity_logo_main.pt")
    score = classifier.predict(element_image)  # 0.0-1.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Input image size (matches EfficientNet-B0 default)
INPUT_SIZE = (224, 224)


def _preprocess_for_classifier(image: np.ndarray) -> np.ndarray:
    """Preprocess a single image for the classifier: resize, to RGB, normalize."""
    if image is None or image.size == 0:
        raise ValueError("Empty image provided")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


# ── Module-level Dataset (picklable on Windows for num_workers > 0) ──────────

class _ElementDataset:
    """Simple dataset for genuine/forged element images."""

    def __init__(self, items: list):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch

        path, label = self.items[idx]
        img = cv2.imdecode(
            np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            img = np.zeros((INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.uint8)
        proc = _preprocess_for_classifier(img)
        tensor = torch.from_numpy(proc).permute(2, 0, 1)  # (C, H, W)
        return tensor, torch.tensor([float(label)], dtype=torch.float32)


class ElementClassifier:
    """
    Binary classifier: genuine (1.0) vs forged (0.0) for a single element type.

    Architecture:
        EfficientNet-B0 backbone (pretrained) → AdaptiveAvgPool → Flatten
        → Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→1) → Sigmoid
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self._build_attempted = False

    def _build_model(self):
        """Build the classifier architecture."""
        if self._build_attempted:
            return
        self._build_attempted = True

        try:
            import torch
            import torch.nn as nn

            class _Classifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    try:
                        from torchvision.models import (
                            efficientnet_b0,
                            EfficientNet_B0_Weights,
                        )

                        backbone = efficientnet_b0(
                            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
                        )
                    except (ImportError, TypeError):
                        from torchvision.models import efficientnet_b0

                        backbone = efficientnet_b0(pretrained=True)

                    # Use all layers except the final classifier
                    self.features = nn.Sequential(*list(backbone.children())[:-1])
                    self.flatten = nn.Flatten()
                    self.head = nn.Sequential(
                        nn.Linear(1280, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 1),
                        # No Sigmoid here — use BCEWithLogitsLoss for AMP safety
                    )

                def forward(self, x):
                    x = self.features(x)
                    x = self.flatten(x)
                    x = self.head(x)
                    return x

            self.model = _Classifier()
            self.model.to(self.device)
            self.model.eval()
            logger.info("ElementClassifier model built successfully")

        except ImportError as e:
            logger.error(f"PyTorch/torchvision not available: {e}")

    @classmethod
    def load(cls, weight_path: str | Path, device: str = "auto") -> "ElementClassifier":
        """Load a trained classifier from a .pt file."""
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        obj = cls(device=device)
        obj._build_model()
        if obj.model is None:
            raise RuntimeError("Cannot build model — PyTorch/torchvision missing")

        state = torch.load(str(weight_path), map_location=device, weights_only=True)
        obj.model.load_state_dict(state)
        obj.model.eval()
        logger.info(f"Loaded ElementClassifier from {weight_path}")
        return obj

    def predict(self, image: np.ndarray) -> float:
        """
        Predict genuineness probability for a single element image.

        Args:
            image: Element crop (BGR, any size)

        Returns:
            Float 0.0 (forged) to 1.0 (genuine)
        """
        if self.model is None:
            self._build_model()
        if self.model is None:
            logger.warning("No model available, returning fallback 0.5")
            return 0.5

        import torch

        preprocessed = _preprocess_for_classifier(image)
        tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
        return float(torch.sigmoid(output).item())

    def train_from_dirs(
        self,
        genuine_dir: Path,
        forged_dir: Path,
        save_path: Path,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        val_split: float = 0.2,
        patience: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the binary classifier from directories of genuine/forged images.

        Args:
            genuine_dir: Directory with genuine augmented images
            forged_dir: Directory with forged images
            save_path: Where to save the trained .pt weights
            epochs: Max training epochs
            batch_size: Batch size
            lr: Learning rate
            val_split: Fraction for validation
            patience: Early stopping patience

        Returns:
            Training stats dict
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, random_split

        # Build fresh model with pretrained backbone
        self._build_attempted = False
        self._build_model()
        if self.model is None:
            return {"status": "error", "message": "Cannot build model"}

        # Collect image paths + labels
        image_paths: List[Tuple[Path, int]] = []
        for p in sorted(genuine_dir.glob("*.png")):
            image_paths.append((p, 1))  # 1 = genuine
        for p in sorted(forged_dir.glob("*.png")):
            image_paths.append((p, 0))  # 0 = forged
        # Also check jpg
        for p in sorted(genuine_dir.glob("*.jpg")):
            image_paths.append((p, 1))
        for p in sorted(forged_dir.glob("*.jpg")):
            image_paths.append((p, 0))

        if len(image_paths) < 10:
            return {
                "status": "error",
                "message": f"Not enough training data: {len(image_paths)} images",
            }

        logger.info(
            f"  Training dataset: {len(image_paths)} images "
            f"({sum(1 for _, l in image_paths if l == 1)} genuine, "
            f"{sum(1 for _, l in image_paths if l == 0)} forged)"
        )

        dataset = _ElementDataset(image_paths)
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        use_cuda = self.device != "cpu"

        # GPU: bigger batch + parallel loading; CPU: conservative
        if use_cuda:
            batch_size = max(batch_size, 64)  # 4GB VRAM handles 64 easily
            num_workers = 4
            torch.backends.cudnn.benchmark = True  # optimize kernels for fixed input size
        else:
            num_workers = 0

        logger.info(
            f"  Device: {self.device.upper()} | batch_size={batch_size} | "
            f"num_workers={num_workers} | AMP={'on' if use_cuda else 'off'}"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
        )

        # Freeze backbone initially, train head only for first 3 epochs
        for param in self.model.features.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

        criterion = nn.BCEWithLogitsLoss()  # AMP-safe, includes sigmoid internally
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        self.model.train()
        best_val_loss = float("inf")
        best_val_acc = 0.0
        epochs_no_improve = 0
        history = []

        # Mixed precision for GPU — ~2x throughput with float16
        use_amp = use_cuda
        scaler = torch.amp.GradScaler(enabled=use_amp)

        for epoch in range(epochs):
            # Unfreeze backbone after epoch 2 (earlier fine-tuning for deep training)
            if epoch == 2:
                for param in self.model.features.parameters():
                    param.requires_grad = True
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr * 0.1)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2
                )
                logger.info("  Epoch 3: Unfreezing backbone for fine-tuning")

            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.device, enabled=use_amp):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * images.size(0)
                preds = (outputs >= 0.0).float()  # logits: >=0 means probability >=0.5
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validate
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    with torch.amp.autocast(device_type=self.device, enabled=use_amp):
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = (outputs >= 0.0).float()  # logits threshold
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "lr": current_lr,
            }
            history.append(epoch_info)
            logger.info(
                f"  Epoch {epoch+1:2d}/{epochs}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.1%} | lr={current_lr:.2e}"
            )

            # Early stopping on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_no_improve = 0
                # Save best model
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(save_path))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        # Reload best weights
        self.model.load_state_dict(
            torch.load(str(save_path), map_location=self.device, weights_only=True)
        )
        self.model.eval()

        return {
            "status": "success",
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc": round(best_val_acc, 4),
            "epochs_trained": len(history),
            "total_images": len(image_paths),
            "history": history,
        }
