"""
ResNet50 training script for logo verification.

Implements transfer learning with:
- Frozen backbone (except layer4)
- Strong regularization for small datasets
- Early stopping
- Data augmentation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogoDataset(Dataset):
    """Dataset for logo classification."""
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = 'train',
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing train/val folders
            split: 'train' or 'val'
            transform: Torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Collect image paths and labels
        self.samples = []
        
        split_dir = self.root_dir / split
        
        # Genuine samples (label = 1)
        genuine_dir = split_dir / 'genuine'
        if genuine_dir.exists():
            for img_path in genuine_dir.glob('**/*.png'):
                self.samples.append((img_path, 1))
            for img_path in genuine_dir.glob('**/*.jpg'):
                self.samples.append((img_path, 1))
        
        # Forged samples (label = 0)
        forged_dir = split_dir / 'forged'
        if forged_dir.exists():
            for img_path in forged_dir.glob('**/*.png'):
                self.samples.append((img_path, 0))
            for img_path in forged_dir.glob('**/*.jpg'):
                self.samples.append((img_path, 0))
        
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def get_transforms(split: str = 'train') -> transforms.Compose:
    """
    Get data transforms for training/validation.
    
    Training includes strong augmentation for regularization.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train(
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Dict
) -> Dict:
    """
    Train the logo classifier.
    
    Args:
        data_dir: Directory containing train/val splits
        output_dir: Directory to save model and logs
        config: Training configuration
        
    Returns:
        Dictionary with training metrics
    """
    # Setup
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    element = config.get('element', 'logo')
    logger.info(f"Training for element: {element}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get training config
    train_config = config.get('training', {})
    epochs = train_config.get('epochs', 50)
    batch_size = train_config.get('batch_size', 16)
    lr = train_config.get('learning_rate', 0.0001)
    weight_decay = train_config.get('weight_decay', 0.01)
    patience = train_config.get('early_stopping_patience', 10)
    num_workers = train_config.get('num_workers', 0)  # 0 for Windows compatibility
    
    # Create datasets
    train_dataset = LogoDataset(data_dir, 'train', get_transforms('train'))
    val_dataset = LogoDataset(data_dir, 'val', get_transforms('val'))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    from models.resnet_classifier import LogoClassifier
    
    model = LogoClassifier(
        freeze_backbone=train_config.get('freeze_backbone', True),
        unfreeze_last_block=train_config.get('unfreeze_last_block', True),
        dropout=train_config.get('dropout', 0.5)
    )
    model = model.to(device)
    
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    logger.info(f"Total parameters: {model.get_total_params():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training loop
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
            f"LR: {current_lr:.6f}"
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f"{element}_resnet50.pt"
            model_path = output_dir / model_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, model_path)
            logger.info(f"Saved best model with val_acc: {val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_acc):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return {
        'best_val_acc': best_val_acc,
        'final_epoch': epoch + 1,
        'history': history
    }


def main():
    """CLI entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train element classifier')
    parser.add_argument('--data', type=str, default='data/logo',
                        help='Data directory')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file path')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from config')
    parser.add_argument('--element', type=str, default='logo',
                        choices=['logo', 'stamp'], help='Element type')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override element
    if 'element' not in config:
        config['element'] = 'logo'
        
    if args.element:
        config['element'] = args.element
        
    # Adjust defaults if element is stamp
    if config['element'] == 'stamp':
        if args.data == 'data/logo':
            args.data = 'data/stamp'
            
    # Override if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Train
    results = train(args.data, args.output, config)
    
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")


if __name__ == '__main__':
    main()
