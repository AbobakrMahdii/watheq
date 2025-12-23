import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os

# ======================================
# 1. الإعدادات الأساسية
# ======================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
MODEL_PATH = "best_resnet18.pth"

# ======================================
# 2. Augmentation للبيانات
# ======================================
train_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(150, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ======================================
# 3. تحميل البيانات
# ======================================
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transforms)
val_dataset = datasets.ImageFolder("dataset/test", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================
# 4. إنشاء موديل ResNet18
# ======================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

# ======================================
# 5. الخسارة والمُحسّن وجدول التعلم
# ======================================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3, verbose=True)

# ======================================
# 6. دوال التدريب والاختبار
# ======================================
def train_one_epoch(loader):
    model.train()
    running_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct += preds.eq(labels.byte()).sum().item()
    return running_loss / len(loader), correct / len(loader.dataset)

def evaluate(loader):
    model.eval()
    running_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += preds.eq(labels.byte()).sum().item()
    return running_loss / len(loader), correct / len(loader.dataset)

# ======================================
# 7. حلقة التدريب الرئيسية مع EarlyStopping
# ======================================
best_val_loss = float("inf")
patience_counter = 0
patience = 5

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # EarlyStopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

# ======================================
# 8. رسم منحنيات التدريب
# ======================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss Curves")
plt.show()

# ======================================
# 9. تحميل أفضل موديل للتقييم النهائي
# ======================================
model.load_state_dict(torch.load(MODEL_PATH))
final_loss, final_acc = evaluate(val_loader)
print(f"🔥 Final Validation Accuracy: {final_acc:.4f}")
