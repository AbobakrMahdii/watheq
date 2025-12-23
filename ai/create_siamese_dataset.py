# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import shutil
import random

# المسارات
SOURCE_DIR = "watheeq_signatures/datasets/custom_signature_dataset_pre"  # عدل حسب مصدر بياناتك
TARGET_DIR = "watheeq_signatures/datasets/siamese_dataset"

# إنشاء المجلدات الأساسية
folders = [
    "train/genuine", "train/forged",
    "val/genuine", "val/forged"
]

for folder in folders:
    path = os.path.join(TARGET_DIR, folder)
    os.makedirs(path, exist_ok=True)

print(" تم إنشاء مجلدات السيامي بنجاح!")

# الحصول على جميع الصور من المصدر
genuine_images = [f for f in os.listdir(os.path.join(SOURCE_DIR, "genuine"))]
forged_images = [f for f in os.listdir(os.path.join(SOURCE_DIR, "forged"))]

# تقسيم البيانات بنسبة 80% تدريب و20% تقييم
def split_and_copy(images, label):
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img in train_imgs:
        src = os.path.join(SOURCE_DIR, label, img)
        dst = os.path.join(TARGET_DIR, "train", label, img)
        shutil.copy(src, dst)

    for img in val_imgs:
        src = os.path.join(SOURCE_DIR, label, img)
        dst = os.path.join(TARGET_DIR, "val", label, img)
        shutil.copy(src, dst)

    print(f" {label}: تم نسخ {len(train_imgs)} للتدريب و {len(val_imgs)} للتقييم")

# نسخ الصور
split_and_copy(genuine_images, "genuine")
split_and_copy(forged_images, "forged")

print("\n تم تجهيز بيانات السيامي بنجاح في:", TARGET_DIR)

