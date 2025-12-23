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
from sklearn.model_selection import train_test_split

# مسارات البيانات الأصلية
src_train_genuine = "datasets/custom_signature_dataset_pre/train/genuine"
src_train_forged = "datasets/custom_signature_dataset_pre/train/forged"

src_val_genuine = "datasets/custom_signature_dataset_pre/val/genuine"
src_val_forged = "datasets/custom_signature_dataset_pre/val/forged"

# مسارات الهدف
dst_train_genuine = "datasets/siamese_dataset/train/genuine"
dst_train_forged = "datasets/siamese_dataset/train/forged"
dst_val_genuine = "datasets/siamese_dataset/val/genuine"
dst_val_forged = "datasets/siamese_dataset/val/forged"

# إنشاء مجلدات الهدف
os.makedirs(dst_train_genuine, exist_ok=True)
os.makedirs(dst_train_forged, exist_ok=True)
os.makedirs(dst_val_genuine, exist_ok=True)
os.makedirs(dst_val_forged, exist_ok=True)

def copy_files(src, dst):
    """ينسخ جميع الصور من المصدر إلى الوجهة"""
    if not os.path.exists(src):
        print(f" المسار غير موجود: {src}")
        return
    files = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

# نسخ الصور للتدريب
copy_files(src_train_genuine, dst_train_genuine)
copy_files(src_train_forged, dst_train_forged)

# نسخ الصور للتحقق
copy_files(src_val_genuine, dst_val_genuine)
copy_files(src_val_forged, dst_val_forged)

print(" تم تجهيز بيانات السيامي بنجاح!")

