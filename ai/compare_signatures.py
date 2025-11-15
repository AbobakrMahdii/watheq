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
# -*- coding: utf-8 -*-
"""
compare_signatures.py
ملف مخصص لمقارنة التواقيع باستخدام:
1. SSIM
2. Cosine Similarity
يدعم الصور وملفات PDF.
"""

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_path

# ======================= 1. استخراج الصورة من PDF =======================
def extract_signature_from_pdf(pdf_path, output_path="temp_signature.png"):
    images = convert_from_path(pdf_path)
    img = np.array(images[0])  # أخذ أول صفحة فقط
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return output_path

# ======================= 2. حساب SSIM =======================
def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

# ======================= 3. حساب Cosine Similarity =======================
def calculate_cosine(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1_flat = img1.flatten().reshape(1, -1)
    img2_flat = img2.flatten().reshape(1, -1)
    return cosine_similarity(img1_flat, img2_flat)[0][0]

# ======================= 4. دالة المقارنة الرئيسية =======================
def compare_signatures(original_path, test_path, threshold=0.75):
    img_original = cv2.imread(original_path)
    img_test = cv2.imread(test_path)

    if img_original is None or img_test is None:
        raise ValueError("❌ خطأ: تحقق من مسارات الصور أو ملفات PDF")

    # حساب SSIM
    img1_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    ssim_score, _ = ssim(img1_gray, img2_gray, full=True)

    # حساب Cosine
    img1_flat = img_original.flatten().reshape(1, -1)
    img2_flat = cv2.resize(img_test, (img_original.shape[1], img_original.shape[0])).flatten().reshape(1, -1)
    cosine_score = cosine_similarity(img1_flat, img2_flat)[0][0]

    decision = "✅ أصلي" if ssim_score >= threshold and cosine_score >= threshold else "❌ مزور"

    return {
        "SSIM": round(ssim_score, 3),
        "Cosine": round(cosine_score, 3),
        "Threshold": threshold,
        "Decision": decision
    }

# ======================= 5. تجربة سريعة =======================
if __name__ == "__main__":
    original = "datasets/custom_signature_dataset_pre/test/genuine/genuine_00024.png"
    test = "datasets/custom_signature_dataset_pre/test/forged/forged_00001.png"

    result = compare_signatures(original, test)
    print("🔹 نتيجة المقارنة:", result)

