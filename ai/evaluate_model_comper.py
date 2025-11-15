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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import seaborn as sns
from compare_signatures import compare_signatures

# ==================== 1. تجهيز المسارات ====================
base_dir = "datasets/custom_signature_dataset_pre/test"
genuine_dir = os.path.join(base_dir, "genuine")
forged_dir = os.path.join(base_dir, "forged")

# فلترة أي ملفات غير صور
def list_images(path):
    return sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

genuine_files = list_images(genuine_dir)
forged_files = list_images(forged_dir)

# ==================== 2. اختبار جميع العتبات ====================
def evaluate_all_thresholds():
    thresholds = np.arange(0.3, 0.96, 0.05)
    best_threshold = 0
    best_f1 = 0
    results_per_threshold = []

    for threshold in thresholds:
        y_true, y_pred = [], []

        for gf in tqdm(genuine_files, desc=f" Testing Threshold={threshold:.2f}"):
            original_path = os.path.join(genuine_dir, gf)
            for ff in forged_files:
                test_path = os.path.join(forged_dir, ff)
                result = compare_signatures(original_path, test_path, threshold=threshold)

                # نعتبر الصور المتشابهة Genuine فقط لو أصلية × أصلية
                y_true.append(0)
                y_pred.append(1 if result["Decision"] == "أصلي " else 0)

            for gf2 in genuine_files:
                test_path = os.path.join(genuine_dir, gf2)
                result = compare_signatures(original_path, test_path, threshold=threshold)
                y_true.append(1)
                y_pred.append(1 if result["Decision"] == "أصلي " else 0)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results_per_threshold.append((threshold, acc, prec, rec, f1))

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return results_per_threshold, best_threshold

# ==================== 3. تشغيل التقييم ====================
if __name__ == "__main__":
    print("\nبدء التقييم ...\n")
    results, best_threshold = evaluate_all_thresholds()

    # ==================== 4. عرض أفضل عتبة ====================
    best_result = max(results, key=lambda x: x[4])
    print(f"\nأفضل عتبة: {best_result[0]:.2f}")
    print(f"الدقة: {best_result[1]:.3f}")
    print(f"Precision: {best_result[2]:.3f}")
    print(f"Recall: {best_result[3]:.3f}")
    print(f"F1-score: {best_result[4]:.3f}")

    # ==================== 5. رسم مقارنة F1 للعتبات ====================
    thresholds = [r[0] for r in results]
    f1_scores = [r[4] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, marker='o', color='blue', linewidth=2)
    plt.title("F1-score vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.show()

