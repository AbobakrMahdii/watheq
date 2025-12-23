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
# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تجربة نموذج ResNet18 Fine-Tuned للتواقيع:
- يدخل صورة توقيع (أصلي أو مزور).
- يطبع القرار (أصلي / مزور) مع درجة الثقة.
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# تجهيز البيانات
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

def load_model(model_path, device):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # 2 Classes: Genuine / Forged
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)
    label = " Genuine (أصلي)" if pred.item() == 1 else " Forged (مزور)"
    return label, confidence.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="مسار المودل المدرب (pth).")
    parser.add_argument("--image", required=True, help="مسار صورة التوقيع.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    image_tensor = preprocess_image(args.image)
    label, confidence = predict(model, image_tensor, device)

    print(f"\n Image: {args.image}")
    print(f" Decision: {label}")
    print(f" Confidence: {confidence:.2f}\n")

if __name__ == "__main__":
    main()

