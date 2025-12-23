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
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# تجهيز الصور
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def load_model(model_path, device):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, device):
    if not os.path.exists(image_path):
        print(f" الملف غير موجود: {image_path}")
        return None, None, None

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = "Genuine (أصلي)" if pred.item()==1 else "Forged (مزور)"
    return label, conf.item(), img

#  نسخة Jupyter: عرض مباشر
def run_in_notebook(model_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    label, conf, img = predict(model, image_path, device)

    if label is not None:
        print("Model: ResNet18 (Fine-Tuned)")
        print("Image:", image_path)
        print("Decision:", label)
        print(f"Confidence: {conf*100:.2f}%")

        # عرض مباشر داخل الجبتور
        plt.imshow(img, cmap="gray")
        plt.title(f"{label} ({conf*100:.2f}%)")
        plt.axis("off")
        plt.show()

