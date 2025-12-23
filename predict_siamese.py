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
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# التحويلات
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_siamese(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

def predict_siamese(model, image_path, ref_path, device):
    img1 = preprocess(image_path).to(device)
    img2 = preprocess(ref_path).to(device)

    with torch.no_grad():
        out = model(img1, img2)
        sim = torch.sigmoid(out).item()

    decision = "Genuine (أصلي)" if sim > 0.5 else "Forged (مزور)"
    return sim, decision

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="اختبار نموذج Siamese للتواقيع")
    parser.add_argument("--model", required=True, help="مسار ملف الموديل (.pth)")
    parser.add_argument("--image", required=True, help="مسار صورة الاختبار")
    parser.add_argument("--ref", required=True, help="مسار الصورة المرجعية")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_siamese(args.model, device)

    sim, decision = predict_siamese(model, args.image, args.ref, device)

    print("Model: Siamese Network")
    print("Test Image:", args.image)
    print("Reference Image:", args.ref)
    print(f"Similarity Score: {sim:.2f}")
    print("Decision:", decision)

    # عرض الصور
    if os.environ.get("DISPLAY") or "ipykernel" in sys.modules:
        fig, axs = plt.subplots(1,2, figsize=(6,3))
        axs[0].imshow(Image.open(args.image), cmap="gray")
        axs[0].set_title("Test Image")
        axs[0].axis("off")

        axs[1].imshow(Image.open(args.ref), cmap="gray")
        axs[1].set_title("Reference")
        axs[1].axis("off")

        plt.suptitle(f"{decision} (Similarity={sim:.2f})")
        plt.show()

