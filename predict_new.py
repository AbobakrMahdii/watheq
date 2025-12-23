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
# predict_new.py
# -*- coding: utf-8 -*-
import argparse, os, glob
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# نفس الإعدادات المستخدمة بالتدريب
IMG_SIZE = 224
MEAN = [0.5, 0.5, 0.5]
STD  = [0.5, 0.5, 0.5]

def load_image(path):
    img = Image.open(path).convert("RGB")  # حتى لو كانت رمادي نخليها 3 قنوات
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0)  # 1x3x224x224
    return tensor

def build_resnet18(num_classes=2):
    # نموذج resnet18 خفيف بدون تحميل أوزان torchvision (علشان بيئة بدون GPU)
    import torch.nn as nn
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@torch.no_grad()
def predict_one(model, img_path, threshold=0.48, device="cpu"):
    x = load_image(img_path).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    p_forged, p_genuine = float(probs[0]), float(probs[1])
    decision = "genuine" if p_genuine >= threshold else "forged"
    return {
        "path": img_path,
        "p_forged": round(p_forged, 4),
        "p_genuine": round(p_genuine, 4),
        "threshold": threshold,
        "decision": decision
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/resnet18_finetuned.pth", help="path to finetuned model")
    ap.add_argument("--image", default=None, help="single image path")
    ap.add_argument("--folder", default=None, help="folder of images (png/jpg)")
    ap.add_argument("--threshold", type=float, default=0.48, help="genuine threshold")
    args = ap.parse_args()

    device = "cpu"
    model = build_resnet18(num_classes=2).to(device)

    # نحاول نحمّل state_dict (وليس model كامل)
    state = torch.load(args.model, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # في بعض سكربتاتنا حفظنا مباشرة state_dict
    if isinstance(state, dict) and any(k.startswith("fc.") or k.startswith("layer") for k in state.keys()):
        model.load_state_dict(state, strict=False)
    else:
        # لو الحفظ كان pure state_dict
        model.load_state_dict(state, strict=False)

    model.eval()

    targets = []
    if args.image:
        targets.append(args.image)
    if args.folder:
        targets += sorted(glob.glob(os.path.join(args.folder, "*.png"))) \
                + sorted(glob.glob(os.path.join(args.folder, "*.jpg"))) \
                + sorted(glob.glob(os.path.join(args.folder, "*.jpeg")))

    if not targets:
        print("No images. Provide --image or --folder.")
        return

    for path in targets:
        if not os.path.isfile(path): 
            continue
        res = predict_one(model, path, threshold=args.threshold, device=device)
        print(f"[{res['decision'].upper()}] p_genuine={res['p_genuine']:.3f}  p_forged={res['p_forged']:.3f}  -> {res['path']}")

if __name__ == "__main__":
    main()

