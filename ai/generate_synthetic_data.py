# -*- coding: utf-8 -*-
"""
Watheq Synthetic Data Generator (v3 — RGB + Color Augmentation)

Generates genuine augmentations and synthetic forgeries from reference images.
Supports both color (RGB) and grayscale references.  Alpha channels are
stripped automatically so transparent PNGs don't crash.

Usage:
    python ai/generate_synthetic_data.py --ref logo.png --out training/identity/logo
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_read(path: Path) -> np.ndarray:
    """Read image handling alpha channels and Unicode paths."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"Cannot read: {path}")
    # Strip alpha channel if present (BGRA → BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        bgr = img[:, :, :3]
        alpha_f = alpha.astype(np.float32) / 255.0
        white = np.full_like(bgr, 255, dtype=np.uint8)
        for c in range(3):
            bgr[:, :, c] = (bgr[:, :, c].astype(np.float32) * alpha_f +
                            white[:, :, c].astype(np.float32) * (1.0 - alpha_f)).astype(np.uint8)
        img = bgr
    # If grayscale, convert to BGR for uniform pipeline
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_max(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    r = max_side / s
    return cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)


def affine(img: np.ndarray, ang: float = 0, sc: float = 1.0,
           tx: float = 0, ty: float = 0) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += [tx, ty]
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def add_gauss_noise(img: np.ndarray, sigma: float = 5) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_hsv_shift(img: np.ndarray) -> np.ndarray:
    """Random hue/saturation/value shift for color augmentation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-8, 8)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.90, 1.10), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ──────────────── genuine augmentations ────────────────

def aug_genuine(base: np.ndarray) -> np.ndarray:
    """Produce a genuine-looking augmentation preserving color."""
    img = base.copy()
    # Geometric
    ang = random.uniform(-7, 7)
    sc = random.uniform(0.92, 1.08)
    tx = random.uniform(-0.03 * img.shape[1], 0.03 * img.shape[1])
    ty = random.uniform(-0.03 * img.shape[0], 0.03 * img.shape[0])
    img = affine(img, ang, sc, tx, ty)
    # Blur
    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    # Noise
    if random.random() < 0.4:
        img = add_gauss_noise(img, sigma=random.uniform(2, 7))
    # Brightness/contrast
    if random.random() < 0.6:
        alpha = random.uniform(0.88, 1.12)
        beta = random.uniform(-12, 12)
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    # Color jitter (hue/sat/val)
    if random.random() < 0.5 and img.ndim == 3:
        img = random_hsv_shift(img)
    # JPEG compression artifact
    if random.random() < 0.3:
        quality = random.randint(60, 85)
        enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


# ──────────────── forgery augmentations ────────────────

def aug_freehand(base: np.ndarray) -> np.ndarray:
    """Simulate a hand-drawn copy (loses fine color detail)."""
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    th = cv2.erode(th, np.ones((2, 2), np.uint8), iterations=1)
    th = cv2.dilate(th, np.ones((2, 2), np.uint8), iterations=1)
    result = 255 - th
    if base.ndim == 3:
        colored = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        tint = random.choice([(245, 240, 235), (235, 240, 245), (240, 245, 235)])
        for c in range(3):
            colored[:, :, c] = np.clip(
                colored[:, :, c].astype(np.float32) * tint[c] / 255.0,
                0, 255).astype(np.uint8)
        return colored
    return result


def aug_tracing(base: np.ndarray) -> np.ndarray:
    """Simulate an edge-traced copy."""
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    h, w = gray.shape[:2]
    if base.ndim == 3:
        canvas = np.full((h, w, 3), 255, np.uint8)
        ys, xs = np.where(edges > 0)
        canvas[ys, xs] = [0, 0, 0]
        return cv2.GaussianBlur(canvas, (3, 3), 0)
    else:
        canvas = np.full((h, w), 255, np.uint8)
        canvas[np.where(edges > 0)] = 0
        return cv2.GaussianBlur(canvas, (3, 3), 0)


def aug_digital(base: np.ndarray) -> np.ndarray:
    """Simulate a digital manipulation (resize + JPEG artifacts)."""
    img = base.copy()
    h, w = img.shape[:2]
    rw = int(w * random.uniform(0.85, 1.2))
    rh = int(h * random.uniform(0.9, 1.15))
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = cv2.addWeighted(img, 1.6, cv2.GaussianBlur(img, (0, 0), 2.0), -0.6, 0)
    if img.ndim == 3:
        canvas = np.full((max(rh, h), max(rw, w), 3), 255, np.uint8)
    else:
        canvas = np.full((max(rh, h), max(rw, w)), 255, np.uint8)
    y0 = random.randint(0, canvas.shape[0] - rh)
    x0 = random.randint(0, canvas.shape[1] - rw)
    canvas[y0:y0 + rh, x0:x0 + rw] = img
    quality = random.randint(30, 65)
    enc = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR if base.ndim == 3 else cv2.IMREAD_GRAYSCALE)


def aug_color_forge(base: np.ndarray) -> np.ndarray:
    """Simulate color-shifted forgery (wrong printer / scanner)."""
    if base.ndim != 3:
        return aug_digital(base)
    img = base.copy()
    shift_type = random.choice(["channel_swap", "heavy_tint", "desaturate"])
    if shift_type == "channel_swap":
        channels = list(range(3))
        random.shuffle(channels)
        img = img[:, :, channels]
    elif shift_type == "heavy_tint":
        tint = np.array([random.randint(180, 255) for _ in range(3)], dtype=np.float32) / 255.0
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) * tint[c], 0, 255).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img = add_gauss_noise(img, sigma=random.uniform(5, 15))
    ang = random.uniform(-3, 3)
    sc = random.uniform(0.95, 1.05)
    img = affine(img, ang, sc)
    return img


def generate_separate(ref_path: Path, out_root: Path,
                      n_genuine: int = 400, n_forged: int = 400):
    """
    Generate augmented genuine + forged training data from a reference image.

    Args:
        ref_path: Path to the clean reference image
        out_root: Output directory (will create out_genuine/ and out_forged/)
        n_genuine: Number of genuine augmentations
        n_forged: Number of forged samples
    """
    img = safe_read(ref_path)
    img = resize_max(img, 512)

    gen_dir = out_root / "out_genuine"
    forg_dir = out_root / "out_forged"
    ensure_dir(gen_dir)
    ensure_dir(forg_dir)

    gen_list = []
    forg_list = []

    # Genuine augmentations (color preserved)
    for i in range(n_genuine):
        g = aug_genuine(img)
        p = gen_dir / f"g_{i:04d}.png"
        cv2.imwrite(str(p), g)
        gen_list.append(str(p).replace("\\", "/"))

    # Forged samples — 4 types split evenly
    forgery_types = [
        ("ff", aug_freehand),
        ("ft", aug_tracing),
        ("fd", aug_digital),
        ("fc", aug_color_forge),
    ]
    each = n_forged // len(forgery_types)
    leftover = n_forged - each * len(forgery_types)

    for code, func in forgery_types:
        n_here = each + (1 if leftover > 0 else 0)
        leftover -= 1 if leftover > 0 else 0
        for i in range(n_here):
            f = func(img)
            p = forg_dir / f"{code}_{i:04d}.png"
            cv2.imwrite(str(p), f)
            forg_list.append(str(p).replace("\\", "/"))

    # Write manifests
    (out_root / "genuine.txt").write_text("\n".join(gen_list), encoding="utf-8")
    (out_root / "forged.txt").write_text("\n".join(forg_list), encoding="utf-8")
    print(f"Done. Saved: {len(gen_list)} genuine, {len(forg_list)} forged")
    print(f"Folders: {gen_dir} | {forg_dir}")


def main():
    ap = argparse.ArgumentParser(description="Watheq Synthetic Data Generator v3")
    ap.add_argument("--ref", required=True, help="Path to clean reference image")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--num_genuine", type=int, default=400)
    ap.add_argument("--num_forged", type=int, default=400)
    args = ap.parse_args()
    generate_separate(Path(args.ref), Path(args.out), args.num_genuine, args.num_forged)


if __name__ == "__main__":
    main()
