# -*- coding: utf-8 -*-
"""
Split custom signature dataset into train/val/test with optional forged subtype fix.
Usage examples (Windows/Jupyter):
!python watheeq_signatures/scripts/split_custom_signature.py ^
  --src datasets/custom_signature_dataset ^
  --dest datasets/custom_split ^
  --train 0.7 --val 0.15 --seed 7 --fix_forged_prefixes

Author: you + Watheeq
"""
import argparse, os, shutil, random
from pathlib import Path

def count_files(p: Path) -> int:
    c = 0
    for _, _, files in os.walk(p):
        c += len(files)
    return c

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def fix_forged_prefixes(root: Path):
    """Move files in forged/ to subfolders based on filename prefix:
       ff_* -> freehand, ft_* -> tracing, fd_* -> digital
    """
    forged = root / "forged"
    if not forged.exists(): 
        return
    (forged / "freehand").mkdir(parents=True, exist_ok=True)
    (forged / "tracing").mkdir(parents=True, exist_ok=True)
    (forged / "digital").mkdir(parents=True, exist_ok=True)
    moved = {"freehand":0, "tracing":0, "digital":0, "other":0}
    for fname in os.listdir(forged):
        src = forged / fname
        if not src.is_file(): 
            continue
        if fname.startswith("ff_"):
            dst = forged / "freehand" / fname
            shutil.move(str(src), str(dst)); moved["freehand"] += 1
        elif fname.startswith("ft_"):
            dst = forged / "tracing" / fname
            shutil.move(str(src), str(dst)); moved["tracing"] += 1
        elif fname.startswith("fd_"):
            dst = forged / "digital" / fname
            shutil.move(str(src), str(dst)); moved["digital"] += 1
        else:
            moved["other"] += 1
    print("[fix] moved:", moved)

def list_images(folder: Path):
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                imgs.append(Path(root) / f)
    return imgs

def split_list(items, train_ratio, val_ratio, seed):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train+n_val]
    test = items[n_train+n_val:]
    return train, val, test

def copy_list(file_list, src_root: Path, dst_root: Path):
    for src in file_list:
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        ensure_dir(dst.parent)
        shutil.copy2(str(src), str(dst))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dataset root (expects genuine/ and forged/)")
    ap.add_argument("--dest", required=True, help="destination root for split dataset")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fix_forged_prefixes", action="store_true",
                    help="organize forged/* into forged/freehand|tracing|digital by filename prefixes ff_/ft_/fd_")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dest)
    assert abs((args.train + args.val) - 1.0) <= 0.999, "train+val must be < 1.0 (test = rest)"
    test_ratio = 1.0 - (args.train + args.val)
    print(f"[split] ratios: train={args.train}, val={args.val}, test={test_ratio:.2f}")

    # optional: fix forged subtypes by prefix
    if args.fix_forged_prefixes:
        print("[split] fixing forged prefixes...")
        fix_forged_prefixes(src)

    # Collect classes (genuine + forged subfolders recursively)
    classes = []
    # genuine as a single class-folder:
    gen_root = src / "genuine"
    if gen_root.exists():
        classes.append(gen_root)
    # forged may have subfolders:
    forged_root = src / "forged"
    if forged_root.exists():
        # include forged itself (flat) and any subdirs:
        has_subdirs = False
        for d in sorted([Path(p) for p in forged_root.iterdir() if Path(p).is_dir()]):
            classes.append(d); has_subdirs = True
        if not has_subdirs:
            classes.append(forged_root)

    print("[split] class folders considered:")
    for c in classes:
        print("  -", c, f"({count_files(c)} files)")

    # Split each class folder independently to preserve class balance
    for c in classes:
        files = list_images(c)
        tr, va, te = split_list(files, args.train, args.val, args.seed)
        # copy
        copy_list(tr, src, dst / "train")
        copy_list(va, src, dst / "val")
        copy_list(te, src, dst / "test")

    # Summary
    print("\n[done] summary:")
    for phase in ["train","val","test"]:
        print(f"  {phase} -> {count_files(dst/phase)} files")
        for sub in ["genuine","forged"]:
            p = dst/phase/sub
            if p.exists():
                print(f"    {sub}: {count_files(p)}")

if __name__ == "__main__":
    main()
