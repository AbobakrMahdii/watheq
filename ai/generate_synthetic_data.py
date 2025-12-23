# -*- coding: utf-8 -*-
import argparse, random
from pathlib import Path
import cv2, numpy as np

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
def resize_max(img, max_side=512):
    h,w = img.shape[:2]; s=max(h,w)
    if s<=max_side: return img
    r = max_side/s
    return cv2.resize(img,(int(w*r),int(h*r)),interpolation=cv2.INTER_AREA)

def affine(img, ang=0, sc=1.0, tx=0, ty=0):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, sc); M[:,2]+=[tx,ty]
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderValue=255)

def add_gauss_noise(img, sigma=5):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)

def binarize_soft(img):
    return cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,31,10)

# ---------- genuine ----------
def aug_genuine(base):
    img=base.copy()
    ang=random.uniform(-7,7); sc=random.uniform(0.92,1.08)
    tx=random.uniform(-0.03*img.shape[1],0.03*img.shape[1])
    ty=random.uniform(-0.03*img.shape[0],0.03*img.shape[0])
    img=affine(img,ang,sc,tx,ty)
    if random.random()<0.6: img=cv2.GaussianBlur(img,(3,3),0)
    if random.random()<0.4: img=add_gauss_noise(img,sigma=random.uniform(2,7))
    if random.random()<0.6:
        alpha=random.uniform(0.9,1.15); beta=random.uniform(-10,10)
        img=np.clip(alpha*img+beta,0,255).astype(np.uint8)
    if random.random()<0.7: img=binarize_soft(img)
    return img

# ---------- forgeries ----------
def aug_freehand(base):
    th=cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,35,15)
    th=cv2.erode(th,np.ones((2,2),np.uint8),iterations=1)
    th=cv2.dilate(th,np.ones((2,2),np.uint8),iterations=1)
    return 255-th

def aug_tracing(base):
    edges=cv2.Canny(base,60,140)
    edges=cv2.dilate(edges,np.ones((2,2),np.uint8),iterations=1)
    h,w=base.shape[:2]; canvas=np.full((h,w),255,np.uint8)
    ys,xs=np.where(edges>0)
    for y,x in zip(ys,xs):
        if 0<=y<h and 0<=x<w: canvas[y,x]=0
    return cv2.GaussianBlur(canvas,(3,3),0)

def aug_digital(base):
    _,th=cv2.threshold(base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h,w=th.shape[:2]
    rw=int(w*random.uniform(0.85,1.2)); rh=int(h*random.uniform(0.9,1.15))
    th=cv2.resize(th,(rw,rh),interpolation=cv2.INTER_NEAREST)
    th=cv2.GaussianBlur(th,(0,0),1.2)
    th=cv2.addWeighted(th,1.6,cv2.GaussianBlur(th,(0,0),2.0),-0.6,0)
    canvas=np.full((max(rh,h),max(rw,w)),255,np.uint8)
    y0=random.randint(0,canvas.shape[0]-rh); x0=random.randint(0,canvas.shape[1]-rw)
    canvas[y0:y0+rh,x0:x0+rw]=th
    enc=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,random.randint(40,75)])[1]
    return cv2.imdecode(enc,cv2.IMREAD_GRAYSCALE)

def generate_separate(ref_path: Path, out_root: Path, n_genuine: int, n_forged: int):
    img=cv2.imread(str(ref_path),cv2.IMREAD_COLOR)
    img=to_gray(img)
    if img is None: raise SystemExit(f"Cannot read: {ref_path}")
    img=resize_max(img,512)

    gen_dir = out_root / "out_genuine"
    forg_dir = out_root / "out_forged"
    ensure_dir(gen_dir); ensure_dir(forg_dir)

    gen_list = []
    forg_list = []

    # genuine
    for i in range(n_genuine):
        g=aug_genuine(img)
        p = gen_dir / f"g_{i:04d}.png"
        cv2.imwrite(str(p), g)
        gen_list.append(str(p).replace("\\","/"))

    # forged split
    each=n_forged//3; leftover=n_forged-each*3
    types=[("ff",aug_freehand),("ft",aug_tracing),("fd",aug_digital)]
    for code,func in types:
        n_here=each+(1 if leftover>0 else 0); leftover-=1 if leftover>0 else 0
        for i in range(n_here):
            f=func(img)
            p = forg_dir / f"{code}_{i:04d}.png"
            cv2.imwrite(str(p), f)
            forg_list.append(str(p).replace("\\","/"))

    # manifests
    (out_root / "genuine.txt").write_text("\n".join(gen_list), encoding="utf-8")
    (out_root / "forged.txt").write_text("\n".join(forg_list), encoding="utf-8")
    print("Done. Saved:", len(gen_list), "genuine,", len(forg_list), "forged")
    print("Folders:", gen_dir, "|", forg_dir)
    print("Lists:", out_root / "genuine.txt", "|", out_root / "forged.txt")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ref",required=True, help="path to clean signature image")
    ap.add_argument("--out",required=True, help="output root (will create out_genuine/ and out_forged/)")
    ap.add_argument("--num_genuine",type=int,default=400)
    ap.add_argument("--num_forged",type=int,default=900)
    args=ap.parse_args()
    generate_separate(Path(args.ref), Path(args.out), args.num_genuine, args.num_forged)

if __name__=="__main__":
    main()
