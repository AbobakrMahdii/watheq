import os, glob, uuid, random, cv2
import numpy as np

src = "datasets/custom_signature_dataset_pre/train/genuine"   # المصدر (بعد preprocess للـ train)
target = 650  # عدّليه حسب احتياجك

files = glob.glob(os.path.join(src, "*.*"))
assert files, "لا توجد صور في مجلد genuine/train"

def affine(img, ang=0, sc=1.0, tx=0, ty=0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, sc)
    M[:, 2] += [tx, ty]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

def vary_ink(img, ksize=2, mode="thin"):
    k = np.ones((ksize, ksize), np.uint8)
    if mode == "thin":
        return cv2.erode(img, k, 1)
    else:
        return cv2.dilate(img, k, 1)

def add_noise(img, sigma=6):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def jpeg_artifact(img, q=60):
    enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
    return cv2.imdecode(enc, 0)

def contrast_brightness(img, alpha=1.0, beta=0):
    out = np.clip(alpha*img + beta, 0, 255).astype(np.uint8)
    return out

def subtle_perspective(img, max_ratio=0.03):
    h, w = img.shape[:2]
    dx = int(w * random.uniform(-max_ratio, max_ratio))
    dy = int(h * random.uniform(-max_ratio, max_ratio))
    src_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
    dst_pts = np.float32([[dx,0],[w-dx,0],[0,h-dy],[w,h+dy]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255)

def one_aug(img):
    out = img.copy()
    # 1) هندسي خفيف
    ang = random.uniform(-7, 7)
    sc  = random.uniform(0.95, 1.05)
    tx  = random.uniform(-0.03*out.shape[1], 0.03*out.shape[1])
    ty  = random.uniform(-0.03*out.shape[0], 0.03*out.shape[0])
    out = affine(out, ang, sc, tx, ty)

    # 2) حبر/سُمك
    if random.random() < 0.5:
        out = vary_ink(out, ksize=random.choice([1,2]), mode=random.choice(["thin","thick"]))

    # 3) ضوضاء/إضاءة
    if random.random() < 0.5:
        out = add_noise(out, sigma=random.uniform(3,7))
    if random.random() < 0.6:
        out = contrast_brightness(out, alpha=random.uniform(0.9,1.1), beta=random.uniform(-10,10))

    # 4) JPEG artifact أو منظور بسيط (واحد منهم فقط غالباً)
    if random.random() < 0.4:
        out = jpeg_artifact(out, q=random.randint(50,80))
    elif random.random() < 0.3:
        out = subtle_perspective(out, max_ratio=0.02)

    return out

i = 0
while len(os.listdir(src)) < target:
    f = random.choice(files)
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    aug = one_aug(img)
    base, ext = os.path.splitext(os.path.basename(f))
    new_path = os.path.join(src, f"{base}_aug_{uuid.uuid4().hex[:6]}{ext}")
    cv2.imwrite(new_path, aug)
    i += 1

print("new genuine count:", len(os.listdir(src)), " (added:", i, ")")
