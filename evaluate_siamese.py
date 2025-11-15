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
# evaluate_siamese.py
# -*- coding: utf-8 -*-
import os, argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# ===================== Config =====================
IMG_SIZE = 224
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
LINSPACE_STEPS = 400  # دقة البحث عن العتبة
RNG_SEED = 7
# ==================================================

def to_tensor(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(arr)

def list_images(folder: Path):
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    out=[]
    for r,_,fs in os.walk(folder):
        for f in fs:
            if Path(f).suffix.lower() in exts:
                out.append(str(Path(r)/f))
    return sorted(out)

def remap_keys(sd: dict):
    out={}
    for k,v in sd.items():
        if k.startswith("module."): k = k[7:]
        if k.startswith("embedding."): k = "head." + k[10:]
        out[k]=v
    return out

def infer_head_dims_from_sd(sd: dict, resnet_out_feats=512, default_hidden=256, default_embed=128):
    hidden, embed = default_hidden, default_embed
    w0 = sd.get("head.0.weight", None)  # (hidden, 512)
    w2 = sd.get("head.2.weight", None)  # (embed, hidden)
    if w0 is not None and w0.dim()==2 and w0.shape[1]==resnet_out_feats:
        hidden = int(w0.shape[0])
    if w2 is not None and w2.dim()==2 and w2.shape[1]==hidden:
        embed = int(w2.shape[0])
    return hidden, embed

class SiameseBackbone(nn.Module):
    def __init__(self, hidden_dim=256, embed_dim=128, pretrained=False):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except Exception as e:
            raise RuntimeError("torchvision مطلوب للتقييم.") from e
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x):
        f = self.backbone(x)
        e = self.head(f)
        return nn.functional.normalize(e, dim=1)

def smart_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt: return ckpt["state_dict"]
    if hasattr(ckpt, "state_dict"): return ckpt.state_dict()
    if isinstance(ckpt, dict): return ckpt
    raise RuntimeError("صيغة checkpoint غير معروفة")

@torch.no_grad()
def embed_many(model, paths, device="cpu"):
    model.eval()
    embs=[]
    for p in paths:
        x = to_tensor(Image.open(p)).unsqueeze(0).to(device).float()
        embs.append(model(x).cpu())
    if len(embs)==0:
        return torch.empty(0, model.head[-1].out_features)
    return torch.cat(embs, dim=0)

def metrics_from_cm(tp, fp, fn, tn):
    tp, fp, fn, tn = map(float, (tp, fp, fn, tn))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2*prec*rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / (tp + fp + fn + tn + 1e-9)
    far  = fp / (fp + tn + 1e-9)  # False Acceptance Rate (مزور تم قبوله)
    frr  = fn / (tp + fn + 1e-9)  # False Rejection Rate (أصلي تم رفضه)
    return acc, prec, rec, f1, far, frr

def choose_threshold(scores, labels, mode="cosine", objective="precision_at_fpr", target_fpr=0.01):
    # mode: cosine => match if s>=thr
    #       distance => match if d<=thr
    smin, smax = float(scores.min()), float(scores.max())
    thrs = np.linspace(smin, smax, num=LINSPACE_STEPS, dtype=np.float32)
    best = None
    for thr in thrs:
        if mode=="cosine":
            pred = (scores >= thr).astype(np.int32)
        else:
            pred = (scores <= thr).astype(np.int32)
        tp = int(((pred==1)&(labels==1)).sum())
        fp = int(((pred==1)&(labels==0)).sum())
        fn = int(((pred==0)&(labels==1)).sum())
        tn = int(((pred==0)&(labels==0)).sum())
        acc, prec, rec, f1, far, frr = metrics_from_cm(tp, fp, fn, tn)
        cand = {"thr":float(thr), "tp":tp, "fp":fp, "fn":fn, "tn":tn,
                "acc":acc,"prec":prec,"rec":rec,"f1":f1,"far":far,"frr":frr}
        if objective=="f1":
            key = f1
            ok = True
        elif objective=="precision_at_fpr":
            ok = (far <= float(target_fpr))
            key = f1 if ok else -1.0
        elif objective=="recall":
            key = rec; ok=True
        else:  # balanced
            key = (f1 + (1.0 - far)) / 2.0; ok=True
        if ok and (best is None or key > best["key"]):
            cand["key"] = key
            best = cand
    # fallback: لو ما تحقق شرط FAR على أي عتبة، نختار أعلى F1
    if best is None:
        for thr in thrs:
            if mode=="cosine":
                pred = (scores >= thr).astype(np.int32)
            else:
                pred = (scores <= thr).astype(np.int32)
            tp = int(((pred==1)&(labels==1)).sum())
            fp = int(((pred==1)&(labels==0)).sum())
            fn = int(((pred==0)&(labels==1)).sum())
            tn = int(((pred==0)&(labels==0)).sum())
            acc, prec, rec, f1, far, frr = metrics_from_cm(tp, fp, fn, tn)
            cand = {"thr":float(thr), "tp":tp, "fp":fp, "fn":fn, "tn":tn,
                    "acc":acc,"prec":prec,"rec":rec,"f1":f1,"far":far,"frr":frr,"key":f1}
            if best is None or f1 > best["key"]:
                best = cand
    return best

def eval_reference_protocol(E_ref, E_pos, E_neg, mode, objective, target_fpr):
    # scores: pos = similarity with ref for genuine, neg for forged
    # cosine => s = dot; distance => d = L2
    if mode=="cosine":
        ref = E_ref.mean(dim=0, keepdim=True)  # تجميع مرجع متعدد
        s_pos = torch.sum(E_pos * ref, dim=1).cpu().numpy() if len(E_pos)>0 else np.array([])
        s_neg = torch.sum(E_neg * ref, dim=1).cpu().numpy() if len(E_neg)>0 else np.array([])
    else:
        ref = E_ref.mean(dim=0, keepdim=True)
        s_pos = torch.cdist(E_pos, ref, p=2.0).squeeze(1).cpu().numpy() if len(E_pos)>0 else np.array([])
        s_neg = torch.cdist(E_neg, ref, p=2.0).squeeze(1).cpu().numpy() if len(E_neg)>0 else np.array([])
    scores = np.concatenate([s_pos, s_neg]).astype(np.float32)
    labels = np.concatenate([np.ones_like(s_pos, dtype=np.int32),
                             np.zeros_like(s_neg, dtype=np.int32)])
    best = choose_threshold(scores, labels, mode=mode, objective=objective, target_fpr=target_fpr)
    return best, scores, labels

def eval_pairwise_protocol(E_pos, E_neg, mode, objective, target_fpr):
    # pairs: pos = (pos,pos) + (neg,neg), neg = (pos,neg)
    rng = np.random.default_rng(RNG_SEED)
    n_pairs = min(len(E_pos), len(E_neg)) * 4
    vals, labs = [], []
    # positive
    for _ in range(max(1, n_pairs//2)):
        i,j = rng.integers(0, len(E_pos), size=2)
        if mode=="cosine":
            s = float(torch.sum(E_pos[i]*E_pos[j]).item())
        else:
            s = float(torch.pairwise_distance(E_pos[i:i+1], E_pos[j:j+1]).item())
        vals.append(s); labs.append(1)
        i,j = rng.integers(0, len(E_neg), size=2)
        if mode=="cosine":
            s = float(torch.sum(E_neg[i]*E_neg[j]).item())
        else:
            s = float(torch.pairwise_distance(E_neg[i:i+1], E_neg[j:j+1]).item())
        vals.append(s); labs.append(1)
    # negative
    for _ in range(n_pairs):
        i = rng.integers(0, len(E_pos)); j = rng.integers(0, len(E_neg))
        if mode=="cosine":
            s = float(torch.sum(E_pos[i]*E_neg[j]).item())
        else:
            s = float(torch.pairwise_distance(E_pos[i:i+1], E_neg[j:j+1]).item())
        vals.append(s); labs.append(0)
    scores = np.array(vals, dtype=np.float32)
    labels = np.array(labs, dtype=np.int32)
    best = choose_threshold(scores, labels, mode=mode, objective=objective, target_fpr=target_fpr)
    return best, scores, labels

def evaluate_split(E_ref, E_genuine, E_forged, protocol, objective, target_fpr):
    # يرجع أفضل نتيجة بين cosine و distance
    res = {}
    for mode in ("cosine", "distance"):
        if protocol == "reference":
            best, scores, labels = eval_reference_protocol(E_ref, E_genuine, E_forged, mode, objective, target_fpr)
        else:
            best, scores, labels = eval_pairwise_protocol(E_genuine, E_forged, mode, objective, target_fpr)
        res[mode] = {"best":best, "scores_min": float(scores.min()) if scores.size>0 else 0.0,
                             "scores_max": float(scores.max()) if scores.size>0 else 0.0}
    # اختر الأفضل على أساس F1 (وهو ما سنثبته لاختبار TEST)
    pick = "cosine" if res["cosine"]["best"]["f1"] >= res["distance"]["best"]["f1"] else "distance"
    return res, pick

def apply_metrics_on_split(E_ref, E_genuine, E_forged, mode, thr, protocol):
    # قياس نهائي بعتبة ثابتة
    if protocol=="reference":
        ref = E_ref.mean(dim=0, keepdim=True)
        if mode=="cosine":
            s_pos = torch.sum(E_genuine * ref, dim=1).cpu().numpy()
            s_neg = torch.sum(E_forged  * ref, dim=1).cpu().numpy()
            pred_pos = (s_pos >= thr).astype(np.int32)
            pred_neg = (s_neg >= thr).astype(np.int32)
        else:
            d_pos = torch.cdist(E_genuine, ref, p=2.0).squeeze(1).cpu().numpy()
            d_neg = torch.cdist(E_forged , ref, p=2.0).squeeze(1).cpu().numpy()
            pred_pos = (d_pos <= thr).astype(np.int32)
            pred_neg = (d_neg <= thr).astype(np.int32)
        tp = int((pred_pos==1).sum())
        fn = int((pred_pos==0).sum())
        fp = int((pred_neg==1).sum())
        tn = int((pred_neg==0).sum())
    else:
        # pairwise: نعيد توليد عينات بسرعة لأجل report مختصر
        rng = np.random.default_rng(RNG_SEED+11)
        n_pairs = min(len(E_genuine), len(E_forged)) * 4
        tp=fp=fn=tn=0
        # pos
        for _ in range(max(1, n_pairs//2)):
            i,j = rng.integers(0, len(E_genuine), size=2)
            if mode=="cosine":
                s = float(torch.sum(E_genuine[i]*E_genuine[j]).item())
                pred = (s >= thr)
            else:
                d = float(torch.pairwise_distance(E_genuine[i:i+1], E_genuine[j:j+1]).item())
                pred = (d <= thr)
            tp += int(pred==1); fn += int(pred==0)
            i,j = rng.integers(0, len(E_forged), size=2)
            if mode=="cosine":
                s = float(torch.sum(E_forged[i]*E_forged[j]).item()); pred = (s >= thr)
            else:
                d = float(torch.pairwise_distance(E_forged[i:i+1], E_forged[j:j+1]).item()); pred = (d <= thr)
            tp += int(pred==1); fn += int(pred==0)
        # neg
        for _ in range(n_pairs):
            i = rng.integers(0, len(E_genuine)); j = rng.integers(0, len(E_forged))
            if mode=="cosine":
                s = float(torch.sum(E_genuine[i]*E_forged[j]).item()); pred = (s >= thr)
            else:
                d = float(torch.pairwise_distance(E_genuine[i:i+1], E_forged[j:j+1]).item()); pred = (d <= thr)
            fp += int(pred==1); tn += int(pred==0)
    acc, prec, rec, f1, far, frr = metrics_from_cm(tp, fp, fn, tn)
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"acc":acc,"prec":prec,"rec":rec,"f1":f1,"far":far,"frr":frr}

def save_hist(scores, labels, thr, mode, out_png):
    try:
        import matplotlib.pyplot as plt
        pos = scores[labels==1]
        neg = scores[labels==0]
        plt.figure(figsize=(7,4))
        bins = 60
        plt.hist(pos, bins=bins, alpha=0.6, label="positive", density=True)
        plt.hist(neg, bins=bins, alpha=0.6, label="negative", density=True)
        plt.axvline(thr, linestyle="--", linewidth=2)
        plt.title(f"{mode} threshold = {thr:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
        print("Saved histogram:", out_png)
    except Exception as e:
        print("Histogram save skipped:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--val_dir", required=True, help=".../val with genuine/ and forged/")
    ap.add_argument("--test_dir", default=None, help=".../test with genuine/ and forged/ (optional)")
    ap.add_argument("--ref_dir", default=None, help="Reference signatures dir (recommended for production)")
    ap.add_argument("--optimize_for", default="precision_at_fpr",
                    choices=["precision_at_fpr","f1","recall","balanced"],
                    help="Default targets low FAR (false acceptance).")
    ap.add_argument("--target_fpr", type=float, default=0.01,
                    help="FPR limit when optimize_for=precision_at_fpr (e.g., 0.01 for 1%).")
    ap.add_argument("--save_json", default="reports/siamese_eval.json")
    ap.add_argument("--save_hist", default=None, help="Optional PNG path to save score histogram on VAL.")
    ap.add_argument("--write_threshold", default=None, help="Optional path to save chosen threshold for production (text).")
    args = ap.parse_args()

    os.makedirs(Path(args.save_json).parent, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Load model ==========
    ckpt = torch.load(args.model_path, map_location="cpu")
    pretrained = bool(ckpt.get("pretrained", False)) if isinstance(ckpt, dict) else False
    sd = remap_keys(smart_state_dict(ckpt))
    hidden, embed = infer_head_dims_from_sd(sd)
    model = SiameseBackbone(hidden_dim=hidden, embed_dim=embed, pretrained=pretrained).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded model. missing:", list(missing)[:6], " unexpected:", list(unexpected)[:6])

    # ========== Embed VAL ==========
    val_g = list_images(Path(args.val_dir)/"genuine")
    val_f = list_images(Path(args.val_dir)/"forged")
    if len(val_g)==0 or len(val_f)==0:
        raise SystemExit("VAL split must contain genuine/ and forged/.")

    E_val_g = embed_many(model, val_g, device=device)
    E_val_f = embed_many(model, val_f, device=device)

    protocol = "reference" if args.ref_dir else "pairwise"
    if protocol=="reference":
        ref_paths = list_images(Path(args.ref_dir))
        if len(ref_paths)==0:
            raise SystemExit("ref_dir is empty.")
        E_ref = embed_many(model, ref_paths, device=device)
        print(f"Protocol: REFERENCE (ref={len(ref_paths)})")
    else:
        E_ref = None
        print("Protocol: PAIRWISE")

    # ========== Evaluate VAL: pick best metric & threshold ==========
    val_res, pick_metric = evaluate_split(E_ref, E_val_g, E_val_f, protocol,
                                          args.optimize_for, args.target_fpr)
    best_val = val_res[pick_metric]["best"]
    print(f"[VAL] metric={pick_metric} thr={best_val['thr']:.4f} "
          f"F1={best_val['f1']:.4f} Prec={best_val['prec']:.4f} Rec={best_val['rec']:.4f} "
          f"FAR={best_val['far']:.4f} FRR={best_val['frr']:.4f}")

    # Histogram (VAL)
    if args.save_hist is not None:
        # rebuild scores/labels for chosen metric on VAL for plotting
        if protocol=="reference":
            if pick_metric=="cosine":
                ref = E_ref.mean(dim=0, keepdim=True)
                s_pos = torch.sum(E_val_g * ref, dim=1).cpu().numpy()
                s_neg = torch.sum(E_val_f * ref, dim=1).cpu().numpy()
            else:
                ref = E_ref.mean(dim=0, keepdim=True)
                s_pos = torch.cdist(E_val_g, ref, p=2.0).squeeze(1).cpu().numpy()
                s_neg = torch.cdist(E_val_f, ref, p=2.0).squeeze(1).cpu().numpy()
            scores = np.concatenate([s_pos,s_neg]).astype(np.float32)
            labels = np.concatenate([np.ones_like(s_pos,dtype=np.int32), np.zeros_like(s_neg,dtype=np.int32)])
        else:
            # pairwise quick sample for histogram
            # (نفس الطريقة المستخدمة في eval_pairwise_protocol)
            rng = np.random.default_rng(RNG_SEED)
            n_pairs = min(len(E_val_g), len(E_val_f)) * 4
            vals, labs = [], []
            for _ in range(max(1, n_pairs//2)):
                i,j = rng.integers(0, len(E_val_g), size=2)
                if pick_metric=="cosine":
                    s = float(torch.sum(E_val_g[i]*E_val_g[j]).item())
                else:
                    s = float(torch.pairwise_distance(E_val_g[i:i+1], E_val_g[j:j+1]).item())
                vals.append(s); labs.append(1)
                i,j = rng.integers(0, len(E_val_f), size=2)
                if pick_metric=="cosine":
                    s = float(torch.sum(E_val_f[i]*E_val_f[j]).item())
                else:
                    s = float(torch.pairwise_distance(E_val_f[i:i+1], E_val_f[j:j+1]).item())
                vals.append(s); labs.append(1)
            for _ in range(n_pairs):
                i = rng.integers(0, len(E_val_g)); j = rng.integers(0, len(E_val_f))
                if pick_metric=="cosine":
                    s = float(torch.sum(E_val_g[i]*E_val_f[j]).item())
                else:
                    s = float(torch.pairwise_distance(E_val_g[i:i+1], E_val_f[j:j+1]).item())
                vals.append(s); labs.append(0)
            scores = np.array(vals, dtype=np.float32)
            labels = np.array(labs, dtype=np.int32)
        save_hist(scores, labels, best_val["thr"], pick_metric, args.save_hist)

    # ========== Apply on TEST (if provided) ==========
    test_report = None
    if args.test_dir:
        test_g = list_images(Path(args.test_dir)/"genuine")
        test_f = list_images(Path(args.test_dir)/"forged")
        if len(test_g)==0 or len(test_f)==0:
            raise SystemExit("TEST split must contain genuine/ and forged/.")
        E_test_g = embed_many(model, test_g, device=device)
        E_test_f = embed_many(model, test_f, device=device)
        test_report = apply_metrics_on_split(E_ref, E_test_g, E_test_f,
                                             mode=pick_metric, thr=best_val["thr"], protocol=protocol)
        print(f"[TEST] metric={pick_metric} thr={best_val['thr']:.4f} "
              f"F1={test_report['f1']:.4f} Prec={test_report['prec']:.4f} Rec={test_report['rec']:.4f} "
              f"FAR={test_report['far']:.4f} FRR={test_report['frr']:.4f}")

    # ========== Save JSON ==========
    rep = {
        "protocol": protocol,
        "optimize_for": args.optimize_for,
        "target_fpr": float(args.target_fpr),
        "val": {
            "cosine": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k,v in val_res["cosine"]["best"].items()},
            "distance": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k,v in val_res["distance"]["best"].items()},
            "picked_metric": pick_metric
        },
        "test": test_report
    }
    os.makedirs(Path(args.save_json).parent, exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("Saved report:", args.save_json)

    # ========== Write chosen threshold (optional) ==========
    if args.write_threshold:
        os.makedirs(Path(args.write_threshold).parent, exist_ok=True)
        with open(args.write_threshold, "w", encoding="utf-8") as f:
            f.write(f"{pick_metric}:{best_val['thr']:.6f}\n")
        print("Saved threshold to:", args.write_threshold)

if __name__ == "__main__":
    main()

