        # evaluate_model.py
# -*- coding: utf-8 -*-
"""
Evaluate a ResNet-18 signature classifier on VAL & TEST.
- Prints Acc/Precision/Recall/F1 + confusion matrices
- Finds best threshold on VAL (by F1) and applies it to TEST
- Saves JSON/TXT reports (handles NumPy types)
- (Optional) Saves plots: confusion matrices + ROC/PR curves
- (Optional) Forgery breakdown by filename prefixes: ff_/ft_/fd_

Usage (Jupyter/Windows):
!python -X utf8 evaluate_model.py ^
  --data_root datasets/custom_signature_dataset_pre ^
  --model_path models/resnet18_finetuned.pth ^
  --batch_size 64 ^
  --forgery_breakdown ^
  --save_plots
"""

import argparse, os, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------- JSON encoder that supports NumPy ----------
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

# ---------- Model / Loader ----------
def build_model(num_classes=2):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def make_loader(root, split, bs=64, workers=0, pin=False):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(os.path.join(root, split), transform=tfm)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
    return ds, dl

# ---------- Eval helpers ----------
@torch.no_grad()
def eval_split(model, loader, device):
    model.eval()
    tp=fp=tn=fn=0; probs=[]; labels=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        logits = model(x)
        p_g = torch.softmax(logits, dim=1)[:,1]   # P(genuine)
        pred = torch.argmax(logits, dim=1)        # 0 forged / 1 genuine
        tp += ((pred==1)&(y==1)).sum().item()
        fp += ((pred==1)&(y==0)).sum().item()
        tn += ((pred==0)&(y==0)).sum().item()
        fn += ((pred==0)&(y==1)).sum().item()
        probs.extend(p_g.cpu().numpy().tolist())
        labels.extend(y.cpu().numpy().tolist())
    total = tp+tn+fp+fn
    acc  = (tp+tn)/max(1,total)
    prec = tp/max(1,(tp+fp))
    rec  = tp/max(1,(tp+fn))
    f1   = 2*prec*rec/max(1e-9,(prec+rec))
    return {
        "acc":acc, "precision":prec, "recall":rec, "f1":f1,
        "confusion":[[tn,fp],[fn,tp]],
        "probs":np.array(probs), "labels":np.array(labels)
    }

def best_thr_from_val(scores, labels, lo=0.30, hi=0.70, steps=41):
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(lo, hi, steps):
        pred = (scores>=thr).astype(int)
        tp=((pred==1)&(labels==1)).sum()
        fp=((pred==1)&(labels==0)).sum()
        fn=((pred==0)&(labels==1)).sum()
        prec=tp/max(1,(tp+fp)); rec=tp/max(1,(tp+fn))
        f1=2*prec*rec/max(1e-9,(prec+rec))
        if f1>best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)

@torch.no_grad()
def per_forgery_breakdown(model, test_root, thr, device):
    # expects forged test files to start with: ff_ / ft_ / fd_
    from PIL import Image
    tfm = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    forged_dir = Path(test_root) / "forged"
    if not forged_dir.exists():
        return {}
    types = {"freehand":"ff_", "tracing":"ft_", "digital":"fd_"}
    stats = {}
    for name, prefix in types.items():
        paths = [p for p in glob.glob(str(forged_dir / "*.*")) if os.path.basename(p).startswith(prefix)]
        if not paths:
            stats[name] = {"N":0, "detect_rate":None}; continue
        tp=fn=0
        for p in paths:
            img = Image.open(p).convert("L")
            x = tfm(img).unsqueeze(0).to(device)
            prob = torch.softmax(model(x), dim=1)[:,1].item()
            pred = 1 if prob>=thr else 0  # 1=genuine, 0=forged
            if pred==0: tp+=1
            else: fn+=1
        stats[name] = {"N":len(paths), "detect_rate": round(tp/max(1,len(paths)),4)}
    return stats

# ---------- Plots (optional) ----------
def save_plots(rep_dir, val_metrics, test_metrics, thr):
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
    except Exception as e:
        print(f"[plots] skipped (missing matplotlib/sklearn?): {e}")
        return

    os.makedirs(rep_dir, exist_ok=True)

    # Confusion matrices
    def plot_cm(cm, title, path):
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(title); plt.colorbar()
        ticks = ['forged(0)','genuine(1)']
        plt.xticks([0,1], ticks); plt.yticks([0,1], ticks)
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i,j]), ha='center', va='center')
        plt.tight_layout(); plt.xlabel('Pred'); plt.ylabel('True')
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

    v_cm = np.array(val_metrics["confusion"])
    t_cm = np.array(test_metrics["confusion"])
    plot_cm(v_cm, "VAL Confusion", os.path.join(rep_dir,"val_confusion.png"))
    plot_cm(t_cm, "TEST Confusion", os.path.join(rep_dir,"test_confusion.png"))

    # ROC / PR (need probabilities and labels)
    for split_name, m in [("VAL", val_metrics), ("TEST", test_metrics)]:
        y_true = m["labels"].astype(int)
        y_score = m["probs"].astype(float)
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{split_name} ROC")
        plt.legend(); plt.savefig(os.path.join(rep_dir, f"{split_name.lower()}_roc.png"), dpi=150, bbox_inches="tight"); plt.close()
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{split_name} PR")
        plt.legend(); plt.savefig(os.path.join(rep_dir, f"{split_name.lower()}_pr.png"), dpi=150, bbox_inches="tight"); plt.close()

    # Threshold line for info
    with open(os.path.join(rep_dir, "threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_threshold_on_VAL = {thr:.4f}\n")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="root with val/test (ImageFolder)")
    ap.add_argument("--model_path", required=True, help=".pth state_dict (ResNet-18)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--report_dir", default="reports")
    ap.add_argument("--thr_lo", type=float, default=0.30)
    ap.add_argument("--thr_hi", type=float, default=0.70)
    ap.add_argument("--thr_steps", type=int, default=41)
    ap.add_argument("--forgery_breakdown", action="store_true")
    ap.add_argument("--save_plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = build_model(num_classes=2)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    # data
    (val_ds, val_loader)   = make_loader(args.data_root, "val",  args.batch_size)
    (test_ds, test_loader) = make_loader(args.data_root, "test", args.batch_size)

    # eval
    val_m  = eval_split(model, val_loader,  device)
    thr, f1v = best_thr_from_val(val_m["probs"], val_m["labels"], args.thr_lo, args.thr_hi, args.thr_steps)
    test_m = eval_split(model, test_loader, device)

    # apply thr to TEST
    probs, labels = test_m["probs"], test_m["labels"]
    pred  = (probs >= thr).astype(int)
    tp=((pred==1)&(labels==1)).sum(); fp=((pred==1)&(labels==0)).sum()
    tn=((pred==0)&(labels==0)).sum(); fn=((pred==0)&(labels==1)).sum()
    acc=(tp+tn)/max(1,(tp+tn+fp+fn))
    prec=tp/max(1,(tp+fp)); rec=tp/max(1,(tp+fn))
    f1=2*prec*rec/max(1e-9,(prec+rec))
    test_conf = [[int(tn), int(fp)], [int(fn), int(tp)]]
    print("\n===== SUMMARY =====")
    print(f"VAL : acc={val_m['acc']:.4f}  prec={val_m['precision']:.4f}  rec={val_m['recall']:.4f}  f1={val_m['f1']:.4f}")
    print(f"      best_thr={thr:.3f} (F1={f1v:.4f})  conf={val_m['confusion']}")
    print(f"TEST: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  conf={test_conf}")

    # breakdown (optional)
    breakdown = None
    if args.forgery_breakdown:
        breakdown = per_forgery_breakdown(model, Path(args.data_root)/"test", thr, device)
        if breakdown:
            print("\n-- Forgery detection rates (TEST) --")
            for k,v in breakdown.items():
                print(f"{k:8s}  N={v['N']:3d}  detect_rate={v['detect_rate']}")

    # save reports
    rep = {
        "val":{
            "acc":val_m["acc"], "precision":val_m["precision"], "recall":val_m["recall"], "f1":val_m["f1"],
            "best_thr":thr, "best_thr_f1":f1v, "confusion":val_m["confusion"]
        },
        "test":{
            "acc":acc, "precision":prec, "recall":rec, "f1":f1, "confusion":test_conf
        }
    }
    if breakdown: rep["forgery_breakdown_test"] = breakdown

    json_path = os.path.join(args.report_dir, "eval_report.json")
    txt_path  = os.path.join(args.report_dir, "eval_report.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("VAL\n")
        f.write(f"acc={val_m['acc']:.4f}  prec={val_m['precision']:.4f}  rec={val_m['recall']:.4f}  f1={val_m['f1']:.4f}\n")
        f.write(f"best_thr={thr:.3f} (F1={f1v:.4f})  conf={val_m['confusion']}\n\n")
        f.write("TEST\n")
        f.write(f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  conf={test_conf}\n")
        if breakdown:
            f.write("\nFORGERY DETECTION (TEST)\n")
            for k,v in breakdown.items():
                f.write(f"{k:8s}  N={v['N']:3d}  detect_rate={v['detect_rate']}\n")

    print(f"\nSaved reports to:\n  {json_path}\n  {txt_path}")

    # optional plots
    if args.save_plots:
        save_plots(args.report_dir, val_m, {**test_m, "confusion":test_conf}, thr)
        print("Saved plots to:", args.report_dir)

if __name__ == "__main__":
    main()
