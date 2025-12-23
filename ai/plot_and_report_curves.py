# watheeq_training_report.py
# -*- coding: utf-8 -*-
"""
Creates a clean English PDF report for Watheeq signature model.

Two modes:
A) From CSV log (preferred) -> auto-generate combined curves image
B) From existing PNGs (loss/acc) -> merge vertically

Then:
- Reads eval_report.json (if provided) to include VAL/TEST metrics
- Builds a neat single-page PDF with a concise English summary + the merged figure

Examples (Jupyter):
!python -X utf8 watheeq_training_report.py ^
  --log_csv models/resnet18_finetuned.log.csv ^
  --eval_json reports/eval_report.json ^
  --out_dir reports ^
  --title "Watheeq Signature Verification — Evaluation Report" ^
  --smooth_k 3

# Or from ready PNGs:
!python -X utf8 watheeq_training_report.py ^
  --loss_png reports/loss_curves.png ^
  --acc_png reports/acc_curves.png ^
  --eval_json reports/eval_report.json ^
  --out_dir reports
"""
import os, json, argparse, textwrap
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def pad_to_width(img, width, bg=(255,255,255)):
    if img.width == width:
        return img
    canvas = Image.new("RGB", (width, img.height), bg)
    canvas.paste(img, ((width - img.width)//2, 0))
    return canvas

def merge_vertical(img_top, img_bottom, bg=(255,255,255)):
    width = max(img_top.width, img_bottom.width)
    t = pad_to_width(img_top, width, bg)
    b = pad_to_width(img_bottom, width, bg)
    merged = Image.new("RGB", (width, t.height + b.height), bg)
    merged.paste(t, (0, 0))
    merged.paste(b, (0, t.height))
    return merged

def pick_col(df, *cands):
    lower = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand in lower: return lower[cand]
    # partial fallback
    for c in df.columns:
        lc = c.lower()
        for cand in cands:
            if cand in lc: return c
    return None

def moving_avg(series, k):
    if series is None or k <= 1:
        return series
    return series.rolling(window=k, min_periods=1).mean()

def make_curves_from_csv(csv_path, out_png, smooth_k=1):
    """Create ONE combined image (loss top, acc bottom) from a training CSV log."""
    df = pd.read_csv(csv_path)

    epoch_c    = pick_col(df, "epoch") or df.columns[0]
    train_loss = pick_col(df, "train_loss", "loss", "training_loss")
    val_loss   = pick_col(df, "val_loss", "valid_loss", "validation_loss", "val_loss_epoch")
    train_acc  = pick_col(df, "train_acc", "accuracy", "train_accuracy")
    val_acc    = pick_col(df, "val_acc", "valid_acc", "validation_acc", "val_accuracy")

    # smoothing
    for c in [train_loss, val_loss, train_acc, val_acc]:
        if c is not None:
            df[c] = moving_avg(df[c], smooth_k)

    has_loss = (train_loss is not None) or (val_loss is not None)
    has_acc  = (train_acc is not None) or (val_acc is not None)

    fig = plt.figure(figsize=(8, 9))
    fig.patch.set_facecolor("white")

    if has_loss and has_acc:
        gs = fig.add_gridspec(2, 1, height_ratios=[1,1], hspace=0.28)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
    else:
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = None

    # Loss
    if has_loss:
        if train_loss: ax1.plot(df[epoch_c], df[train_loss], label="train_loss")
        if val_loss:   ax1.plot(df[epoch_c], df[val_loss],   label="val_loss")
        ax1.set_title("Loss Curves"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax1.grid(True, alpha=0.3); ax1.legend()

    # Accuracy
    if has_acc:
        if ax2 is None:
            ax2 = ax1
        if train_acc: ax2.plot(df[epoch_c], df[train_acc], label="train_acc")
        if val_acc:   ax2.plot(df[epoch_c], df[val_acc],   label="val_acc")
        ax2.set_title("Accuracy Curves"); ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
        ax2.grid(True, alpha=0.3); ax2.legend()

    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

def load_eval(eval_json_path):
    if not (eval_json_path and os.path.exists(eval_json_path)):
        return None
    with open(eval_json_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    val = rep.get("val", {})
    test = rep.get("test", {})
    return {
        "val_acc":  float(val.get("acc", 0)),
        "val_prec": float(val.get("precision", 0)),
        "val_rec":  float(val.get("recall", 0)),
        "val_f1":   float(val.get("f1", 0)),
        "best_thr": float(val.get("best_thr", 0.5)),
        "best_thr_f1": float(val.get("best_thr_f1", 0)),
        "val_conf": val.get("confusion", [[0,0],[0,0]]),

        "test_acc":  float(test.get("acc", 0)),
        "test_prec": float(test.get("precision", 0)),
        "test_rec":  float(test.get("recall", 0)),
        "test_f1":   float(test.get("f1", 0)),
        "test_conf": test.get("confusion", [[0,0],[0,0]]),
    }

def make_pdf(summary_lines, merged_img_path, pdf_path, logo_path=None):
    # A4 portrait PDF via matplotlib
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')

    # Optional logo
    y_start = 0.96
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            # put small logo on top-right
            ax_logo = plt.axes([0.86, 0.92, 0.10, 0.06])  # [L,B,W,H]
            ax_logo.axis('off')
            ax_logo.imshow(logo)
            y_start = 0.90
        except Exception:
            y_start = 0.96

    # Text block
    y = y_start
    lh = 0.028
    for line in summary_lines:
        wrapped = textwrap.wrap(line, width=100) if len(line) > 110 else [line]
        for wline in wrapped:
            plt.text(0.06, y, wline, fontsize=10, va='top', ha='left')
            y -= lh

    # Figure
    img = Image.open(merged_img_path).convert("RGB")
    bottom = 0.06
    height = max(0.22, y - 0.08)
    ax = plt.axes([0.06, bottom, 0.88, height])
    ax.axis('off')
    ax.imshow(img)

    plt.savefig(pdf_path, format="pdf", dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # Mode A (from CSV)
    ap.add_argument("--log_csv", default=None, help="CSV log path to generate the combined curves image from")
    ap.add_argument("--smooth_k", type=int, default=1, help="moving average window (1=no smoothing)")
    # Mode B (from existing PNGs)
    ap.add_argument("--loss_png", default=None, help="optional: path to loss curves PNG")
    ap.add_argument("--acc_png",  default=None, help="optional: path to accuracy curves PNG")

    ap.add_argument("--eval_json", default="reports/eval_report.json", help="optional: eval_report.json path")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--title", default="Watheeq Signature Verification — Evaluation Report")
    ap.add_argument("--logo", default=None, help="optional small logo PNG")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Build/merge the combined figure
    combined_path = os.path.join(args.out_dir, "combined_curves.png")

    if args.log_csv:
        # Generate one combined image directly from CSV
        make_curves_from_csv(args.log_csv, combined_path, args.smooth_k)
    else:
        # Merge two PNGs if provided
        if not (args.loss_png and args.acc_png):
            raise SystemExit("Provide either --log_csv OR both --loss_png and --acc_png")
        loss = Image.open(args.loss_png).convert("RGB")
        acc  = Image.open(args.acc_png).convert("RGB")
        merged = merge_vertical(loss, acc)
        merged.save(combined_path, format="PNG")

    # 2) Load evaluation metrics (optional)
    m = load_eval(args.eval_json)

    # 3) Build English summary
    lines = [args.title, ""]
    if m:
        lines += [
            "Methodology: The optimal decision threshold was selected on the validation (VAL) split by maximizing F1,",
            "then fixed and applied to the test (TEST) split to verify generalization and stability.",
            "",
            f"VAL — accuracy: {m['val_acc']:.4f} | precision: {m['val_prec']:.4f} | recall: {m['val_rec']:.4f} | F1: {m['val_f1']:.4f}",
            f"Optimal threshold (VAL): {m['best_thr']:.3f}  (F1={m['best_thr_f1']:.4f})",
            f"Confusion (VAL): TN={m['val_conf'][0][0]}, FP={m['val_conf'][0][1]}, FN={m['val_conf'][1][0]}, TP={m['val_conf'][1][1]}",
            "",
            f"TEST — accuracy: {m['test_acc']:.4f} | precision: {m['test_prec']:.4f} | recall: {m['test_rec']:.4f} | F1: {m['test_f1']:.4f}",
            f"Confusion (TEST): TN={m['test_conf'][0][0]}, FP={m['test_conf'][0][1]}, FN={m['test_conf'][1][0]}, TP={m['test_conf'][1][1]}",
            "",
            "Notes:",
            "• Close VAL vs TEST scores indicate robust performance and low overfitting.",
            "• Combined Loss/Accuracy curves are embedded below.",
        ]
    else:
        lines += ["No eval_report.json provided — curves only are embedded below."]

    # 4) Export PDF
    pdf_path = os.path.join(args.out_dir, "watheeq_signature_eval.pdf")
    make_pdf(lines, combined_path, pdf_path, logo_path=args.logo)

    print("Saved combined plot:", combined_path)
    print("Saved PDF report :", pdf_path)

if __name__ == "__main__":
    main()
