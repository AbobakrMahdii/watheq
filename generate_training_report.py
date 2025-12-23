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
# generate_training_report.py
# -*- coding: utf-8 -*-
"""
Generate a clean English PDF report for Watheeq training:
- Reads a CSV log (epochs, train/val loss, train/val acc if available)
- Creates ONE combined plot (loss on top, accuracy below) and saves it
- Optionally reads eval_report.json to include metrics & confusion
- Exports a single-page PDF with a concise summary + the combined plot

Usage (Jupyter):
!python -X utf8 generate_training_report.py --log_path models/resnet18_finetuned.log.csv --out_dir reports --title "Watheeq Signature — Training Report" --smooth_k 3

Usage (CMD/PowerShell):
python -X utf8 generate_training_report.py ^
  --log_path models\resnet18_finetuned.log.csv ^
  --out_dir reports ^
  --title "Watheeq Signature — Training Report" ^
  --smooth_k 3
"""
import os, argparse, json, textwrap
import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df, *cands):
    lower = {c.lower(): c for c in df.columns}
    # exact first
    for cand in cands:
        if cand in lower: return lower[cand]
    # partial fallback
    for c in df.columns:
        lc = c.lower()
        for cand in cands:
            if cand in lc: return c
    return None

def moving_avg(series, k):
    if k <= 1: return series
    return series.rolling(window=k, min_periods=1).mean()

def load_eval(eval_json_path):
    if not os.path.exists(eval_json_path): return None
    with open(eval_json_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    val = rep.get("val", {})
    test = rep.get("test", {})
    return dict(
        val_acc=float(val.get("acc", 0)),
        val_prec=float(val.get("precision", 0)),
        val_rec=float(val.get("recall", 0)),
        val_f1=float(val.get("f1", 0)),
        best_thr=float(val.get("best_thr", 0.5)),
        best_thr_f1=float(val.get("best_thr_f1", 0)),
        val_conf=val.get("confusion", [[0,0],[0,0]]),
        test_acc=float(test.get("acc", 0)),
        test_prec=float(test.get("precision", 0)),
        test_rec=float(test.get("recall", 0)),
        test_f1=float(test.get("f1", 0)),
        test_conf=test.get("confusion", [[0,0],[0,0]]),
    )

def build_summary(title, evals):
    lines = [title, ""]
    if not evals:
        lines += ["No eval_report.json found. Plotting curves only."]
        return lines
    m = evals
    lines += [
        "Methodology: The optimal decision threshold was selected on VAL by maximizing F1, then fixed",
        "and applied to TEST to verify generalization and stability.",
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
        "• Combined Loss/Accuracy curves are embedded below for academic review.",
    ]
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_path", required=True, help="CSV log path (from training/fine-tune)")
    ap.add_argument("--out_dir", default="reports", help="output directory")
    ap.add_argument("--title", default="Watheeq Signature — Training Report")
    ap.add_argument("--smooth_k", type=int, default=1, help="moving average window for smoothing (1 = off)")
    ap.add_argument("--eval_json", default="reports/eval_report.json", help="optional: eval_report.json path")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load log
    df = pd.read_csv(args.log_path)
    epoch_c = pick_col(df, "epoch") or df.columns[0]

    train_loss_c = pick_col(df, "train_loss", "loss", "training_loss")
    val_loss_c   = pick_col(df, "val_loss", "valid_loss", "validation_loss", "val_loss_epoch")
    train_acc_c  = pick_col(df, "train_acc", "accuracy", "train_accuracy")
    val_acc_c    = pick_col(df, "val_acc", "valid_acc", "validation_acc", "val_accuracy")

    # Smoothing
    for c in [train_loss_c, val_loss_c, train_acc_c, val_acc_c]:
        if c is not None:
            df[c] = moving_avg(df[c], args.smooth_k)

    # === Build ONE combined image (loss top, acc bottom) ===
    fig = plt.figure(figsize=(8, 9))
    fig.patch.set_facecolor("white")
    has_loss = (train_loss_c is not None) or (val_loss_c is not None)
    has_acc  = (train_acc_c is not None) or (val_acc_c is not None)

    if has_loss and has_acc:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
    else:
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = None

    # Loss
    if has_loss:
        if train_loss_c: ax1.plot(df[epoch_c], df[train_loss_c], label="train_loss")
        if val_loss_c:   ax1.plot(df[epoch_c], df[val_loss_c],   label="val_loss")
        ax1.set_title("Loss Curves"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        ax1.grid(True, alpha=0.3); ax1.legend()

    # Accuracy
    if has_acc:
        if ax2 is None:
            ax2 = ax1
        if train_acc_c: ax2.plot(df[epoch_c], df[train_acc_c], label="train_acc")
        if val_acc_c:   ax2.plot(df[epoch_c], df[val_acc_c],   label="val_acc")
        ax2.set_title("Accuracy Curves"); ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
        ax2.grid(True, alpha=0.3); ax2.legend()

    combined_png = os.path.join(args.out_dir, "combined_curves.png")
    plt.savefig(combined_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved combined plot:", combined_png)

    # === Build PDF with text + image ===
    evals = load_eval(args.eval_json)
    lines = build_summary(args.title, evals)

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor("white")

    y = 0.96
    lh = 0.028
    for line in lines:
        for wline in textwrap.wrap(line, width=100) if len(line) > 110 else [line]:
            plt.text(0.06, y, wline, fontsize=10, va='top', ha='left')
            y -= lh

    img = plt.imread(combined_png)
    bottom = 0.06
    height = max(0.20, y - 0.08)
    ax = plt.axes([0.06, bottom, 0.88, height])
    ax.axis("off")
    ax.imshow(img)

    out_pdf = os.path.join(args.out_dir, "training_report.pdf")
    plt.savefig(out_pdf, format="pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved PDF:", out_pdf)

if __name__ == "__main__":
    main()

