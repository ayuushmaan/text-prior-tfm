"""Paper figures for TFM3.0 draft. Reads results/*.json, writes paper/figures/*.png.

Usage:  py -3.11 paper/figures.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def smooth(x, w=25):
    x = np.asarray(x, dtype=float)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def fig_train_curves():
    d = json.loads((RES / "curves_500.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    steps = np.arange(1, 1501)
    ax.plot(steps, smooth(d["histC_500"]), label="control (numeric-only)")
    ax.plot(steps, smooth(d["histT_500"]), label="fusion, grounded text (8 cols)")
    ax.plot(steps, smooth(d["uhist_500"]), label="fusion, shuffled text (8 cols)")
    ax.set(xlabel="pretraining steps (500 tables, mask-train 0.3)",
           ylabel="train loss (25-step MA)", title="Text fits train best — including ungrounded text")
    ax.legend(fontsize=9)
    fig.savefig(OUT / "fig1_train_curves.png")
    print("fig1 ok")


def fig_three_way():
    d = json.loads((RES / "results_500_3way.json").read_text())
    labels = ["clean", "masked 50%"]
    arms = [("ctrl", "control"), ("tmpl8", "+grounded"), ("shuf8", "+shuffled")]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (key, name) in enumerate(arms):
        vals = [d["eval"][f"mask_{m}"][key] for m in (0.0, 0.5)]
        ax.bar(x + (i - 1) * w, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set(ylabel="held-out ICL accuracy (100 tables)",
           title="Only grounded text helps; shuffled capacity adds nothing")
    ax.legend(fontsize=9)
    ax.set_ylim(0.65, 0.85)
    fig.savefig(OUT / "fig2_three_way.png")
    print("fig2 ok")


def fig_icl():
    d = json.loads((RES / "summary.json").read_text())["icl_nanotabpfn"]
    ctx = d["ctx"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(ctx, d["ctrl_acc"], "o-", label="numeric-only")
    ax.plot(ctx, d["fuse_acc"], "s--", label="+text fusion")
    ax.set(xlabel="in-context rows (NanoTabPFN, trained on 100-row tables)",
           ylabel="accuracy (4 tables x 200 test rows)",
           title="ICL scales with context; fusion trails throughout")
    ax.legend(fontsize=9)
    fig.savefig(OUT / "fig3_icl_context.png")
    print("fig3 ok")


if __name__ == "__main__":
    fig_train_curves()
    fig_three_way()
    fig_icl()
    print("figures in", OUT)
