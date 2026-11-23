"""Figure 2(b,c) from measured GRPO group counts and held-out episode outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs/figures"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})
# palette shared with make_plots.py
C = {"zs": "#bdbdbd", "grpo": "#08519c", "xfer": "#fd8d3c", "xgrpo": "#a63603", "weak": "#6a51a3"}
H = 1.76  # same height as coverage.pdf (126.5 pt)
W = 2.25


def save(fig, name, preview):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview) / name.replace(".pdf", ".png"), dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print("wrote", OUT / name)


def bars(ax, labels, values, colors, notes, ymax):
    xs = range(len(values))
    ax.bar(xs, values, color=colors, width=0.62, edgecolor="black", linewidth=0.3, zorder=3)
    for x, v, note in zip(xs, values, notes):
        ax.text(x, v + ymax * 0.025, note, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, lw=0.4, color="#e5e5e5", zorder=0)
    ax.tick_params(axis="x", length=0)


def fig_signal(preview):
    groups = json.loads((OUT.parent / "observations.json").read_text())["groups"]
    vals = [100 * a / b for a, b in groups]
    fig, ax = plt.subplots(figsize=(W, H))
    bars(ax, ["weak\nstart", "strong,\nzero-shot", "strong,\ntransfer SFT"], vals,
         [C["weak"], C["grpo"], C["xgrpo"]], [f"{a} of {b}" for a, b in groups], 48)
    ax.set_ylabel("groups with a gradient (%)")
    fig.tight_layout(pad=0.2)
    save(fig, "obs_signal.pdf", preview)


def fig_diversity(preview):
    vals = json.loads((OUT.parent / "observations.json").read_text())["gaps"]
    fig, ax = plt.subplots(figsize=(W, H))
    bars(ax, ["zero-shot", "transfer\nSFT", "transfer\nSFT$\\to$GRPO"], vals,
         [C["zs"], C["xfer"], C["xgrpo"]], [f"{v}" for v in vals], 28)
    ax.set_ylabel("pass@8 $-$ pass@1 (points)")
    fig.tight_layout(pad=0.2)
    save(fig, "obs_diversity.pdf", preview)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="optional directory for PNG previews")
    args = ap.parse_args()
    fig_signal(args.preview)
    fig_diversity(args.preview)
