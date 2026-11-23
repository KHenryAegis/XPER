"""Figure 2(a): exact-replay coverage by step depth, from the archived test and training depth counts in data/records.

    python -B scripts/figures/make_plots.py        # writes outputs/figures/coverage.pdf
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures"
PREVIEW = Path(os.environ["PREVIEW"]) if os.environ.get("PREVIEW") else None

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "lines.linewidth": 1.2, "lines.markersize": 3.5,
})
COL = 3.4    # column width, inches
C = {"src": "#7f7f7f", "zs": "#bdbdbd", "selfexp": "#9ecae1", "rft": "#4292c6", "grpo": "#08519c",
     "xfer": "#fd8d3c", "icl": "#d94801", "xgrpo": "#a63603", "weak": "#6a51a3"}


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    if PREVIEW:
        fig.savefig(PREVIEW / name.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- coverage by depth
def fig_coverage():
    records = ROOT / "data/records"
    test = json.loads((records / "coverage_test.json").read_text())
    tel = json.loads((records / "coverage_telecom_train.json").read_text())
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    d = list(range(1, 11))
    ax.bar(d, [100 * test["by_depth"][str(i - 1)] for i in d], color=C["zs"], width=0.7,
           label="strong target, test tasks")
    for key, lab, c, mk in [("strong_selfexp", "strong, self-exp. rollouts (train)", C["selfexp"], "o"),
                            ("strong_rft", "strong, RFT rollouts (train)", C["rft"], "s"),
                            ("strong_grpo", "strong, GRPO rollouts (train)", C["grpo"], "^"),
                            ("weak_selfexp", "weak, self-exp. rollouts (train)", C["weak"], "x")]:
        ax.plot(d, [100 * tel[key]["by_depth"][str(i - 1)] for i in d], marker=mk, color=c, label=lab)
    ax.set_xticks(d)
    ax.set_xlabel("step depth $t$")
    ax.set_ylabel("steps covered (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "coverage.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_coverage()
