"""Fig. 3: XPER overview for Section 4 (single column).

Writes outputs/figures/overview.pdf. Topology follows sections/method.tex and sections/discussion.tex:
  - every rollout of every model is written to the store with provenance and charged calls (Sec. 4.1)
  - store = trajectories / steps / per-task prefix trie (Sec. 4.1)
  - consumers: (1) exact replay as a cache (Sec. 3.1), (2) finite injection into all-failure GRPO groups
    (Sec. 4.3), (3) offline transfer SFT / ICL (Sec. 4.2), with the gate probe choosing among offline options (Sec. 7.1)
  - the trained target's rollouts are written back into the store, which accumulates across generations
Section numbers are drawn as text: update SEC below if sections are renumbered (the LaTeX caption uses \\cref).
No result numbers appear in this figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from teaser import (BAD, ENV_E, ENV_F, GREY, INK, OK, OURS_F, SRC, SRC_F, TGT, TGT_F,  # noqa: E402
                    arrow, cross, rbox, rollout, setup_font, text)

OUT = Path(__file__).resolve().parents[2] / "outputs/figures/overview.pdf"
SEC = {"store": "§4.1", "offline": "§4.2", "inject": "§4.3", "replay": "§3.1", "gate": "§7.1"}

W, H = 3.33, 2.62
FS, FS_S, FS_T = 6.6, 5.9, 7.0
STORE_F = "#EEF4FB"


def badge(ax, cx, cy, n, color):
    ax.add_patch(Circle((cx, cy), 0.058, fc=color, ec="none", zorder=8))
    text(ax, cx, cy + 0.003, str(n), size=5.6, color="white", weight="bold", z=9)


def table_glyph(ax, x, y, w=0.15, h=0.12, color=SRC):
    rbox(ax, x, y, w, h, "white", color, lw=0.5, r=0.01, z=6)
    for k in (1, 2):
        ax.plot([x, x + w], [y + k * h / 3, y + k * h / 3], color=color, lw=0.4, zorder=7)


def trie_glyph(ax, x, y, w=0.20, h=0.15, color=SRC):
    root = (x, y + h / 2)
    mids = [(x + w / 2, y + h * 0.2), (x + w / 2, y + h * 0.8)]
    leaves = [(x + w, y), (x + w, y + h * 0.4), (x + w, y + h)]
    for m in mids:
        ax.plot([root[0], m[0]], [root[1], m[1]], color=color, lw=0.6, zorder=6)
    for m, lv in ((mids[0], leaves[0]), (mids[0], leaves[1]), (mids[1], leaves[2])):
        ax.plot([m[0], lv[0]], [m[1], lv[1]], color=color, lw=0.6, zorder=6)
    for p in [root] + mids + leaves:
        ax.add_patch(Circle(p, 0.017, fc=color, ec="none", zorder=7))


def render(path, preview=None):
    setup_font()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    L, R = 0.05, 3.10  # content bounds; x in (R, W) is the write-back lane

    # ---- producers: any model interacting with the environment
    y0, h0 = 0.05, 0.32
    rbox(ax, L, y0, 0.82, h0, SRC_F, SRC)
    text(ax, L + 0.41, y0 + 0.11, "Source model", size=FS_T, weight="bold", color=SRC)
    text(ax, L + 0.41, y0 + 0.23, "earlier generation", size=FS_S, color=GREY)
    ex = 1.17
    rbox(ax, ex, y0, 0.76, h0, ENV_F, ENV_E)
    text(ax, ex + 0.38, y0 + 0.11, "Tool env.", size=FS_T, weight="bold")
    text(ax, ex + 0.38, y0 + 0.23, "charged per call", size=FS_S, color=GREY)
    tx = 2.24
    rbox(ax, tx, y0, R - tx, h0, TGT_F, TGT)
    text(ax, (tx + R) / 2, y0 + 0.11, "Target model", size=FS_T, weight="bold", color=TGT)
    text(ax, (tx + R) / 2, y0 + 0.23, "new generation", size=FS_S, color=GREY)
    arrow(ax, (L + 0.84, y0 + h0 / 2), (ex - 0.02, y0 + h0 / 2), both=True, lw=0.7, ms=5)
    arrow(ax, (ex + 0.78, y0 + h0 / 2), (tx - 0.02, y0 + h0 / 2), both=True, lw=0.7, ms=5)

    # ---- every rollout is written to the store
    sy, sh = 0.58, 0.62
    for x in (L + 0.41, (tx + R) / 2):
        arrow(ax, (x, y0 + h0 + 0.02), (x, sy - 0.01), color=SRC, lw=0.9, ms=6)
    text(ax, (L + R) / 2 + 0.02, 0.475, "write every rollout + provenance + charged calls",
         size=FS_S, color=SRC, style="italic")

    # ---- experience store
    rbox(ax, L, sy, R - L, sh, STORE_F, SRC, lw=0.9, r=0.05)
    text(ax, L + 0.08, sy + 0.085, f"Experience store ({SEC['store']})", size=FS_T, weight="bold",
         color=SRC, ha="left")
    cg, cy, ch = 0.065, sy + 0.17, 0.38
    cw = (R - L - 0.16 - 2 * cg) / 3
    comps = [("trajectories", "model, version,\nreward, calls"),
             ("steps", "canonical action,\nobservation"),
             ("trie nodes", "per-task prefix\nof actions")]
    for i, (name, body) in enumerate(comps):
        cx = L + 0.08 + i * (cw + cg)
        rbox(ax, cx, cy, cw, ch, "white", "#B9CCE6", lw=0.6, r=0.03, z=4)
        text(ax, cx + 0.07, cy + 0.09, name, size=FS, weight="bold", ha="left")
        text(ax, cx + 0.07, cy + 0.255, body, size=FS_S, color=GREY, ha="left")
        if i < 2:
            table_glyph(ax, cx + cw - 0.21, cy + 0.04)
        else:
            trie_glyph(ax, cx + cw - 0.27, cy + 0.035)

    # ---- consumers
    by, bh = 1.53, 0.66
    xs = [(L, 0.92), (1.04, 1.06), (2.17, R - 2.17)]
    store_bot = sy + sh
    # (1) exact replay as a cache: measured, rarely hits
    x, w = xs[0]
    rbox(ax, x, by, w, bh, "#F5F5F5", "#9A9A9A", lw=0.7, ls=(0, (3, 2)))
    badge(ax, x + 0.11, by + 0.11, 1, GREY)
    text(ax, x + 0.21, by + 0.11, "Exact replay", size=FS, weight="bold", ha="left")
    text(ax, x + 0.08, by + 0.32, "cache lookup\nin the trie", size=FS_S, color=GREY, ha="left")
    cross(ax, x + 0.13, by + 0.545, 0.09, lw=1.2)
    text(ax, x + 0.21, by + 0.545, f"rarely hits ({SEC['replay']})", size=FS_S, color=BAD, ha="left")
    arrow(ax, (x + w / 2, store_bot + 0.01), (x + w / 2, by - 0.01), color=GREY, lw=0.8, ls=(0, (3, 2)),
          ms=5)
    text(ax, x + w / 2 + 0.05, (store_bot + by) / 2, "lookup", size=FS_S, color=GREY, ha="left")

    # (2) finite injection: the contribution
    x, w = xs[1]
    rbox(ax, x, by, w, bh, OURS_F, OK, lw=1.1)
    badge(ax, x + 0.11, by + 0.11, 2, OK)
    text(ax, x + 0.21, by + 0.11, f"Injection ({SEC['inject']})", size=FS, weight="bold", ha="left")
    s, g = 0.07, 0.018
    gx, gy = x + 0.08, by + 0.23
    for k in range(9):
        c, r_ = k % 3, k // 3
        rollout(ax, gx + c * (s + g), gy + r_ * (s + g), s, "ss" if (c, r_) == (0, 2) else "tf", lw=0.45)
    text(ax, gx + 3 * (s + g) + 0.04, by + 0.405, "1 stored success\ninto all-failure\ngroups, capped",
         size=FS_S, color=INK, ha="left")
    arrow(ax, (x + w / 2, store_bot + 0.01), (x + w / 2, by - 0.01), color=OK, lw=1.2, ms=6)
    text(ax, x + w / 2 + 0.05, (store_bot + by) / 2, "successes", size=FS_S, color=OK, ha="left")

    # (3) offline transfer, chosen by the gate probe
    x, w = xs[2]
    gy0, gh = store_bot + 0.08, 0.17
    rbox(ax, x + 0.06, gy0, w - 0.12, gh, "white", INK, lw=0.6, r=0.03, z=4)
    text(ax, x + w / 2, gy0 + gh / 2, f"gate probe ({SEC['gate']})", size=FS_S, weight="bold", z=6)
    arrow(ax, (x + w / 2, store_bot + 0.01), (x + w / 2, gy0 - 0.005), lw=0.7, ms=4.5)
    arrow(ax, (x + w / 2, gy0 + gh + 0.005), (x + w / 2, by - 0.01), lw=0.7, ms=4.5)
    rbox(ax, x, by, w, bh, "white", SRC, lw=0.7)
    badge(ax, x + 0.11, by + 0.11, 3, SRC)
    text(ax, x + 0.21, by + 0.11, f"Offline ({SEC['offline']})", size=FS, weight="bold", ha="left")
    text(ax, x + 0.08, by + 0.30, "SFT: into weights", size=FS_S, ha="left")
    text(ax, x + 0.08, by + 0.43, "ICL: into prompt", size=FS_S, ha="left")
    text(ax, x + 0.08, by + 0.56, "or no reuse", size=FS_S, color=GREY, ha="left")

    # ---- trained target, whose rollouts are written back
    ty, th = 2.36, 0.22
    for (x, w), col in zip(xs[1:], (OK, SRC)):
        arrow(ax, (x + w / 2, by + bh + 0.01), (x + w / 2, ty - 0.01), color=col, lw=0.9, ms=5)
    rbox(ax, L, ty, R - L, th, TGT_F, TGT)
    text(ax, (L + R) / 2, ty + th / 2, "Trained target: its rollouts are written back for the next model",
         size=FS_S + 0.2, color=INK)
    lane, back_y = R + 0.12, sy + sh / 2
    ax.plot([R + 0.01, lane, lane], [ty + th / 2, ty + th / 2, back_y], color=TGT, lw=0.9, zorder=3)
    arrow(ax, (lane, back_y), (R + 0.02, back_y), color=TGT, lw=0.9, ms=5)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if preview:
        Path(preview).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=400)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="optional PNG preview path")
    render(OUT, ap.parse_args().preview)
