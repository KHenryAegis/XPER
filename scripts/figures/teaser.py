"""Fig. 1 teaser: three ways a new model can use an environment it shares with an older model.

    (a) re-explore every generation (status quo)
    (b) reuse the stored log as a cache (exact replay)        -> fails: targets diverge at action 2
    (c) reuse the stored log by learning (finite injection)   -> ours

Writes outputs/figures/teaser.pdf (vector, Linux Biolinum to match acmart) and, with --preview PATH, a PNG.
Every number is copied from the paper; nothing here is measured:
  (a) weak telecom target, no injection: 0.10% test pass@1 at 44.5-45.2k target calls  (tab:injection)
      source exploration: 44,414 calls                                                  (sec:setup)
  (b) strong telecom target: 7.5% of steps covered, 0 of 3,991 episodes,
      85% of episodes leave the trie at their second action                             (sec:coverage)
  (c) real-source injection, <=100 exposures: telecom 0.10 -> 43.44%, math 2.61 -> 19.58% (tab:injection)
      advantage bars use the true group-normalised values for 1 success among 9 members.
"""

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/figures/teaser.pdf"

W, H = 7.0, 1.92  # inches; figure* in acmart is ~7.0in wide

INK, GREY, MUTED = "#262626", "#7A7A7A", "#A8A8A8"
PANEL_F, PANEL_E = "#F7F7F7", "#CFCFCF"
SRC, SRC_F = "#2F6DB5", "#DDE8F6"  # source model / stored experience
TGT, TGT_F = "#D47A2E", "#FBE7D6"  # new target model
OK, OK_F, OURS_F = "#2E8B57", "#DFF1E6", "#F3FAF5"
BAD, BAD_F = "#C2414B", "#F8E0E2"
ENV_F, ENV_E = "#E6E6E6", "#9A9A9A"

FS, FS_S, FS_CAP = 7.0, 6.2, 8.0  # body, secondary, sub-caption (pt)


def setup_font():
    family = ["Helvetica", "Arial", "DejaVu Sans"]
    try:
        path = subprocess.run(["kpsewhich", "LinBiolinum_R.otf"], capture_output=True, text=True).stdout.strip()
    except OSError:
        path = ""
    if path:
        d = Path(path).parent
        for f in ("LinBiolinum_R.otf", "LinBiolinum_RB.otf", "LinBiolinum_RI.otf"):
            if (d / f).exists():
                fm.fontManager.addfont(str(d / f))
        family = ["Linux Biolinum O"] + family
    plt.rcParams.update({"font.family": family, "pdf.fonttype": 42, "mathtext.fontset": "custom",
                         "mathtext.rm": family[0], "mathtext.it": family[0] + ":italic"})


# ----------------------------------------------------------------------------- primitives
def rbox(ax, x, y, w, h, fc, ec, lw=0.7, r=0.04, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(ax, x, y, s, size=FS, color=INK, ha="center", va="center", weight="normal", z=6, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=z,
            linespacing=1.05, **kw)


def check(ax, cx, cy, s, color=OK, lw=1.3, z=7):
    ax.plot([cx - 0.45 * s, cx - 0.12 * s, cx + 0.48 * s], [cy, cy + 0.33 * s, cy - 0.38 * s],
            color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=z)


def cross(ax, cx, cy, s, color=BAD, lw=1.2, z=7):
    for dy in (1, -1):
        ax.plot([cx - 0.36 * s, cx + 0.36 * s], [cy - dy * 0.36 * s, cy + dy * 0.36 * s],
                color=color, lw=lw, solid_capstyle="round", zorder=z)


def arrow(ax, p0, p1, color=INK, lw=0.8, both=False, ls="-", z=4, rad=0.0, ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=ms,
                                 lw=lw, color=color, ls=ls, shrinkA=0, shrinkB=0, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def cylinder(ax, x, y, w, h, fc, ec, lw=0.8, z=2):
    e = 0.16 * w
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, fc=fc, ec="none", zorder=z))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=ec, lw=lw, zorder=z + 1)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=ec, lw=lw, zorder=z + 1)
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, fc=fc, ec=ec, lw=lw, zorder=z + 1))
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, fc=fc, ec=ec, lw=lw, zorder=z + 2))


def trash(ax, cx, top, w, h, color=GREY, lw=0.8, z=3):
    ax.plot([cx - 0.62 * w, cx + 0.62 * w], [top, top], color=color, lw=lw * 1.3, zorder=z,
            solid_capstyle="round")
    ax.plot([cx - 0.18 * w, cx - 0.18 * w, cx + 0.18 * w, cx + 0.18 * w],
            [top, top - 0.10 * h, top - 0.10 * h, top], color=color, lw=lw, zorder=z)
    xs = [cx - 0.5 * w, cx - 0.4 * w, cx + 0.4 * w, cx + 0.5 * w]
    ax.plot(xs, [top + 0.08 * h, top + h, top + h, top + 0.08 * h], color=color, lw=lw, zorder=z)
    for f in (-0.18, 0.0, 0.18):
        ax.plot([cx + f * w, cx + f * w], [top + 0.25 * h, top + 0.85 * h], color=color, lw=lw * 0.8,
                zorder=z)


def rollout(ax, x, y, s, kind, z=5, lw=0.7):
    """Square glyph for one episode: 'tf' target failure, 'ss' source success."""
    if kind == "tf":
        rbox(ax, x, y, s, s, TGT_F, TGT, lw=lw, r=0.018, z=z)
        cross(ax, x + s / 2, y + s / 2, s * 0.62, lw=0.9, z=z + 1)
    else:
        rbox(ax, x, y, s, s, SRC_F, SRC, lw=lw * 1.6, r=0.018, z=z)
        check(ax, x + s / 2, y + s / 2, s * 0.62, lw=1.1, z=z + 1)


def verdict(ax, x, w, lines, good):
    y, h = 1.30, 0.27
    rbox(ax, x + 0.06, y, w - 0.12, h, OK_F if good else BAD_F, "none", r=0.03, z=2)
    (check if good else cross)(ax, x + 0.19, y + h / 2, 0.11, lw=1.4)
    text(ax, x + 0.31, y + h / 2, lines, size=FS_S + 0.2, ha="left", color=INK)


# ----------------------------------------------------------------------------- panels
PANELS = {"a": (0.04, 2.08), "b": (2.20, 2.18), "c": (4.46, 2.50)}
TOP, BOT = 0.04, 1.62
CAPTIONS = {"a": "(a) Re-explore every generation",
            "b": "(b) Reuse the log as a cache",
            "c": "(c) Reuse the log by learning"}


def panel_a(ax, x0, w):
    bw, bh = 0.60, 0.34
    ys, yt = 0.24, 0.82  # source / target row tops
    rbox(ax, x0 + 0.08, ys, bw, bh, SRC_F, SRC)
    text(ax, x0 + 0.38, ys + 0.12, "Source", weight="bold", color=SRC)
    text(ax, x0 + 0.38, ys + 0.25, "4B-Instruct", size=FS_S, color=GREY)
    rbox(ax, x0 + 0.08, yt, bw, bh, TGT_F, TGT)
    text(ax, x0 + 0.38, yt + 0.12, "New target", weight="bold", color=TGT)
    text(ax, x0 + 0.38, yt + 0.25, "8B-Base", size=FS_S, color=GREY)

    ex, ew = x0 + 1.12, 0.38
    rbox(ax, ex, ys, ew, yt + bh - ys, ENV_F, ENV_E)
    text(ax, ex + ew / 2, (ys + yt + bh) / 2 - 0.07, "Tool", weight="bold")
    text(ax, ex + ew / 2, (ys + yt + bh) / 2 + 0.07, "env.", weight="bold")

    for y, lab in ((ys + bh / 2, "44k calls"), (yt + bh / 2, "44k again")):
        arrow(ax, (x0 + 0.08 + bw + 0.02, y), (ex - 0.02, y), both=True, lw=0.8)
        text(ax, (x0 + 0.70 + ex) / 2, y - 0.075, lab, size=FS_S, color=INK)

    # source log is dropped after its updates
    tx = x0 + 1.80
    arrow(ax, (ex + ew + 0.02, ys + bh / 2), (tx - 0.12, ys + bh / 2), lw=0.7, color=GREY)
    trash(ax, tx, ys + 0.07, 0.15, 0.19)
    text(ax, tx, ys + 0.40, "log dropped", size=FS_S, color=GREY)

    # the new target's GRPO groups contain no success
    s, g = 0.075, 0.02
    gx = x0 + 1.58
    arrow(ax, (ex + ew + 0.02, yt + bh / 2), (gx - 0.03, yt + bh / 2), lw=0.7, color=GREY)
    for i in range(8):
        rollout(ax, gx + (i % 4) * (s + g), yt + 0.06 + (i // 4) * (s + g), s, "tf", lw=0.55)
    text(ax, gx + 2 * s + 1.5 * g, yt + 0.33, "0 of 8 succeed", size=FS_S, color=BAD)

    verdict(ax, x0, w, "GRPO from scratch, 44k calls:\n0.1% pass@1", good=False)


def panel_b(ax, x0, w):
    cols = [x0 + 0.20 + i * 0.42 for i in range(5)]
    r = 0.032
    yroot = 0.58
    src_nodes = {
        (0, yroot): [(1, yroot)],
        (1, yroot): [(2, 0.28), (2, 0.50), (2, 0.72)],
        (2, 0.28): [(3, 0.22), (3, 0.36)],
        (2, 0.50): [(3, 0.52)],
        (2, 0.72): [(3, 0.68), (3, 0.82)],
        (3, 0.22): [(4, 0.22)], (3, 0.36): [(4, 0.36)], (3, 0.52): [(4, 0.52)],
        (3, 0.68): [(4, 0.68)], (3, 0.82): [(4, 0.82)],
    }
    nodes = set(src_nodes)
    # target follows the shared first action (cache hit), then leaves the trie
    ax.plot([cols[0], cols[1]], [yroot, yroot], color=TGT, lw=4.2, alpha=0.35, zorder=2,
            solid_capstyle="round")
    for (d, y), kids in src_nodes.items():
        for kd, ky in kids:
            ax.plot([cols[d], cols[kd]], [y, ky], color=SRC, lw=0.9, zorder=3)
            nodes.add((kd, ky))
    for d, y in nodes:
        ax.add_patch(Circle((cols[d], y), r if d else r * 1.25, fc=SRC if d else INK, ec="none", zorder=5))

    ymiss = 1.04
    path = [(cols[1], yroot), (cols[2], ymiss), (cols[3], ymiss), (cols[4], ymiss)]
    ax.plot([p[0] for p in path], [p[1] for p in path], color=TGT, lw=1.1, ls=(0, (3, 1.6)), zorder=4)
    for px, py in path[1:]:
        ax.add_patch(Circle((px, py), r, fc="white", ec=TGT, lw=1.0, zorder=5))
    cross(ax, cols[2], ymiss, 0.14, lw=1.5, z=8)

    text(ax, cols[0], yroot + 0.12, "task", size=FS_S, color=GREY)
    text(ax, cols[1], 0.14, "$t$=1", size=FS_S, color=GREY)
    text(ax, cols[2], 0.14, "$t$=2", size=FS_S, color=GREY)
    check(ax, cols[1], yroot - 0.12, 0.12, lw=1.4)
    text(ax, cols[1] - 0.01, yroot + 0.12, "hit", size=FS_S, color=OK)
    text(ax, cols[4] + 0.06, 0.12, "stored source paths", size=FS_S, color=SRC, ha="right")
    text(ax, cols[2] + 0.05, ymiss + 0.15, "miss: 85% of target episodes\nleave the trie here", size=FS_S,
         color=BAD, ha="left")
    text(ax, cols[4] + 0.06, ymiss - 0.11, "target", size=FS_S, color=TGT, ha="right")

    verdict(ax, x0, w, "Serves 7.5% of target steps,\n0 of 3,991 episodes", good=False)


def panel_c(ax, x0, w):
    # experience store holding source successes
    cx, cw = x0 + 0.10, 0.44
    cylinder(ax, cx, 0.22, cw, 0.52, SRC_F, SRC)
    text(ax, cx + cw / 2, 0.37, "Source", size=FS_S, color=SRC, weight="bold")
    text(ax, cx + cw / 2, 0.48, "log", size=FS_S, color=SRC, weight="bold")
    for i in range(3):
        rollout(ax, cx + 0.06 + i * 0.115, 0.56, 0.09, "ss", lw=0.5)

    # one GRPO group: 8 failed target rollouts + 1 injected source success
    gx, gy, s, g = x0 + 0.80, 0.24, 0.15, 0.04
    gw = 3 * s + 2 * g + 0.10
    rbox(ax, gx, gy, gw, 3 * s + 2 * g + 0.10, "white", PANEL_E, lw=0.7, r=0.03, z=2)
    text(ax, gx + gw / 2, 0.14, "GRPO group", size=FS_S, weight="bold")
    cells = [(c, rr) for rr in range(3) for c in range(3)]
    for (c, rr) in cells:
        kind = "ss" if (c, rr) == (0, 2) else "tf"
        rollout(ax, gx + 0.05 + c * (s + g), gy + 0.05 + rr * (s + g), s, kind)
    inj_y = gy + 0.05 + 2 * (s + g) + s / 2
    arrow(ax, (cx + cw + 0.01, 0.62), (gx + 0.035, inj_y), color=SRC, lw=1.1, rad=0.25, ms=7)
    text(ax, (cx + gx + gw) / 2 - 0.02, 1.04, "inject one stored success\nonly when all 8 rollouts fail",
         size=FS_S, color=SRC)

    # group-normalised advantages, true values for 1 success among 9 members
    ax_x, base = gx + gw + 0.24, 0.50
    arrow(ax, (gx + gw + 0.03, base), (ax_x - 0.05, base), lw=0.8)
    text(ax, ax_x + 0.30, 0.14, "advantage", size=FS_S, weight="bold")
    mean, sd = 1 / 9, (1 / 9 * 8 / 9) ** 0.5
    scale = 0.30 / ((1 - mean) / sd)
    bw, bg = 0.045, 0.024
    ax.plot([ax_x - 0.02, ax_x + 9 * bw + 8 * bg + 0.02], [base, base], color=GREY, lw=0.6, zorder=3)
    for i in range(9):
        a = ((1 if i == 0 else 0) - mean) / sd
        hgt = a * scale
        ax.add_patch(Rectangle((ax_x + i * (bw + bg), base - max(hgt, 0)), bw, abs(hgt),
                               fc=SRC if i == 0 else TGT, ec="none", zorder=4))
    text(ax, ax_x + 0.30, base + 0.13, "non-zero → gradient", size=FS_S, color=OK)
    text(ax, ax_x + 0.30, base + 0.25, "(no injection: all zero)", size=FS_S, color=GREY)

    # finite dose, then target-only training
    tx, ty, th = ax_x - 0.04, 0.98, 0.07
    rbox(ax, tx, ty, 0.24, th, SRC_F, SRC, lw=0.6, r=0.015, z=3)
    rbox(ax, tx + 0.26, ty, 0.44, th, TGT_F, TGT, lw=0.6, r=0.015, z=3)
    text(ax, tx, ty - 0.08, "≤100 injections", size=FS_S, color=SRC, ha="left")
    text(ax, tx + 0.70, ty + th + 0.08, "then target-only rounds", size=FS_S, color=TGT, ha="right")

    verdict(ax, x0, w, "Equal target calls, pass@1:\ntelecom 0.1 → 43.4%, math 2.6 → 19.6%", good=True)


def render(path, preview=None):
    setup_font()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    for key, (x0, w) in PANELS.items():
        ours = key == "c"
        rbox(ax, x0, TOP, w, BOT - TOP, OURS_F if ours else PANEL_F, OK if ours else PANEL_E,
             lw=1.0 if ours else 0.7, r=0.06, z=0)
        text(ax, x0 + w / 2, 1.76, CAPTIONS[key], size=FS_CAP, weight="bold")
        {"a": panel_a, "b": panel_b, "c": panel_c}[key](ax, x0, w)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if preview:
        Path(preview).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="optional PNG preview path")
    args = ap.parse_args()
    render(OUT, args.preview)
    print("wrote", OUT)
