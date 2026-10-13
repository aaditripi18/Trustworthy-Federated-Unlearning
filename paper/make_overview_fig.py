"""Fig. 1 of the paper: functional block diagram of the study.

One top-to-bottom layered diagram:

  input data (three blocks)
  dashed container "federated simulation"
      central pipeline: federated training -> original model -> fine-tuning
      helper blocks on the right: pretrained weights, frozen reference, explanation loss
      layer "models compared":  Oracle | Plain | EPU
      layer "evaluation":       utility | forgetting | explanation | efficiency
  two bars underneath: seeds, hardware

    python paper/make_overview_fig.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "savefig.dpi": 600,
})

INK, ARROW = "#1f1f1f", "#333333"
RED = ("#f6ddd8", "#b4574d")        # input data
GREEN = ("#dcecdd", "#5d9163")      # central pipeline
BLUE = ("#dae5f3", "#557bb0")       # helper blocks
ORANGE = ("#f7e6cf", "#b8843c")     # models compared
PURPLE = ("#e3dff5", "#7768b6")     # evaluation
GREY = ("#ececec", "#8a8a8a")       # run conditions
CONT = ("#f7f7f7", "#9a9a9a")       # container

W = 716.0
Y0, Y1 = -20.0, 440.0                # drawing units; 100 units = 1 inch
FS_T, FS_S, FS_H = 7.6, 6.5, 7.2


def txt(ax, x, y, s, fs=FS_S, ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, color=INK, ha=ha, va=va, zorder=6, **kw)


def block(ax, x, y, w, h, title, lines=(), c=GREEN, lw=0.9, step=12.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4.5",
                                fc=c[0], ec=c[1], lw=lw, zorder=3))
    rows = [(title, True)] + [(ln, False) for ln in lines]
    y0 = y + h / 2 + step * (len(rows) - 1) / 2
    for i, (s, bold) in enumerate(rows):
        txt(ax, x + w / 2, y0 - i * step - 0.3, s, fs=FS_T if bold else FS_S,
            fontweight="bold" if bold else "normal")


def seg(ax, xs, ys):
    ax.plot(xs, ys, color=ARROW, lw=0.9, solid_capstyle="butt", solid_joinstyle="miter",
            zorder=2)


def head(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=0.42,head_width=0.19",
                                 mutation_scale=9, lw=0.9, color=ARROW, shrinkA=0, shrinkB=0,
                                 zorder=2.5))


def route(ax, pts):
    xs, ys = zip(*pts)
    seg(ax, xs[:-1], ys[:-1])
    head(ax, pts[-2], pts[-1])


def dashed(ax, xs, ys):
    ax.plot(xs, ys, color=ARROW, lw=0.9, ls=(0, (3, 2)), zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(W / 100, 0.80 * (Y1 - Y0) / 100))   # compact rows
    ax.set_xlim(0, W); ax.set_ylim(Y0, Y1); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ---------------------------------------------------------------- input data
    txt(ax, W / 2, 433, "Input data", fs=FS_H)
    iw, ig, iy, ih = 204.0, 28.0, 396.0, 30.0
    ix = [24 + i * (iw + ig) for i in range(3)]
    block(ax, ix[0], iy, iw, ih, "NIH ChestX-ray14", ["112,120 radiographs, 14 findings"], RED)
    block(ax, ix[1], iy, iw, ih, "Patient-level split", ["70 / 15 / 15, patient-disjoint"], RED)
    block(ax, ix[2], iy, iw, ih, "Non-IID client partition",
          [r"Dirichlet ($\alpha$ = 0.5) over primary finding"], RED)
    head(ax, (ix[0] + iw, iy + ih / 2), (ix[1], iy + ih / 2))
    head(ax, (ix[1] + iw, iy + ih / 2), (ix[2], iy + ih / 2))

    # ---------------------------------------------------------------- container
    ax.add_patch(FancyBboxPatch((12, 16), 692, 364, boxstyle="round,pad=0,rounding_size=6",
                                fc=CONT[0], ec=CONT[1], lw=0.9, ls=(0, (4, 3)), zorder=1))
    txt(ax, 24, 370.5, "Federated simulation: FedAvg, DeiT-Small, $K$ = 5 clients", fs=FS_H,
        ha="left", fontweight="bold")

    # ---------------------------------------------------------------- training
    cx0, cw = 196.0, 262.0
    cc = cx0 + cw / 2
    rx0, rw = 506.0, 186.0
    rc = rx0 + rw / 2
    block(ax, cx0, 322, cw, 32, "Federated training",
          ["2 local epochs, weighted BCE, warm-up + cosine LR"], GREEN)
    block(ax, rx0, 322, rw, 32, "Pretrained DeiT-Small", ["ImageNet weights"], BLUE)
    block(ax, cx0, 272, cw, 32, r"Original model  $\theta_o$",
          ["best validation round (early stopping)"], GREEN)
    block(ax, rx0, 272, rw, 32, r"Frozen reference  $\theta_o$",
          [r"rollout map $\mathbf{M}_{\theta_o}(x)$, no gradient"], BLUE)
    px = ix[2] + iw / 2
    route(ax, [(px, iy), (px, 388), (cc, 388), (cc, 354)])
    head(ax, (rx0, 338), (cx0 + cw, 338))
    head(ax, (cc, 322), (cc, 304))
    head(ax, (cx0 + cw, 288), (rx0, 288))

    # ---------------------------------------------------------------- EPU local update
    ux, uw = 56.0, 180.0                 # left column of the update
    uc = ux + uw / 2
    ax_, aw = 272.0, 186.0               # middle column
    ac = ax_ + aw / 2
    dx = 220.0
    head(ax, (dx, 272), (dx, 250))
    txt(ax, dx + 6, 261.5,
        "erasure request from client $u$:  EPU local update at each remaining client",
        ha="left", fs=6.4, style="italic")
    block(ax, ux, 218, uw, 32, r"Trainable model  $\theta$",
          [r"DeiT-Small, starts from $\theta_o$"], GREEN)
    block(ax, ax_, 218, aw, 32, "Attention rollout",
          [r"12 layers to one map $\mathbf{M}_{\theta}(x)$"], GREEN)
    block(ax, rx0, 218, rw, 32, r"Explanation loss  $\mathcal{L}_{\mathrm{exp}}$",
          [r"MSE of $\mathbf{M}_{\theta}(x)$ and $\mathbf{M}_{\theta_o}(x)$"], BLUE)
    head(ax, (ux + uw, 234), (ax_, 234))
    head(ax, (ax_ + aw, 234), (rx0, 234))
    head(ax, (rc, 272), (rc, 250))
    block(ax, ux, 166, uw, 32, r"Task loss  $\mathcal{L}_{\mathrm{task}}$",
          [r"weighted BCE on logits $\hat{y}$"], GREEN)
    block(ax, ax_, 166, aw, 32, "EPU objective",
          [r"$\mathcal{L}_{\mathrm{task}}+\lambda\,\mathcal{L}_{\mathrm{exp}}$;  "
           r"$D_u$ is never read"], GREEN)
    head(ax, (uc, 218), (uc, 198))
    head(ax, (ux + uw, 182), (ax_, 182))
    route(ax, [(rc, 218), (rc, 182), (ax_ + aw, 182)])
    txt(ax, rc + 5, 199, r"$\times\lambda$", ha="left", fs=7.0)
    # backpropagation into the trainable model only
    bx_ = 300.0
    dashed(ax, [bx_, bx_, 46, 46], [166, 157, 157, 234])
    head(ax, (46, 234), (ux, 234))
    lab = txt(ax, 150, 157, r"backpropagation (updates $\theta$ only)", fs=6.2)
    lab.set_bbox(dict(fc=CONT[0], ec="none", pad=1.2))

    # ---------------------------------------------------------------- models compared
    mw, mg, my, mh = 208.0, 22.0, 104.0, 34.0
    mx = [24 + i * (mw + mg) for i in range(3)]
    mc = [x + mw / 2 for x in mx]
    txt(ax, 440, 157.5, "Models compared", fs=FS_H, ha="left")
    block(ax, mx[0], my, mw, mh, r"Oracle  $\theta^{*}$", ["retrained without client $u$"],
          ORANGE)
    block(ax, mx[1], my, mw, mh, r"Plain  $\theta_u$", [r"fine-tuning with $\lambda$ = 0"],
          ORANGE)
    block(ax, mx[2], my, mw, mh, r"EPU  $\theta_u$  (proposed)",
          [r"fine-tuning with $\lambda$ > 0"], ORANGE, lw=2.0)
    head(ax, (mc[1], 166), (mc[1], my + mh))
    route(ax, [(mc[1], 149), (mc[2], 149), (mc[2], my + mh)])
    ox = 34.0
    route(ax, [(cx0, 338), (ox, 338), (ox, my + mh)])
    txt(ax, (ox + cx0) / 2, 344.5, "repeated without $u$", fs=6.4, style="italic")

    # ---------------------------------------------------------------- evaluation
    ew, eg, ey, eh = 155.0, 16.0, 24.0, 46.0
    ex = [24 + i * (ew + eg) for i in range(4)]
    ec = [x + ew / 2 for x in ex]
    bus = 95.0
    for c in mc:
        seg(ax, [c, c], [my, bus])
    seg(ax, [ec[0], ec[3]], [bus, bus])
    for c in ec:
        head(ax, (c, bus), (c, ey + eh))
    txt(ax, (ec[1] + ec[2]) / 2, 83.5, "Evaluation, held-out test split", fs=FS_H)
    for i, (t, ls2) in enumerate([
            ("Utility", ["AUROC, mAP,", "F1, precision, recall"]),
            ("Forgetting", ["MIA AUC, forget loss,", "distance to oracle"]),
            ("Explanation", ["cosine, Pearson, IoU, MSE", r"to $\theta_o$ and to $\theta^{*}$"]),
            ("Efficiency", ["wall-clock time,", "speed-up vs. oracle"])]):
        block(ax, ex[i], ey, ew, eh, t, ls2, PURPLE, step=12.4)

    # ---------------------------------------------------------------- run conditions
    block(ax, 12, -15, 338, 24, "Three seeds: split, partition and training repeated", (), GREY)
    block(ax, 366, -15, 338, 24, "One NVIDIA Tesla T4; PyTorch and timm", (), GREY)

    fig.savefig(OUT / "fig_overview.png")
    plt.close(fig)
    print("wrote fig_overview")


if __name__ == "__main__":
    main()
