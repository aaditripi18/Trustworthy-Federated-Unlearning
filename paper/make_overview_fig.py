"""Fig. 1 of the paper: block diagram of the study in three panels.

  (a) Original model    data, patient-level split, non-IID partition, FedAvg training
  (b) Unlearning        the retrained Oracle, and the EPU local update at a remaining client
  (c) Evaluation        models compared and the four groups of metrics

    python paper/make_overview_fig.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "savefig.dpi": 600,
})

INK = "#000000"
PANEL = "#9a9a9a"                    # thin outline of the three panels
BLUE, BLUE_FILL = "#1f4e79", "#dce6f1"   # the explanation path of EPU
GREY_FILL = "#e6e6e6"                # frozen model, forgotten client
W, H = 2000.0, 1300.0                # drawing units; the figure is 7.16 in wide
FS_P, FS_T, FS_S = 7.4, 6.7, 6.3     # panel title, block title, block text
STEP = 26.5                          # line spacing inside a block


def txt(ax, x, y, s, fs=FS_S, ha="center", va="center", **kw):
    kw.setdefault("color", INK)
    return ax.text(x, y, s, fontsize=fs, ha=ha, va=va, zorder=6, **kw)


def rows(ax, x, y_top, items):
    """Stack of text lines; items are (string, bold)."""
    for i, (s, bold) in enumerate(items):
        txt(ax, x, y_top + i * STEP, s, fs=FS_T if bold else FS_S,
            fontweight="bold" if bold else "normal")


def box(ax, x0, y0, x1, y1, title=None, lines=(), fc="#ffffff", ec=INK, lw=0.7, cy=None):
    """Rectangle with corners (x0, y0)-(x1, y1); a bold title and plain lines, centred."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, zorder=3))
    items = ([(title, True)] if title else []) + [(s, False) for s in lines]
    cy = (y0 + y1) / 2 if cy is None else cy
    rows(ax, (x0 + x1) / 2, cy - STEP * (len(items) - 1) / 2, items)


def arrow(ax, pts, dashed=False, lw=0.7):
    """Polyline through pts with an arrow head at the last point."""
    ls = (0, (3.2, 2.2)) if dashed else "-"
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=INK, lw=lw, ls=ls, zorder=4, solid_capstyle="butt",
                solid_joinstyle="miter")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=6.5,
                                 color=INK, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=5))


def main():
    fig = plt.figure(figsize=(7.16, 7.16 * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                 # y grows downwards, as on the page
    ax.axis("off")

    # ------------------------------------------------------------------ panels
    for x0, x1 in ((14, 391), (425, 1330), (1352, 1986)):
        ax.add_patch(Rectangle((x0, 10), x1 - x0, 1233, fc="none", ec=PANEL, lw=0.6, zorder=1))
    txt(ax, 30, 40, "(a) Original model", fs=FS_P, ha="left", fontweight="bold")
    txt(ax, 441, 40, "(b) Unlearning after the erasure request of client $u$", fs=FS_P,
        ha="left", fontweight="bold")
    txt(ax, 1368, 40, "(c) Evaluation", fs=FS_P, ha="left", fontweight="bold")
    txt(ax, 1670, 68, "on the held-out test split", fs=FS_S, style="italic")

    # ------------------------------------------------------------------ (a)
    box(ax, 33, 92, 372, 204, "NIH ChestX-ray14", ["112,120 radiographs", "14 findings"])
    box(ax, 33, 254, 372, 366, "Patient-level split", ["70 / 15 / 15", "patient-disjoint"])
    box(ax, 33, 416, 372, 606, "Non-IID partition", [r"Dirichlet $\alpha$ = 0.5, $K$ = 5"],
        cy=509)
    for i, (a, b) in enumerate(((48, 83), (93, 156), (166, 214), (224, 304), (314, 357))):
        ax.add_patch(Rectangle((a, 548), b - a, 42, fc=GREY_FILL if i == 0 else "#ffffff",
                               ec=INK, lw=0.6, zorder=4))
    txt(ax, 65.5, 568, "$u$")
    box(ax, 33, 656, 372, 869, "FedAvg training",
        ["DeiT-Small, ImageNet init.", "2 local epochs, weighted BCE", "warm-up + cosine LR",
         "early stopping on validation"])
    box(ax, 33, 947, 372, 1059, r"Original model $\theta_o$", ["best validation round"], lw=1.3)
    for y0, y1 in ((204, 254), (366, 416), (606, 656), (869, 947)):
        arrow(ax, [(202, y0), (202, y1)])
    arrow(ax, [(372, 511), (408, 511), (408, 179), (458, 179)])      # partition -> Oracle
    arrow(ax, [(372, 1003), (458, 1003)], lw=1.0)                    # theta_o -> unlearning
    ax.add_patch(Rectangle((398, 965), 30, 32, fc="#ffffff", ec="none", zorder=2))
    txt(ax, 413, 982, r"$\theta_o$", fs=FS_T)

    # ------------------------------------------------------------------ (b)
    box(ax, 458, 114, 1297, 243, r"Oracle $\theta^*$",
        ["retrained from the pretrained weights without client $u$,",
         r"same FedAvg procedure (no use of $\theta_o$)"])
    ax.add_patch(Rectangle((458, 316), 839, 889, fc="none", ec=BLUE, lw=1.2, zorder=2))
    txt(ax, 481, 353, r"EPU: local update at each remaining client $k \neq u$", fs=FS_T,
        ha="left", fontweight="bold", color=BLUE)

    box(ax, 480, 393, 587, 729, None, ["Batch", "$(x, y)$", "from", "$D_k$,", r"$k \neq u$"])
    box(ax, 631, 393, 855, 511, r"Frozen $\theta_o$", ["(no gradient)"], fc=GREY_FILL)
    box(ax, 893, 393, 1073, 511, None, ["Rollout map", r"$M_{\theta_o}(x)$"],
        fc=BLUE_FILL, ec=BLUE)
    box(ax, 631, 612, 855, 729, r"Trainable $\theta$", [r"(initialised at $\theta_o$)"])
    box(ax, 893, 612, 1073, 729, None, ["Rollout map", r"$M_{\theta}(x)$"],
        fc=BLUE_FILL, ec=BLUE)
    ax.add_patch(Rectangle((1106, 393), 168, 336, fc=BLUE_FILL, ec=BLUE, lw=0.7, zorder=3))
    rows(ax, 1190, 508, [("Explanation", True), ("loss", True),
                         (r"$\mathcal{L}_{exp}$ =", False), ("MSE of the", False),
                         ("two maps", False)])
    box(ax, 631, 819, 855, 936, "Task loss", ["weighted BCE", "on logits"])
    box(ax, 893, 819, 1274, 936, "EPU objective",
        [r"$\mathcal{L}_{EPU} = \mathcal{L}_{task} + \lambda\,\mathcal{L}_{exp}$",
         "$D_u$ is never read"])
    box(ax, 631, 1059, 1274, 1154, r"Server: FedAvg, $T_u$ rounds $\rightarrow$ $\theta_u$",
        [r"over the remaining clients; Plain uses $\lambda$ = 0"])

    arrow(ax, [(587, 452), (631, 452)])               # batch -> frozen model
    arrow(ax, [(587, 670), (631, 670)])               # batch -> trainable model
    arrow(ax, [(855, 452), (893, 452)])
    arrow(ax, [(1073, 452), (1106, 452)])
    arrow(ax, [(855, 670), (893, 670)])
    arrow(ax, [(1073, 670), (1106, 670)])
    arrow(ax, [(743, 729), (743, 819)])               # logits -> task loss
    arrow(ax, [(855, 877), (893, 877)])               # task loss -> objective
    arrow(ax, [(1190, 729), (1190, 819)])             # explanation loss -> objective
    txt(ax, 1232, 772, r"$\times\lambda$", fs=FS_T)
    arrow(ax, [(950, 936), (950, 975), (606, 975), (606, 710), (631, 710)], dashed=True)
    txt(ax, 777, 997, r"backprop, $\theta$ only", style="italic")
    arrow(ax, [(1173, 936), (1173, 1059)])            # local model returned to the server
    txt(ax, 1097, 997, r"return $\theta_k$, $n_k$", style="italic")

    arrow(ax, [(1297, 179), (1380, 179)])             # Oracle -> evaluation
    arrow(ax, [(1274, 1106), (1380, 1106)])           # unlearned model -> evaluation
    for y in (152, 1083):
        ax.add_patch(Rectangle((1322, y - 16), 32, 30, fc="#ffffff", ec="none", zorder=2))
    txt(ax, 1338, 152, r"$\theta^*$", fs=FS_T)
    txt(ax, 1338, 1083, r"$\theta_u$", fs=FS_T)

    # ------------------------------------------------------------------ (c)
    box(ax, 1380, 126, 1581, 1154)
    rows(ax, 1480.5, 559, [("Models", True), ("compared", True), ("", False),
                           (r"Original $\theta_o$", False), (r"Oracle $\theta^*$", False),
                           (r"Plain ($\lambda$ = 0)", False), (r"EPU ($\lambda$ > 0)", False)])
    ax.add_patch(Rectangle((1637, 126), 333, 1028, fc="none", ec=PANEL, lw=0.6, zorder=1))
    arrow(ax, [(1581, 640), (1637, 640)])
    box(ax, 1654, 159, 1953, 377, "Utility", ["macro-AUROC, mAP,", "precision, recall, F1"])
    box(ax, 1654, 405, 1953, 623, "Forgetting",
        ["membership-inference", "AUC, forget loss,", "distance to oracle"])
    box(ax, 1654, 651, 1953, 869, "Explanation",
        ["cosine, Pearson,", r"MSE, IoU vs. $\theta_o$", r"(stability) and $\theta^*$",
         "(oracle fidelity)"])
    box(ax, 1654, 897, 1953, 1114, "Efficiency", ["wall-clock time,", "speed-up vs. oracle"])

    txt(ax, 1000, 1274, "Three seeds (42, 43, 44): data split, client partition and training are "
        "repeated for every seed.   Single NVIDIA Tesla T4; PyTorch and timm.", style="italic")

    fig.savefig(OUT / "fig_overview.png")
    plt.close(fig)
    print("wrote fig_overview")


if __name__ == "__main__":
    main()
