"""Figures required by the BCSE497J Project-I report template.

Fig. 1  Gantt chart              (section 2.5 Project Plan)
Fig. 3  Data flow diagram        (section 4.2.1, mandatory)
Fig. 4  Use case diagram         (section 4.2.2, mandatory)
Fig. 5  Class diagram            (section 4.2.3)
Fig. 6  Sequence diagram         (section 4.2.4)

Fig. 2 is the system architecture, which already exists as architecture.png.
Everything is drawn with matplotlib so the report can be rebuilt from source
with no external diagramming tool.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

OUT = Path(__file__).resolve().parent / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.dpi": 220, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

INK, BLUE, GREY = "#1a1b20", "#2563eb", "#6b7280"
FILL, FILL2, ACC = "#eef2ff", "#f3f4f6", "#dbeafe"


def box(ax, x, y, w, h, text, fill=FILL, edge=BLUE, fs=8.5, bold=False,
        radius=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                linewidth=1.1, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(ax, p, q, text="", style="-|>", color=GREY, fs=7.5, dy=0.07,
          ls="-", rad=0.0, boxed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.0, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        # an opaque label box keeps the caption legible where it crosses a
        # neighbouring arrow or box edge
        bb = dict(facecolor="white", edgecolor="none", pad=0.8) if boxed else None
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center",
                va="bottom", fontsize=fs, color=INK, bbox=bb, zorder=5)


def canvas(w, h, xlim, ylim, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=INK, pad=8)
    return fig, ax


# ====================================================== Fig. 1 — Gantt chart
TASKS = [
    ("Literature review and research gap analysis",      0.0, 2.0, "Planning"),
    ("Dataset acquisition and patient-disjoint split",   1.5, 1.5, "Data"),
    ("Non-IID Dirichlet partitioning across clients",    2.5, 1.0, "Data"),
    ("DeiT-Small adaptation to single-channel CXR",      3.0, 1.5, "Model"),
    ("FedAvg client/server implementation",              3.5, 2.0, "Model"),
    ("Differentiable attention-rollout module",          5.0, 1.5, "Method"),
    ("EPU objective  L = L_task + λ·L_exp",              6.0, 1.5, "Method"),
    ("Oracle retrain and plain unlearning baselines",    6.5, 1.5, "Method"),
    ("Membership-inference forgetting verifier",         7.5, 1.0, "Eval"),
    ("Multi-seed experiment campaigns (42/43/44)",       8.0, 2.0, "Eval"),
    ("λ-sweep ablation and figure generation",           9.5, 1.0, "Eval"),
    ("Streamlit demonstrator and reproducibility pack", 10.0, 1.5, "Delivery"),
    ("Report, review deck and paper draft",             11.0, 1.5, "Delivery"),
]
PHASE_C = {"Planning": "#94a3b8", "Data": "#38bdf8", "Model": BLUE,
           "Method": "#7c3aed", "Eval": "#0f766e", "Delivery": "#b45309"}
MONTHS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb",
          "Mar", "Apr", "May", "Jun", "Jul"]

fig, ax = plt.subplots(figsize=(11.0, 3.7))
for i, (name, start, dur, phase) in enumerate(TASKS):
    y = len(TASKS) - i - 1
    ax.barh(y, dur, left=start, height=0.58, color=PHASE_C[phase],
            edgecolor="white", linewidth=0.8, zorder=3)
ax.set_yticks(range(len(TASKS)))
ax.set_yticklabels([t[0] for t in reversed(TASKS)], fontsize=8.5)
ax.set_xticks(range(len(MONTHS)))
ax.set_xticklabels(MONTHS, fontsize=8.5)
ax.set_xlim(0, 12.5)
ax.set_xlabel("Project timeline (months)", fontsize=9.5)
ax.grid(axis="x", alpha=0.3, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_C.values()]
ax.legend(handles, PHASE_C.keys(), ncol=6, fontsize=8, frameon=False,
          loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.savefig(OUT / "fig1_gantt.png")
plt.close(fig)

# ================================================ Fig. 3 — Data flow diagram
fig, ax = canvas(11.0, 4.7, (0, 128), (0, 60))

box(ax, 1, 45, 21, 8, "NIH ChestX-ray14\n(external data store)", FILL2, GREY, bold=True)
box(ax, 1, 29, 21, 8, "Hospital client\n(data owner / actor)", FILL2, GREY, bold=True)
box(ax, 1, 11, 21, 8, "Researcher\n(external entity)", FILL2, GREY, bold=True)

box(ax, 42, 45, 24, 8, "1.0  Partition\npatient-disjoint · non-IID", ACC)
box(ax, 42, 32, 24, 8, "2.0  Federated training\nlocal epochs + FedAvg")
box(ax, 42, 19, 24, 8, "3.0  Unlearning\noracle / plain / EPU")
box(ax, 42, 6, 24, 8, "4.0  Evaluation\n4 axes · 3 seeds")

box(ax, 86, 45, 20, 8, "D1  Client shards", FILL2, GREY)
box(ax, 86, 32, 20, 8, "D2  Global model\ncheckpoints", FILL2, GREY)
box(ax, 86, 19, 20, 8, "D3  Unlearned\ncheckpoints", FILL2, GREY)
box(ax, 86, 6, 20, 8, "D4  Metrics, figures,\nmanifest", FILL2, GREY)

box(ax, 111, 30, 16, 10, "Report, deck\nand live\ndemonstrator", ACC, BLUE, bold=True)

# external entity  ->  process
arrow(ax, (22, 49), (42, 49), "images + labels", boxed=True)
arrow(ax, (22, 33), (42, 36), "forget request", dy=0.9, boxed=True)
arrow(ax, (22, 15), (42, 10), "configuration", dy=-1.3, boxed=True)

# process  ->  data store, and the return leg one band lower
arrow(ax, (66, 49), (86, 49), "shards", boxed=True)
arrow(ax, (86, 46.5), (66, 38.5), "shards read", dy=-1.6, boxed=True)
arrow(ax, (66, 37), (86, 37), "global weights", boxed=True)
arrow(ax, (86, 33.5), (66, 25.5), "warm start", dy=-1.6, boxed=True)
arrow(ax, (66, 23), (86, 23), "unlearned weights", boxed=True)
arrow(ax, (86, 20.5), (66, 12.5), "every model", dy=-1.6, boxed=True)
arrow(ax, (66, 10), (86, 10), "scores and timings", boxed=True)

arrow(ax, (106, 10), (114, 30), "", rad=-0.25)
arrow(ax, (119, 30), (119, 23), "")
ax.text(119, 21, "Reviewer", ha="center", fontsize=8.5, color=INK)

ax.text(64, 57.5, "Level-1 Data Flow Diagram", ha="center", fontsize=10.5,
        fontweight="bold", color=INK)
ax.text(64, 1.5, "Rounded boxes = processes · square boxes = external entities "
                 "and data stores · arrows = data in motion",
        ha="center", fontsize=7.5, color=GREY)
fig.savefig(OUT / "fig3_dfd.png")
plt.close(fig)

# ================================================= Fig. 4 — Use case diagram
fig, ax = canvas(9.0, 5.8, (0, 100), (0, 62))

ax.add_patch(Rectangle((26, 4), 48, 54, fill=False, edgecolor=GREY,
                       linewidth=1.2))
ax.text(50, 55.5, "Federated Unlearning System", ha="center", fontsize=9.5,
        fontweight="bold", color=INK)

USE_CASES = [
    (50, 50, "Partition data\nacross clients"),
    (50, 43, "Train global model\n(FedAvg)"),
    (50, 36, "Submit forget request"),
    (50, 29, "Perform unlearning\n(plain / EPU)"),
    (50, 22, "Retrain oracle\n(reference)"),
    (50, 15, "Verify forgetting\n(membership inference)"),
    (50, 8.5, "Measure explanation\nfidelity"),
]
for x, y, label in USE_CASES:
    ax.add_patch(Ellipse((x, y), 30, 5.6, facecolor=FILL, edgecolor=BLUE,
                         linewidth=1.1))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color=INK,
            linespacing=1.35)


def actor(ax, x, y, name):
    ax.plot([x], [y + 2.0], marker="o", ms=6, mfc="white", mec=INK, mew=1.2)
    ax.plot([x, x], [y + 1.7, y - 0.6], color=INK, lw=1.2)
    ax.plot([x - 1.8, x + 1.8], [y + 1.0, y + 1.0], color=INK, lw=1.2)
    ax.plot([x, x - 1.5], [y - 0.6, y - 2.4], color=INK, lw=1.2)
    ax.plot([x, x + 1.5], [y - 0.6, y - 2.4], color=INK, lw=1.2)
    # clear of the stick figure's legs, which end at y - 2.4
    ax.text(x, y - 5.4, name, ha="center", va="top", fontsize=8.5, color=INK,
            fontweight="bold", linespacing=1.35)


actor(ax, 10, 44, "Hospital\nclient")
actor(ax, 10, 20, "Aggregation\nserver")
actor(ax, 90, 32, "Researcher /\nauditor")

for uy in (50, 43, 36):
    arrow(ax, (14, 44), (35, uy), color=GREY)
for uy in (43, 29, 22):
    arrow(ax, (14, 20), (35, uy), color=GREY)
for uy in (15, 8.5, 22):
    arrow(ax, (86, 32), (65, uy), color=GREY)

ax.text(50, 60, "Use Case Diagram", ha="center", fontsize=10.5,
        fontweight="bold", color=INK)
fig.savefig(OUT / "fig4_usecase.png")
plt.close(fig)

# ===================================================== Fig. 5 — Class diagram
fig, ax = canvas(10.6, 6.4, (0, 132), (0, 102))


def cls(ax, x, y, w, name, attrs, ops, h_head=5.0, line=3.1):
    h = h_head + line * (len(attrs) + len(ops)) + 1.6
    ax.add_patch(Rectangle((x, y - h), w, h, facecolor="white",
                           edgecolor=BLUE, linewidth=1.1))
    ax.add_patch(Rectangle((x, y - h_head), w, h_head, facecolor=ACC,
                           edgecolor=BLUE, linewidth=1.1))
    ax.text(x + w / 2, y - h_head / 2, name, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=INK)
    yy = y - h_head - 2.0
    for a in attrs:
        ax.text(x + 1.2, yy, a, fontsize=7.2, color=INK, va="center")
        yy -= line
    ax.plot([x, x + w], [yy + line / 2, yy + line / 2], color=BLUE, lw=0.9)
    yy -= 0.6
    for o in ops:
        ax.text(x + 1.2, yy, o, fontsize=7.2, color=INK, va="center")
        yy -= line
    return (x, y, w, h)


C1X, C2X, C3X, CW3 = 2, 40, 76, 36

cls(ax, C1X, 96, 30, "NIHChestXrayDataset",
    ["- image_dir : Path", "- labels : DataFrame", "- image_size : int"],
    ["+ __getitem__(i)", "+ patient_disjoint_split()"])          # bottom 73.9
cls(ax, C1X, 68, 30, "DirichletPartitioner",
    ["- num_clients : int = 5", "- alpha : float = 0.5"],
    ["+ partition(dataset)"])                                    # bottom 52.1
cls(ax, C1X, 42, 30, "ViTClassifier",
    ["- backbone : DeiT-Small", "- num_labels : int = 14"],
    ["+ forward(x)", "+ forward_with_maps(x)"])                  # bottom 23.0

cls(ax, C2X, 96, 30, "FederatedClient",
    ["- client_id : int", "- loader : DataLoader"],
    ["+ local_train(weights)", "+ num_samples()"])
cls(ax, C2X, 68, 30, "FedAvgServer",
    ["- clients : List[Client]", "- rounds : int = 30"],
    ["+ aggregate(updates)", "+ run()"])
cls(ax, C2X, 42, 30, "AttentionRollout",
    ["- discard_ratio : float"],
    ["+ rollout(attn_maps)", "+ cls_map(grid)"])

cls(ax, C3X, 96, CW3, "UnlearningMethod  «abstract»",
    ["# retained : List[Client]", "# seed : int"],
    ["+ run() : state_dict"])                                    # bottom 80.1
cls(ax, C3X, 74, CW3, "OracleRetrain / PlainFineTune",
    ["- lr : float", "- rounds : int"],
    ["+ run()"])                                                 # bottom 58.1
cls(ax, C3X, 52, CW3, "EPUFineTune",
    ["- lam : float", "- reference : ViTClassifier"],
    ["+ explanation_loss()", "+ run()"])                         # bottom 33.0
cls(ax, C3X, 26, CW3, "EvaluationHarness",
    ["- seeds : List[int] = [42, 43, 44]"],
    ["+ score_all()"], h_head=4.4)                               # bottom 13.8

arrow(ax, (32, 86), (40, 86), "loads", color=GREY, fs=7)
arrow(ax, (17, 73.9), (17, 68), "", color=GREY)
arrow(ax, (32, 60), (40, 60), "shards", color=GREY, fs=7)
arrow(ax, (32, 33), (40, 33), "explains", color=GREY, fs=7)
arrow(ax, (70, 60), (76, 86), "invokes", color=GREY, fs=7, rad=0.15)
arrow(ax, (94, 80.1), (94, 74), "", style="-|>", color=BLUE)
arrow(ax, (94, 58.1), (94, 52), "", style="-|>", color=BLUE)
arrow(ax, (70, 30), (76, 42), "L_exp", color=GREY, fs=7)
arrow(ax, (94, 33), (94, 26), "scores", color=GREY, fs=7)
ax.text(115, 88, "△  inheritance", fontsize=7.5, color=BLUE)
ax.text(115, 84, "→  association", fontsize=7.5, color=GREY)

ax.text(66, 100, "Class Diagram — principal classes and their relationships",
        ha="center", fontsize=10.5, fontweight="bold", color=INK)
fig.savefig(OUT / "fig5_class.png")
plt.close(fig)

# ================================================== Fig. 6 — Sequence diagram
fig, ax = canvas(11.0, 4.6, (0, 100), (0, 60))

LIFELINES = [(12, "Hospital\nclient r"), (34, "Retained\nclients"),
             (56, "Aggregation\nserver"), (78, "Evaluation\nharness")]
for x, name in LIFELINES:
    box(ax, x - 9, 52, 18, 6, name, ACC, BLUE, fs=8.5, bold=True)
    ax.plot([x, x], [52, 4], color=GREY, lw=0.9, ls=(0, (4, 3)))

STEPS = [
    (56, 34, 48, "1: broadcast global weights"),
    (34, 56, 43, "2: return local updates"),
    (12, 56, 38, "3: return local updates (pre-removal)"),
    (56, 56, 33, "4: FedAvg aggregate → Original model"),
    (12, 56, 28, "5: forget request (right to be forgotten)"),
    (56, 34, 23, "6: warm-start fine-tune on retained clients only"),
    (34, 56, 18, "7: EPU updates, L = L_task + λ·L_exp"),
    (56, 78, 13, "8: submit Original, Plain, EPU, Oracle"),
    (78, 56, 8, "9: AUROC · MIA AUC · attention fidelity · wall-time"),
]
for src, dst, y, label in STEPS:
    if src == dst:
        ax.add_patch(FancyArrowPatch((src, y + 1.6), (src, y - 1.0),
                                     arrowstyle="-|>", mutation_scale=10,
                                     linewidth=1.0, color=BLUE,
                                     connectionstyle="arc3,rad=2.6"))
        ax.text(src + 7, y + 0.2, label, fontsize=7.5, color=INK, va="center")
    else:
        arrow(ax, (src, y), (dst, y), color=BLUE)
        ax.text((src + dst) / 2, y + 0.8, label, ha="center", fontsize=7.5,
                color=INK)

ax.text(50, 58.5, "Sequence Diagram — one federated round, then the "
                  "unlearning request", ha="center", fontsize=10.5,
        fontweight="bold", color=INK)
ax.text(50, 1.5, "Steps 1–4 repeat for 30 rounds before the forget request "
                 "arrives; steps 6–7 replace the full retrain the oracle performs.",
        ha="center", fontsize=7.5, color=GREY)
fig.savefig(OUT / "fig6_sequence.png")
plt.close(fig)

print("report figures written to", OUT)
for f in sorted(OUT.glob("*.png")):
    print("  ", f.name)
