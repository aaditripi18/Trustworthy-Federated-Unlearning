"""Qualitative attention-rollout figure for the paper (Fig. 2).

Runs the real seed-42 checkpoints on CPU and renders, for held-out TEST-split
radiographs, the rollout map of Original / Plain / EPU(lambda=100) / Oracle.
Images are chosen by a fixed rule (the first N patients, in filename order, with a
test-split image that carries at least one finding; one image per patient) --
not hand-picked.

Run with the project venv (needs torch + timm):
    .venv/Scripts/python.exe paper/make_qualitative_fig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.nih import NIH_LABELS, load_nih_manifest  # noqa: E402
from src.data.synthetic import split_by_patient  # noqa: E402
from src.explain.metrics import cosine_similarity, iou  # noqa: E402
from src.explain.torch_rollout import forward_with_maps  # noqa: E402
from src.models.vit import build_model  # noqa: E402

SEED = 42
N_IMAGES = 3
EXP = ROOT / "ec2_final_run/tfu/experiments"
SAMPLE = ROOT / "ec2_final_run/demo_sample"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "qualitative_maps.npz"

CKPTS = {
    "Original": EXP / f"full_experiment/seed_{SEED}/original.pt",
    "Plain": EXP / f"exp_a1/seed_{SEED}/plain.pt",
    "EPU ($\\lambda$=100)": EXP / f"exp_a1/seed_{SEED}/epu_lambda100.pt",
    "Oracle": EXP / f"full_experiment/seed_{SEED}/oracle.pt",
}
_MEAN, _STD = 0.5, 0.25   # identical to src/data/transforms.py


def pick_images():
    m = load_nih_manifest(SAMPLE / "Data_Entry_2017_v2020.csv")
    m = split_by_patient(m, 0.15, 0.15, SEED)
    have = sorted(p.name for p in (SAMPLE / "images").glob("*.png"))
    m = m[m.image_id.isin(have) & (m.split == "test")].sort_values("image_id")
    label_cols = [f"label_{i}" for i in range(len(NIH_LABELS))]
    m = m[m[label_cols].sum(axis=1) > 0]
    picks = []
    m = m.drop_duplicates("patient_id")          # one radiograph per patient
    for _, r in m.head(N_IMAGES).iterrows():
        names = [NIH_LABELS[i].replace("_", " ") for i in range(len(NIH_LABELS))
                 if r[f"label_{i}"] == 1]
        picks.append((r.image_id, names))
    return picks


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(dict(cfg["model"], pretrained=False),
                        cfg["data"]["image_size"], cfg["data"]["num_labels"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def compute():
    picks = pick_images()
    disp, xs = [], []
    for image_id, _ in picks:
        im = Image.open(SAMPLE / "images" / image_id).convert("L").resize((224, 224), Image.BILINEAR)
        d = np.asarray(im, dtype=np.uint8)
        disp.append(d)
        xs.append((d.astype(np.float32) / 255.0 - _MEAN) / _STD)
    x = torch.from_numpy(np.stack(xs)[:, None]).float()
    maps = {}
    for name, path in CKPTS.items():
        model = load_model(path)
        gs = model.patch_embed.grid_size
        with torch.no_grad():
            _, mp = forward_with_maps(model, x, (int(gs[0]), int(gs[1])))
        maps[name] = mp.numpy()
        print("computed", name)
    np.savez(CACHE, disp=np.stack(disp), ids=np.array([p[0] for p in picks]),
             labels=np.array(["|".join(p[1]) for p in picks]),
             **{f"map{i}": maps[k] for i, k in enumerate(CKPTS)})


def render():
    z = np.load(CACHE, allow_pickle=False)
    disp, ids, labels = z["disp"], z["ids"], z["labels"]
    names = list(CKPTS)
    maps = [z[f"map{i}"] for i in range(len(names))]
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 7})
    n = len(disp)
    fig, axes = plt.subplots(n, 5, figsize=(3.5, 0.92 * n + 0.2))
    for r in range(n):
        axes[r, 0].imshow(disp[r], cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(labels[r].replace("|", "\n"), fontsize=5.5, rotation=90, labelpad=2)
        for c, (name, mp) in enumerate(zip(names, maps), start=1):
            up = np.asarray(Image.fromarray((mp[r] * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
            axes[r, c].imshow(disp[r], cmap="gray", vmin=0, vmax=255)
            axes[r, c].imshow(up, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            if name != "Original":
                cs = cosine_similarity(mp[r], maps[0][r])
                io = iou(mp[r], maps[0][r], 0.5)
                axes[r, c].set_xlabel(f"cos {cs:.4f}\nIoU {io:.2f}", fontsize=5.5, labelpad=1.5)
                print(ids[r], name, f"cos={cs:.4f} iou={io:.3f}")
        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_linewidth(0.4)
    for c, t in enumerate(["Input"] + names):
        axes[0, c].set_title(t, fontsize=6.5, pad=2.5)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.12, wspace=0.05, hspace=0.36)
    fig.savefig(OUT / "fig_qualitative.png", dpi=600)
    plt.close(fig)
    print("wrote", OUT / "fig_qualitative.png", "| images:", list(ids), list(labels))


if __name__ == "__main__":
    if not CACHE.exists() or "--recompute" in sys.argv:
        compute()
    render()
