# Live Demo — Attention-Fidelity-Preserving Federated Unlearning with Vision Transformers for Chest X-Rays

A CPU-friendly Streamlit dashboard for the project review. It tells the whole
story using the **real trained checkpoints** — no live training, nothing faked:

1. **Diagnosis** — the federated (Original) DeiT-Small model predicts 14 chest
   X-ray findings and shows *where it looks* via attention rollout.
2. **Unlearning (before → after)** — Original vs Plain vs EPU vs Oracle on the
   same image, plus the measured metrics proving the forgotten client is no
   longer identifiable (membership-inference ≈ 0.5) while explanations stay
   stable (attention preserved).
3. **Results** — the measured comparison table and result figures.

## One-time setup

From the repository root, in your project virtual environment (the one you
trained with, so `torch` + `timm` are already present):

```bash
pip install -r demo/requirements-demo.txt
```

## Run it

```bash
streamlit run demo/streamlit_app.py
```

Your browser opens at `http://localhost:8501`. Everything runs on **CPU** —
loading a checkpoint and scoring one image takes a couple of seconds.

## What it expects on disk (defaults, all editable in the sidebar)

- **Checkpoints:** `ec2_final_run/tfu/experiments/full_experiment/seed_42/…`
  (contains `original.pt`, `plain.pt`, `epu_lambda1.0.pt`, `epu_lambda0.5.pt`,
  `oracle.pt`). Switch seeds in the sidebar.
- **Sample X-rays:** `ec2_final_run/demo_sample/images/` (the 80 images kept
  aside before deleting the full dataset). You can also upload any chest X-ray.
- **Label CSV (optional):** `ec2_final_run/demo_sample/Data_Entry_2017_v2020.csv`
  — used only to show the ground-truth findings for a sample image.
- **Figures:** `…/full_experiment/figures/*.png`.

If your folders differ, just edit the three paths at the top of the sidebar.

## Suggested 10–15 minute presentation flow

1. **(1 min) Framing.** Hospitals train one model together without sharing data
   (federated learning); a hospital later exercises its *right to be forgotten*,
   so we must remove its influence — *federated unlearning* — without a full,
   expensive retrain, and without breaking the model's clinical explanations.
2. **(3 min) Tab 1 — Diagnosis.** Pick an X-ray. Show the predicted findings and
   the attention heatmap ("this is the evidence the model uses"). Optionally
   compare to the ground-truth label.
3. **(5 min) Tab 2 — Unlearning.** Choose *Plain* then *EPU*. Show the two
   attention maps side by side (they stay aligned → explanation preserved), then
   the metrics table: **MIA ≈ 0.50** (the client is forgotten), **explanation
   cosine ≈ 0.96** (reasoning preserved), **speedup > 1** (cheaper than the
   Oracle's full retrain).
4. **(3 min) Tab 3 — Results.** Walk the figures and the one-line finding.
5. **(1 min) Honesty + scope.** State clearly: this is a research prototype
   (TRL 4), validated in simulation on the NIH benchmark; the EPU explanation
   gain over Plain is real but small (within noise), and full clinical
   deployment is future work.

## Notes

- **No live training in the demo.** A real unlearning run takes ~40+ minutes on
  a GPU and cannot complete in 15 minutes on a laptop. Instead the demo loads
  the *already-unlearned* real checkpoints and shows the genuine before/after —
  which is both faster and more reliable for a live audience.
- The app reuses your project code (`src/models/vit.py`, `src/explain/
  torch_rollout.py`, `src/data/nih.py`) so preprocessing and attention rollout
  are identical to the experiments.
