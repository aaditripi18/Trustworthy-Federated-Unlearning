# Live Demo — Attention-Fidelity-Preserving Federated Unlearning with Vision Transformers for Chest X-Rays

A CPU-friendly Streamlit dashboard for the project review. It tells the whole
story using the **real trained checkpoints** — no live training, nothing faked:

1. **Diagnosis** — the federated (Original) DeiT-Small model predicts 14 chest
   X-ray findings and shows *where it looks* via attention rollout.
2. **Unlearning (before → after)** — Original vs Plain vs EPU vs Oracle on the
   same image, with the measured metrics: attention maps stay closer to the
   Original with EPU, and a calibrated membership audit shows the trace of the
   forgotten client reduced but not removed.
3. **Results** — setting A and setting B, the membership audit, the
   gradient-ascent baseline and the verdict, read from the result files in
   `ec2_final_run/` and `paper/`.

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
   the metrics table. Note that the loss-threshold MIA is ≈ 0.50 for every
   model, the Original included, so it does not by itself show forgetting; the
   calibrated audit on page 02 does detect a trace, which unlearning lowers.
4. **(3 min) Tab 3 — Results.** Walk setting A, the λ-sweep, the membership
   audit, the gradient-ascent baseline and the verdict.
5. **(1 min) Honesty + scope.** State clearly: this is a research prototype
   (TRL 4), validated in simulation on the NIH benchmark with three seeds. EPU
   keeps attention maps closer to the pre-unlearning model in every seed, does
   not bring them closer to the retrain oracle, and after gradient ascent keeps
   slightly more of the membership trace. Clinical deployment is future work.

## Notes

- **No live training in the demo.** A real unlearning run takes ~40+ minutes on
  a GPU and cannot complete in 15 minutes on a laptop. Instead the demo loads
  the *already-unlearned* real checkpoints and shows the genuine before/after —
  which is both faster and more reliable for a live audience.
- The app reuses your project code (`src/models/vit.py`, `src/explain/
  torch_rollout.py`, `src/data/nih.py`) so preprocessing and attention rollout
  are identical to the experiments.
