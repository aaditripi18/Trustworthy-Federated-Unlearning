# Trustworthy-Federated-Unlearning — Attention-Fidelity-Preserving Federated Unlearning (CXR / ViT)

Research implementation of explanation-fidelity-preserving federated unlearning
for trustworthy chest X-ray diagnosis with Vision Transformers.

Built incrementally under a **Minimum Working Research System** roadmap. Each
milestone leaves the repository runnable and testable.

## Structure
```
Trustworthy-Federated-Unlearning/
├── configs/
│   ├── default.yaml          # seed, logging, smoke params            (M0)
│   ├── data.yaml             # synthetic subset, splits, clients, skew (M1)
│   ├── model.yaml            # ViT + centralized-training config       (M2)
│   ├── federated.yaml        # FedAvg rounds/epochs/history            (M3)
│   └── unlearn.yaml          # forget client, method, oracle/finetune  (M4)
├── src/
│   ├── utils/                # seed, logging, config                   (M0)
│   ├── data/                 # synthetic, partition, transforms,       (M1)
│   │                         #   dataset, central loaders
│   ├── models/vit.py         # ViT/DeiT model factory                  (M2)
│   ├── eval/                 # metrics, engine, experiment framework   (M2/M7)
│   ├── federated/            # FedAvg strategy, client, server, history (M3)
│   ├── unlearning/           # oracle, methods (registry), verify, epu    (M4/M6)
│   └── explain/              # rollout, extract, metrics, torch_rollout   (M5/M6)
├── scripts/
│   ├── smoke_test.py         # M0 verifier
│   ├── prepare_data.py       # M1 data driver
│   ├── train_central.py      # M2 centralized training
│   ├── eval.py               # M2 checkpoint evaluation
│   ├── run_federated.py      # M3 federated (FedAvg) simulation
│   ├── verify_m3.py          # M3 artifact verification + history_summary.json
│   ├── run_unlearn.py        # M4 oracle + unlearning + forgetting report
│   ├── run_explain.py        # M5 explanation extraction + similarity report
│   ├── run_epu.py            # M6 explanation-preserving unlearning + 4-way report
│   ├── run_evaluation.py     # M7 multi-seed, multi-lambda evaluation + tables
│   └── run_final.py          # M8 full NIH run: pipeline + eval + tables + figures
├── docs/DATASET_NIH.md       # NIH download, layout, checklist, smoke test  (M8)
├── tests/                    # ...unlearn, explain, epu, evaluation, nih, m8_outputs
├── requirements.txt · pyproject.toml · .gitignore · README.md
├── data/                     # generated: manifest.csv, partition.json
└── experiments/              # generated: checkpoints, metrics
```

## Setup
```bash
pip install -r requirements.txt
```
Backend note: the core stack is PyTorch + timm (+ scikit-learn for metrics),
required from **M2** onward. Where practical the code degrades gracefully so the
scaffold/tests still run without the full stack — the M0 smoke test uses a NumPy
forward fallback, the M1 loaders use a NumPy loader when torch is absent, and the
M2 metrics use a NumPy fallback when scikit-learn is absent. Model training/eval
(M2) require torch + timm.

## Run
```bash
cd Trustworthy-Federated-Unlearning
python scripts/smoke_test.py       # M0 -> "M0 SMOKE TEST PASSED"
python scripts/prepare_data.py     # M1 -> "M1 DATA PIPELINE OK"
python scripts/train_central.py    # M2 -> "M2 CENTRAL BASELINE OK" (needs torch+timm)
python scripts/eval.py             # M2 -> "M2 EVAL OK"
python scripts/run_federated.py    # M3 -> "M3 FEDERATED OK" (needs torch+timm)
python scripts/run_unlearn.py      # M4 -> "M4 UNLEARN OK" (needs torch+timm; uses M3 model)
python scripts/run_explain.py      # M5 -> "M5 EXPLAIN OK" (needs torch+timm; uses M3/M4 models)
python scripts/run_epu.py          # M6 -> "M6 EPU OK" (needs torch+timm; 4-way comparison)
python scripts/run_evaluation.py   # M7 -> "M7 EVAL OK" (multi-seed x multi-lambda; writes tables)
pytest                             # all milestones
```

## Status
- **M0 — scaffold & environment:** complete.
- **M1 — data pipeline (non-IID, patient-disjoint):** complete.
- **M2 — centralized ViT baseline + multi-label metrics:** complete.
- **M3 — federated FedAvg simulation + per-round history:** complete.
- **M4 — federated unlearning + retrain oracle + forgetting verification:** complete.
- **M5 — explanation extraction (attention rollout) + explanation metrics:** complete.
- **M6 — explanation-preserving unlearning (EPU) + 4-way comparison:** complete.
- **M7 — reproducible evaluation framework (4 methods x 4 axes, multi-seed, lambda ablation):** complete.
- **M8 — real NIH ChestX-ray14 integration + final experiment driver:** code complete (Steps A/B/D);
  the real training run (Step C) is executed locally once NIH is downloaded (see `docs/DATASET_NIH.md`).

Design note (M8): the pipeline is dataset-agnostic; `configs/data_nih.yaml` +
`configs/model_nih.yaml` (DeiT-Small@224) select NIH via `data.source: nih`, while
the synthetic `configs/data.yaml` stays the default (all tests still pass). Only the
data layer changed (NIH loader wiring, partition scheme, `image_loader` propagation
incl. the forget-set loader); no algorithm/model/eval changes. `scripts/run_final.py`
reuses `src/eval/experiment.py` and writes `final_*` CSV/JSON, `reproducibility.json`,
and `experiments/figures/*.{png,pdf}`.

Design note (M7): evaluation is separate from training (`src/eval/experiment.py`).
Each seed retrains its own chain; results are aggregated (mean/std) and written to
`evaluation_results.csv`, `multi_seed_summary.csv`, `evaluation_tables.csv`,
`ablation_lambda.csv`, `evaluation_summary.json`. Findings are computed from the
data (which lambda, if any, is closest to the oracle; whether plain beats EPU) —
reported as measured, not tuned for positive results.

Design note (M6): **EPU** adds `L = L_task + lambda * L_exp` to the federated
fine-tune, where `L_exp` (differentiable attention rollout, `torch_rollout.py`)
keeps explanations close to the frozen original on remaining-client data. It's a
registry method (`method: epu`) toggled by config; the oracle is used only for
evaluation. `run_epu.py` reports Original -> Unlearned -> EPU -> Oracle across
diagnostic performance, forgetting (incl. MIA), explanation similarity, and the
preservation gain.

Design note (M4): the retrain **oracle** (from-scratch FedAvg excluding the
forgotten client) is the gold standard. Unlearning methods are a **registry**
(`src/unlearning/methods.py`) — the first is `finetune_retain` (warm-started
continued FedAvg on remaining clients); FedEraser update-replay / gradient-ascent
plug in without changing the pipeline. Forgetting is quantified vs original & oracle
via forget-set loss, membership-inference AUC, and prediction-distance-to-oracle.

Design note (M3): a native FedAvg loop is used (not the Flower runtime) so the
per-round client updates are captured directly in `federated_history.pkl`, which
the M4 update-replay unlearning needs. The `client/strategy/history` modules are
Flower-agnostic; a Flower server adapter can be added later without changing them.
