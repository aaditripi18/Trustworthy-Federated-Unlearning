# Project Handoff — for ChatGPT (or any AI collaborator)

> Paste this whole file to fully hand off the project. It is a factual status
> document. Nothing here is a finished experimental result — the full experiment
> has **not** yet been run to completion (we are waiting on AWS GPU quota).

---

## 1. Project identity

- **Title:** Attention-Fidelity-Preserving Federated Unlearning for Trustworthy Chest X-Ray Diagnosis using Vision Transformers (DeiT)
- **GitHub repository:** https://github.com/aaditripi18/Trustworthy-Federated-Unlearning
- **Current branch:** `main`
- **Commit state:** single initial commit (`Initial implementation of explanation-preserving federated unlearning`). New files not yet committed: `PROJECT_BRIEFING.md`, `HANDOFF.md`, `data/` (dataset, gitignored).

## 2. Current milestone completed

**M8 is complete.** All milestones M0 → M8 are implemented, tested, and verified. The system runs end-to-end on the real NIH dataset (smoke test passed). We are past coding and at the **experiment-execution stage**.

| Milestone | Scope | Status |
|-----------|-------|--------|
| M0 | Scaffolding, config system, logging, reproducibility utils, test harness | ✅ done |
| M1 | Data module: manifest, patient-level split, non-IID partition, dataset, loaders | ✅ done |
| M2 | DeiT/ViT model + training/predict engine + diagnostic metrics | ✅ done |
| M3 | Federated learning (FedAvg server + client + aggregation + history) | ✅ done |
| M4 | Unlearning baselines (oracle, finetune_retain) + forgetting verification (MIA) | ✅ done |
| M5 | Explainability: attention-rollout extraction + explanation metrics | ✅ done |
| M6 | **EPU** — explanation-preserving unlearning (attention-consistency loss) | ✅ done |
| M7 | Evaluation framework: 4-method × multi-seed × multi-λ, tables, figures | ✅ done |
| M8 | Real NIH ChestX-ray14 integration + final driver + reproducibility capture | ✅ done |

## 3. Core idea (one paragraph)

Federated learning trains a shared chest-X-ray model across hospitals without sharing data. Privacy law requires the ability to remove one hospital's data influence (federated *unlearning*). Existing methods restore accuracy and verify forgetting but ignore whether the model's **explanations** survive. Our contribution, **Explanation-Preserving Unlearning (EPU)**, adds an attention-consistency loss during unlearning: `L = L_task + λ·L_exp`, where `L_exp` is the MSE between the current model's attention-rollout map and the *frozen original* model's map on the **remaining** clients' data. Everything is judged against a **retrain-from-scratch oracle** (the gold standard) on four axes: diagnosis, forgetting (MIA), explanation fidelity, and efficiency.

## 4. Architecture

```
                 configs/*.yaml  (all hyperparameters, seeds, paths)
                        │
        ┌───────────────┴───────────────┐
        │            src/                │
        │                                │
   data/ ──► builds patient-disjoint, non-IID federated partitions
        │      (NIH adapter + synthetic fallback + transforms + cache)
        ▼
   models/vit.py  ──► DeiT-Small (timm), 1-channel, multi-label head
        ▼
   federated/  ──► FedAvg simulation (server loop + client local train)
        ▼
   unlearning/ ──► oracle (retrain) │ finetune_retain (plain) │ epu (ours)
        │              └── verify.py: MIA / forget-loss / pred-distance
        ▼
   explain/   ──► attention rollout extraction + map-comparison metrics
        ▼
   eval/      ──► engine (train/predict), metrics, experiment (4-way × seeds
                  × λ), figures. This is the orchestrator.
        ▼
   scripts/run_final.py ──► full NIH run → final_*.csv/json + figures
```

Design pattern highlights: **registry pattern** for unlearning methods (`@register("epu")`) and explanation methods; **torch-optional** code with NumPy fallbacks so the non-GPU parts run and test anywhere; strict separation of *pure* (unit-tested) helpers from *torch* (heavy) code.

## 5. Directory structure (actual)

```
Trustworthy-Federated-Unlearning/
├── configs/
│   ├── default.yaml          # seed, logging, smoke params
│   ├── data.yaml             # synthetic subset (default, keeps tests torch-free)
│   ├── data_nih.yaml         # REAL NIH data config (used by run_final)
│   ├── model.yaml            # ViT + centralized-training config
│   ├── model_nih.yaml        # DeiT-Small NIH model config
│   ├── federated.yaml        # FedAvg rounds/epochs/lr/history
│   ├── unlearn.yaml          # forget client, method, finetune rounds, λ
│   ├── explain.yaml          # attention-rollout / probe settings
│   └── eval.yaml             # seeds [42,43,44], lambdas [0.1,0.5,1.0,2.0]
├── src/
│   ├── data/        nih.py, partition.py, dataset.py, transforms.py,
│   │                synthetic.py, central.py
│   ├── models/      vit.py
│   ├── federated/   server.py, client.py, strategy.py, history.py
│   ├── unlearning/  oracle.py, methods.py (finetune_retain), epu.py, verify.py
│   ├── explain/     extract.py, torch_rollout.py, rollout.py (numpy), metrics.py
│   ├── eval/        engine.py, metrics.py, experiment.py, figures.py
│   └── utils/       config.py, logging.py, repro.py, seed.py
├── scripts/
│   ├── smoke_test.py         # M0 verifier
│   ├── prepare_data.py       # M1 data driver
│   ├── train_central.py      # M2 centralized training
│   ├── eval.py               # M2 checkpoint eval
│   ├── run_federated.py      # M3
│   ├── run_unlearn.py        # M4
│   ├── run_explain.py        # M5
│   ├── run_epu.py            # M6 (4-way comparison)
│   ├── run_evaluation.py     # M7 (multi-seed × multi-λ, writes tables)
│   ├── run_final.py          # M8 FULL NIH run (main training entrypoint)
│   ├── compare_smoke.py      # diff a smoke run vs frozen baseline (equivalence check)
│   └── verify_m3.py
├── tests/                    # test_{smoke,partition,model_forward,fed_smoke,
│                             #  unlearn_smoke,explain,epu,evaluation,nih,
│                             #  m8_outputs,cache}.py
├── docs/DATASET_NIH.md       # NIH download, layout, checklist
├── data/                     # NIH dataset (gitignored): images/ + CSV
├── experiments/              # outputs: final_*.csv/json, figures/, _smoke_baseline/
├── README.md · pyproject.toml · requirements.txt
├── PROJECT_BRIEFING.md · HANDOFF.md
```

## 6. Module implementation status

### Fully implemented & verified
- **Data:** `nih.py` (real 112,120-image adapter + decode-once cache), `partition.py` (IID + Dirichlet, patient-disjoint), `dataset.py`, `transforms.py`, `synthetic.py`, `central.py`.
- **Model:** `models/vit.py` (DeiT-Small, 1-channel, multi-label).
- **Federated:** `server.py` (FedAvg loop + progress logging), `client.py`, `strategy.py`, `history.py`.
- **Unlearning:** `oracle.py`, `methods.py` (`finetune_retain`), `epu.py` (EPU + attention-consistency local step), `verify.py` (MIA / forget-loss / pred-distance).
- **Explain:** `torch_rollout.py` (attention rollout via hook), `extract.py`, `metrics.py` (cosine/Pearson/IoU/MSE).
- **Eval:** `engine.py`, `metrics.py`, `experiment.py` (full 4-way × seed × λ orchestration), `figures.py`.
- **Utils:** `config.py`, `logging.py`, `repro.py`, `seed.py`.
- **Tests:** 38 passing, 7 torch-gated skips.

### Partially implemented / scaffolded (honest)
- **Flower framework integration:** the FL loop is a **custom sequential FedAvg simulation**, not the Flower runtime yet. A Flower adapter is planned (the code is structured so `server.py`/`client.py`/`strategy.py` can be swapped without touching the algorithm). Not required for results.
- **NumPy fallbacks** (`explain/rollout.py`, NumpyLoader, metric fallbacks) exist so the pipeline imports/tests without torch/sklearn. They are scaffolding for the sandbox, **not** the path used for the real GPU run (torch path is authoritative).
- **Explanation methods registry:** only attention rollout is implemented; the registry allows adding Grad-CAM-style variants later (future work, not needed for the core claim).

## 7. Remaining TODOs

1. **[BLOCKING] AWS GPU quota approval** — waiting on Amazon to approve the quota / vCPU limit for a **g4dn.xlarge** (GPU) instance. Training cannot start until approved.
2. Launch the **full experiment** (`run_final.py`) once the instance is available; sync `experiments/` outputs back.
3. Generate/verify final figures (λ-ablation, trade-off, oracle comparison, runtime).
4. Write the conference-quality paper (target: MICCAI / NeurIPS / ICML standard).
5. (Optional) Flower runtime adapter; additional explanation methods; commit the new markdown docs.

## 8. How to run

**Environment:** Python + `requirements.txt` (PyTorch, `timm`, torchvision, NumPy, pandas, scikit-learn, matplotlib). GPU strongly recommended for training.

```bash
git clone https://github.com/aaditripi18/Trustworthy-Federated-Unlearning
cd Trustworthy-Federated-Unlearning
pip install -r requirements.txt
# Download NIH ChestX-ray14 into data/nih/ (see docs/DATASET_NIH.md)
pytest                              # sanity: 38 passed, 7 skipped
```

**Per-milestone smoke commands (dev/verification):**
```bash
python scripts/smoke_test.py        # M0
python scripts/prepare_data.py      # M1
python scripts/train_central.py     # M2 (needs torch+timm)
python scripts/run_federated.py     # M3
python scripts/run_unlearn.py       # M4
python scripts/run_explain.py       # M5
python scripts/run_epu.py           # M6 (4-way comparison)
python scripts/run_evaluation.py    # M7 (multi-seed × multi-λ, writes tables)
```

## 9. Required commands for TRAINING (the full experiment)

This is the single entrypoint for the real run. It uses `configs/data_nih.yaml` +
`configs/model_nih.yaml` + the shared federated/unlearn/explain/eval configs.

```bash
# 1) Make sure NIH data is present at data/nih/ (images/ + Data_Entry_2017_v2020.csv)
# 2) Ensure FULL config (not smoke): eval.yaml seeds=[42,43,44], lambdas=[0.1,0.5,1.0,2.0];
#    data_nih.yaml subset_size: null, num_clients: 5; federated.yaml num_rounds: 3
python scripts/run_final.py
```

Outputs land in `experiments/`: `final_results.csv`, `final_multi_seed_summary.csv`,
`final_tables.csv`, `final_ablation.csv`, `final_summary.json`, `reproducibility.json`,
and `experiments/figures/*.{png,pdf}`.

**Target hardware:** AWS **g4dn.xlarge** (GPU) — *current plan*. Estimated a few hours
for 3 seeds × 4 λ values (exact time to be measured on first run).

**Optional smoke re-verification** (proves optimizations didn't change results):
```bash
# apply the smoke config (num_clients:3, subset_size:150, num_rounds:1,
# finetune_rounds:1, seeds:[42], lambdas:[0.5]), then:
python scripts/run_final.py
python scripts/compare_smoke.py     # exit 0 = byte-identical to frozen baseline
```

## 10. Known bugs / caveats

- **Sandbox can't install torch/timm/scikit-learn** (proxy blocks the wheels). All
  torch-dependent runs (training, smoke, full experiment) must run on a real
  machine / AWS. The torch-free parts (data, partitioning, metric logic, cache
  correctness) are what get verified in the sandbox.
- **Image cache RAM:** with `cache_images: true`, the full run caches all 112,120
  decoded images ≈ **5.6 GB RAM**. Fine on a 16 GB instance, but set
  `cache_images: false` if memory-constrained.
- **Config mode discipline:** the repo may be left in smoke OR full mode. Before the
  real run, confirm FULL config (seeds/lambdas/subset_size/rounds as in §9).
- **CSV filename:** must be `Data_Entry_2017_v2020.csv` (already set in config).
- Previously fixed and worth knowing: a partition O(P·N) slowdown (now vectorized),
  a forget-set loader that used the wrong image source under NIH (fixed), and an AUC
  tie-handling edge case (fixed). All covered by tests now.

## 11. Design decisions that MUST NOT be changed

These are locked. Changing them silently invalidates the science.

1. **EPU objective form:** `L = L_task + λ·L_exp` with `L_exp` = MSE of attention-rollout
   maps vs the **frozen original** model on **remaining-client** data only. The
   forgotten client's data is **never** used during unlearning.
2. **Retrain oracle is the gold standard.** Every method is evaluated by closeness to
   it. Do not redefine the oracle.
3. **Attention Rollout** is captured via a hook on `attn_drop` with `fused_attn=False`,
   handling 1 (ViT) or 2 (DeiT) prefix tokens. Do not swap the explanation method
   for the core comparison.
4. **Patient-disjoint splits** + non-IID Dirichlet(α=0.5) partitioning. No patient
   leakage across train/val/test.
5. **No LR scheduler** (constant LR); optimizer is **Adam**; loss is
   **BCEWithLogitsLoss** (multi-label). Model is **DeiT-Small@224**, pretrained,
   1-channel input.
6. **Seeds are fixed** (42, 43, 44). Reproducibility (seed/env/hardware capture) is
   mandatory.
7. **Honest reporting is non-negotiable.** Results are reported exactly as measured;
   never tune the algorithm to make EPU look better. (A prior smoke run showed EPU@0.5
   beating plain on explanation closeness — reported because it was measured, not
   because it was wanted.)
8. **Performance optimizations are result-preserving ONLY.** The decode-once cache,
   eval-only DataLoader workers, pin_memory, datamodule reuse, and progress logging
   must remain bit-identical to the unoptimized path. Train loaders stay at
   `num_workers=0` (so the augmentation RNG stream can't diverge). Any change here
   must pass `tests/test_cache.py` and `scripts/compare_smoke.py`.
```
