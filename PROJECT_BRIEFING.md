# Project Briefing — Context Handoff

> Paste this whole document to bring an AI assistant up to speed on the project.

---

## 1. Project Title

**Attention-Fidelity-Preserving Federated Unlearning for Trustworthy Chest X-Ray Diagnosis using Vision Transformers (DeiT)**

## 2. One-line summary

We remove a hospital's data influence from a federated chest-X-ray model (federated *unlearning*) **without degrading the model's explanations** — using a Vision Transformer (DeiT) and an attention-consistency loss. The goal is a publication-quality, fully reproducible research project.

## 3. The problem & motivation

- **Federated learning (FL)** lets multiple hospitals train a shared diagnostic model on chest X-rays without pooling patient data.
- Privacy law (the "right to be forgotten") requires the ability to remove one hospital's data influence from an already-trained model. That process is **federated unlearning (FU)**.
- Existing FU methods only care about (a) restoring accuracy and (b) proving the data was forgotten. **They ignore explanation fidelity.**
- In medical AI, a model that predicts correctly but *looks at the wrong region* is untrustworthy. No current method guarantees the model's attention/explanations survive unlearning.

## 4. Research gap & novelty (our contribution)

**Explanation-Preserving Unlearning (EPU):** during unlearning, we add an **attention-consistency loss** that keeps the current model's attention-rollout maps close to the *original* (pre-unlearning) model's maps, computed on the **remaining** clients' data.

- Core objective: `L = L_task + λ · L_exp`
  - `L_task` = BCEWithLogitsLoss (multi-label classification)
  - `L_exp` = MSE between the current model's attention-rollout map and the frozen original model's map
  - `λ` = weight controlling the forgetting-vs-explanation-preservation trade-off (we ablate λ ∈ {0.1, 0.5, 1.0, 2.0})
- The forgotten client's data is **never** used during EPU, so preservation cannot secretly re-teach it (verified downstream via membership-inference attack).

## 5. Method components

- **Model:** DeiT-Small (`deit_small_patch16_224`) from `timm`, pretrained, adapted to 1-channel (grayscale) input, multi-label head for 14 findings.
- **Explanation method:** Attention Rollout (Abnar & Zuidema, 2020), captured via a hook on `attn_drop` with `fused_attn=False`; handles 1 (ViT) or 2 (DeiT) prefix tokens.
- **FL algorithm:** FedAvg simulation (sequential clients, weighted averaging).
- **Unlearning baselines being compared:**
  1. **Original** — the fully trained federated model (before forgetting).
  2. **Retrain Oracle** — retrain from scratch excluding the forgotten client. This is the *gold standard* every method is judged against.
  3. **Plain unlearning (`finetune_retain`)** — warm-started continued FedAvg on remaining clients only.
  4. **EPU (ours)** — same as plain but with the attention-consistency loss, at each λ.

## 6. Dataset

- **NIH ChestX-ray14** — 112,120 images, 14 disease findings, multi-label.
- CSV: `Data_Entry_2017_v2020.csv`; images in a flat `images/` folder.
- **Patient-disjoint splits** (no patient leakage across train/val/test).
- **Non-IID federated partition** across 5 simulated hospitals using Dirichlet(α=0.5).

## 7. Evaluation — four axes, judged against the oracle

1. **Diagnostic performance:** macro-AUC, mAP, macro-F1, micro-F1.
2. **Forgetting effectiveness:** membership-inference attack (MIA) AUC on the forgotten client's data (should drop toward the oracle's), forget-set loss.
3. **Explanation fidelity:** cosine similarity / Pearson / IoU / MSE of attention maps vs the oracle's maps (this is the headline contribution metric).
4. **Efficiency:** wall-clock time, rounds, speedup vs the oracle.

Run across **3 seeds (42, 43, 44)**; report mean ± std. **Honest reporting is mandatory** — results are reported as measured, never tuned to favor EPU.

## 8. Tech stack

Python, PyTorch, `timm`, Flower (planned adapter), torchvision, NumPy, pandas, scikit-learn, matplotlib. Modular package layout under `src/`.

## 9. Methodology / working style (important rules we follow)

- **Project Bible governance:** design sections are Drafted → Reviewed → Improved → Approved → **LOCKED**; locked sections are never silently changed.
- **Milestone-based development (M0–M8):** each milestone has objective, deliverables, verification, and a stop-for-approval gate. Repo stays working/testable after each.
- **The AI acts as a critical collaborator** (senior ML engineer, FL expert, XAI researcher, conference reviewer) — challenges assumptions, doesn't just agree.
- **Reproducibility is required:** fixed seeds, logged environment/hardware, deterministic preprocessing.
- **No fabricated results, ever.**

## 10. Implementation stages completed (Milestones M0–M8)

The whole system was built incrementally following a "Minimum Working Research System" roadmap. **Every milestone below is implemented, verified, and complete.** Each left the repo in a working, testable state.

- **M0 — Project scaffolding.** Repository structure, config system (YAML), logging, reproducibility utilities (seed control, environment/hardware capture), test harness. *Status: done.*
- **M1 — Data module.** Manifest building, patient-level splitting (no leakage), non-IID client partitioning (IID + Dirichlet schemes), `CXRDataset`, data loaders, transforms. Originally on synthetic data so the pipeline runs without torch. *Status: done.*
- **M2 — Model + training engine.** DeiT/ViT model builder (`timm`), 1-channel adaptation, multi-label head, shared train/predict loop (`train_one_epoch`, `collect_predictions`), diagnostic metrics (AUC, mAP, F1). *Status: done.*
- **M3 — Federated learning.** FedAvg server loop, client parameter (de)serialization + local training, weighted-average aggregation, per-round history. *Status: done.*
- **M4 — Unlearning baselines + forgetting verification.** Retrain oracle (gold standard), `finetune_retain` (plain unlearning), and forgetting metrics: membership-inference attack (MIA) AUC, forget-set loss, prediction-distance to oracle. *Status: done.*
- **M5 — Explainability.** Attention Rollout extraction for ViT/DeiT (hook-based), explanation comparison metrics (cosine, Pearson, IoU, MSE between attention maps). *Status: done.*
- **M6 — EPU (core contribution).** Explanation-Preserving Unlearning: the attention-consistency loss (`L = L_task + λ·L_exp`) added into the local training step against a frozen original model. Registered as a pluggable method. *Status: done.*
- **M7 — Evaluation framework.** The full four-method comparison (Original / plain / EPU@each λ / Oracle) across the four axes, multi-seed aggregation (mean±std), IEEE-style result tables, honest-findings summary, and publication figures. *Status: done.*
- **M8 — Real NIH dataset integration + final driver.** NIH ChestX-ray14 adapter wired into the same pipeline, the `run_final.py` end-to-end driver, and reproducibility capture. *Status: done.*

### Post-milestone hardening (also complete)
- **Real data verified** on all 112,120 NIH images (datamodule build ~4.4s; batch shape `(32, 1, 224, 224)`); two real pre-training bugs found and fixed.
- **Smoke test** (tiny config) ran end-to-end successfully and produced valid outputs (metrics, CSVs, JSONs, figures).
- **Result-preserving performance optimizations** added and proven bit-identical by tests: decode-once image cache (~75× faster repeat loads), eval-only DataLoader workers, `pin_memory`, datamodule reuse per seed.
- **Training-loop progress logging** added (round / client / epoch / batch / loss) — logging-only, zero behavior change.
- **Test suite:** 38 passed, 7 torch-gated skips.

### Repository structure (high level)
```
configs/     YAML configs (data, model, federated, unlearn, eval, explain)
src/data/    NIH adapter, partitioning, dataset, transforms, synthetic
src/models/  DeiT/ViT builder
src/federated/  FedAvg server + client + aggregation + history
src/unlearning/ oracle, finetune_retain, EPU, forgetting verification (MIA)
src/explain/    attention-rollout extraction + explanation metrics
src/eval/       training/predict engine, metrics, experiment framework, figures
src/utils/      seed, logging, reproducibility capture
scripts/     run_federated / run_unlearn / run_epu / run_final / compare_smoke
tests/       full torch-free + torch-gated test suite
```

## 11. Current status — AWS stage (waiting on quota)

**Everything is implemented and verified. We are now at the training/experiment stage.**

- The plan is to run the **full experiment on an AWS g5.xlarge GPU instance** (estimated ~1.5–3 hours for 3 seeds × 4 λ values).
- **We are currently WAITING FOR AWS to approve the service quota / vCPU limit increase for the g5.xlarge (GPU) instance.** Training cannot start until that quota request is approved.
- Local CPU laptop training is impractical (~days), which is why we are going through AWS.
- **Nothing is blocked on code** — the moment the quota is approved, we launch the full run.

## 12. What's pending / next

1. **AWS quota approval** (in progress — waiting on Amazon).
2. Launch the **full experiment** on g5.xlarge once the instance is available.
3. Collect final metrics across the four evaluation axes; sync outputs back.
4. Generate final figures (λ-ablation, trade-off, oracle comparison, runtime).
5. Write the conference-quality paper (target standard: MICCAI / NeurIPS / ICML).

## 13. Sandbox constraint (technical note)

The isolated coding sandbox cannot install torch/timm/scikit-learn (proxy blocks the wheels). So torch-dependent runs (training, smoke, full experiment) happen on the user's machine / AWS; the sandbox is used for torch-free verification (data, partitioning, metrics logic, cache correctness).

---

*This is a factual status brief. Nothing here should be presented as a finished result — the full experiment has not yet been run to completion.*
