# Every Metric, Term and Diagram — Explained

Companion to the study guide. Part A is the one they are most likely to probe.

---

## PART A — What "fidelity" actually means

**Everyday meaning:** faithfulness. How closely a copy reproduces the original.
A "high-fidelity" recording sounds like the real performance.

**In this project:**

> **Attention fidelity = does the unlearned model still look at the same places
> as the reference model, on the same X-ray?**

You take one chest X-ray, push it through two models, and get two heatmaps
showing where each model looked. Fidelity is **how similar those two heatmaps
are**. High fidelity = the evidence didn't move. Low fidelity = the model is now
looking somewhere else, even if its prediction is identical.

That is the entire project in one sentence: *accuracy says whether the answer is
right; fidelity says whether the reasoning is still the same.*

### The honest caveat (a sharp reviewer may raise this)

In the wider explainable-AI literature, "fidelity" sometimes means something
slightly different — *how truthfully an explanation reflects the model's real
decision process* (tested by deleting the highlighted pixels and seeing if the
prediction collapses).

**We use it in the sense of explanation *preservation*: similarity to a
reference model's explanation.** If challenged, say exactly that:

> I use fidelity to mean similarity to a reference model's explanation —
> explanation preservation. I am not claiming to measure how truthfully the
> rollout reflects the network's causal reasoning; that would need a
> deletion/insertion test, which is future work.

That answer will earn you more credit than defending a word.

### Two references — say which one you mean, every time

| Fidelity to… | Question it answers | Our result |
|---|---|---|
| **the Original model** | "Does it still reason like the model the doctor validated?" | EPU wins clearly: 0.9760 → 0.9987 |
| **the Oracle** | "Does it reason like a model that never saw the client?" | EPU ≈ Plain, no significant difference |

---

## PART B — Every metric in the results tables

### B1 · Diagnosis (is the prediction right?)

**Precision** = of the cases I flagged, how many were real.
`TP / (TP + FP)`. Ours: **0.148** — about 1 in 7 flags is correct.

**Recall (sensitivity)** = of the real cases, how many I caught.
`TP / (TP + FN)`. Ours: **0.552** — we catch over half.

> Precision and recall trade off. We deliberately favour recall via the
> positive-weighted loss: for screening, a missed finding is worse than one sent
> for a second look.

**F1** = harmonic mean of the two, `2PR / (P + R)`. A single number balancing them.

**AUROC** — Area Under the Receiver Operating Characteristic curve.
Sweep the decision threshold from 0 to 1 and plot true-positive rate against
false-positive rate. The area under that curve is the AUROC.

- **0.5 = random guessing. 1.0 = perfect.**
- Interpretation: *the probability that the model scores a randomly chosen sick
  patient higher than a randomly chosen healthy one.*
- **Threshold-free** — that's why it's the headline metric. It doesn't depend on
  where you set the cut-off.

**macro vs micro:**
- **macro** — compute AUROC separately for each of the 14 findings, then average.
  Every disease counts equally, including rare ones. **This is our headline: 0.805.**
- **micro** — pool every prediction together and compute once. Dominated by the
  common findings.

**mAP** — mean Average Precision. Area under the *precision–recall* curve,
averaged over the 14 findings. Ours: **0.206**.

> Why report it as well as AUROC? Because when positives are very rare, AUROC
> can look flattering — a huge number of easy true negatives inflates it. mAP
> ignores true negatives, so it's the harsher, more honest view of a rare-disease
> task.

### B2 · Forgetting (did the client really go?)

**MIA AUC** — Membership Inference Attack AUC.
An attacker asks: *was this specific record in the training set?* It works
because models are systematically more confident on data they trained on. The
attacker builds a classifier on that confidence signal, and we report its AUC.

- **0.5 = the attacker is guessing = the client is forgotten.** ← we WANT 0.5
- 1.0 = every training record perfectly identifiable = total leakage.

Ours: **0.505**. The oracle — which genuinely never saw the client — gets
**0.499**. We match the gold standard.

> This is the only metric in the project where a *lower/mid* value is the good
> result. Don't let that trip you up.

**Forget-set loss** — the model's loss on the forgotten client's data. After
successful unlearning it should rise to look like the loss on data the model
never saw.

**Prediction distance to oracle** — how far the model's outputs are from the
oracle's outputs on the same test images. Lower = behaves more like the model
that never saw the client.

### B3 · Explanation fidelity (does it still look in the same place?)

All four compare **two 14 × 14 attention maps** for the same X-ray.

**Cosine similarity** — treat each map as a 196-number vector; measure the angle
between them.

```
    cos(a, b) = (a · b) / (‖a‖ ‖b‖)
```
1.0 = identical pattern. Insensitive to overall brightness — it compares *shape*.
Ours: **0.9760 (Plain) → 0.9987 (EPU λ=100)**.

**Pearson correlation** — the same idea, but each map has its mean subtracted
first. So it measures whether the two maps *vary together*, ignoring any
constant offset. Ours: 0.9296 → 0.9964.

**IoU** — Intersection over Union. Threshold both maps at **0.5** to get two
binary regions, then:

```
    IoU = |A ∩ B| / |A ∪ B|
```
This is the strictest and most physical of the four: it asks whether **the same
pixels** are highlighted, not merely whether the patterns correlate.
Ours: **0.6029 → 0.8707**.

> Quote this one when challenged. Cosine can be high for two blurry maps that
> merely have similar shape; IoU going 0.60 → 0.87 means the highlighted regions
> genuinely overlap.

**MSE** — Mean Squared Error, the average squared difference per cell. **Lower is
better.** Ours: 0.0070 → 0.0005, about **14× smaller**.

### B4 · Efficiency

**Wall-clock time** — real elapsed seconds. Original 4.8 h, Oracle 3.5 h,
Plain 0.7 h, EPU 0.9 h.

**Speed-up vs oracle** = `oracle_time / method_time`.
Plain **5.2×**, EPU **3.9×**. The oracle is 1.0 by definition.

> Why is EPU slower than Plain? Every step needs a second forward pass through
> the frozen original model and a backward pass through the attention rollout.
> That's the cost of the method — and it's still ~4× cheaper than retraining.

### B5 · The "±" symbol

Every number is written `mean ± standard deviation` **across the three seeds
(42, 43, 44)** — not across images.

> It tells you how much the number moves if you just rerun the experiment with
> different randomness. It is the yardstick for whether a difference is real: our
> EPU gain is **2.4–3.3× larger than Plain's own ± value**, which is why we can
> claim it isn't noise.

---

## PART C — Every term in the architecture diagram

**Stage 1 — Data & Partitioning**

| Term | What it means |
|---|---|
| **NIH ChestX-ray14** | Public benchmark: 112,120 frontal chest X-rays, 14 findings. Labels were mined automatically from radiology reports, so they carry known noise. |
| **Grayscale · 224² · 1-channel** | X-rays have no colour. Resized to 224×224 (the ViT's input size); 1 channel instead of RGB's 3. |
| **Patient-disjoint split** | All images of one patient stay on one side of the train/test divide. Without it the same patient appears in both and scores are inflated — a well-known trap on this dataset. |
| **Non-IID partition, Dirichlet α = 0.5** | Clients get *different* disease mixes, as real hospitals would. α controls the skew: small α = extreme skew, large α = uniform. 0.5 = moderately skewed, the standard choice. |

**Stage 2 — Federated Training (FedAvg)**

| Term | What it means |
|---|---|
| **DeiT-Small clients ×5** | Five simulated hospitals, each holding a copy of the model and its own private data. |
| **Local epochs (×2)** | Each round, a client passes over its own data twice before reporting back. |
| **pos-weight BCE** | Binary Cross-Entropy where positive examples are weighted up, because findings are rare. Without it the model learns to answer "no" to everything. |
| **Warmup + cosine LR** | Learning-rate schedule: rise gently for 3 rounds (warm-up), then decay along a cosine curve to ~0. Warm-up stops early instability; cosine decay lets it settle. |
| **Weighted averaging** | FedAvg: the server averages client weights in proportion to how much data each holds. |
| **30 rounds** | 30 broadcast → train → return → average cycles. |
| **Best-checkpoint + early stop** | Keep the version that scored best on validation macro-AUROC; stop if it stops improving. |
| **Original global model** | The finished pre-unlearning model. Everything is later compared against this. |

**Stage 3 — Federated Unlearning**

| Term | What it means |
|---|---|
| **Forget client r** | The client (client 0) that invokes its right to be forgotten. |
| **Right to be forgotten** | The legal right to have your data's influence removed. |
| **Oracle — retrain from scratch** | Train again from zero on the 4 remaining clients. Correct **by definition**; the gold standard. |
| **Plain unlearning (warm-start)** | Start from the Original's weights and keep training on the retained clients only. Cheap; nothing protects explanations. |
| **Warm-start** | Begin from an already-trained model instead of random weights. |
| **EPU ★ (proposed)** | The same warm-start fine-tune **plus** the attention-fidelity loss. |
| **L = L_task + λ·L_exp** | The objective. Task loss plus λ times the explanation-consistency loss. λ = 0 recovers Plain exactly. |
| **Unlearned global model** | The output of any of the three procedures. |

**Stage 4 — Evaluation (4 axes, 3 seeds)**

| Term | What it means |
|---|---|
| **Held-out test set** | Images no model trained on, patient-disjoint from training. |
| **Mean ± std** | Averaged over seeds 42/43/44, with the spread. |
| **Attention Rollout** | The method that turns 12 layers of attention into one 14×14 heatmap. |
| **Membership-inference AUC** | The forgetting verifier (Part B2). |
| **Forget-loss · dist → oracle** | Supporting forgetting evidence. |
| **Wall-time · rounds · speed-up** | The efficiency axis. |

**Stage 5 — Artifacts & Reproducibility**

| Term | What it means |
|---|---|
| **checkpoints .pt** | Saved PyTorch model weights — 30 of them, so any result can be re-derived. |
| **final_*.csv** | Per-seed metric tables, one row per configuration. |
| **final_summary.json** | The aggregated mean ± std tables. |
| **reproducibility.json** | The environment manifest: seed, Python and library versions, GPU model, git commit. This is what lets someone else rebuild the exact setup. |

**★** marks the proposed contribution — the EPU box and the explanation-fidelity
axis. Nothing else in the diagram is new.

---

## PART D — What each figure technically specifies

**Fig. Dose-response in λ** *(two panels)*
- **x-axis:** the explanation weight λ (1, 10, 30, 100).
- **y-axis:** left panel = attention cosine to the Original; right = IoU to the Original.
- **Blue points + error bars:** mean ± 1 s.d. over the 3 seeds.
- **Grey dashed line + shaded band:** Plain unlearning's mean and its ±1 s.d.
- **What it proves:** both curves rise monotonically with λ and sit entirely
  above Plain's noise band. Monotonic response to the knob = the loss term is
  doing the work, not chance.

**Fig. Is the gain real? (paired gain)**
- **Bars:** for each λ and each seed, `EPU − Plain` on cosine to the Original,
  compared **within the same seed** (pairing removes between-seed variance).
- **Red dotted line:** Plain's own across-seed standard deviation, 0.0068.
- **What it proves:** all 12 bars are positive and all clear the noise line. Not
  an average that hides a bad run — every individual comparison wins.

**Fig. What the gain costs**
- **Left:** membership-inference AUC per configuration, with the chance line at 0.50.
- **Right:** macro-AUROC per configuration.
- **What it proves:** both are flat across every λ. The fidelity gain is not
  bought with privacy or accuracy.

**Fig. Cost of unlearning**
- **Left:** wall-clock hours per method. **Right:** speed-up relative to the oracle.
- **What it proves:** unlearning avoids the full retrain; EPU's extra cost is the
  attention-rollout backward pass.

**Fig. λ ablation / Oracle comparison** *(the training run's own two plots)*
- These plot fidelity **to the ORACLE**, not the Original — which is why the
  ablation looks flat. Same runs, same files, different reference model. Both
  references are in the deck deliberately.

**Data Flow Diagram** — the *movement of data*. Rounded boxes are processes,
square boxes are external entities and data stores, arrows are data in motion.
Its point: raw images enter only the partitioning process; after that, everything
crossing a client boundary is weights or metrics.

**Use Case Diagram** — *who can ask the system to do what*. Three actors
(hospital client, aggregation server, researcher/auditor) and seven use cases.
Its point: the auditor's verification use cases are separate from the server's,
so forgetting can be checked by someone other than whoever performed it.

**Class Diagram** — the *static code structure*. The key detail: the three
unlearning procedures inherit from one abstract `UnlearningMethod`, so
`EPUFineTune` differs from `PlainFineTune` by an `explanation_loss()` method and
a `λ` attribute — the code itself shows it's a controlled ablation.

**Sequence Diagram** — the *order of messages in time*. One federated round
(broadcast → local train → return → aggregate), repeated 30 times, then the
forget request and the fine-tune. Its point: the only things on the arrows are
weights and scores — never an image, label or patient ID.
