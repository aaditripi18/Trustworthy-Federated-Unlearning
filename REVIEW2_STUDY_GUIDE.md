# Review-2 Study Guide

**Attention-Fidelity-Preserving Federated Unlearning with Vision Transformers for Chest X-Rays**

Read Part 1 and Part 7 twice. Everything else once. If you only have an hour,
learn Part 1, the four numbers in Part 6, and the answers in Part 8.

---

## PART 1 — The 60-second pitch (learn this word-perfect)

> Hospitals cannot share chest X-rays with each other, because patient images
> are protected data. **Federated learning** solves that: each hospital trains
> the model on its own machine and only the model weights are sent to a central
> server, which averages them. No image ever leaves the hospital.
>
> But privacy law also gives a hospital the **right to be forgotten**. If it
> leaves the group, its influence has to be removed from the shared model. The
> correct way is to retrain the whole model from scratch without it, which is
> extremely expensive. **Federated unlearning** is the study of cheaper ways to
> get the same effect.
>
> Existing unlearning methods check two things: did accuracy recover, and is the
> client's data statistically gone. **Nobody checks whether the model still
> looks at the same part of the lung.** That matters in radiology, because a
> doctor accepts a prediction by looking at the highlighted region. A model can
> keep the exact same accuracy while looking somewhere completely different, and
> no standard metric would catch it.
>
> So we propose **EPU — Explanation-Preserving Unlearning**. While the model is
> being unlearned, we add one extra term to the loss that keeps its attention
> map close to the original model's attention map. We then measure whether that
> costs anything, and it does not.

**If you say nothing else, say that.**

---

## PART 2 — The four building blocks

### 2.1 Federated Learning and FedAvg

The algorithm is **FedAvg** (McMahan et al., 2017). One round:

1. Server sends the current global weights **w<sub>t</sub>** to all clients.
2. Each client *k* trains locally on its own data for a couple of epochs, and
   gets **w<sub>t</sub><sup>k</sup>**.
3. Clients send weights back (never data).
4. Server averages them, **weighted by how much data each client has**:

```
                 K
    w        =   Σ   (n_k / n) · w_t^k
     t+1        k=1
```

where `n_k` = number of samples at client *k*, and `n = Σ n_k`.

**Our setting:** 5 clients, 30 rounds, 2 local epochs per round.

**Why weighted?** A hospital with 20,000 images should influence the average
more than one with 2,000.

### 2.2 Non-IID and the Dirichlet split

Real hospitals do not see the same diseases in the same proportions. A chest
clinic sees more pneumonia; a cancer centre sees more masses. We simulate that.

- **IID** = every client has the same label distribution (unrealistic, easy).
- **Non-IID** = clients have different label distributions (realistic, harder —
  FedAvg converges worse).

We use a **Dirichlet distribution with α = 0.5** to decide how much of each of
the 14 findings goes to each client.

- **α → 0** : extreme skew, each client sees almost one disease.
- **α → ∞** : uniform, back to IID.
- **α = 0.5** : moderately skewed. The standard choice in federated papers.

We also make the split **patient-disjoint**: all images of one patient stay on
one side. Without this, the same patient appears in train and test and the
scores are inflated — this is a real, well-known trap with NIH ChestX-ray14.

### 2.3 Vision Transformer (DeiT-Small)

- The image (224×224) is cut into **16×16 patches** → 14 × 14 = **196 patches**.
- Each patch is flattened and linearly projected into a vector ("token").
- A special **[CLS] token** is prepended. Its final state is what the classifier
  reads.
- 12 transformer layers, 6 attention heads each, embedding dimension 384,
  **≈ 22 million parameters**.
- We change two things: **1 input channel** (X-rays are grayscale, not RGB) and
  **14 sigmoid outputs**.

**Why 14 sigmoids and not softmax?** This is **multi-label**, not multi-class —
a patient can have Atelectasis *and* Effusion *and* Infiltration at the same
time. Softmax forces the probabilities to sum to 1, which is wrong here. Each
finding gets its own independent yes/no.

**Why a Vision Transformer instead of a CNN?**
This is the question they will most likely ask. Answer:

> Because attention is an *intrinsic* explanation. In a CNN you have to bolt on
> a method like Grad-CAM afterwards to guess where it looked. In a transformer,
> the attention weights are already part of the forward computation — the model
> tells you where it looked as a by-product of predicting. And because it is
> just matrix multiplication, it is **differentiable**, so I can put it inside a
> loss function. That is the whole reason my method is possible.

### 2.4 Attention Rollout — how we read the explanation

Raw attention from a single layer is a poor explanation, because information
gets mixed across 12 layers and residual connections carry information *around*
the attention.

**Attention Rollout** (Abnar & Zuidema, 2020) fixes this:

1. Take the attention matrix of layer *l*, averaged over the 6 heads: **A<sub>l</sub>**
2. Add the identity to account for the residual connection, then re-normalise
   each row so it sums to 1:

```
    Â_l  =  normalise( A_l + I )
```

3. Multiply through all the layers to trace information flow back to the input:

```
    R  =  Â_L · Â_(L-1) · … · Â_2 · Â_1
```

4. Take the **[CLS] row** of R → one relevance value per patch.
5. Reshape the 196 values into a **14 × 14 map**, min-max normalise to [0, 1],
   upsample over the X-ray → the heatmap you show.

**The one property that matters:** every step is a matrix multiplication, so the
whole thing is differentiable. That is what lets it go inside a loss.

---

## PART 3 — The three unlearning methods

One client (**client 0**) says "forget me". We produce three models:

| Method | What it does | Honest description |
|---|---|---|
| **Oracle** | Retrains from scratch on the 4 remaining clients | Correct **by definition**. This is the gold standard we measure everything against. Most expensive. |
| **Plain** | Warm-starts from the trained model, fine-tunes on the 4 remaining clients | Cheap and standard. Nothing protects the explanations. |
| **EPU (ours)** | Same as Plain, **plus** an attention-consistency term | The contribution. |

**Why the oracle matters:** without it you can only *claim* forgetting. With it
you can say "my model's attack score matches the model that genuinely never saw
the client". That turns an assertion into a measurement.

---

## PART 4 — The equation (learn this properly, they will ask)

```
    L  =  L_task  +  λ · L_exp
```

**L_task** — the normal classification loss: binary cross-entropy with logits,
with a **per-class positive weight**. The weight is needed because the findings
are rare — for most labels, the vast majority of images are negative, so without
weighting the model would just predict "no" for everything.

**L_exp** — the explanation term. For the same X-ray, compute the attention
rollout map of (a) the model currently being fine-tuned and (b) the **frozen
original model**, and penalise the distance between them.

**λ (lambda)** — how strongly you insist on keeping the old explanation.

**The single most important property:**

> **λ = 0 gives back plain unlearning exactly.**

Say this if they push on rigour. It means EPU and Plain differ in **one term and
nothing else** — same data, same schedule, same seeds, same learning rate. So it
is a **controlled ablation**, not a comparison of two different systems. Any
difference we measure has to come from that term.

We tested **λ = 1, 10, 30, 100** (four values, two orders of magnitude).

---

## PART 5 — Architecture (walk the diagram left to right)

Five stages. Practise saying them in order:

1. **Data and partitioning** — NIH ChestX-ray14 (112,120 images, 14 findings),
   grayscale, 224×224, patient-disjoint split, Dirichlet α = 0.5 across 5 clients.
2. **Federated training (FedAvg)** — DeiT-Small on each client, 30 rounds × 2
   local epochs, Adam at 2×10⁻⁴, 3 warm-up rounds then cosine decay,
   best-checkpoint on validation macro-AUROC. **Output: the Original model.**
3. **Unlearning** — client 0 issues a forget request; three procedures run:
   Oracle / Plain / EPU ★.
4. **Evaluation — 4 axes, 3 seeds** — diagnosis, forgetting, explanation
   fidelity, efficiency.
5. **Artifacts** — 30 checkpoints, per-seed CSVs, figures, reproducibility
   manifest (seed, library versions, GPU, git commit).

★ = the proposed contribution. If you only point at one box, point at that one.

---

## PART 6 — The numbers (memorise ONLY these six)

| What | Number | What to say |
|---|---|---|
| Diagnosis | **macro-AUROC 0.805** | "The federated model scores 0.805, and unlearning leaves it unchanged at 0.806." |
| Cost | **5.2× cheaper** | "Plain unlearning is 5.2× faster than a full retrain; EPU 3.9×." |
| Forgetting | **MIA ≈ 0.50** | "Membership inference sits at chance — 0.505 for us, 0.499 for the oracle." |
| Fidelity gain | **0.9760 → 0.9987** | "Attention cosine to the original rises from 0.976 with Plain to 0.9987 at λ = 100." |
| Overlap gain | **IoU 0.60 → 0.87** | "The regions don't just correlate better, they overlap better." |
| Significance | **all 12, 2.4–3.3× noise** | "Positive in all 12 paired comparisons; the effect is 2.4 to 3.3 times the baseline's own seed-to-seed variation." |

*(12 = 4 lambdas × 3 seeds.)*

**The metrics in one line each** — you do not need more than this:

- **macro-AUROC** — area under the ROC curve, averaged over the 14 findings.
  0.5 = random, 1.0 = perfect.
- **mAP** — mean average precision. Better than AUROC when positives are rare.
- **MIA AUC** — can an attacker tell this record was used in training?
  **0.5 = cannot tell = forgotten.** This is the one metric where you *want* 0.5.
- **Cosine similarity** — angle between two heatmaps flattened into vectors.
  1.0 = identical pattern.
- **IoU** — threshold both maps at 0.5, then intersection ÷ union. Measures
  whether the *same regions* light up.
- **MSE** — plain squared error between the two maps. Lower is better.
- **Speed-up vs oracle** — how many times faster than the full retrain.

---

## PART 7 — The one subtle thing (this is where you could fumble)

**We measure explanation fidelity against TWO different reference models.**

1. **Against the Original model** (the one that was validated). EPU wins clearly:
   0.9760 → 0.9987.
2. **Against the Oracle**. EPU and Plain are statistically **indistinguishable** —
   no significant difference.

**And here is the finding:** the oracle's own cosine to the Original is **0.9612
— which is LOWER than Plain's 0.9760.**

What that means, in plain words:

> A model that genuinely never saw the client naturally looks at things slightly
> differently. So "preserve the reasoning the doctor validated" and "behave like
> a model that never saw the client" are **two different goals**, and they pull
> in opposite directions. We measured the gap between them instead of assuming
> they were the same thing. We report both references — we don't pick the
> flattering one.

**If they point at the λ-ablation figure and say "this looks flat":**

> That figure is the training run's own output and it plots fidelity to the
> **oracle**, where I've just said there is no significant effect. My claim is
> about fidelity to the **original model**, which is the quantity EPU actually
> optimises. Same runs, same files, two different reference models — both are in
> the deck.

---

## PART 8 — Questions they will ask, and what to say

**"Isn't your result circular? You optimise that metric, so of course it improves."**
> The *direction* is by construction, yes — I'd be worried if it didn't improve.
> What the experiment measures is the **price**. Adding a constraint to one
> objective normally damages another. Here forgetting is unchanged — membership
> inference is identical to Plain and to the oracle at every λ — and macro-AUROC
> stays within a fraction of a point. That the price is zero is the result. The
> clean dose-response from λ=1 to λ=100 also shows the mechanism is the loss
> term itself and not a seed artefact.

**"Why is precision only 0.148?"**
> Because of the positive weighting in the loss. The 14 findings are rare, so I
> weight positives up, which pushes the model toward recall — 0.552 recall
> against 0.148 precision. For a screening tool, missing a finding is worse than
> flagging one for review. Also the NIH labels are mined automatically from
> radiology reports and are known to be noisy, so absolute numbers on this
> benchmark are modest across the whole literature.

**"How do you know forgetting actually happened?"**
> A membership-inference attack. It asks whether an attacker can tell that a
> record was in the training set — models are more confident on data they were
> trained on. I report the AUC: 0.5 means the attacker is guessing. I get 0.505,
> and the oracle — which truly never saw the client — gets 0.499. I match the
> gold standard.

**"Why does the Original take 4.8 hours but the Oracle only 3.5?"**
> The Original trains on all five clients; the oracle retrains on four. Less
> data per round, so it's faster. The speed-ups are all quoted relative to the
> oracle, which is the fair reference.

**"Why 5 clients?"**
> It's a simulation on a single GPU. Five is enough to produce genuine non-IID
> heterogeneity and a meaningful "remove one of them" question, while keeping a
> three-seed campaign inside about a day of GPU time.

**"What's actually new here?"**
> Three things together, which I could not find combined anywhere: treating
> explanation fidelity as a **differentiable objective optimised during**
> unlearning rather than a picture drawn afterwards; verifying forgetting
> against a **retrain-from-scratch oracle** instead of asserting it; and
> reporting **three seeds** with the effect size measured against run-to-run
> noise.

**"What are the limitations?"**
> It's a simulation on one GPU, not a real hospital network — TRL 4. Clients are
> processes, so there's no secure aggregation or differential privacy. And
> explanation quality is measured against reference models, not against
> radiologist annotation — a reader study is the obvious next step.

**"Why not use Grad-CAM?"**
> Grad-CAM is built for convolutional feature maps and is applied after training.
> I needed something native to the transformer and differentiable so it could sit
> inside the loss. Attention rollout is both.

---

## PART 9 — If you don't know the answer

Do not invent. Use one of these:

- *"I haven't measured that, so I don't want to guess — but I can tell you what I
  did measure, which is …"*
- *"That's outside what I tested. My scope was X; that would be the natural next
  experiment."*
- *"I'd have to check the config to answer precisely."*

A reviewer respects "I measured this, not that" far more than a confident wrong
answer. Your entire project is built on the claim that you measure things
instead of asserting them — answering that way *is* the argument.

---

## PART 10 — Two-minute pre-review checklist

- Slide 2 — **replace the template's sample approval email with your guide's.**
- The date at the bottom of every slide auto-updates to whatever day you open it.
- Have the live demo ready as a fallback: it runs on CPU from the saved
  checkpoints, so it works with no GPU and no internet.
- Know these six numbers cold: **0.805 · 5.2× · 0.50 · 0.9760→0.9987 · 0.60→0.87 · all 12.**
- One sentence you can always fall back on:
  *"Unlearning that removes a hospital's data without changing where the model
  looks — and I measured that it costs nothing."*
