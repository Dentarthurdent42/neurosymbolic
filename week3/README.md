# Week 3 — Metacognition Techniques & Capstone-Track Experiments

**Goal:** Implement PyEDCR on a hierarchical classifier; build and compare UQ methods.

**Deliverables:**
- `notebooks/edcr_cifar100.ipynb` — Experiment 2 (EDCR)
- `notebooks/uq_comparison.ipynb` — Experiment 3 (UQ)

---

## Daily schedule

### Day 1 (4 h) — EDCR theory

| Time | Task |
|------|------|
| 75 min | Xi et al., "Rule-Based Error Detection and Correction to Operationalize Movement Trajectory Classification," IJCAI STRL 2025, arXiv:2308.14250 |
| 60 min | Kricheli et al., "Error Detection and Constraint Recovery in Hierarchical Multi-Label Classification Without Prior Knowledge," CIKM 2024 — **this is the f-EDR paper implemented in PyEDCR** |
| 45 min | Shakarian, Simari, Bastian, "Probabilistic Foundations for Metacognition via Hybrid-AI," AAAI Spring Symposium 2025 |
| ~20 min | Watch: Shakarian EDCR talk — youtu.be/d_OV4lap_rk |

---

### Day 2 (4 h) — Experiment 2: EDCR on hierarchical classifier

Clone `github.com/lab-v2/PyEDCR` (already installed via `setup.sh`).

**Task:**
1. Load CIFAR-100 with the 20-superclass / 100-class hierarchy
2. Fine-tune ResNet-50 (or use frozen DINOv2-small) — target ~70% top-1 on the val split
3. Apply f-EDR to learn error-detection rules without prior constraints
4. Report class-level precision/recall **before** vs. **after** EDCR correction
5. Print the top-10 learned rules

**TRAP formula:** `f′(x; g(f(x), θ))` — adaptation via post-hoc symbolic correction of a frozen f.

**Success criterion:**
- ≥ 3% macro-F1 lift on at least one class
- ≥ 5 human-interpretable rules of the form "if predicted fine-label = X and condition C, suspect error"

**Notebook:** `notebooks/edcr_cifar100.ipynb`

---

### Day 3 (4 h) — NASR + Concept Induction

| Time | Task |
|------|------|
| 75 min | Cornelio et al., "Learning Where and When to Reason in Neuro-Symbolic Inference," ICLR 2023 — extract the Mask-Predictor pattern |
| 75 min | Dalal, Sarker, Barua, Hitzler, "Explaining Deep Learning Hidden Neuron Activations Using Concept Induction," arXiv:2301.09611 |
| 60 min | Evans et al., "Making Sense of Raw Input," *Artif. Intell.* 299, 2021 — Apperception Engine |

---

### Day 4 (4 h) — Uncertainty quantification theory

| Time | Task |
|------|------|
| 60 min | Sensoy, Kaplan, Kandemir, "Evidential Deep Learning," NeurIPS 2018, arXiv:1806.01768 |
| 45 min | Ye, Chen, Wei, Zhan, "Uncertainty Regularized Evidential Regression," AAAI 2024 |
| 75 min | Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction," 2022 tutorial |
| ~20 min | Skim: Gal & Ghahramani, "Dropout as Bayesian Approximation" (MC-Dropout) |

---

### Day 5 (4 h) — Experiment 3: UQ comparison

**Task:** On CIFAR-10, train one ResNet-18 and apply four UQ methods:
1. Softmax temperature scaling
2. MC-Dropout (T=30 passes)
3. Evidential Deep Learning
4. Split-conformal prediction (MAPIE, α = 0.1)

Evaluate: accuracy vs. coverage on in-distribution + OOD detection on CIFAR-10-C / SVHN.

**Success criterion:** A single plot of selective accuracy vs. coverage for all four methods,
plus a paragraph on which best feeds the Adaptation arm of TRAP.

**Notebook:** `notebooks/uq_comparison.ipynb`

**Cluster B videos** (during compute waits):
- Ufuk Topcu — "Multi-Modal Pre-Trained Models in Verifiable Sequential Decision-Making"
- Gavin Strunk — "Uncertainty Quantification's Role in Metacognition"
- Taylor Johnson — "Metacognition in Autonomous CPS with NN Verification, Repair, and Monitoring"

---

## Week 3 checkpoint

- [ ] EDCR notebook runs; ≥ 3% macro-F1 lift demonstrated
- [ ] Learned rules printed and readable
- [ ] UQ comparison plot exists with all four methods
- [ ] TRAP formulas `f′(…)` and the UQ arm of Adaptation are annotated in both notebooks
