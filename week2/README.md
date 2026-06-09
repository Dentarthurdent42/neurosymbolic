# Week 2 — Neurosymbolic Foundations & First Hands-On

**Goal:** Working knowledge of LTN, DeepProbLog, ABL; one reproduced experiment.

**Primary deliverable:** `notebooks/mnist_addition_ablkit.ipynb` — Experiment 1 complete,
with a plot of test accuracy vs. (loops × noise level) and a one-paragraph TRAP annotation.

---

## Daily schedule

### Day 1 (4 h) — Semantic Loss + DeepProbLog

| Time | Task |
|------|------|
| 75 min | Xu et al., "A Semantic Loss Function for Deep Learning with Symbolic Knowledge," ICML 2018 |
| 90 min | Manhaeve et al., "Neural Probabilistic Logic Programming in DeepProbLog," *Artif. Intell.* 2021 |
| 60 min | Code-along: DeepProbLog MNIST-Addition tutorial |

Extract: how DPL reduces queries to weighted model counting and where gradients flow.

---

### Day 2 (4 h) — Logic Tensor Networks

| Time | Task |
|------|------|
| 90 min | Badreddine et al., "Logic Tensor Networks," *Artif. Intell.* 2022, arXiv:2012.13635 — first half (Real Logic + groundings) |
| 2.5 h | LTNtorch tutorial notebooks — "binary classification" + "multiclass with constraints" |

---

### Day 3 (4 h) — Scallop + provenance semirings

| Time | Task |
|------|------|
| 90 min | Li, Huang, Naik, "Scallop: A Language for Neurosymbolic Programming," PACMPL/PLDI 2023 |
| 2.5 h | `pip install scallopy`; run MNIST-Addition end-to-end |

Extract: how the provenance semiring abstraction unifies discrete, probabilistic, and
differentiable reasoning.

---

### Day 4 (4 h) — Abductive Learning + Experiment 1 start

| Time | Task |
|------|------|
| 60 min | Zhou, "Abductive Learning," *Science China Inf. Sci.* 2019 |
| 45 min | Huang et al., "ABLkit," *Frontiers of Computer Science* 2024 |
| 2.5 h | **Experiment 1 start**: clone `github.com/AbductiveLearning/ABLkit`; run MNIST-Addition example |

---

### Day 5 (4 h) — Experiment 1 finish + Compendium reading

| Time | Task |
|------|------|
| 2.5 h | **Experiment 1 finish** (see success criterion below) |
| 2 h | Shakarian et al., *Neuro Symbolic Reasoning and Learning* (Springer Briefs 2023) — Ch. 6 (LNN) + Ch. 7 (NeurASP) |

**Cluster B videos** (slot 60–90 min anywhere this week):
- Zhe Xu — "Interpretable and Data-Efficient Learning for Autonomous Systems"
- YooJung Choi — "Tractable Probabilistic Reasoning for Trustworthy AI"
- Visar Berisha — "A Theoretically-Grounded Framework for Assured ML"

---

## Experiment 1 — MNIST-Addition with ABLkit

**TRAP formula:** `f(g(x), x)` — the perceptual layer is corrected by symbolic abduction.

**Success criterion:**
- A plot of test accuracy vs. (loops × noise level) comparing LeNet vs. ResNet-18 backbones
- A one-paragraph reflection identifying the TRAP cell and formula

**Notebook:** `notebooks/mnist_addition_ablkit.ipynb`

**Modification from the ABLkit example:**
1. Swap the perceptual backbone between LeNet (default) and ResNet-18
2. Vary label noise: 0%, 10%, 20%, 30%
3. Log convergence (number of abduction loops to reach plateau)
4. Plot a 3D surface or heatmap: backbone × noise → accuracy at convergence

---

## Week 2 checkpoint

- [ ] Can describe LTN, DeepProbLog, Scallop, and ABL in three sentences each
- [ ] MNIST-Addition notebook runs end-to-end with both backbones
- [ ] Plot of accuracy vs. loops × noise level exists
- [ ] TRAP cell `f(g(x), x)` is annotated in the notebook
