# Metacognitive AI — Neurosymbolic Focus (ABLkit + PyEDCR track)

4-week curriculum following the TRAP framework (Wei et al., NeSy 2024, arXiv:2406.12147).  
Tool pair: **ABLkit** (Week 2) + **PyEDCR** (Weeks 3–4).

---

## TRAP Framework

| Component | Failure mode addressed | Formal mode | Tool in this repo |
|-----------|------------------------|-------------|-------------------|
| **T**ransparency | Hallucination / opaque outputs | `g(f(x), θ)` — meta-monitor wraps f | LLM critic (Week 4) |
| **R**easoning | Reasoning error | `f(x; g(θ))` — symbolic prior g shapes f | ABLkit (Week 2) |
| **A**daptation | Distribution shift | `f′(x; g(f(x), θ))` — post-hoc rule correction | PyEDCR (Week 3) |
| **P**erception | Perception / grounding error | `f(g(x), x)` — symbolic abduction corrects perception | ABLkit (Week 2, Ex. 1) |

The four formulas are the spine of every experiment.  Re-derive them before each session, not the acronym.

---

## Weeks at a glance

| Week | Focus | Primary output |
|------|-------|----------------|
| 1 | TRAP + NSAI taxonomies (reading-heavy) | TRAP reference diagram (`week1/trap_reference.md`) |
| 2 | ABLkit · MNIST-Addition · LTN intro | `week2/notebooks/mnist_addition_ablkit.ipynb` |
| 3 | PyEDCR · hierarchical CIFAR-100 · UQ | `week3/notebooks/edcr_cifar100.ipynb` + `uq_comparison.ipynb` |
| 4 | TRAP-Lite capstone · position paper | `week4/trap_lite/` + `week4/demo.py` |

---

## TRAP-Lite pipeline (Week 4 capstone)

```
Input image x
      │
      ▼
┌─────────────┐    TRAP formula
│  Perception │  f(x)            ← frozen DINOv2 / ResNet-50
│      f      │
└──────┬──────┘
       │  predicted label + logits
       ▼
┌─────────────┐
│  Reasoning  │  g(f(x), θ)      ← PyEDCR learned rules
│      g      │    (f-EDR over hierarchy)
└──────┬──────┘
       │  corrected prediction or error flag
       ▼
┌─────────────┐
│  Adaptation │  f′(x; g(f(x),θ))← split-conformal abstention
│  (conformal)│    when g cannot correct
└──────┬──────┘
       │  prediction set  (or ABSTAIN)
       ▼
┌─────────────┐
│Transparency │  g(f(x), θ)      ← LLM critic narrative
│  (critic)   │    (Claude / local Llama)
└─────────────┘
```

---

## Quick setup

```bash
# 1. clone (already done)
# 2. create env and install
bash setup.sh

# 3. activate
source .venv/bin/activate

# 4. launch notebooks
jupyter lab

# 5. (Week 4) one-command capstone demo
python week4/demo.py --help
```

> **Dependency note:** PyEDCR is installed from GitHub source in `setup.sh` because the
> PyPI package lags the CIKM-2024 f-EDR extension implemented in `lab-v2/PyEDCR`.
> ABLkit requires SWI-Prolog on the PATH; on Debian/Ubuntu: `apt install swi-prolog`.

---

## Repo layout

```
neurosymbolic/
├── README.md
├── requirements.txt
├── setup.sh
├── .gitignore
│
├── notes/
│   └── bibliography.md        ← running annotated bibliography (add 1 para/paper)
│
├── week1/
│   ├── README.md              ← daily reading schedule + tasks
│   └── trap_reference.md      ← TRAP diagram + formula derivations (fill in)
│
├── week2/
│   ├── README.md
│   └── notebooks/
│       └── mnist_addition_ablkit.ipynb   ← Experiment 1
│
├── week3/
│   ├── README.md
│   └── notebooks/
│       ├── edcr_cifar100.ipynb           ← Experiment 2
│       └── uq_comparison.ipynb           ← Experiment 3
│
└── week4/
    ├── README.md
    ├── demo.py                ← one-command TRAP-Lite demo
    └── trap_lite/
        ├── __init__.py
        ├── perception.py      ← f  (frozen backbone)
        ├── reasoning.py       ← g  (PyEDCR rules)
        ├── adaptation.py      ← conformal abstention
        ├── transparency.py    ← LLM critic
        └── pipeline.py        ← end-to-end orchestration
```

---

## Key papers (anchor)

- **TRAP**: Wei et al., "Metacognitive AI: Framework and the Case for a Neurosymbolic Approach," NeSy 2024, arXiv:2406.12147
- **ABLkit**: Huang et al., *Frontiers of Computer Science* Vol. 18 No. 6, 186354, Dec 2024
- **PyEDCR / f-EDR**: Kricheli et al., CIKM 2024, DOI 10.1145/3627673.3679918
- **EDCR (original)**: Xi et al., arXiv:2308.14250, IJCAI STRL Workshop 2025

Full bibliography → `notes/bibliography.md`
