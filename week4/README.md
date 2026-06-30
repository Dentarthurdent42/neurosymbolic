# Week 4 — Integration, Capstone, and Synthesis Deliverables

**Goal:** Build TRAP-Lite end-to-end; draft position paper; finalize bibliography and slide deck.

---

## Deliverables checklist

- [ ] `trap_lite/` package runs end-to-end via `python demo.py`
- [ ] Results table: top-1 accuracy of f alone vs. f + EDCR vs. f + EDCR + conformal abstention
- [ ] Confusion-matrix delta plot + 3 qualitative LLM-critic examples
- [ ] 6–8 page position paper (draft)
- [ ] `notes/bibliography.md` — 25–35 entries annotated
- [ ] 12–15 slide deck

---

## TRAP-Lite architecture

Each component maps to one TRAP formula:

| Component | File | Formula | What it does |
|-----------|------|---------|--------------|
| Perception | `trap_lite/perception.py` | — | Frozen DINOv2 / ResNet-50 backbone |
| Reasoning | `trap_lite/reasoning.py` | `f′(x; g(f(x), θ))` | PyEDCR f-EDR rule engine |
| Adaptation | `trap_lite/adaptation.py` | `f′(x; g(f(x), θ))` | Split-conformal abstention when rules can't correct |
| Transparency | `trap_lite/transparency.py` | `g(f(x), θ)` | LLM critic — natural-language justification |
| Pipeline | `trap_lite/pipeline.py` | all four | Orchestrates f → g → adaptation → critic |

---

## Daily schedule

### Day 1 (4 h) — Foundation-model self-reflection

| Time | Task |
|------|------|
| 60 min | Shinn et al., "Reflexion," NeurIPS 2023, arXiv:2303.11366 |
| 45 min | Madaan et al., "Self-Refine," NeurIPS 2023, arXiv:2303.17651 |
| 30 min | Toy, MacAdam, Tabor, "Metacognition is All You Need?" 2023 |
| 45 min | Kadavath et al., "Language Models (Mostly) Know What They Know," 2022 |
| 75 min | Cluster C videos: Tianlong Chen (sparsity), Chaudhuri (symbolic RL agents), Da et al. (UQ of LLM explanations) |

---

### Day 2 (4 h) — Cognitive architectures

| Time | Task |
|------|------|
| 60 min | Anderson et al., "An Integrated Theory of the Mind" (ACT-R), *Psych. Review* 2004 — §1–3 only |
| 45 min | Laird, Lebiere, Rosenbloom, "A Standard Model of the Mind," *AI Magazine* 2017 |
| 60 min | Laird, Lebiere, Rosenbloom & Stocco, "A Proposal to Extend the Common Model of Cognition with Metacognition," arXiv:2506.07807, 2025 — **bridge from CogArch to TRAP** |
| 30 min | SOAR overview (Laird 2012, intro chapter) |

---

### Day 3 (6 h) — Capstone build

Run: `python week4/demo.py --help` to verify all components are wired.

Build order:
1. `perception.py` — load frozen backbone, return logits + feature vector
2. `reasoning.py` — load trained PyEDCR model from Week 3, expose `.correct(pred, features)`
3. `adaptation.py` — fit conformal calibration set; expose `.predict_set(logits, alpha)`
4. `transparency.py` — call LLM with structured prompt; return justification string
5. `pipeline.py` — chain all four; implement the results table logic
6. `demo.py` — CLI entry point with `--image`, `--split` (val/test), `--coverage`

**Success criterion (meet ≥ 1 of 3):**
- (a) Measurable accuracy lift on hierarchy-consistency metric
- (b) Calibrated coverage within ±2% of nominal
- (c) LLM-critic flagged cases have substantively higher empirical error rates

---

### Day 4 (4 h) — Position paper draft

Draft 6–8 pages:
1. Why metacognition, why now, why NSAI
2. TRAP framework recap (with your own diagram)
3. Survey grouped by TRAP cell — EDCR/NASR for `f′(…)`, ABL/concept-induction for `f(g(x),x)`, LTN/DeepProbLog for `f(x;g(θ))`, Reflexion/Self-Refine for `g(f(x),θ)`
4. Identified gap (defend one: UQ-aware EDCR under domain shift; hierarchical conformal with logical constraints; LLM-as-symbolic-engine for g)
5. Concrete proposed experiment (2–3 pages)

Finalize `notes/bibliography.md` — target 25–35 entries, 3–4 sentences each.

---

### Day 5 (2 h) — Slide deck

12–15 slides:
1. Title
2. Motivating failure (your own taxonomy)
3. TRAP diagram
4–7. One method slide per TRAP cell
8–10. Experiment slides (Ex. 1, 2, 3)
11–12. Capstone results
13. Gap-and-proposal
14. Conclusions
15. References

Record yourself presenting in 20 minutes.

---

## Submission targets (after Week 4)

- NeSy 2026 workshop
- METACOG-26 (if continued)
- AAAI Spring Symposium track on metacognition
