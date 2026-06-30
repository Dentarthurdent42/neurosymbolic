# Week 1 — Foundations: Metacognition, TRAP, and NSAI Landscape

**Goal:** Internalize TRAP, the four formal modes, developmental-psychology origins, and
the Kautz / Garcez taxonomies.

**Deliverable:** `trap_reference.md` filled in from memory by end of Day 5.

---

## Daily schedule

### Day 1 (4 h) — The anchor paper, twice

| Time | Task |
|------|------|
| 90 min | First pass: Wei et al., "Metacognitive AI: Framework and the Case for a Neurosymbolic Approach," NeSy 2024, arXiv:2406.12147 |
| 75 min | Second pass: extract the 4×2 table (formula → TRAP cell + NSAI method) |
| 45 min | Watch Shakarian METACOG-23 overview + Lebiere "Architectural approach to metacognition" |

| modality | formula | explanation |
|----------|---------|-------------|
| transparency | $g(f(x), \theta)$, $g\|f$ | generates explanations based on input, $x$, and parameters, $\theta$, of the model $f$. represents function $f$ with a series of $g$ |
| reasoning | $f(x; g(\theta))$ | self-reflection through $g$ informs decision-making through $f$ |
| adaptability | $f\prime(x; g(f(x), \theta)$, $g(x)?f(x):h(x)$ | adapts $f$ to $f\prime$ based on metacognitive assessment of $g$ and params $\theta$. $g$ decides between $f$ or $h$ based on $x$ |
| perception | $f(g(x), x)$ | $f$ is the primary processing function of input $x$ and the metacognitive assessment $g(x)$ to refine its interpretation. $g$ evaluates accuracy and limitations of sensory processing |

**Self-assessment:** For each failure mode (hallucination, reasoning error, distribution shift,
perception error) — which formula and which NSAI technique applies?

---

### Day 2 (4 h) — Historical roots

| Time | Task |
|------|------|
| 60 min | Flavell, "Metacognition and Cognitive Monitoring," *Am. Psychologist* 1979 |
| 75 min | Cox & Raja (eds.), *Metareasoning*, MIT Press 2011 — Ch. 1 + Ch. 3 (Zilberstein) |
| 15 min | Nelson & Narens 1994 — monitoring vs. control dichotomy |
| 45 min | Stocco METACOG-25 keynote, "What is 'meta' in metacognition?" |

---

### Day 3 (4 h) — NSAI taxonomies

| Time | Task |
|------|------|
| 90 min | Garcez & Lamb, "Neurosymbolic AI: The 3rd Wave," *Artif. Intell. Rev.* 56(11), 2023, arXiv:2012.05876 |
| 60 min | Kautz, "The Third AI Summer," *AI Magazine* 43(1), 2022 — six-type taxonomy |
| 80 min | Hitzler, Sarker & Eberhart, *Compendium of Neurosymbolic AI*, IOS Press 2023 — Chs. 1–4 only |

**Output:** Annotate `trap_reference.md` with where each TRAP formula sits in Kautz's taxonomy.

---

### Day 4 (4 h) — Shakarian & Wei book + working definition

| Time | Task |
|------|------|
| ~180 min | *Metacognitive Artificial Intelligence* (Shakarian & Wei, Cambridge UP, ISBN 978-1-009-52245-8, 2025) — Ch. 1 + Ch. 4 (Nirenburg, McShane, Ferguson on mutual trust) |
| 45 min | De Smet et al., "Defining Neurosymbolic AI," arXiv:2507.11127, 2025 |

---

### Day 5 (4 h) — Sprint: videos + writing

| Time | Task |
|------|------|
| ~180 min | Cluster A videos: Wei, Nirenburg, Krishnaswamy METACOG-23 talks + Lebiere again |
| ~60 min | Write a 2-page memo: "Why does the TRAP paper argue NSAI is the right substrate? Where are the weakest points?" → seed for Week 4 position paper |

---

## Week 1 checkpoint

- [ ] Can draw the TRAP diagram with all four formulas from memory
- [ ] Can classify any given paper into one TRAP cell in < 60 s
- [ ] Can recite Kautz's six types with one example each
- [ ] `trap_reference.md` is filled in

---

## Cluster A videos (use this week)

- Christian Lebiere (CMU) — "An architectural approach to metacognition" — youtu.be/0a3YPGKnGmM
- Sergei Nirenburg (RPI) — "Mutual Trust in Human-AI Teams" — youtu.be/KmMe4QrQiRQ
- Hua Wei (ASU) — "Trustworthy Decision Making through Uncertainty Reasoning" — youtu.be/YZLJlRvPrj4
- Nikhil Krishnaswamy (CSU) — "Reasoning About Anomalous Object Interaction" — youtube.com/watch?v=5ZHzEHZbK0Y
- Andrea Stocco (UW, METACOG-25 keynote) — "What is 'meta' in metacognition?" — youtu.be/F-vXM9_larU
- Bonnie Johnson (NPS, METACOG-25 keynote) — "Synthetic Metacognition for Managing Tactical Complexity" — youtu.be/2p9m6BH1wZE
