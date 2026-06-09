# TRAP Reference — Fill in from memory by end of Week 1

> Instructions: complete every table cell and formula derivation without looking at the paper.
> Then verify against arXiv:2406.12147. Discrepancies are your learning signal.

---

## 1 · The four TRAP components

| Letter | Full name | The failure mode it addresses | One-sentence definition |
|--------|-----------|-------------------------------|-------------------------|
| T | | | |
| R | | | |
| A | | | |
| P | | | |

---

## 2 · The four formal modes

Derive each formula from first principles, then label which TRAP component it operationalises.

| # | Formula | TRAP component | What f does | What g does | Example NSAI method |
|---|---------|----------------|-------------|-------------|---------------------|
| 1 | `g(f(x), θ)` | | | | |
| 2 | `f(x; g(θ))` | | | | |
| 3 | `f′(x; g(f(x), θ))` | | | | |
| 4 | `f(g(x), x)` | | | | |

---

## 3 · Kautz six-type taxonomy annotation

Map each of Kautz's types and note which TRAP formula(s) it implements.

| Kautz type | Notation | Description | TRAP formula |
|------------|----------|-------------|--------------|
| symbolic Neuro symbolic | | | |
| Symbolic[Neuro] | | | |
| Neuro → Symbolic | | | |
| Neuro∪compile[Symbolic] | | | |
| Neuro[Symbolic] | | | |
| Neuro_{Symbolic} | | | |

---

## 4 · Failure-mode × formula × tool

| Failure mode | Formula | Method in this repo | Week |
|--------------|---------|---------------------|------|
| Hallucination / opacity | `g(f(x), θ)` | LLM critic | 4 |
| Reasoning error | `f(x; g(θ))` | ABLkit (consistency KB) | 2 |
| Distribution shift | `f′(x; g(f(x), θ))` | PyEDCR f-EDR rules | 3 |
| Perception / grounding error | `f(g(x), x)` | ABLkit (abduction loop) | 2 |

---

## 5 · Nelson & Narens dichotomy

| Level | Function | Maps to TRAP |
|-------|----------|--------------|
| Object level | | |
| Meta level — monitoring | | |
| Meta level — control | | |

---

## 6 · Week 1 memo seed

"Why does the TRAP paper argue NSAI is the right substrate?  
Where are the weakest points of that argument?"

*(Write 2 pages here or in a separate file. This becomes §1–2 of the Week 4 position paper.)*
