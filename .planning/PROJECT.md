# CAB420 Assignment 2 — Enron Authorship Identification

## What This Is

A machine learning group assignment investigating email authorship identification on the Enron Email Corpus. Three group members each implement one distinct ML approach to classify emails by author, then compare results across methods in a final written report and video presentation. This is worth 30% of the CAB420 Machine Learning subject at QUT.

## Core Value

A fair, reproducible comparison of three genuinely diverse ML approaches for authorship classification — the methods must span different paradigms (e.g. traditional, deep learning, transformer) and be evaluated on identical data splits.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Implement 3 distinct ML methods for email authorship classification (one per group member)
- [ ] Curate Enron subset: top 10–20 most prolific authors with sufficient per-class email counts
- [ ] Preprocess email bodies: strip headers that reveal authorship, clean text, tokenise
- [ ] Apply consistent 70/15/15 stratified train/val/test split across all methods
- [ ] Evaluate all methods using: accuracy, macro F1, per-class precision/recall, training time, inference time
- [ ] Report majority-class baseline alongside all results
- [ ] Write 10–15 page report (excl. front matter, references, appendices)
- [ ] Cover ≥10 existing published methods in Related Work section
- [ ] Record ~5 minute pre-recorded video presentation
- [ ] Submit optional 1–2 page project proposal for feedback (no marks)

### Out of Scope

- Full 150-user classification — compute constraints, focus on top 10–20 authors
- Novel algorithm proposal — comparing existing methods is the goal
- Email metadata features (To/From/Subject) — stripped to avoid trivial authorship leakage
- Mobile or web deployment — academic research deliverable only

## Context

- **Dataset:** Enron Email Dataset (Klimt & Yang, 2004), ~500k emails from ~150 Enron employees. Working with a curated subset of top 10–20 authors by email count.
- **Task type:** Multi-class text classification (authorship attribution)
- **Paradigm diversity required:** Methods must span at least two of: traditional ML, deep sequence models, transformer-based models
- **Existing work:** There is prior literature on Enron authorship attribution to benchmark against
- **Group members:** Mackenzie Stone (N11608081), Alexander Venn, Johnsen Villocino (N10530819)
- **Methods not yet finalised** — research phase will recommend the best 3 approaches

## Constraints

- **Timeline:** Final report + video due 29 May 2026 — approximately 4 weeks from project start
- **Compute:** Methods must be feasible on standard student hardware (no HPC assumed)
- **Scope:** Exactly 3 methods, one per group member — assignment requirement
- **Diversity:** All 3 methods must be genuinely different (not 3 variants of the same model class)
- **Reproducibility:** Identical data splits across all methods — required for fair comparison
- **Language/Tools:** Python expected (consistent with CAB420 practicals)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Top 10–20 authors only | Sufficient per-class data; feasible compute | — Pending |
| Strip email headers | Prevents trivial authorship leakage via To/From fields | — Pending |
| Stratified 70/15/15 split | Balanced class representation; consistent across all members | — Pending |
| Methods TBD via research | Research phase will identify best 3 approaches for this task | — Pending |

---
*Last updated: 2026-04-27 after initialization*
