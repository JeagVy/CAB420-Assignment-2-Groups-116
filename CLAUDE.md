# CAB420 Assignment 2 — Enron Authorship Identification

## Project

CAB420 Machine Learning group assignment (QUT, 30% weight). Three members each implement one distinct ML approach to email authorship classification on the Enron corpus, then write a joint report and record a video.

**Group 116:**
- Johnsen Villocino (N10530819) — DistilBERT fine-tuning *(this repo)*
- Alexander Venn — SVM + char n-gram TF-IDF
- Mackenzie Stone (N11608081) — BiLSTM + GloVe

**Deadline:** Sunday 11:59 PM (June 1, 2026)

## GSD Workflow

This project uses GSD for planning and execution. Planning artifacts are in `.planning/`.

```
/gsd-plan-phase 1    ← start here
/gsd-plan-phase 2
/gsd-plan-phase 3
/gsd-plan-phase 4
```

**Current state:** Roadmap approved, ready to execute Phase 1.

## Critical Constraints

- **Shared data first:** Phase 1 builds `shared_data.csv` — all three methods depend on identical splits. Johnsen builds and shares it with the group before individual implementation begins.
- **No data leakage:** Never use email headers (To/From/Subject), signatures, forwarded content, or email addresses as features. Strip them in preprocessing.
- **Reproducibility:** All splits use `random_state=42`. Fix this everywhere.
- **Student hardware:** No HPC. DistilBERT must train on CPU or consumer GPU.
- **Record timing:** Wrap every training loop with `time.time()`. The rubric requires it.

## Repository Layout

```
notebook.ipynb               ← main notebook
enron_mail_20150507/         ← raw Enron maildir data
.planning/
  PROJECT.md                 ← project context and goals
  REQUIREMENTS.md            ← 14 v1 requirements with REQ-IDs
  ROADMAP.md                 ← 4 phases with success criteria
  STATE.md                   ← current phase tracker
  config.json                ← GSD settings (YOLO mode, parallel)
  research/
    SUMMARY.md               ← recommended 3-method suite + pitfalls
    STACK.md                 ← library versions and setup
    FEATURES.md              ← feature strategy per method
    ARCHITECTURE.md          ← model architectures and benchmarks
    PITFALLS.md              ← 26 specific pitfalls to avoid
```

## Method Assignment

| Method | Person | Library |
|--------|--------|---------|
| SVM + char n-gram TF-IDF | Alex | scikit-learn |
| BiLSTM + GloVe | Mackenzie | PyTorch |
| DistilBERT fine-tuning | Johnsen | HuggingFace transformers |

## Key Technical Decisions

- Top 10–15 authors (minimum 200 emails each after filtering)
- Strip everything after `"-----Original Message-----"` (forwarded content)
- Filter emails < 20 words (auto-generated content)
- 70/15/15 stratified split, `random_state=42`, saved to `shared_data.csv`
- DistilBERT: `DistilBertForSequenceClassification`, AdamW LR=2e-5, 3–5 epochs, 256 token max
- Evaluate: accuracy, macro F1, per-class precision/recall, training time, inference time, majority baseline
