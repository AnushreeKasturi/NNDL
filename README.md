# Clause Risk Review

Multi-label classification of commercial contract clauses into six risk-relevant
categories, on [CUAD v1](https://www.atticusprojectai.org/cuad) — 510 contracts
annotated by lawyers at The Atticus Project.

**Labels:** Cap on Liability · Non-Compete · License Grant · Audit Rights ·
Termination for Convenience · Insurance

```bash
pip install -e ".[analysis]"
legal-risk-prepare                     # CUAD spans + document-level splits
legal-risk-analyze                     # dataset report and figures
legal-risk-train-cnn --epochs 12       # the CNN baseline
legal-risk-evaluate --run_dir outputs/cnn --split test
```

Everything runs from the command line, so a notebook never holds logic. The
[notebooks](./notebooks) are Colab runners that install and call these commands.

## Results so far

Held-out test split: 76 contracts, 3,314 windows, never seen in training.

| Model | Window F1 | Document F1 | Status |
|---|---:|---:|---|
| TextCNN | **0.663** | **0.840** | trained |
| BERT | — | — | awaiting a GPU run |
| Legal-BERT | — | — | awaiting a GPU run |
| Longformer | — | — | awaiting a GPU run |

Per class, CNN, window level:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Insurance | 0.877 | 0.814 | 0.844 | 70 |
| License Grant | 0.806 | 0.767 | 0.786 | 146 |
| Cap on Liability | 0.760 | 0.731 | 0.745 | 104 |
| Audit Rights | 0.651 | 0.793 | 0.715 | 87 |
| Termination for Convenience | 0.491 | 0.540 | 0.514 | 50 |
| Non-Compete | 0.400 | 0.351 | 0.374 | 57 |

Rarity and difficulty track each other: the weakest label is also the rarest.

**Run-to-run variance is about ±0.03 macro F1** on identical settings, from float
nondeterminism on GPU. A difference smaller than that is not evidence of
anything, and should be run across several seeds before it is claimed.

## The two problems

### Problem 1 — how should legal text be represented?

BERT against Legal-BERT. They share an architecture, a parameter count (109,486,854
each), a context window and a chunk width, differing only in pretraining corpus, so
any gap between them measures domain adaptation and nothing else.

The hypothesis is specific: the gain should concentrate in labels carried by legal
vocabulary and be smallest for labels carried by ordinary English. Read the
per-class columns, not the macro score. See [docs/model-selection.md](./docs/model-selection.md).

### Problem 2 — how should whole contracts be represented?

Contracts run to a median of 5,039 words and a maximum of 47,733.

**97% exceed BERT's 512-token window. 70% exceed Longformer's 4,096 as well.**

So long context does not remove the need for chunking — both models train on
chunks. What differs is how much context one window holds: ~300 words against
~2,600. The experiment measures whether a clause is easier to recognise with more
contract around it, which is narrower and more defensible than "long-context models
handle long documents".

A null result is a real finding. Most of these clauses are locally signalled:
*"shall maintain insurance"* is recognisable from its own sentence.

## Dataset challenges

Generated in full by `legal-risk-analyze` into
[docs/dataset-report.md](./docs/dataset-report.md).

| Challenge | Evidence | Consequence |
|---|---|---|
| Documents far exceed context windows | 97% over 512 tokens; longest 47,733 words | Chunking is mandatory for every model |
| Severe negative dominance | 85% of windows carry no label | Weighted loss, tuned thresholds, no accuracy |
| Sparse positives per label | as few as 301 positive windows | High variance; report per-class support |
| Labels co-occur | 275 contracts carry Cap on Liability, 182 of those also License Grant | Six sigmoids, not a softmax |
| Annotation is span-level, supervision is window-level | 3,123 spans mapped by overlap | The overlap rule is a modelling choice |
| Legal English is domain-specific and OCR-noisy | irregular whitespace and page furniture throughout | Motivates the Legal-BERT comparison |

The positive rate per label runs 1.5%–4.2% against every window a classifier sees.
A balanced loss needs weights of 22×–68×.

## How it works

```text
CUAD_v1.json
  └─ prepare    spans extracted by category, splits assigned per document
       └─ chunking   overlapping word windows; a window takes a label when it
       │             meaningfully overlaps an annotated span
       └─ training   one shared loop for every model
            └─ evaluate   window-level and document-level, thresholds from validation
                 └─ compare / dashboard / demo
```

Four decisions worth knowing before reading the code:

**Windows are counted in words, not tokens.** A word window is tokenizer-agnostic,
so all four models train on identical boundaries and a score difference is a
difference between models rather than between tokenizers.

**Splits are per document and persisted.** Splitting per window would put
overlapping text on both sides. The assignment is written to disk so every model is
scored on an identical test set.

**Thresholds are tuned per label on validation, never on test.** They range 0.22
to 0.83, because each label has its own positive rate and a single 0.5 cut is
arbitrary.

**A contract is flagged when its strongest window clears the threshold.** The task
is existential. Averaging across windows instead drops document macro F1 from 0.77
to 0.62 on the same predictions.

## Layout

```text
src/legal_risk_classifier/
  cuad.py chunking.py splits.py      data layer — standard library only
  analysis.py figures.py analyze.py  dataset report
  textcnn.py vocab.py                CNN baseline
  pretrained.py                      BERT / Legal-BERT / Longformer
  training.py datasets.py metrics.py one training loop for every model
  pooling.py evaluate.py compare.py  window and document scoring
  attribution.py runtime.py demo.py  interpretability and the browser demo
  export_dashboard.py                results page data
notebooks/                           Colab runners
docs/                                generated reports and the model justification
tests/                               64 checks, no pytest required
```

## Tests

```bash
python  tests/test_data_pipeline.py      # 12 — standard library only
python  tests/test_analysis.py           #  9 — standard library only
python  tests/test_model_layer.py        # 14 — needs torch
python  tests/test_pretrained.py         # 11 — needs torch, runs offline
python  tests/test_pooling.py            #  7 — needs torch
python  tests/test_interpretability.py   # 11 — needs torch
```

No framework. Each file runs as a script and prints what passed.

Tests carry the reasoning behind decisions, so a decision cannot be quietly
reversed: that a whole CUAD question is not a category, that a clause must not
label windows which do not contain it, that mean pooling buries a true positive,
that BERT and Legal-BERT differ only in checkpoint, and that metrics expose an
all-negative model which accuracy would rate above 95%.

## Deployment

The results dashboard is a static page published to GitHub Pages, regenerated
from whatever runs exist:

```bash
./scripts/publish-dashboard.sh outputs/cnn
```

The demo is a HuggingFace Space holding a checkpoint and a six-line entry point
that installs the model code from this repository, so it cannot drift from the
pipeline the results were measured on. See [deploy/](./deploy/README.md).

## Demo

```bash
pip install -e ".[demo]"
legal-risk-demo --run_dir outputs/cnn          # --share for a public link in Colab
```

Paste a contract; get the six labels scored against their own thresholds, the
flagged passages with confidence and character offsets, and the phrases that
triggered each flag.

Attribution works because a convolutional filter is a literal n-gram detector:
max-over-time pooling records which position fired it hardest, so the explanation
is an actual phrase from the contract. It already earns its place — asked why it
flagged Cap on Liability, the model returns the same `product liability insurance`
phrases it uses for Insurance, meaning it keys on the shared word "liability"
rather than on capping language.

## Team deliverables

| Owner | Deliverable | Where |
|---|---|---|
| A — data | Cleaned, chunked dataset + exploration | `cuad.py`, `chunking.py`, `splits.py`, notebook 1 |
| B — embeddings | BERT vs Legal-BERT + justification | `pretrained.py`, `docs/model-selection.md`, notebook 3 |
| C — long documents | Longformer + pooling | `pretrained.py`, `pooling.py`, notebook 4 |
| D — evaluation | Metrics, interpretability, demo, report | `metrics.py`, `attribution.py`, `demo.py`, notebook 5 |

License: [MIT](./LICENSE)
