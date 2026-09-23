# Clause-Level Risk Detection in Commercial Contracts

Multi-label classification of contract clauses into six risk-relevant categories,
using the [CUAD v1](https://www.atticusprojectai.org/cuad) corpus of 510
expert-annotated commercial contracts.

**Labels:** Cap on Liability · Non-Compete · License Grant · Audit Rights ·
Termination for Convenience · Insurance

## Status

The repository is being rebuilt as a research project. The FastAPI / Next.js /
Postgres / Redis scaffold was removed: it served none of the project's
deliverables and none of its grading criteria, while still needing maintenance.

What lands next, one pull request per layer:

| Layer | Contents |
|---|---|
| Data pipeline | CUAD span extraction, sliding-window chunking, document-level splits |
| Dataset analysis | Length distribution, class balance, label co-occurrence |
| CNN baseline | TextCNN over trainable embeddings |
| Pretrained models | BERT, Legal-BERT, Longformer behind a shared training loop |
| Evaluation | Per-class P/R/F1, PR-AUC, chunk-to-document pooling, model comparison |
| Notebooks | Colab runners for each experiment |

## Layout

```text
src/legal_risk_classifier/   training, evaluation and inference package
data/                        CUAD v1 corpus (JSON, per-contract TXT, master CSV)
```

License: [MIT](./LICENSE)
