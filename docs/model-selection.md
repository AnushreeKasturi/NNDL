# Why these models

Four models, chosen so that each comparison isolates exactly one variable.
Adding a fourth architecture would add a result; it would not add an answer.

| Model | Role | Pretraining | Context | Parameters |
|---|---|---|---|---|
| TextCNN | Baseline floor | none — embeddings learned from 358 contracts | chunk | ~4M |
| `bert-base-uncased` | General-domain control | BooksCorpus + English Wikipedia, ~3.3B words | 512 | 110M |
| `nlpaueb/legal-bert-base-uncased` | Domain-adapted | 12GB of legislation, court cases and contracts | 512 | 110M |
| `allenai/longformer-base-4096` | Long context | RoBERTa, continued on long documents | 4096 | 149M |

## The two experiments

The models are not four attempts at the same thing. They are arranged as two
controlled comparisons, each answering one of the project's core problems.

### Problem 1 — does legal pretraining help?

**BERT vs Legal-BERT.** These two share an architecture, a parameter count, a
tokenizer vocabulary size, a context window and a training recipe. They differ
in one respect: what they were pretrained on. Any difference in score is
therefore attributable to the pretraining corpus, which is precisely the
question. A comparison against, say, RoBERTa would confound domain with
architecture and answer nothing cleanly.

The hypothesis is specific rather than general. Legal-BERT's advantage should
concentrate in the labels whose signal is vocabulary the general model has
barely seen — "indemnify", "liquidated damages", "force majeure", "no event
shall" — and should be smallest for labels signalled by ordinary English such
as Insurance. A uniform improvement across all six labels would be evidence
that something other than domain knowledge is responsible.

The CNN baseline is what makes this measurable. A pretrained model that cannot
beat convolutions over embeddings learned from 358 contracts is not earning
its 110M parameters, and the CNN's per-class results give each comparison a
floor rather than only a relative ranking.

### Problem 2 — does longer context help?

**BERT (512) vs Longformer (4096).** Full self-attention costs O(n²) in
sequence length, which is why BERT stops at 512 tokens. Longformer replaces it
with sliding-window local attention plus global attention on selected tokens,
reducing the cost to O(n) and making 4096 tokens affordable.

The dataset report reframes what this experiment can show. 97% of contracts
exceed 512 tokens, but **70% also exceed 4096**, so Longformer does not remove
the need for chunking — both models are trained on chunks. What changes is how
much context a single chunk holds: roughly 300 words for BERT against roughly
2,600 for Longformer. The experiment therefore measures whether a clause is
easier to recognise with more surrounding contract around it, which is a
narrower and more honest claim than "long-context models handle long
documents".

There is a real chance the answer is no, and the design should be able to say
so. Most of these clauses are locally signalled: "shall maintain insurance" is
recognisable from its own sentence. If Longformer does not win, that is a
finding about the task, not a failed experiment — and it comes with a cost
argument attached, since Longformer is 35% larger and materially slower to
train.

## Why `[CLS]` gets global attention

Longformer's local attention window cannot pool a sequence on its own: no
position sees the whole chunk. The classification head reads the `[CLS]`
representation, so `[CLS]` is marked global — it attends to every token and
every token attends to it. Without this the head is reading a position that
only ever saw its own neighbourhood, and the model's long context is wasted
on it.

## Why chunk width differs between models, and nothing else does

Every model is trained through the same loop, on the same document-level
splits, with the same weighted loss and the same per-label threshold tuning.
The one deliberate difference is chunk width, because it is bounded by the
context window — and that bound is the variable Problem 2 is about. Holding
chunk width constant across models would make the Problem 2 comparison
impossible to run, since Longformer's extra capacity would simply go unused.

## Models considered and rejected

**RoBERTa, DeBERTa, ELECTRA.** Stronger general-domain encoders, but adding
one changes architecture and pretraining together. They would improve a
leaderboard and blur the experiment.

**CaseHOLD-BERT, legal RoBERTa variants.** A second legal model measures
variance between legal models, which is not one of our questions.

**A generative LLM prompted for clause extraction.** A different paradigm with
no fine-tuning on our splits, so its score would not be comparable to the
others — and the assignment asks for trained classifiers.

**Hierarchical BERT over chunk embeddings.** The obvious next step if chunk
pooling proves to be the bottleneck. It is deferred rather than dismissed:
worth building only once the evaluation shows that document-level pooling,
not chunk-level classification, is what limits the score.
