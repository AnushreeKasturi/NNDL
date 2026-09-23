# Deploying

Two surfaces, deployed separately because they have different needs: the
dashboard is a static file, the demo needs a Python process holding a model.

## Dashboard — Hugging Face Space (static)

```bash
pip install -e ".[hub,analysis]"
huggingface-cli login

python -m legal_risk_classifier.export_dashboard \
    --run_dir outputs/cnn --repo_id <your-username>/clause-risk-review --push
```

Rerun it after any training run; the page is generated, so the published
dashboard picks up the new numbers with nothing edited by hand.

### Why not GitHub Pages

GitHub Pages was tried first and is the wrong tool here. A custom domain
attached to a user-level Pages site makes *every* project repository serve as a
subpath of that domain, so enabling Pages on this repository published the
project under a personal portfolio. There is no per-repository way to opt out of
the domain while keeping Pages.

A static Space avoids that entirely: its own URL, no relationship to any other
site, free, and it uses the same account as the model and the demo.

If Pages was ever enabled on this repository, turn it off under
Settings -> Pages -> Source: None. The REST API refuses to deactivate it.

## Demo — HuggingFace Spaces

The Space holds a checkpoint and a six-line `app.py`; the model code is
installed from this repository, so the demo cannot drift from the pipeline the
results were measured on.

```bash
pip install huggingface_hub
huggingface-cli login                      # needs a token with write access

huggingface-cli repo create nndl-clause-risk --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<your-username>/nndl-clause-risk
cd nndl-clause-risk

cp ../NNDL/deploy/space/{app.py,requirements.txt,README.md} .
mkdir -p run
cp ../NNDL/outputs/cnn/{model.pt,vocab.json,metrics.json,training.json} run/

git lfs install && git lfs track "run/model.pt"
git add -A && git commit -m "Deploy clause risk demo" && git push
```

The checkpoint is about 16 MB, which is why it goes through Git LFS. Free
Spaces run on CPU; the CNN scores a contract in well under a second there.

To deploy a transformer instead, point `run/` at that run directory — but a
fine-tuned BERT checkpoint is roughly 440 MB, and a free CPU Space will be slow
enough to notice.

## Model — Hugging Face Hub

A training run saves what the training loop needs, which is not what a stranger
needs: the transformer checkpoints are wrapper state dicts with `backbone.`
prefixed keys, so uploading one unchanged hands people a file they cannot load.

`legal-risk-export-hf` converts a run into the standard layout, names the six
outputs in the config so the model is self-describing, and writes a model card
carrying that run's real numbers, its per-label thresholds and its limitations.

```bash
pip install -e ".[hub]"
huggingface-cli login                      # a token with write access

legal-risk-export-hf --run_dir outputs/legal_bert \
    --repo_id <your-username>/cuad-clause-risk-legal-bert --push
```

Leave `--push` off to inspect `outputs/<run>/hf` first.

A transformer exported this way loads with plain `transformers`, no repository
required, and reproduces the training wrapper exactly:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(name)
model.config.id2label      # {0: 'Cap on Liability', 1: 'Non-Compete', ...}
```

The TextCNN is not a transformers architecture, so its card says so and points
at `CNNRuntime` rather than promising a `from_pretrained` that would fail.

## Why not one deployment for both

A static page costs nothing to host and never sleeps. A Space holding a model
sleeps after inactivity and takes time to wake. Keeping the results on Pages
means the thing most people open is always instant, and the demo is there for
anyone who wants to try their own contract.
