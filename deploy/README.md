# Deploying

Two surfaces, deployed separately because they have different needs: the
dashboard is a static file, the demo needs a Python process holding a model.

## Dashboard — GitHub Pages

The page is generated from the current runs, so publishing means regenerating
and replacing one file:

```bash
./scripts/publish-dashboard.sh outputs/cnn
```

It builds the `gh-pages` branch with git plumbing rather than checking it out,
so it is safe to run from a dirty working tree.

**One-time setup** (needs repository admin): Settings → Pages → Source:
*Deploy from a branch* → Branch: `gh-pages`, folder `/ (root)` → Save.

Live at https://anushreekasturi.github.io/NNDL/ within a minute of saving.

Rerun the script after any new training run and the published page picks up the
new numbers. Nothing is edited by hand.

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

## Why not one deployment for both

A static page costs nothing to host and never sleeps. A Space holding a model
sleeps after inactivity and takes time to wake. Keeping the results on Pages
means the thing most people open is always instant, and the demo is there for
anyone who wants to try their own contract.
