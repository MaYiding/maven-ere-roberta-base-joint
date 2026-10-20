# MAVEN-ERE Joint RoBERTa-base Training

This repository adapts the official THU-KEG/MAVEN-ERE four-task joint baseline with
complete resume state, official-protocol validation and independent Hugging Face
exports. See [README.md](README.md) for Chinese instructions and upstream documentation.

The input must contain pre-identified event mentions and temporal expressions.
This model predicts within-document coreference, temporal, causal and subevent
relations; it is not an end-to-end event detector. Subevent pairs are `[parent, child]`.

## Trained checkpoint (2026-10-20)

[Hugging Face weights](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint) · [Run details and usage](docs/TRAINING_RUN.md)

Seed 42 training stopped after 23 epochs, using microbatch **8** and accumulation **4**.
On all 710 validation documents with gold mentions and annotated TIMEX, one best checkpoint
achieved coreference mean F1 **90.08**, temporal **53.08**, causal **27.34**, subevent **24.35**,
and overall **48.71**. Reloading the release model and reevaluating reproduced every metric exactly.
No hidden-test score is claimed. Repositories are initially private.
Exact training source: `369d9df4262f3fb4c22073d8c6c8021fa78698f8`; later commits add release documentation and evidence.


## Training

Use Python 3.11 and a BF16-capable NVIDIA GPU. Run these commands on the training server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python scripts/prepare_data.py
python scripts/download_base.py
python -m pytest -q tests
python train_joint.py --base_model base_model --data_dir data/MAVEN_ERE \
  --output_dir runs/roberta-joint-seed42 --cache_dir runs/tokenization-cache \
  --batch_size 4 --grad_accum 8 --max_epochs 100 --seed 42
```

Use a separate output directory with `--smoke` before full training. It exercises
real training documents, positive labels, training-state restoration and HF
roundtrips. Full training can reuse its tokenization cache, but not its run directory.

Defaults are microbatch 4, accumulation 8, nominal effective batch 32 documents,
512-token chunks, BF16, gradient checkpointing, seed 42 and up to 100 epochs.
Encoder AdamW uses lr 2e-5, eps 1e-6 and no weight decay; head Adam uses lr 3e-4.
Encoder scheduling uses actual optimizer updates, 200 warmup updates and linear
decay. Loss weights are 0.4/2/4/4 for coreference/temporal/causal/subevent.
Validation runs each epoch on 710 documents; patience is 10 evaluations.

Coreference NLL is averaged over mentions and then documents. Relation CE is
averaged over valid pairs per microbatch. Thus 4x8 differs in pair weighting from
the upstream README's 8x4. Scorer width 200 follows released code, while the paper
appendix states 150. This is a documented adaptation, not an exact reproduction.

## Resume, validation and prediction

Append `--resume` to the original command. `latest.pt` contains both optimizers,
scheduler, epoch, fixed sample order, next microbatch and Python/NumPy/Torch/CUDA
RNG states at an optimizer boundary. Data, batch and schedule signatures must
match. SIGTERM/SIGINT requests a checkpoint and exit at the next update boundary.

`best_joint` is a symlink to one complete model selected by upstream overall F1,
including safetensors, config, custom source and tokenizer. Its validation scores
are copied to `best_dev_metrics.json`. `progress.json` tracks the current run.
Training symmetrizes SIMULTANEOUS/BEGINS-ON; official evaluation preserves raw
annotation directions. Both protocols are explicitly recorded.

```bash
python train_joint.py --predict-only --checkpoint runs/roberta-joint-seed42/best_joint \
  --data-dir data/MAVEN_ERE --split test \
  --prediction-file runs/roberta-joint-seed42/test_predictions.jsonl
```

Full training automatically predicts the test candidates using the best model.
Test gold labels are hidden; no test score is claimed. `AutoModel.from_pretrained`
with `trust_remote_code=True` loads the complete export after source review. Its
forward requires input_ids, attention_mask, event_spans, splits and events_idx.
Use the prediction entry point for preprocessing and JSONL decoding.

## Publishing later

`scripts/package_release.py` produces a `huggingface/` directory, training-code
archive and SHA-256 manifest after successful full training. It requires the
launch-time `source_manifest.json`; commit code and run `python scripts/snapshot_source.py`
to create it before training. It never creates remote
repositories or uploads files. Corpus text, environments, optimizer states and
caches are excluded from the public code repository.

Upstream commit: `ac81a9711a69f43f55bfbc50b3bb573fd11c64b0`.
Retain GPL-3.0 and attribution notices. RoBERTa's MIT attribution is recorded in
[UPSTREAM_NOTICE.md](UPSTREAM_NOTICE.md).
