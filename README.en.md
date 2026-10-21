# MAVEN-ERE Joint RoBERTa-base Training

## Latest verified single-joint improvement: positive4 interim snapshot

`positive4-256-seed42-snapshot-u750/` has CPU overall **51.7822** on the full710-document validation set: **+0.2825points** over the preceding positive2 single, and **+3.0684points** over the original root single. It contains one encoder and all task heads, one forward per document. Full public CLI and independent AutoModel weights, predictions and metrics match exactly.

| F1 % | Original root single | Preceding positive2 single | Positive4 CPU single | Delta vs preceding |
|---|---:|---:|---:|---:|
| coreference_mean_f1 | 90.08368906 | 90.56372401 | 90.20028320 | -0.36344081 |
| temporal_f1 | 53.08093519 | 54.42049630 | 52.45616097 | -1.96433534 |
| causal_f1 | 27.33665533 | 31.07405202 | 33.28013849 | +2.20608647 |
| subevent_f1 | 24.35389549 | 29.94070344 | 31.19226171 | +1.25155827 |
| overall_f1 | 48.71379377 | 51.49974394 | 51.78221109 | +0.28246715 |

Selected at epoch9/update750 of a still-running formal100-epoch run. All15formal experiments continue. The root still loads the old baseline: select this subdirectory and use default padding without `--reference-padding`. [Full metrics, training and inference](docs/POSITIVE4_SINGLE_SNAPSHOT.md).

CPU/publicCLI and independentAutoModel match exactly on all710documents, predictions, metrics and state tensors. CPU overall51.78221109033361 and temporal52.456160965460995 differ from saved GPU51.78218842947946 and52.45607032204441: one validation document has one additional saved-GPU BEFORE pair that CPU omits. Full document/pair and metric differences are in validation.json. GPU logits are unavailable; no numerical margin or exact cause is asserted, and no thresholds were changed.

## Latest verified single-joint improvement: positive2 interim snapshot

`positive2-256-seed42-snapshot-u750/` has CPU overall **51.4997** on the full710-document validation set: **+2.0301points** over the preceding corrected512 single, and **+2.7860points** over the original root single. It contains one encoder and all task heads, one forward per document. Full public CLI and independent AutoModel weights, predictions and metrics match exactly.

| F1 % | Original root single | Preceding corrected512 single | Positive2 CPU single | Delta vs preceding |
|---|---:|---:|---:|---:|
| coreference_mean_f1 | 90.08368906 | 89.92380233 | 90.56372401 | +0.63992168 |
| temporal_f1 | 53.08093519 | 53.10557320 | 54.42049630 | +1.31492311 |
| causal_f1 | 27.33665533 | 31.46712390 | 31.07405202 | -0.39307189 |
| subevent_f1 | 24.35389549 | 23.38204593 | 29.94070344 | +6.55865751 |
| overall_f1 | 48.71379377 | 49.46963634 | 51.49974394 | +2.03010760 |

The parent formal run completed all100epochs/9,200updates/36,500microsteps/184validations. Its final best remains epoch9/update750, not an epoch100 weight; all8model/source/tokenizer hashes are unchanged. All15formal experiments continue (6/15complete). The root still loads the old baseline: select this subdirectory and use default padding without `--reference-padding`. [Full metrics, training and inference](docs/POSITIVE2_SINGLE_SNAPSHOT.md).

## Verified single-joint improvement: interim update1500 snapshot

`corrected-512-seed42-snapshot-u1500/` contains one complete joint model. On the same710-document validation protocol, overall49.4696 versus oldsingle48.7138 (+0.7558points). Causal+4.1305, temporal+0.0246, coreference mean-0.1599, subevent-0.9718points. PublicCLI and independent AutoModel full710 predictions/metrics and all state tensors match exactly.
Its parent run has now completed all100epochs/9,200updates; the selected best checkpoint remains epoch17/update1500. The full15-run campaign continues. [Commands and detailed evidence](docs/CORRECTED_SINGLE_SNAPSHOT.md). The Hub root still loads the oldbaseline; select the new subdirectory and use default padding (do not add `--reference-padding`).


## Verified interim reference models (2026-10-20)

[Model weights](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint),
[dataset mirror](https://huggingface.co/datasets/MaYiding/MAVEN-ERE), and
[detailed results and four-model inference commands](docs/REFERENCE_CHECKPOINTS.md).

The Hub root remains the older single joint checkpoint (validation overall 48.7138).
New `reference-seed42/{coreference,temporal,causal,subevent}/` directories each contain
one complete joint model with its own encoder and all four heads. Combining their
selected prediction fields uses four complete models and four forwards per document.
Its actual CPU validation overall is 50.0361814071 (+1.3224 points versus the older
single model). This does not establish equal-cost or single-model improvement.
All 710 validation documents passed legacy/converted/AutoModel and documented CLI
prediction/metric checks. CPU and original GPU scores are reported separately;
one temporal relation differs across devices. The reference run completed 100 epochs.
The full 15-run, 100 epochs-per-run campaign and five-seed analysis continue after this
interim publication; no hidden-test scores or completed multi-seed reproduction are claimed.


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
