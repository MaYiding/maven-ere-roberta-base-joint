# Completed training run: RoBERTa joint, seed 42

This page records the initial single-checkpoint baseline. The subsequent full reproduction
campaign and its interim task-selected checkpoint releases are documented separately in
[REFERENCE_CHECKPOINTS.md](REFERENCE_CHECKPOINTS.md). Publishing an interim result does
not end the remaining experiments.

The model is available at [Hugging Face](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint).
Repository visibility is managed by the current release; authenticate if a repository is still private.

Training completed on 2026-10-20 with early stopping after 23 epochs and 2,116 optimizer updates,
using a single RTX 4090 D, BF16, microbatch 8 and gradient accumulation 4.
The frozen training commit is `369d9df4262f3fb4c22073d8c6c8021fa78698f8`.
The first publication commits added documentation and evidence. Later reproduction commits
also add the isolated reference runner, controlled candidate settings and export/evaluation tools;
they do not change the frozen source or weights of this baseline.

## Validation

All scores are percentages on the 710-document validation split, using gold event mentions
and annotated TIMEX. One complete checkpoint was selected by official overall F1.
The packaged model was reloaded through Transformers AutoModel and reevaluated on
all 710 documents; every recorded metric matched exactly.

| Metric | F1 |
|---|---:|
| Coreference B3 | 97.6302 |
| Coreference CEAF-e | 97.3049 |
| Coreference MUC | 78.2982 |
| Coreference BLANC | 87.1015 |
| Coreference mean | 90.0837 |
| Temporal | 53.0809 |
| Causal | 27.3367 |
| Subevent | 24.3539 |
| Overall | 48.7138 |

Overall is the mean of coreference mean F1 and the other three task F1 scores.
Official temporal scoring retains raw annotated directions; training symmetrizes
SIMULTANEOUS and BEGINS-ON. Test labels are hidden. Predictions were generated for
857 test documents, but no test score is claimed.

## Reproduce and use

Follow the main README for data and base-model download. Use the frozen training
commit above for exact source reproduction and `requirements-lock.txt` for the captured
environment. For this published run, override the CLI defaults with:

```bash
python train_joint.py --base-model base_model --data-dir data/MAVEN_ERE \
  --output-dir runs/roberta-joint-seed42 --cache-dir runs/tokenization-cache \
  --batch_size 8 --grad_accum 4 --max_epochs 100 --seed 42
```

```bash
hf download MaYiding/maven-ere-roberta-base-joint --local-dir checkpoint
python train_joint.py --predict-only --checkpoint checkpoint \
  --data-dir data/MAVEN_ERE --split valid --prediction-file predictions/valid.jsonl
```

The model requires supplied event mentions and temporal expressions. It does not
detect events from raw text. It predicts within-document relations in English.
Add `--device cpu --no-bf16` for CPU inference.

[Detailed scores and evidence](../results/roberta-joint-seed42/) include source hashes,
release checksums and full-validation verification. The inference weights are
504,795,436 bytes, SHA-256 `6a47ebc48a2d4c5c45714bfbe4a59791c4f182f521341631b0e93e116973480a`.
Full data, environments, caches and the optimizer/RNG resume checkpoint are excluded.
The code and checkpoint use GPL-3.0, with the RoBERTa MIT notice retained.
