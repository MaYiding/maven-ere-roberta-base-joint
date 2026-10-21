# Positive2 seed42: validated three-model task combination

## Verified positive2 seed42 task combination: three complete joint models

The optional `positive2-256-seed42-task-combination/` package scores **52.00153038** CPU overall on all710 official validation documents, **+1.96534897 points** over the published reference42 four-model CPU combination. It uses **three independent complete RoBERTa-base encoders with all task heads and three forwards per document**. Temporal and subevent share the existing `positive2-256-seed42-snapshot-u750/` checkpoint. Two new complete checkpoints supply coreference and causal fields. No head splicing or weight averaging.

| F1 % | Reference42 CPU, four models | Positive2 CPU, three models | Delta points |
|---|---:|---:|---:|
| MUC | 79.80519481 | 79.56058589 | -0.24460892 |
| B3 | 97.74255953 | 97.78058903 | +0.03802950 |
| CEAF-e | 97.47629174 | 97.47092468 | -0.00536705 |
| BLANC | 88.09889190 | 88.50683704 | +0.40794514 |
| Coreference mean | 90.78073449 | 90.82973416 | +0.04899967 |
| Temporal | 53.10764361 | 54.42049630 | +1.31285269 |
| Causal | 31.09946825 | 32.81518762 | +1.71571936 |
| Subevent | 25.15687927 | 29.94070344 | +4.78382416 |
| Overall | 50.03618141 | 52.00153038 | +1.96534897 |

The reference uses predict-only reference-padding; positive2 uses default padding trimming. Both results use CPU FP32, the same710-document raw-direction official valid protocol, gold event mentions and annotated TIMEX. The MUC and CEAF-e declines are retained above. The published strongest single remains positive4 CPU51.78221109; this combination is a separate inference option with three encoders, not a new single-model score.

All three complete checkpoints passed actual public CLI and independent AutoModel full710 predictions, metrics and state equality. The prior joint-model gates were reused only after exact checkpoint identity, all8 file hashes and saved full710 parsed predictions matched; only the other two complete models were newly evaluated. The public combination script also reproduced all710 combined predictions exactly. Original GPU and actual CPU outputs are equal for all three retained checkpoints. All four coreference metrics and all three relation metrics, checkpoint mappings and hashes are in validation.json and combination.json.

The seed42 parent completed100epochs/36500microsteps/9200updates/184validations, FP32/batch8/accumulation4, positive relation weight2.0, no early stopping. All15 formal runs have now completed actual100epochs with fresh counter and history audits. Final five-seed CPU statistics, paired confidence intervals, remaining reloads and release readbacks continue. Earlier sections retain historical release-time snapshots. This is a selected single-seed validation result, not a paper hidden-test score or completed five-seed claim. Existing versions remain available. GPL-3.0, RoBERTa MIT and upstream attribution are retained; the existing MAVEN-ERE data mirror preserves its MAVEN/Wikipedia licenses and all4480 source attributions.

[Three-model download and inference instructions](https://github.com/MaYiding/maven-ere-roberta-base-joint/blob/main/docs/POSITIVE2_TASK_COMBINATION.md). Pin the immutable HF release revision recorded in the accompanying Git publication manifest. The root model remains the original single; select the documented subdirectories explicitly.

Prediction equality refers to parsed JSON documents and canonical prediction hashes; raw JSON key ordering is not an equality requirement.

## Download and infer

Use the immutable HF revision in `results/positive2-256-seed42-task-combination/publication_manifest.json` (filled with the published commit). The two new directories and the existing joint directory contain full checkpoints; keep their original configuration and tokenizer files.

```bash
REVISION=1b308bc516da9539fe837cff95a0677c4659a6ae
hf download MaYiding/maven-ere-roberta-base-joint --revision "$REVISION" \
  --include 'positive2-256-seed42-task-combination/**' 'positive2-256-seed42-snapshot-u750/**' \
  --local-dir checkpoints
mkdir -p predictions/positive2-seed42
python train_joint.py --predict-only --checkpoint checkpoints/positive2-256-seed42-task-combination/coreference \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --seed 42 \
  --prediction-file predictions/positive2-seed42/coreference.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoints/positive2-256-seed42-snapshot-u750 \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --seed 42 \
  --prediction-file predictions/positive2-seed42/joint.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoints/positive2-256-seed42-task-combination/causal \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --seed 42 \
  --prediction-file predictions/positive2-seed42/causal.valid.jsonl
python scripts/combine_task_predictions.py \
  --coreference-predictions predictions/positive2-seed42/coreference.valid.jsonl \
  --temporal-relations-predictions predictions/positive2-seed42/joint.valid.jsonl \
  --causal-relations-predictions predictions/positive2-seed42/causal.valid.jsonl \
  --subevent-relations-predictions predictions/positive2-seed42/joint.valid.jsonl \
  --expected-documents 710 --output predictions/positive2-seed42/combined.valid.jsonl
```

Use default trimming, without `--reference-padding`. Inputs require annotated event mentions, TIMEX and token locations; this is not raw-text event detection. Keep the three full original predictions. For unscored test predictions change all three splits to test and expected documents to857; hidden-test relation gold is unavailable. Do not score the unannotated test as validation.

The training implementation differs from the author runner in corrected dummy-label loss, scheduling/resume behavior and positive relation weight2.0. Coreference selection uses upstream-any; temporal/causal/subevent select their task F1, and the joint snapshot selects overall F1. The fixed candidate winner was selected by maximin four-task delta, then overall, then fixed candidate order. A task-selected combination is separate from one joint-overall-selected checkpoint. Sources and exact training/data/base hashes accompany each new model. The original reference CPU/GPU device difference remains recorded in its own release; it is not rounded away or attributed to unmeasured GPU logits.

See [validation.json](../results/positive2-256-seed42-task-combination/validation.json), [combination.json](../results/positive2-256-seed42-task-combination/combination.json), [publication manifest](../results/positive2-256-seed42-task-combination/publication_manifest.json), and [licenses and upstream attribution](../UPSTREAM_NOTICE.md).
