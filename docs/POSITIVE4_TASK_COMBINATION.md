# Positive4 seed42: three complete joint models

Positive4 seed42 CPU Overall **52.14502736**, +0.14349698 points versus preceding positive2 CPU52.00153038 on the same710-document valid protocol. Three independent complete joint encoders, three forwards per document. Coreference uses one new full model; temporal/causal share another; subevent reuses the existing positive4 joint single. No head splicing or averaging. Both combinations use CPUFP32, batch8, default trimming. All task declines are disclosed. The strongest single remains positive4 CPU51.78221109, one encoder/all heads/one forward. Fixed five-seed winner remains positive2; positive4 is a posthoc single-seed release review.

| F1 % | Positive2 CPU | Positive4 CPU | Delta points |
|---|---:|---:|---:|
| MUC | 79.56058589 | 78.52802161 | -1.03256428 |
| B3 | 97.78058903 | 97.73238598 | -0.04820306 |
| CEAF-e | 97.47092468 | 97.37842644 | -0.09249825 |
| BLANC | 88.50683704 | 87.95527399 | -0.55156305 |
| Coreference mean | 90.82973416 | 90.39852700 | -0.43120716 |
| Temporal | 54.42049630 | 53.13718065 | -1.28331565 |
| Causal | 32.81518762 | 33.85214008 | +1.03695246 |
| Subevent | 29.94070344 | 31.19226171 | +1.25155827 |
| Overall | 52.00153038 | 52.14502736 | +0.14349698 |

Four new full710 CLI/AutoModel passes plus two exact reused prior-single passes and the public combiner gate passed. All state/prediction/metric/canonical checks are exact within CPU; original GPU differences remain in validation.json without unmeasured numerical-cause claims. All15 training runs and fixed positive2 five-seed statistics are complete. Validation is not hidden-test performance.

## Download and infer

Use the immutable HF revision recorded in the accompanying publication manifest.

```bash
REVISION=32cec123f43723b0eb47dd3406347c3b0f537df4
hf download MaYiding/maven-ere-roberta-base-joint --revision "$REVISION" --include "positive4-256-seed42-task-combination/**" "positive4-256-seed42-snapshot-u750/**" --local-dir checkpoints
mkdir -p predictions/positive4-seed42
python train_joint.py --predict-only --checkpoint checkpoints/positive4-256-seed42-task-combination/coreference --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --seed 42 --prediction-file predictions/positive4-seed42/coreference.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoints/positive4-256-seed42-task-combination/temporal --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --seed 42 --prediction-file predictions/positive4-seed42/temporal-causal.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoints/positive4-256-seed42-snapshot-u750 --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --seed 42 --prediction-file predictions/positive4-seed42/joint.valid.jsonl
python scripts/combine_task_predictions.py --coreference-predictions predictions/positive4-seed42/coreference.valid.jsonl --temporal-relations-predictions predictions/positive4-seed42/temporal-causal.valid.jsonl --causal-relations-predictions predictions/positive4-seed42/temporal-causal.valid.jsonl --subevent-relations-predictions predictions/positive4-seed42/joint.valid.jsonl --expected-documents 710 --output predictions/positive4-seed42/combined.valid.jsonl
```

Use default trimming without `--reference-padding`. For unscored test prediction change every split to test and expected-documents to857; hidden relation gold is unavailable. Keep all three complete original predictions.
