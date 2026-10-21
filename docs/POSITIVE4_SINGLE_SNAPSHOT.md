# Positive4：已核验的单联合模型中期快照

本次权重选于正式 seed42 训练的第9轮、update750。父run已于2026-10-21T04:05:34Z完整完成100轮、9,200次更新、36,500个microsteps和184次验证。最终选中权重仍来自第9轮/update750，全部8个模型、配置、分词和源码文件哈希与先前全710核验包完全一致，并非第100轮权重。当前7/15项完成，剩余8项及最终五种子统计、重载核验和发布继续。模型包含一个完整 RoBERTa-base 编码器及全部四任务头，约126M参数、每篇一次forward。全部15项正式实验和最终五种子统计仍须完成。

| F1 % | 原始根目录单模型 | 上一版 positive2 单模型 | positive4 CPU 单模型 | 相比上一版增减 |
|---|---:|---:|---:|---:|
| 共指四项均值 | 90.08368906 | 90.56372401 | 90.20028320 | -0.36344081 |
| 时间 | 53.08093519 | 54.42049630 | 52.45616097 | -1.96433534 |
| 因果 | 27.33665533 | 31.07405202 | 33.28013849 | +2.20608647 |
| 子事件 | 24.35389549 | 29.94070344 | 31.19226171 | +1.25155827 |
| Overall | 48.71379377 | 51.49974394 | 51.78221109 | +0.28246715 |

CPU/publicCLI and independentAutoModel match exactly on all710documents, predictions, metrics and state tensors. CPU overall51.78221109033361 and temporal52.456160965460995 differ from saved GPU51.78218842947946 and52.45607032204441: one validation document has one additional saved-GPU BEFORE pair that CPU omits. Full document/pair and metric differences are in validation.json. GPU logits are unavailable; no numerical margin or exact cause is asserted, and no thresholds were changed.

全部结果使用相同710篇valid、官方原始方向评测、gold事件提及及TIMEX。公开CLI与独立AutoModel重载的所有state tensor、完整预测和指标严格一致；原始GPU与CPU的差异详见validation.json，不把valid当作论文隐藏test。

相比原始单模型 Overall +3.06841732点，相比上一版 +0.28246715点。各任务增减完整列出，不保证多种子或隐藏test也提高。它与四完整模型组合分开报告，未拼接或平均权重。

## 下载与推理

HF根目录默认仍为旧模型，必须指定本子目录。本模型使用默认padding裁剪，**不要加`--reference-padding`**。输入需要已给出的事件提及、TIMEX及token位置；不负责原始文本事件检测。

```bash
hf download MaYiding/maven-ere-roberta-base-joint \
  --revision 7e539b7632f399806b1220c666f8f86a0b2caeee \
  --include "positive4-256-seed42-snapshot-u750/*" --local-dir checkpoint
hf download MaYiding/MAVEN-ERE --repo-type dataset \
  --revision 682fb313d7bec8c0db71715f0d59cd1b7dfa2e2e --local-dir data/MAVEN_ERE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python train_joint.py --predict-only \
  --checkpoint checkpoint/positive4-256-seed42-snapshot-u750 \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 \
  --prediction-file predictions/positive4-u750.valid.jsonl
```

## 训练协议

固定训练来源`8a49171cfe160d85877ffea5e7247c2141bcf3d4`；公开推理trainer SHA `14c3e00302020f0219710a6a304f5cbdbd6d6f63ebea4a8fd58488c0949c9c54`。256-token分块、隐藏维度200、3层Linear，positive_class_weight=4.0，FP32、batch8、累积4、seed42、100轮、禁用early stopping，每50次更新在完整valid评测。精确参数、数据及基础模型哈希见training_recipe.json。

```bash
python train_joint.py --base-model base_model --data-dir data/MAVEN_ERE \
  --output-dir runs/positive4-256-seed42 --cache-dir runs/tokenization-cache \
  --seed 42 --max_epochs 100 --batch_size 8 --grad_accum 4 \
  --eval_every_updates 50 --save_every_updates 100 --early_stopping_patience 0 \
  --no-bf16 --export-task-bests --skip-test-predictions --no-eval-epoch-end \
  --coreference-selection upstream-any --max_length 256 --scorer_hidden_size 200 \
  --scorer_num_layers 3 --positive_class_weight 4.0
```

相对作者参考实现，corrected driver修复dummy标签loss、按实际update调度和完整resume；本次是预定六候选之一。六候选已全部完成，按预定规则选中positive2；这与当前最佳公开single为positive4独立。后续reference及positive2各四个种子继续。[选择证据](CANDIDATE_SELECTION.md)。原始数据三JSONL未修改，GPL、RoBERTa MIT、MAVEN/Wikipedia许可及全部来源署名保留。

[全量核验](../results/positive4-256-seed42-snapshot-u750/validation.json) · [训练参数与来源](../results/positive4-256-seed42-snapshot-u750/training_recipe.json) · [发布清单](../results/positive4-256-seed42-snapshot-u750/publication_manifest.json)

[Publication hash and anonymous-readback receipt](../results/positive4-256-seed42-snapshot-u750/publication_receipt.json)

[完整100轮训练及权重不变核验](../results/positive4-256-seed42-snapshot-u750/training_completion.json)

The parent formal run completed all100epochs /9,200optimizer updates /36,500microsteps /184validations at2026-10-21T04:05:34Z. Its final selected checkpoint remains epoch9/update750, not an epoch100 weight. All8model/config/tokenizer/source hashes exactly match the previously full710validated public snapshot. No model inference or weight upload was repeated. See positive4-256-seed42-snapshot-u750/training_completion.json. Seven of15formal runs are complete; the remaining8runs and final five-seed checks continue.
