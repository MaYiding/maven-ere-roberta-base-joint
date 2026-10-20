# Positive2：已核验的单联合模型中期快照

本次权重选于正式 seed42 训练的第9轮、update750，父 run 计划完整100轮且仍在继续。模型包含一个完整 RoBERTa-base 编码器及全部四任务头，约126M参数、每篇一次forward。全部15项正式实验和最终五种子统计仍须完成。

| F1 % | 原始根目录单模型 | 上一版 corrected512 单模型 | positive2 CPU 单模型 | 相比上一版增减 |
|---|---:|---:|---:|---:|
| 共指四项均值 | 90.08368906 | 89.92380233 | 90.56372401 | +0.63992168 |
| 时间 | 53.08093519 | 53.10557320 | 54.42049630 | +1.31492311 |
| 因果 | 27.33665533 | 31.46712390 | 31.07405202 | -0.39307189 |
| 子事件 | 24.35389549 | 23.38204593 | 29.94070344 | +6.55865751 |
| Overall | 48.71379377 | 49.46963634 | 51.49974394 | +2.03010760 |

全部结果使用相同710篇valid、官方原始方向评测、gold事件提及及TIMEX。公开CLI与独立AutoModel重载的所有state tensor、完整预测和指标严格一致；原始GPU与CPU的差异详见validation.json，不把valid当作论文隐藏test。

相比原始单模型 Overall +2.78595017点，相比上一版 +2.03010760点。各任务增减完整列出，不保证多种子或隐藏test也提高。它与四完整模型组合分开报告，未拼接或平均权重。

## 下载与推理

HF根目录默认仍为旧模型，必须指定本子目录。本模型使用默认padding裁剪，**不要加`--reference-padding`**。输入需要已给出的事件提及、TIMEX及token位置；不负责原始文本事件检测。

```bash
hf download MaYiding/maven-ere-roberta-base-joint \
  --revision 1f4fa66ad77bf1e63d82892de2130c8dc394513e \
  --include "positive2-256-seed42-snapshot-u750/*" --local-dir checkpoint
hf download MaYiding/MAVEN-ERE --repo-type dataset \
  --revision 682fb313d7bec8c0db71715f0d59cd1b7dfa2e2e --local-dir data/MAVEN_ERE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python train_joint.py --predict-only \
  --checkpoint checkpoint/positive2-256-seed42-snapshot-u750 \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 \
  --prediction-file predictions/positive2-u750.valid.jsonl
```

## 训练协议

固定训练来源`8a49171cfe160d85877ffea5e7247c2141bcf3d4`；公开推理trainer SHA `14c3e00302020f0219710a6a304f5cbdbd6d6f63ebea4a8fd58488c0949c9c54`。256-token分块、隐藏维度200、3层Linear，positive_class_weight=2.0，FP32、batch8、累积4、seed42、100轮、禁用early stopping，每50次更新在完整valid评测。精确参数、数据及基础模型哈希见training_recipe.json。

```bash
python train_joint.py --base-model base_model --data-dir data/MAVEN_ERE \
  --output-dir runs/positive2-256-seed42 --cache-dir runs/tokenization-cache \
  --seed 42 --max_epochs 100 --batch_size 8 --grad_accum 4 \
  --eval_every_updates 50 --save_every_updates 100 --early_stopping_patience 0 \
  --no-bf16 --export-task-bests --skip-test-predictions --no-eval-epoch-end \
  --coreference-selection upstream-any --max_length 256 --scorer_hidden_size 200 \
  --scorer_num_layers 3 --positive_class_weight 2.0
```

相对作者参考实现，corrected driver修复dummy标签loss、按实际update调度和完整resume；本次是预定六候选之一。当前发布不提前宣告最终winner，仍按预先固定规则完成候选与后续种子。原始数据三JSONL未修改，GPL、RoBERTa MIT、MAVEN/Wikipedia许可及全部来源署名保留。

[全量核验](../results/positive2-256-seed42-snapshot-u750/validation.json) · [训练参数与来源](../results/positive2-256-seed42-snapshot-u750/training_recipe.json) · [发布清单](../results/positive2-256-seed42-snapshot-u750/publication_manifest.json)

[公开上传回读凭证](../results/positive2-256-seed42-snapshot-u750/publication_receipt.json)
