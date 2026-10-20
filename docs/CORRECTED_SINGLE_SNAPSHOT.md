# 单联合模型中期快照：corrected512 seed42

这个快照取自正式训练第17轮、update1500。父训练仍须跑满100轮，完整15次实验及五种子统计继续进行。
一个完整共享编码器和全部四个任务头，每篇文档一次 forward。原始 GPU、实际公开命令 CPU、AutoModel CPU 的710篇完整预测、全部指标完全一致，重载权重逐 tensor 相同。

| valid710，F1% | 旧单模型 | 新单模型 | 增减（点） |
|---|---:|---:|---:|
| 共指均值 | 90.08368906 | 89.92380233 | -0.15988673 |
| 时间 | 53.08093519 | 53.10557320 | +0.02463801 |
| 因果 | 27.33665533 | 31.46712390 | +4.13046857 |
| 子事件 | 24.35389549 | 23.38204593 | -0.97184956 |
| Overall | 48.71379377 | 49.46963634 | +0.75584257 |

这是同一完整验证协议上的整体提升，部分任务下降已如实列出。单seed、验证集调参不能证明五种子或隐藏test上的提升。旧根目录模型和 reference 四模型组合均保留。

[模型子目录](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint/tree/main/corrected-512-seed42-snapshot-u1500) · [完整验证记录](../results/corrected-512-seed42-snapshot-u1500/validation.json)

该版本使用默认 padding 裁剪，与 reference 组合的固定padding协议不同，**不要加 `--reference-padding`**。配置保存512-token分块，输入仍需已标注的事件提及/TIMEX和token偏移。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hf download MaYiding/maven-ere-roberta-base-joint \
  --revision fd86462a71ecb54206c05e51ecc9e662b4348134 \
  --include "corrected-512-seed42-snapshot-u1500/*" --local-dir checkpoint
python train_joint.py --predict-only \
  --checkpoint checkpoint/corrected-512-seed42-snapshot-u1500 \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 \
  --prediction-file predictions/corrected512-u1500.valid.jsonl
```

存档复现应将模型revision固定为公开验证记录中的HF commit，数据revision为682fb313d7bec8c0db71715f0d59cd1b7dfa2e2e。训练源码8a49171cfe160d85877ffea5e7247c2141bcf3d4，公开推理trainer SHA14c3e00302020f0219710a6a304f5cbdbd6d6f63ebea4a8fd58488c0949c9c54。

GPL、RoBERTa MIT、上游署名和引用随模型保留；原数据的MAVEN/Wikipedia许可层独立保留。这个快照发布不会停止或缩减原定15次完整实验。

[Published model revision and verification receipt](../results/corrected-512-seed42-snapshot-u1500/publication_receipt.json)
