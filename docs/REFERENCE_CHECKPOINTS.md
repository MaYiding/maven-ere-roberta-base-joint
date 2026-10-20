# 作者参考 seed 42：中期发布与四模型组合

截至 2026-10-20，作者参考 seed 42 已完成 100 epochs、36,500 micro steps、9,125
optimizer updates，FP32，微批 8、梯度累积 4、256-token 分块，scorer hidden 200、
3 层 Linear。本次正式 campaign 计划 15 次训练，每次均须完成 100 epochs；目前完成
1/15。提前公开经核验的已有产物，不会停止或缩减后续实验，也不代表五 seed 实验已经完成。

本页尚处于发布前记录：四套 HF 导出必须各自完成 CPU 全量 710 篇 valid 重载复核，
并从实际重载预测重建组合后，才可公开为已验证权重。小样本 canary 不能代替全量验证。
目前没有已经证实的单共享 checkpoint 整体超过旧版的结论。

## 模型目录与结果口径

[HF 模型仓库](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint) 的根目录继续
保留旧版单共享编码器模型、原配置及原 metrics。直接加载仓库根目录得到的是旧模型，
不是下表的四模型组合。新增参考模型的目标目录如下：

| 子目录 | 原始选中 micro step | 原始选中 optimizer update | 组合取用字段 |
|---|---:|---:|---|
| `reference-seed42/coreference/` | 1,800 | 450 | `coreference` |
| `reference-seed42/temporal/` | 2,800 | 700 | `temporal_relations` |
| `reference-seed42/causal/` | 23,400 | 5,850 | `causal_relations` |
| `reference-seed42/subevent/` | 7,200 | 1,800 | `subevent_relations` |

每个子目录都是一个约 126M 参数的完整四任务联合模型，各有完整 RoBERTa-base 编码器
和全部任务头，并非只含一个分类头。四模型组合使用约 4 × 126M 个独立参数、每篇文档
四次 forward，再各取相应任务的预测字段。没有跨 checkpoint 拼接权重或共享编码器。
单个子目录能够输出四任务，但不能把组合成绩当作该子目录的全部任务成绩。

以下是完成 run 原始预测在官方 `evaluate.py`、原始有向标注下的 valid F1（百分数）：

| 指标 | 旧版单模型 | 参考 seed 42 四模型组合 |
|---|---:|---:|
| 共指均值 | 90.0836890588 | 90.7807344934 |
| 时间 | 53.0809351898 | 53.1077737217 |
| 因果 | 27.3366553346 | 31.0994682515 |
| 子事件 | 24.3538954914 | 25.1568792732 |
| Overall | 48.7137937687 | 50.0362139350 |

组合共指子项为 B3 97.7425595346、CEAF-e 97.4762917372、MUC 79.8051948052、
BLANC 88.0988918965。共指均值为四项 F1 的算术平均；overall 为共指均值、时间、因果、
子事件四项的算术平均。这组原始参考结果仍须与最终 HF 导出的完整复核结果精确核对。

组合相对旧单模型的 overall 差为 +1.3224 个百分点，资源及选模口径不同，不能解释为
同模型数量或同计算成本的单模型改善。单 checkpoint 的四任务结果须来自它自己的全部
710 篇预测，逐一列出后才能作对应比较。当前不提供尚未完成的单模型复核分数。

valid 需要 gold event mentions 和 annotated TIMEX，本模型不从原始文本检测事件。
所有成绩均为 validation；test 857 篇只有候选 event mentions（含 distractors）和 TIMEX，
关系金标隐藏。训练中双向扩展 SIMULTANEOUS / BEGINS-ON 的 augmented 指标与本表
原始方向指标分别记录。没有 test F1，也不宣称严格达到论文隐藏 test 或五 seed 平均成绩。

## 下载数据和模型

在本仓库环境安装 `requirements-train.txt`，在计划推理的计算环境执行下列命令。
仓库未公开时，需使用有访问权的 Hugging Face 账号认证。

```bash
hf download MaYiding/MAVEN-ERE --repo-type dataset \
  --revision 682fb313d7bec8c0db71715f0d59cd1b7dfa2e2e --local-dir data/MAVEN_ERE
hf download MaYiding/maven-ere-roberta-base-joint \
  --include "reference-seed42/*" --local-dir checkpoint
```

数据仓库根目录保留 `train.jsonl` / `valid.jsonl` / `test.jsonl` 原始文件名，分别为
2,913 / 710 / 857 文档，总计 4,480。HF datasets 的 `annotated` 配置包含 train / validation，
`test_candidates` 配置包含候选 test；后者没有关系金标。下载命令固定数据镜像 revision；
复现已发布权重时还应将模型下载的 `--revision` 固定为对应发布清单记录的 HF commit。

## 四次完整模型推理及组合

以下例子用 CPU FP32，保证每个任务使用对应完整模型。也可在可用 GPU 上运行；如果比较
已经记录的精确重载结果，应保持评估精度、设备及来源一致。配置自带分块长度，无须改成
CLI 默认的 512。输出目录请使用新的空目录，保留四份原始预测及模型 revision。

```bash
mkdir -p predictions/reference-seed42
python train_joint.py --predict-only --checkpoint checkpoint/reference-seed42/coreference \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 \
  --prediction-file predictions/reference-seed42/coreference.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoint/reference-seed42/temporal \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 \
  --prediction-file predictions/reference-seed42/temporal.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoint/reference-seed42/causal \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 \
  --prediction-file predictions/reference-seed42/causal.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoint/reference-seed42/subevent \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 \
  --prediction-file predictions/reference-seed42/subevent.valid.jsonl
python scripts/combine_task_predictions.py \
  --coreference-predictions predictions/reference-seed42/coreference.valid.jsonl \
  --temporal-relations-predictions predictions/reference-seed42/temporal.valid.jsonl \
  --causal-relations-predictions predictions/reference-seed42/causal.valid.jsonl \
  --subevent-relations-predictions predictions/reference-seed42/subevent.valid.jsonl \
  --expected-documents 710 --output predictions/reference-seed42/combined.valid.jsonl
```

combiner 检查文档 ID 一致、无重复，并只按指定字段组合；它不训练或修改权重，也不计算
F1。使用官方评测器和原始 valid 金标计分。生成 test 候选预测时，将四次推理都改为
`--split test`，改用独立的 test 输出名，组合检查设为 `--expected-documents 857`；不能把
无关系金标的 test 当作 validation 评分。

HF `AutoModel.from_pretrained(..., trust_remote_code=True)` 可加载每个独立导出；使用前
审阅随包自定义源码。`forward` 还需要事件 span 与分块/索引输入，推荐使用上面保留分词、
span 对齐、TIMEX 排序及 JSONL 解码的入口，不能直接当成普通文本分类 pipeline。

## 发布前 CPU 全量复核

`scripts/evaluate_reference_exports.py` 只读已完成参考 run，CPU 导出、重载四模型，并核对
原始模型与新 HF 模型的全部权重、预测和官方指标。来源模型代码固定为
`8a49171cfe160d85877ffea5e7247c2141bcf3d4`；`--source-root` 必须指向这个版本的独立
checkout，而不是任意最新分支。工具与发布文档的后续提交另行记录，不改变冻结训练版本。

以下均为可替换的本地相对路径，不指向训练机器。`--reference-dir` 应是已完成参考 runner
的目录，含 `reference_summary.json` 等原始产物；输出必须为新建或空目录，并在所有只读
输入之外。基线 metrics 与 manifest 须来自旧版同一 710 文档验证分割。

```bash
python scripts/evaluate_reference_exports.py \
  --source-root ../maven-ere-frozen-8a49171 \
  --reference-dir completed/reference-seed42/reference \
  --base-model base_model --data-dir data/MAVEN_ERE \
  --baseline-metrics completed/old-single/best_dev_metrics.json \
  --baseline-manifest completed/old-single/split_manifest.json \
  --output-dir assessment/reference-seed42 --threads 4 --batch-size 8 --expected-documents 710
```

先在另一新目录加 `--canary-only` 可检查少量文档兼容性；它的结果不是完整验证或发布验收。
公开须确认完整报告为 `status: completed`、`split: valid`、`documents: 710`，四个模型
`hf_automodel_roundtrip_passed: true`，以及组合由四份真实 AutoModel 全量预测重建、指标与
原始完成记录一致。公开证据应保留必要哈希、每任务选中位置、全部单模型分数及组合分数，
去掉内部绝对路径；不上传优化器、RNG、缓存、凭据或工具的私有输入副本。

## 来源与许可

- 训练及兼容导出冻结来源：`8a49171cfe160d85877ffea5e7247c2141bcf3d4`；作者上游来源：
  [`THU-KEG/MAVEN-ERE`](https://github.com/THU-KEG/MAVEN-ERE)，commit
  `ac81a9711a69f43f55bfbc50b3bb573fd11c64b0`。原始模型/数据工作请引用
  [Wang et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.60/)。
- 改编代码及 checkpoint 分发保留 GPL-3.0；基础
  [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) 保留 MIT 归属。
  参考 runner 的旧版 Transformers AdamW 适配另保留 Apache-2.0 声明，见
  [上游声明](../UPSTREAM_NOTICE.md)。
- [数据镜像](https://huggingface.co/datasets/MaYiding/MAVEN-ERE) 重新分发官方 MAVEN-ERE 1.0
  原始三个 split，不提供新标注，也不是作者官方发布账号。数据保留 MAVEN-ERE 的 GPL
  许可声明，以及来源 [MAVEN](https://github.com/THU-KEG/MAVEN-dataset) 的 CC BY-SA 4.0
  与 Wikipedia 文本的 CC BY-SA 3.0 署名/来源说明；按镜像数据卡和随附许可区分各组成。
  模型卡的 GPL 标签不能替代所有数据组成的许可。

单 seed 记录、四模型组合与旧版单模型都只是验证集证据。完整 15 次实验结束前，不把本页
称为已完成的多 seed 复现或最终改进结论。
