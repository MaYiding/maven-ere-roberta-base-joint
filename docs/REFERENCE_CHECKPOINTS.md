# 作者参考 seed 42：中期发布与四模型组合

截至 2026-10-20，作者参考 seed 42 已完成 100 epochs、36,500 micro steps、9,125
optimizer updates，FP32，微批 8、梯度累积 4、256-token 分块，scorer hidden 200、
3 层 Linear。本次正式 campaign 计划 15 次训练，每次均须完成 100 epochs；作者参考seed42已经完成，全部15次矩阵仍在进行。提前公开经核验的已有产物，不会停止或缩减后续实验，也不代表五 seed 实验已经完成。

四套模型各自的 legacy CPU、转换 CPU 和 AutoModel CPU
三路全710篇预测、指标及重载权重已严格通过。最终 JSON 汇总的 NumPy布尔问题已修复，
恢复时重新核对全部12路预测、来源及权重哈希，原失败证据保留。
四套实际公开推理命令也已经完成全部710篇核验，完整预测、所有官方指标及重新生成的组合均与 AutoModel 输出严格一致。
本文四个参考 checkpoint 单独的 overall 均未超过旧版。另一个 corrected512 单 joint 快照已经独立核验提升，见 [单模型说明](CORRECTED_SINGLE_SNAPSHOT.md)。

## 模型目录与结果口径

[HF 模型仓库](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint) 的根目录继续
保留旧版单共享编码器模型、原配置及原 metrics。直接加载仓库根目录得到的是旧模型，
不是下表的四模型组合。新增参考模型的目标目录如下：

| 子目录 | 原始选中 micro step | 原始选中 optimizer update | 组合取用字段 |
|---|---:|---:|---:|
| `reference-seed42/coreference/` | 1,800 | 450 | `coreference` |
| `reference-seed42/temporal/` | 2,800 | 700 | `temporal_relations` |
| `reference-seed42/causal/` | 23,400 | 5,850 | `causal_relations` |
| `reference-seed42/subevent/` | 7,200 | 1,800 | `subevent_relations` |

每个子目录都是一个约 126M 参数的完整四任务联合模型，各有完整 RoBERTa-base 编码器
和全部任务头，并非只含一个分类头。四模型组合使用约 4 × 126M 个独立参数、每篇文档
四次 forward，再各取相应任务的预测字段。没有跨 checkpoint 拼接权重或共享编码器。
单个子目录能够输出四任务，但不能把组合成绩当作该子目录的全部任务成绩。

以下是完成 run 的原始 GPU 预测在官方 `evaluate.py`、原始有向标注下的 valid F1
（百分数）；CPU列为实际完整验证结果：

| 指标 | 旧版单模型 | 参考 GPU 四模型组合 | 导出 CPU 四模型组合 |
|---|---:|---:|---:|
| 共指均值 | 90.0836890588 | 90.7807344934 | 90.7807344934 |
| 时间 | 53.0809351898 | 53.1077737217 | 53.1076436102 |
| 因果 | 27.3366553346 | 31.0994682515 | 31.0994682515 |
| 子事件 | 24.3538954914 | 25.1568792732 | 25.1568792732 |
| Overall | 48.7137937687 | 50.0362139350 | 50.0361814071 |

组合共指子项为 B3 97.7425595346、CEAF-e 97.4762917372、MUC 79.8051948052、
BLANC 88.0988918965。共指均值为四项 F1 的算术平均；overall 为共指均值、时间、因果、
子事件四项的算术平均。这组原始 GPU 结果作为历史来源保留，不写成最终导出的 CPU 成绩。

原始 GPU 组合相对旧单模型的 overall 差为 +1.3224 个百分点，资源及选模口径不同，不能解释为
同模型数量或同计算成本的单模型改善。单 checkpoint 的四任务结果须来自它自己的全部
710 篇预测，逐一列出后才能作对应比较。单模型复核结果见下表。
四个任务所选单模型的 CPU 全710篇结果如下，列顺序为共指均值、时间、因果、子事件及 overall（F1 %）：

| 选中任务 | 共指均值 | 时间 | 因果 | 子事件 | Overall |
|---|---:|---:|---:|---:|---:|
| coreference | 90.78073449 | 44.55126289 | 15.80985286 | 1.74757282 | 38.22235577 |
| temporal | 90.28431509 | 53.10764361 | 22.45346162 | 9.71270235 | 43.88953067 |
| causal | 89.89255920 | 50.49797950 | 31.09946825 | 20.85265181 | 48.08566469 |
| subevent | 89.78541875 | 51.45488693 | 25.56110859 | 25.15687927 | 47.98957339 |

这些是四个任务所选 checkpoint 的结果，不能称为全部训练评测点中的全局最优单 joint。
四者的整体分数均低于旧单模型，专项增减与四模型组合分别报告。


valid 需要 gold event mentions 和 annotated TIMEX，本模型不从原始文本检测事件。
所有成绩均为 validation；test 857 篇只有候选 event mentions（含 distractors）和 TIMEX，
关系金标隐藏。训练中双向扩展 SIMULTANEOUS / BEGINS-ON 的 augmented 指标与本表
原始方向指标分别记录。没有 test F1，也不宣称严格达到论文隐藏 test 或五 seed 平均成绩。

## v1 差异诊断与 v2 验收边界

v1 在 TEMPORAL 完整 710 篇预测比较中发现：1 篇文档的 CPU 预测较原始 GPU 多出
1 条 BEFORE，时间 F1 从原始 GPU 的 **53.10777372173359** 变为 CPU 的
**53.10764361015359**。这是真实的跨设备预测差异，不能抹平或把原始 GPU 数值写进
CPU 报告。它不是四模型 CPU 组合最终分数；其他模型及组合仍须完整 v2 复核。

对该文档所在的原始 batch 8 重放，legacy 与转换模型的 **CPU temporal logits 逐位相同**，
相关 CPU top-2 logit margin 为 **1.66893e-6**。这些证据支持设备/后端数值差异的解释；
没有原始 GPU logits，因此未测得 GPU margin，也未直接测得 GPU 与 CPU 的逐 logit 差。
不修改类别阈值、权重或选中 checkpoint 来追平历史 GPU 预测。

v2 在新输出目录运行，保留 v1 失败证据。验收包含四个完整模型的全 710 篇
legacy CPU → converted CPU → HF AutoModel CPU 预测及指标严格一致，并核对转换映射和
重载权重。组合必须从四份已重载的 CPU 全量预测重新生成。原始 GPU 结果单列，另保存
实际 CPU 单模型/组合分数与跨设备差异报告；同设备转换一致性不等于跨设备完全复现。
实际 CPU overall为50.036181407057285；公开命令级全量验收也已通过。

公开推理增加仅用于 `--predict-only` 的 `--reference-padding`：保留 checkpoint 的完整
256-token 输入宽度以匹配参考评测。901 上已比对完整710篇：两种数据管线的逐样本 token、
事件位置和标签全部相同，但默认裁剪末尾 padding 会改变75/89个批次的张量宽度。
这个选项不修改训练、权重或分类阈值；默认训练行为保留。13项远端预测/训练契约测试通过。参考模式同时将共指分组按首提及位置固定顺序；
首次CLI全710篇的708条差异仅是分组排列，成员、关系边及全部指标均相同，旧记录保留。
下列真实 CLI 已对四套模型逐一预测710篇，并严格比对 v2 AutoModel 的全部预测和指标。
校验脚本曾因派生均值的浮点加法顺序不同中断；原始任务指标与预测完全一致。统一为
MUC/B3/CEAF/BLANC 顺序后，明确恢复并重新核对前三份实际CLI输出，再完成第四份，
所有派生均值、组合指标及预测哈希均严格相同，原失败证据保留，未放宽阈值。

## 下载数据和模型

在本仓库环境安装 `requirements-train.txt`，在计划推理的计算环境执行下列命令。
仓库未公开时，需使用有访问权的 Hugging Face 账号认证。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
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
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --reference-padding \
  --prediction-file predictions/reference-seed42/coreference.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoint/reference-seed42/temporal \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --reference-padding \
  --prediction-file predictions/reference-seed42/temporal.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoint/reference-seed42/causal \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --reference-padding \
  --prediction-file predictions/reference-seed42/causal.valid.jsonl
python train_joint.py --predict-only --checkpoint checkpoint/reference-seed42/subevent \
  --data-dir data/MAVEN_ERE --split valid --device cpu --no-bf16 --eval_batch_size 8 --reference-padding \
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
原始 legacy CPU、转换 CPU 及新 HF CPU 模型的权重映射、全部预测和官方指标。
原始 GPU 记录作为单独对照，允许显式报告其与实际 CPU 的差异。来源模型代码固定为
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
  --output-dir assessment/reference-seed42-v2 --threads 4 --batch-size 8 --expected-documents 710
```

先在另一新目录（例如 `assessment/canary-v2`）加 `--canary-only` 可检查少量文档兼容性；
它的结果不是完整验证或发布验收。
公开须确认完整报告为 `status: completed`、`split: valid`、`documents: 710`，四个模型
`hf_automodel_roundtrip_passed: true`，legacy CPU 与 converted CPU 的全量预测/指标严格
一致，以及组合由四份真实 AutoModel 全量预测重建。公开证据须分别保留原始 GPU 和实际
CPU 指标及差异、必要哈希、每任务选中位置、全部单模型分数及组合分数，
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

[Validated metrics and model checksums](../results/reference-seed42/publication_manifest.json)
