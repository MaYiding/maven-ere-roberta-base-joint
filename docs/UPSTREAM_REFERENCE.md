# 作者公开代码参考实验

`scripts/run_upstream_reference.py` 从本地 Git 对象库的固定提交
`ac81a9711a69f43f55bfbc50b3bb573fd11c64b0` 读取原始 `joint/*`、`utils/*`、
`evaluate.py` 和 `LICENSE`，在一个新的运行目录内生成原始快照与带兼容补丁的执行快照。
它不读取当前已修复的 `joint/src/utils.py` 作为训练实现，不下载源码，也不修改原数据或 `/data` 系统目录。

这是**作者公开代码的参考实验**。论文附录与公开代码本身存在差别，例如论文写隐藏维度 150，
公开代码实际使用 200。完成运行不自动表示复现了论文结果；论文表 7/8 是测试集多次运行结果，
本 runner 只能在公开验证集上计算分数，测试集只生成预测。

## 固定训练协议

| 项目 | 参考协议 |
|---|---|
| 训练轮数 | 默认 100，支持显式 `--epochs` |
| batch / accumulation | 8 / 4，名义有效 batch 为 32 文档 |
| 文档分块长度 | 256，保留作者 tokenizer 与 collator |
| 精度 | FP32，无 AMP、梯度缩放或裁剪 |
| 分类头 / encoder 学习率 | `3e-4` / `2e-5` |
| 分类头优化器 | 原始 `torch.optim.Adam` |
| encoder 优化器 | 旧版 Hugging Face AdamW 数学语义，`eps=1e-6, weight_decay=0` |
| warmup | 200 次 optimizer update |
| scheduler 总步数分母 | **原样保留** `len(train_dataloader) * epochs` |
| 累积边界 | **原样保留**全局 micro step 模 4，跨 epoch 携带，不补最后残余更新 |
| 验证周期 | 每 200 个 micro step，验证 batch 同样为 8 |
| best checkpoint | 四个任务分别保存；共指任一指标提升即按原代码覆盖 |
| 共指目标与损失 | **原样保留** `fill_expand` 与原始 clamp/sum/log 损失，包括首提及 dummy 缺陷 |
| gradient checkpointing | 默认关闭；只有显式传参才启用并记录 |

同一名义 batch 下，改成 batch 1 / accumulation 32 会改变每微批关系对的加权方式；本 runner
不会在 OOM 时自动降 batch、启用混合精度或替换训练流程。OOM 和其他失败会保留日志与失败状态。

## 仅做的兼容与运行补丁

1. 把基础模型、数据和输出路径指向参数提供的本地目录，禁止隐式联网加载模型。
2. 把误写的 `BertConfig` 检查改为 `RobertaConfig`，保留 `RobertaModel`、DataParallel、分类头及其初始化顺序。
3. 固定 encoder 为 eager attention，接近原时代实现，避免新版 Transformers 默认切换 SDPA。
4. 携带 Transformers v4.24.0 AdamW 的独立适配实现，保留**未进行 bias correction 的二阶矩开方后加 epsilon**、
   bias-corrected step size、以及 Adam 更新后执行 weight decay 的顺序。直接替换 `torch.optim.AdamW`
   即使 `eps` 和 `weight_decay` 相同，也不保证数学语义相同。
5. 删除未使用的 `pyparsing.col` import，替换旧 Hungarian 模块中的 `np.bool` / `np.int` 类型别名。
6. 验证函数增加 `torch.no_grad()`，不改变模型的 eval 模式或预测计算。
7. 明确从本次运行自己写出的可信 checkpoint 加载完整对象，适配新版 `torch.load` 默认值。
8. 添加不改变参数初始化、训练 loss 或更新顺序的日志、进度和训练后验证导出。
9. 给隔离的 `src` 增加空包标记，避免碰到环境中其他同名已安装包。
10. 每次 optimizer 更新前只读检查全部梯度是否有限，不裁剪、不修改；最终逐个检查实际加载的
    四个 best checkpoint 权重。遇到非有限值立即失败并记录对应参数名。

旧 AdamW 来源：
[Transformers v4.24.0 optimization.py](https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/optimization.py#L249-L350)。
它保留原版权和 Apache-2.0 notice；运行目录会从明确的系统许可证位置或已安装 Transformers
包元数据复制完整 Apache-2.0。完整训练缺少该许可证副本会在启动模型前失败。
作者源码的 GPLv3 许可证一并保存在原始与执行快照中。

每次修改均以固定原始文本为锚点；锚点不符合预期即停止，避免未知版本被静默修补。

## 远端执行

这些命令只应在用户授权的 LXC 901 训练环境执行。路径均是示意参数。

```bash
python scripts/run_upstream_reference.py \
  --base-model /path/to/local/roberta-base-snapshot \
  --data-dir /path/to/MAVEN_ERE \
  --output-dir /path/to/reference-seed42 \
  --seed 42 --epochs 100
```

运行目录必须不存在或为空。runner 不覆盖历史实验，也不声称能从作者不完整的 checkpoint
无损续训；中断应作为中断实验保留，再使用新目录执行。

如果显存需要，可以显式添加 `--gradient-checkpointing`。它默认关闭，不允许自动尝试后
悄悄成为新的默认配置。启用时会在源码 manifest、runtime 和最终 summary 中保留标记。

只提取并进行 AST 静态检查、不 import PyTorch 或执行模型：

```bash
python scripts/run_upstream_reference.py \
  --base-model /path/to/local/roberta-base-snapshot \
  --data-dir /path/to/MAVEN_ERE \
  --output-dir /path/to/reference-preparation \
  --prepare-only
```

## Canary

```bash
python scripts/run_upstream_reference.py \
  --base-model /path/to/local/roberta-base-snapshot \
  --data-dir /path/to/MAVEN_ERE \
  --output-dir /path/to/reference-canary \
  --canary
```

Canary 从真实训练集固定选择 64 篇，覆盖分块最多的文档、事件最多的文档、三个关系任务的
正例和非单例共指簇。默认两个 optimizer update 对应 8 个微批，完整覆盖 64 篇。验证和
测试各使用原顺序前 16 篇。选择列表保存到 `canary_*_ids.json`。

`--max-steps` 单位为 optimizer update；canary 最多为 2。显式限制为 1 时可能尚未遍历
所选的全部 64 篇，不能声称覆盖已全部经过前向/反向。完整运行若显式限制 max steps，
也属于截断实验，不能进入正式比较。

为检查导出链路，max-update 边界会额外触发一次验证。仅 canary 中，如果某任务 F1
一直为零导致原代码从未产生 best，会写带 `canary_final_state_diagnostic_fallback` 标记的
最终状态快照。它不是真正的 best，`reference_summary.json` 会列明 fallback 任务。
完整实验绝不补造 best。Canary 状态为 `smoke`，不进入正式排行榜，不判定论文达标。

## 产物与评测口径

运行目录包含：

- `original/`：Git blob 的原始字节，`joint/src/utils.py` 保持作者版本。
- `executed/`：实际执行源码、两个明确标记的运行辅助模块和空包标记。
- `source_manifest.json`：固定提交、逐文件原始/执行 SHA256、完整补丁 SHA256、参数与协议。
- `compatibility.patch`：包括新增辅助模块的实际统一 diff。
- `dataset_manifest.json`：split 数量、SHA256 与跨 split 文档 ID 不重叠校验。
- `artifacts/log.txt`：作者 stdout 日志；`console.log`：stderr、进度条与异常 traceback。
- `events.jsonl` / `progress.json`：逐微批、逐更新、验证及选模进度。
- `artifacts/best_*` / `best_selection.json`：四个任务各自的 checkpoint 与选模身份。
- `validation_by_checkpoint.json`：每个 best 在相同验证集上的原始作者扩展标签指标。
- `raw_direction_validation_by_checkpoint.json`：每个 best 的对应任务在根 `evaluate.py` 原始方向口径下的指标。
- `weights_finite_by_checkpoint.json`：逐个 best checkpoint 全部浮点权重的有限值检查。
- `best_valid_metrics.json` / `best_valid_predictions.jsonl`：四个 best 组合的验证结果与预测。
- `artifacts/test_prediction.jsonl`：四个 best 组合的测试集预测，**没有测试分数**。
- `reference_summary.json`：campaign 使用的稳定字段，见下文。

作者训练标签把 `SIMULTANEOUS` / `BEGINS-ON` 补成双向，根 `evaluate.py` 按原始标注方向评分。
两个口径独立保存；公平比较改进 driver 时使用 `official_protocol`，不能把另一口径的较高值
替换进来。最终结果由 **4 个 checkpoint** 组合，不应标成单一联合模型 checkpoint 的结果。

`reference_summary.json` 的主要字段：

```json
{
  "status": "completed | smoke | interrupted | failed",
  "official_protocol": {
    "coreference": {"muc_f1": 0.0, "b_cubed_f1": 0.0, "ceaf_f1": 0.0, "blanc_f1": 0.0},
    "temporal": {"temporal_f1": 0.0},
    "causal": {"causal_f1": 0.0},
    "subevent": {"subevent_f1": 0.0},
    "coreference_mean_f1": 0.0,
    "overall_f1": 0.0
  },
  "task_checkpoints": {"coreference": {"path": "...", "sha256": "...", "selection": "..."}},
  "training": {"requested_epochs": 100, "epochs_completed": 100, "optimizer_updates": 0},
  "experiment_complete": true,
  "leaderboard_eligible": true,
  "test_scored": false,
  "paper_reproduction_success": null
}
```

`official_protocol` 的 P/R/F1 均为百分数；原始作者 classification report 为 `[0,1]`，
单独存为 `upstream_augmented_protocol_fraction`。示例省略的 P/R 字段实际保留。
只有正常完成 100 轮、未截断、非 canary 的运行才会标记 `leaderboard_eligible=true`；
该标记表示实验协议完整，**不表示达到论文结果**。
