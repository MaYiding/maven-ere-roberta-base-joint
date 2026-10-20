# MAVEN-ERE 联合模型的验证集对齐与改进协议

本协议适用于独立改进 driver `train_joint.py`。作者参考 runner 与它完全隔离，
用于记录原实现行为；不得把参考 runner 的缺陷修复和候选模型收益混为同一实验。
同口径验证中相比上一版有明确进步的模型可作为中期成果公开；发布前须完成导出重载与
完整验证。提前发布不结束实验：全部 15 次正式训练各 100 轮及最终五种子对照仍须完成。
单模型与四个完整 checkpoint 的任务组合分别标识，详见 [参考快照说明](REFERENCE_CHECKPOINTS.md)。

## 验证边界

固定官方 MAVEN-ERE 1.0 的 train 2,913 / valid 710 / test 857 文档，保留原始哈希及
文档不相交检查。模型得到 gold event mentions 和 annotated TIMEX，预测四类关系；
这不是从原始文本发现事件的端到端系统。test 的关系金标不可得，任何本地结果都必须
标为 validation。论文发表的 test 平均分不能直接作为本地 valid 的成功阈值。

每次完整 valid 验证使用未改动的上游 `evaluate.py`，按原始有向标注计分；另记训练标签
双向扩展 SIMULTANEOUS / BEGINS-ON 后的正类 micro 指标。共指分别报告 B3、CEAF-e、
MUC、BLANC；overall 为这四项共指 F1 的均值与其他三任务 F1 的四项平均。
报告单 seed 和五个固定 seed（13、21、42、87、100）的均值与标准差，不把一次最好
seed 当作平均收益。参考与候选采用同一数据、验证计分与选择规则，均训练 100 轮；
原版累积尾批处理与改进 driver 的更新次数差异必须单独披露。

## 单因素候选与运行设置

兼容原 driver 的默认值仍是 scorer hidden 200、3 层 Linear、正类权重 1、overall 早停。
新增候选均新建输出目录，不能以更改签名的方式接续旧训练：

| 对照 | scorer hidden | Linear 层数 | 关系正类权重 |
|---|---:|---:|---:|
| 256-token 当前实现基准 | 200 | 3 | 1 |
| 512-token 长度对照 | 200 | 3 | 1 |
| 256-token 头宽对照 | 150 | 3 | 1 |
| 256-token 附录字面头解释 | 150 | 2 | 1 |
| 256-token 正类权重对照一 | 200 | 3 | 2 |
| 256-token 正类权重对照二 | 200 | 3 | 4 |

“two-layer perceptron” 的层数存在解释空间，150/2 只是把附录字面解释为两层 Linear，
不是证明作者实际训练时隐藏采用此结构；公开参考代码是 200/3。150/3 与 150/2 的配对
用于进一步区分宽度和深度的影响。类别权重只用于时间/因果/子事件 CE：NONE 权重为 1，每个正类为指定值，
忽略 -100；损失为加权 NLL 之和除以有效样本权重之和。共指边际 NLL 和四任务总损失权重
保持不变。正类权重 1 保持旧损失语义。

正式 campaign 共 15 次训练：先运行作者参考 seed 42，再运行上述六个候选的 seed 42，
之后完成作者参考其余四个 seed，最后将预先选定 winner 扩展到其余四个 seed。
候选选择预先固定为：计算四任务相对于参考
seed 42 的 F1 差，先最大化四项差值中的最小值，再以 overall F1 打破平局。primary
comparison 固定为候选的 task-combined 与作者参考的各任务 best 组合；共指使用四项
均值，所有数值来自同一评分协议。只选择一个 winner；best_joint 单快照结果另行报告，
不另外挑一个 single winner，也不把单快照与组合混称为同一种模型。选择依据、六组完整结果及 winner 的五 seed 结果全部保留，
seed 42 参与过选择这一事实也须披露。

正式配对设置固定 FP32、100 轮、patience 0、微批 8 / 累积 4。候选每 50 optimizer
updates 及最后一次更新验证，关闭额外的 epoch-end 验证；可用以下候选命令，并仅按表
改变对应参数：

```bash
python train_joint.py --base-model base_model --data-dir data/MAVEN_ERE \
  --output-dir runs/candidate-seed42 --cache-dir runs/tokenization-cache \
  --batch_size 8 --grad_accum 4 --max_epochs 100 --seed 42 --no-bf16 --max_length 256 \
  --scorer-hidden-size 150 --scorer-num-layers 2 --positive-class-weight 1 \
  --export-task-bests --early-stopping-monitor any-task \
  --early_stopping_patience 0 --eval_every_updates 50 --no-eval-epoch-end \
  --coreference-selection upstream-any --skip-test-predictions
```

`any-task` 在任一任务满足自己的选择改善规则时重置 patience；默认 `overall` 只看
联合 overall。小规模探索可显式使用 patience 20，但不能把不同早停预算的分数直接作为
正式配对收益；正式 campaign 用 patience 0 禁用早停并完成配置的 100 轮预算。
微批 8 / 累积 4 和微批 4 / 累积 8 的关系对权重不同，即使有效文档批同为 32 也不可视为
等价设置。分块长度、精度、scheduler、两优化器、梯度检查点、head 结构、类别权重以及选择
策略必须在配对比较中固定或明确披露。

完整 train 的微批数为每轮 365。作者原版跨 epoch 的累积计数在 100 轮产生 9,125 次
optimizer updates，每 200 微批验证，约 182 次选模机会；改进 driver 在每轮尾批 flush，
产生 9,200 次更新，每 50 次更新验证，约 184 次选模机会。两者均为 100 轮，不能称更新
预算完全相同。末次强制验证在关闭 epoch-end 时仍执行；只有最后更新恰好已验证时避免
重复。原版与改进 driver 的优化器设置、scheduler、梯度缩放/裁剪、尾批 flush 和数值稳定
处理属于本次被测 treatment，须与头结构及类别权重效果一起如实报告，不能称逐比特复现
作者训练流程。

## 单模型选择与多 checkpoint 组合

`best_joint` 始终是按 overall 选出的一个完整联合模型。启用 `--export-task-bests` 后，
按配置的共指规则及 temporal F1、causal F1、subevent F1 保存
`best_coreference`、`best_temporal`、`best_causal`、`best_subevent`。每个出口指向一个
包含同一时间点编码器与全部任务头的完整 HF 快照，禁止把不同时间点的头拼成“单模型”。
一次验证的多个出口可共享同一物理快照；稳定状态每个出口只保留一个目标，最多五个快照。

共指默认 `--coreference-selection mean`，严格改善四项 F1 均值时替换快照，兼容旧 run。
正式 campaign 使用 `upstream-any` 对齐作者选择规则：分别维护 MUC、B3、CEAF-e、BLANC
的历史最大 F1，任一项严格创新高就替换共指的完整快照。新快照的实际均值可能下降；
该快照成绩始终记录它当时的四项值及均值，历史逐项 maxima 另存为
`coreference_metric_bests`，不能拼成一份虚构模型成绩。每个任务的选中身份明确记录
`selection_rule`，`best_joint` 仍独立按整体均值选择。

`history.jsonl` 每次验证记录 epoch、optimizer update、全部官方与 augmented 指标、改善标记、
各任务实际 checkpoint ID、相对目录、共指 selector 及逐项历史 maxima。
`progress.json`、`training_summary.json` 记录各项选择结果与
实际导出身份。若未启用 task 导出，各项最佳成绩仍被跟踪，但明确标为 `exported: false`，
不得宣称存在相应可下载权重。

结束时从四项完整快照各取对应任务的验证预测，生成
`task_best_combined_predictions.jsonl` 和 `task_best_combined_metrics.json`，
并重新用官方评测器计分。文件明确标注 `is_single_model: false` 和每个任务的来源；这是
多 checkpoint 的任务组合，不是单一联合模型，也不是对不同 seed 做投票集成。与作者每任务
挑最佳的流程比较时，可报告该组合，必须同时报告 `best_joint` 单模型结果。

## 中断恢复与最终确认

继续使用原命令加 `--resume`。完整恢复需要保留运行目录、`latest.pt` 与其引用的 `exports/`。
checkpoint 保存模型、两个优化器、scheduler、RNG、样本顺序、下一微批、任务选择结果、
共指 selector、逐项历史 maxima 和验证历史。新开关进入训练签名，默认值兼容旧 run；
旧记录只有均值时不会虚构逐项历史。
导出先写新完整快照、原子替换选择链接；只有 latest.pt 提交成功才清理旧目录。恢复时重新
建立已提交的链接并回滚未提交的 history，避免断电留下的较新链接冒充已恢复的最佳状态。
默认语义的旧 checkpoint 可恢复；旧 checkpoint 不存在的任务历史不会被凭空重建。
改变头结构、类别权重、验证/选择策略或训练签名须新建实验。

探索可跳过 test 预测。候选确定后从 `best_joint` 生成 test candidate predictions；若还需
多 checkpoint test 组合，应分别运行四项快照并保留每项来源，不能把其成绩称为 test F1。
最终在 LXC 901 从实际导出重载，覆盖全部 710 篇 valid，核对指标、快照 ID、文件哈希及
模型结构。只有同协议验证集证据充分后才能讨论论文水平的接近或改进；隐藏 test 无评分时，
不宣称获得与论文完全一致的 test 复现结果。
