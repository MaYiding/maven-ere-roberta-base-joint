# 固定规则的候选选择结果

六个seed42候选均完整完成100轮。按事先固定的“最大化四任务相对reference42最小F1增量，再overall，再候选原顺序”选择 **positive2-256**。该规则使用task-best组合的相同raw-valid协议，独立于single模型排名。

| 候选 | 最小任务增量（点） | 组合Overall F1 |
|---|---:|---:|
| positive2-256 | +0.0489996660 | 52.0015303788 |
| positive4-256 | -0.3822074926 | 52.1450273608 |
| width150-256 | -0.8606106863 | 49.7278278145 |
| corrected-512 | -0.9155939213 | 50.1991155645 |
| appendix-two-layer-256 | -1.0967329148 | 49.2964281144 |
| corrected-256 | -1.9770288206 | 49.4338238077 |

positive2四任务增量均为正，最小增量为共指+0.04899966598点；positive4组合Overall较高，但共指-0.38220749258点，因此按预定规则排序第二。未因公开single改善而修改选择规则。

原始GPU选择依据保持不变。positive2和positive4的seed42组合后来均通过完整710篇CPU CLI/AutoModel核验，CPU/GPU和任务模型成本分别披露。全部15项正式训练均完成100轮，固定positive2五种子CPU均值、样本SD和配对95%CI已完成。[完整报告](FINAL_REPRODUCTION_REPORT.md)保留全部四项共指、三类关系指标及选择/padding/实现差异。valid不是隐藏test，positive4不被追加为五种子训练arm。

[原始选择证据](../results/candidate_selection_seed42.json)
