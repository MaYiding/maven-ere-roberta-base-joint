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

上述组合仅是原始GPU训练统计，不冒充已核验的CPU single或新公开组合包。positive2和positive4各自的task-best组合均使用四个任务字段、三个不同完整checkpoint。当前已核验公开single最佳仍为positive4，实际CPU Overall51.78221109033361。

当前7/15项完成，余下reference与positive2的seed13/21/87/100共8项仍须各跑满100轮，之后报告同一valid协议五种子均值、样本SD和配对95%CI，并完成最终冻结重载、报告和发布回读。valid不是论文隐藏test。

[完整分项增量和选择证据](../results/candidate_selection_seed42.json)
