# MAVEN-ERE：完整复现与改进研究

规定的15项正式实验均实际完成100轮，总计1500轮。固定winner positive2五种子CPU任务组合Overall为 **51.5469 ±0.3519**，作者参考为 **49.7728 ±0.3367**（均值±样本SD，n−1）；配对增量 **+1.7741点**，95%t置信区间 **[+1.0901,+2.4581]**。valid不是论文隐藏test，单模型与多完整模型组合分别报告。

[Final report](docs/FINAL_REPRODUCTION_REPORT.md) · [Five-seed CPU data](results/final-campaign/five_seed_cpu.json) · [All15 run metrics and checkpoint hashes](results/final-campaign/formal_runs.json) · [Formal training protocol](docs/REPRODUCTION_PROTOCOL.md) · [Author runner](docs/UPSTREAM_REFERENCE.md)

最新可选positive4 seed42三完整模型组合CPU Overall **52.14502736**，比此前positive2 CPU52.00153038提高 **0.14349698点**。最强单模型仍为positive4 **51.78221109**（一个完整编码器和全部任务头，每篇一次forward）；组合为三编码器三次forward。固定五种子winner仍为positive2。[全部任务增减与三次CLI复现命令](docs/POSITIVE4_TASK_COMBINATION.md)。

仓库根仍加载最初48.71379377单模型，请显式选择子目录。输入需要gold event mentions、TIMEX和token位置，不是原始文本事件发现。保留GPL-3.0、RoBERTaMIT、上游及旧优化器许可证、MAVEN/WikipediaCCBY-SA和4480篇来源署名；既有数据镜像未重复上传，旧checkpoint/历史发布修订均保留。

[Models](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint) · [Existing licensed data mirror](https://huggingface.co/datasets/MaYiding/MAVEN-ERE). The root loads the old48.71379377 baseline; select a documented child or combination. Inputs require gold event mentions, TIMEX and token locations. GPL-3.0/RoBERTaMIT/legacy-optimizer notices and MAVEN/WikipediaCCBY-SA/all4480source attributions are retained. No dataset duplicate, test score or extra positive4 five-seed training is claimed. Historical revisions and checkpoints remain available.
