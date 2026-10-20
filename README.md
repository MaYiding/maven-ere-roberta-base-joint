# MAVEN-ERE Joint RoBERTa-base Training

本仓库在 THU-KEG/MAVEN-ERE 官方实现上，补充可断点续训、可独立导出的四任务联合训练流程。
英文说明见 [README.en.md](README.en.md)。原始项目说明保留在本文后半部分。

模型输入需要已标出的事件提及和时间表达；它不负责从原始文本发现全部事件。
一个共享 RoBERTa-base 编码器同时预测事件共指、六类时间关系、两类因果关系和子事件关系。
子事件有向边 `[parent, child]` 的第二项属于第一项。

## 已训练模型（2026-10-20）

[Hugging Face 权重](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint) · [训练记录与使用方法](docs/TRAINING_RUN.md)

seed 42 正式训练已完成，23 轮提前停止，实际批设置为 **8 × 4**。
710 篇 valid 上最佳同一检查点的 F1：共指均值 **90.08**、时间 **53.08**、因果 **27.34**、子事件 **24.35**，overall **48.71**。
使用金标事件提及和已标注 TIMEX；完整发布模型已重跑验证，全部指标一致。测试金标未公开，无测试成绩。
仓库初始为私有。精确训练代码 commit 为 `369d9df4262f3fb4c22073d8c6c8021fa78698f8`；后续提交只补充发布文档与证据。


## 训练

建议使用 Python 3.11、支持 BF16 的 NVIDIA GPU。只在计划训练的服务器上执行以下步骤。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
python scripts/prepare_data.py
python scripts/download_base.py
python -m pytest -q tests
python train_joint.py --base_model base_model --data_dir data/MAVEN_ERE \
  --output_dir runs/roberta-joint-seed42 --cache_dir runs/tokenization-cache \
  --batch_size 4 --grad_accum 8 --max_epochs 100 --seed 42
```

完整数据首次分词会写入缓存。正式训练前可在独立目录使用 `--smoke`：它运行真实训练文档的短试验，覆盖长文档和四类关系正例，检查损失、完整训练状态恢复及 HF 导出重载。
正式训练不要复用 smoke 的输出目录；两者可以共享分词缓存。

默认设置：微批4文档、梯度累积8次、名义有效批32文档、512-token最大分块、BF16、梯度检查点、seed 42、最多100轮。
编码器 AdamW 学习率2e-5、eps=1e-6；任务头 Adam 学习率3e-4。编码器按实际优化器更新步数进行200步 warmup 和线性衰减。
四任务损失权重为0.4／2／4／4。每轮在710篇 valid 文档上验证；连续10次验证未提升则停止。

共指损失先对每篇文档的提及求均值，再对文档求均值；其他任务对每个微批内有效事件对求均值。
因此4×8与上游README的8×4虽然名义有效批相同，事件对权重仍存在差别。
分类头隐藏维度200沿用公开代码，论文附录写150。本仓库是记录差异的工程复现，不宣称逐项复现论文结果。

## 断点、验证和推理

同一命令追加 `--resume`，从输出目录的 `latest.pt` 恢复。checkpoint 只在梯度更新边界保存，包含两个优化器、scheduler、轮次、样本排列、下一微批位置及 Python／NumPy／Torch／CUDA RNG。
恢复时数据、批大小、累积次数、训练总轮数等签名必须一致。SIGTERM／SIGINT 会在下个更新边界保存并退出。

`best_joint` 指向按官方 overall F1 选出的**同一个完整联合模型**，包含 safetensors、配置、自定义模型源码和 tokenizer。
`best_dev_metrics.json` 是该快照的验证结果；`progress.json` 记录当前状态。
训练标签会双向扩展 SIMULTANEOUS／BEGINS-ON，官方 `evaluate.py` 使用原始标注方向；两套指标会分别记录。

```bash
python train_joint.py --predict-only --checkpoint runs/roberta-joint-seed42/best_joint \
  --data-dir data/MAVEN_ERE --split test \
  --prediction-file runs/roberta-joint-seed42/test_predictions.jsonl
```

正式训练结束会自动用最佳模型生成 test 预测。测试金标未公开，所有本地成绩均标为 valid，不宣称测试成绩。
`AutoModel.from_pretrained(checkpoint, trust_remote_code=True)` 可以加载导出的完整模型；使用前审阅导出的自定义源码。
其 forward 需要 `input_ids`、`attention_mask`、`event_spans`、`splits`、`events_idx`。
推荐使用上面的推理入口，以保留完整的分词、span对齐和JSONL解码流程。

## 后续发布

训练代码在 GitHub 发布，HF 只上传模型包。正式训练成功后，`scripts/package_release.py` 可生成
`huggingface/`、训练代码归档和SHA-256清单，不会创建远程仓库或上传文件。
它要求启动时保存的 `source_manifest.json`；提交代码后运行 `python scripts/snapshot_source.py` 生成来源清单。
数据正文、虚拟环境、优化器状态及训练缓存均不加入公开代码仓库。

上游commit：`ac81a9711a69f43f55bfbc50b3bb573fd11c64b0`。
保留GPL-3.0许可证、上游署名及RoBERTa的MIT归属，详见 [UPSTREAM_NOTICE.md](UPSTREAM_NOTICE.md)。

---

# Original MAVEN-ERE Project
Source code and dataset for EMNLP 2022 paper ["MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction"](https://arxiv.org/abs/2211.07342).

## Overview
- MAVEN-ERE is the first unified large-scale dataset for events relation extraction. It contains 4480 documents, with 103,193 coreference chains, 1,216,217 temporal relations, 15,841 subevent relations, and 57,992 causal relations. 
  - Temporal relations include: `BEFORE`, `OVERLAP`, `CONTAINS`, `SIMULTANEOUS`, `BEGINS-ON`, and `ENDS-ON`
  - Causal relations include: `CAUSE` and `PRECONDITION`
- The annotation rule of above relations mainly follows [RED guideline](https://github.com/timjogorman/RicherEventDescription/blob/master/guidelines.md).
- For more details, please see [the paper](https://arxiv.org/abs/2211.07342).
![illustration](https://user-images.githubusercontent.com/55496186/196908089-e240d099-cdfd-4286-9046-e7e9d764562e.png)

## Requirements
```shell
pip install -r requirements.txt
```

## MAVEN-ERE Dataset

### Get the data
The dataset (ver. 1.0) can be obtained from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1) or [Google Drive](https://drive.google.com/file/d/1fxomYO6zPl5DDrDr_HeWFK14s8BpW1z-/view?usp=sharing).

You can also download `MAVEN-ERE` and other processed datasets used in our experiments with the scripts in `/data`:
```bash
cd data
bash download_maven.sh
bash download.sh
cd ../
```
Note that the annotations of the test set are kept unpublic to maintain a fair evaluation environment. To get evaluation results on the test set, you need to submit your predictions to CodaLab following our [instructions](https://github.com/THU-KEG/MAVEN-ERE#get-test-results-from-codalab).

### Data Format
Each `.jsonl` file is a subset of `MAVEN-ERE` and each line in the files is a json string for a document. For the `train.jsonl` and `valid.jsonl`, the json format sample is as below:

```JSON5
{
  "id": "f28bce270df5a122c09365002d247e76", // an unique string for each document
  "title": "United States occupation of Nicaragua", // the tiltle of the document
  "tokens": [ // a list for tokenized document content. each item is a tokenized sentence
    ["The", "United", "States", "occupation", "of", "Nicaragua", "from", 
    "1912", "to", "1933", "was", "part", "of", "the", "Banana", "Wars", ",", 
    "when", "the", "US", "military", "intervened", "in", "various", 
    "Latin", "American", "countries", "from", "1898", "to", "1934", "."],
  ],
  "sentences": [ // untokenized sentences of the document. each item is a sentence (string)
      "The United States occupation of Nicaragua from 1912 to 1933 was part of the Banana Wars, when the US military intervened in various Latin American countries from 1898 to 1934.",
  ],
  "events": [ // a list for annotated events, each item is a dict for an event (coreference chain)
        {
            "id": "EVENT_0341c5cced5545ffe7c543b7a155bfa8", // an unique string for the event (coreference chain)
            "type": "Choosing", // the event type
            "type_id": 25, // the numerical id for the event type, consistent with MAVEN
            "mention": [ // a list for the coreferential event mentions of the chain, each item is a dict. they have coreference relations to each other
                {
                    "id": "a75ba55cadad23555a0ffc9454088687", // an unique string for the event mention
                    "trigger_word": "assumed", // a string of the trigger word or phrase
                    "sent_id": 3, // the index of the corresponding sentence, starts with 0
                    "offset": [1, 2] // the offset of the trigger words in the tokens list
                }
            ]
        }
  ],
  "TIMEX": [ // a list for annotated temporal expressions (TIMEX), each item is a dict for a TIMEX
    {
      "id": "TIME_833e41f3304210094101eca59905055e", // an unique string for the TIMEX
      "mention": "1912", // a string of the mention of the TIMEX
      "type": "DATE", // the type of the TIMEX
      "sent_id": 0, // the index of the corresponding sentence, starts with 0
      "offset": [7, 8] // the offset of the trigger words in the tokens list
    },
  ],
  "temporal_relations": { // a list for annotated temporal relations between events (and TIMEXs)
    "BEFORE": [ // a list for temporal relations of BEFORE type
      ["EVENT_id_1", "TIME_id_2"] // a temporal relation instance, means EVENT_id_1 BEFORE TIME_id_2
    ],
    "OVERLAP": [ // all the following types are similar
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "CONTAINS": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "SIMULTANEOUS": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "ENDS-ON": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
    "BEGINS-ON": [
      ["EVENT_id_1", "EVENT_id_2"]
    ],
  },
  "causal_relations": { // a list for annotated causal relations between events
    "CAUSE": [  // a list for causal relations of CAUSE type
      ["EVENT_id_1", "EVENT_id_2"] // a causal relation instance, means EVENT_id_1 CAUSE EVENT_id_2
    ],
    "PRECONDITION": [ // the PRECONDITION type is similar
      ["EVENT_id_1", "EVENT_id_2"]
    ],
  },
  "subevent_relations": [ // a list for annotated subevent relations between events
    ["EVENT_id_1", "EVENT_id_2"] // a subevent relation instance, means EVENT_id_2 is a subevent of EVENT_id_1
  ]
}
```

- For the `test.jsonl`, the format is similar but we hide the ground truth annotations to organize a fair evaluation challenge. To get evaluation results on the test set, you need to submit the prediction results to our [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/8691#learn_the_details). 
- To avoid leak the test set of the original MAVEN event detection challenge, the candidate event mentions we offered here have more than the golden annotations. But we only evaluate your predictions for the golden event mentions.

```JSON5
{
  "id": "f28bce270df5a122c09365002d247e76", // an unique string for each document
  "title": "United States occupation of Nicaragua", // the tiltle of the document
  "tokens": [ // a list for tokenized document content. each item is a tokenized sentence
    ["The", "United", "States", "occupation", "of", "Nicaragua", "from", 
    "1912", "to", "1933", "was", "part", "of", "the", "Banana", "Wars", ",", 
    "when", "the", "US", "military", "intervened", "in", "various", 
    "Latin", "American", "countries", "from", "1898", "to", "1934", "."],
  ],
  "sentences": [ // untokenized sentences of the document. each item is a sentence (string)
      "The United States occupation of Nicaragua from 1912 to 1933 was part of the Banana Wars, when the US military intervened in various Latin American countries from 1898 to 1934.",
  ],
  "event_mentions": [ // a list for event mentions (and distractors), you need to predict the relations between the given mentions
    {
      "id": "a75ba55cadad23555a0ffc9454088687", // an unique string for the event mention
      "trigger_word": "assumed", // a string of the trigger word or phrase
      "sent_id": 3, // the index of the corresponding sentence, starts with 0
      "offset": [1, 2], // the offset of the trigger words in the tokens list
      "type": "Choosing", // the event type
      "type_id": 25, // the numerical id for the event type, consistent with MAVEN
    }
  ],
  "TIMEX": [ // a list for annotated temporal expressions (TIMEX), each item is a dict for a TIMEX
    {
      "id": "TIME_833e41f3304210094101eca59905055e", // an unique string for the TIMEX
      "mention": "1912", // a string of the mention of the TIMEX
      "type": "DATE", // the type of the TIMEX
      "sent_id": 0, // the index of the corresponding sentence, starts with 0
      "offset": [7, 8] // the offset of the trigger words in the tokens list
    }
  ]
}
```

## How to run experiments
- [Coreference Relation Experiments](coreference/README.md)
- [Temporal Relation Experiments](temporal/README.md)
- [Causal Relation Experiments](causal/README.md)
- [Subevent Relation Experiments](subevent/README.md)
- [Joint Training Experiments](joint/README.md)

## Get Test Results from CodaLab
To get the test results, you can submit your predictions to our permanent [CodaLab competition](https://codalab.lisn.upsaclay.fr/competitions/8691#learn_the_details).

You need to name your result file as `test_prediction.jsonl` and compress it into zip format file named `submission.zip` for submission. Each line in the submission file should be a `json` string encoding the prediction results for one document. The json format is as below:

```JSON5
{
  "id": "f28bce270df5a122c09365002d247e76", // an unique string for each document, mandatory
  "coreference": [ // a list for predicted coreference clusters, each item is a cluster of event mentions having coreference relations with each other
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a list for a predicted cluster, each item is the id of an event mention
  ],
  "temporal_relations": { // a list for predicted temporal relations between event mentions (not events) and TIMEXs
    "BEFORE": [ // a list for predicted temporal relations of BEFORE type
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a temporal relation instance, its items shall be id of event mentions or TIMEXs
    ],
    "OVERLAP": [ // all the following types are similar
      ["a75ba55cadad23555a0ffc9454088687", "TIME_id_1"]
    ],
    "CONTAINS": [
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
    "SIMULTANEOUS": [
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
    "ENDS-ON": [
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
    "BEGINS-ON": [
      ["555a0ffc9454a75ba08868755cadad23", "TIME_id_2"]
    ],
  },
  "causal_relations": { // a list for predicted causal relations between event mentions (not events)
    "CAUSE": [  // a list for causal relations of CAUSE type
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a causal relation instance, its items shall be id of event mentions
    ],
    "PRECONDITION": [ // the PRECONDITION type is similar
      ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"]
    ],
  },
  "subevent_relations": [ // a list for predicted subevent relations between event mention (not events)
    ["a75ba55cadad23555a0ffc9454088687", "555a0ffc9454a75ba08868755cadad23"] // a subevent relation instance, its items shall be id of event mentions
  ]
}
```

For the detailed implementations of our evaluations, please refer to the [evaluation script](evaluate.py).

## Citation
```bibtex
@inproceedings{wang-chen-etal2022MAVENERE,
  title = {MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference, Temporal, Causal, and Subevent Relation Extraction},
  author = {Xiaozhi Wang and Yulin Chen and Ning Ding and Hao Peng and Zimu Wang and Yankai Lin and Xu Han and Lei Hou and Juanzi Li and Zhiyuan Liu and Peng Li and Jie Zhou},
  booktitle = {Proceedings of EMNLP},
  year = {2022},
}
```
## Contact
- wangxz20@mails.tsinghua.edu.cn
- yl-chen21@mails.tsinghua.edu.cn
