# Upstream attribution

This project is adapted from THU-KEG/MAVEN-ERE:
https://github.com/THU-KEG/MAVEN-ERE

Upstream revision: ac81a9711a69f43f55bfbc50b3bb573fd11c64b0.
The original GPL-3.0 LICENSE is retained. Modifications add an independently
serializable four-task model, corrected coreference dummy labels and stable loss,
optimizer-step scheduling, complete checkpoint restoration, official validation,
dataset/base-model provenance, and local release packaging.

The RoBERTa-base foundation weights are provided by FacebookAI:
https://huggingface.co/FacebookAI/roberta-base

The base model card identifies its license as MIT. Retain the foundation model's
copyright and license notices when redistributing adapted weights. Source license:
https://github.com/facebookresearch/fairseq/blob/main/LICENSE

MAVEN-ERE builds on MAVEN and English Wikipedia. The MAVEN-ERE paper describes
MAVEN's CC BY-SA 4.0 and Wikipedia's CC BY-SA 3.0 terms. This repository supplies
official download scripts and hashes rather than redistributing the corpus in
the code or model release.

References:

- Wang et al. 2022. MAVEN-ERE: A Unified Large-scale Dataset for Event Coreference,
  Temporal, Causal, and Subevent Relation Extraction. EMNLP, 926–941.
  https://aclanthology.org/2022.emnlp-main.60/
- Wang et al. 2020. MAVEN: A Massive General Domain Event Detection Dataset. EMNLP.
  https://github.com/THU-KEG/MAVEN-dataset
- Liu et al. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach.
  https://arxiv.org/abs/1907.11692
